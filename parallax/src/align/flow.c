/* Dense flow: rendering through a lattice field, the shared-Hessian inverse-compositional
 * solve, the frozen model, single-image fitting and the population loop. See flow.h. */
#include "align/flow.h"

#include "align/align_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_DOF = 2 * PLX_LAT_MAX_NODES };

plx_fl_params plx_fl_default(void) {
    plx_fl_params p = {
        .cg = plx_cg_default(),
        .weighting = PLX_FL_W_INVVAR,
        .lambda0 = 0.0f,
        .damp = 1.0f,
        .tol = 0.01f,
    };
    return p;
}

/* ---- rendering ------------------------------------------------------------------------ */
/* map (2 channels, Ω+ with `margin`): photo coordinates W(q - margin) of every Ω+ pixel. */
static void warp_map(plx_sim T, const plx_lattice *L, const float *u, int margin, plx_img *map) {
    for (int qy = 0; qy < map->h; qy++)
        for (int qx = 0; qx < map->w; qx++) {
            double x = qx - margin, y = qy - margin, v[2], ix, iy;
            plx_lat_eval(L, u, x, y, v);
            plx_sim_apply(T, x + v[0], y + v[1], &ix, &iy);
            PLX_PX(map, qx, qy, 0) = (float)ix;
            PLX_PX(map, qx, qy, 1) = (float)iy;
        }
}

void plx_fl_render_gray(const plx_img *gray, plx_sim T, const plx_lattice *L, const float *u, int margin,
                        plx_interp interp, plx_img *out, plx_img *valid) {
    PLX_ASSERT(out->w == PLX_CANON_W + 2 * margin && out->h == PLX_CANON_H + 2 * margin && out->c == gray->c);
    plx_img map = plx_canon_new(2, margin);
    warp_map(T, L, u, margin, &map);
    plx_warp_map(gray, &map, interp, out, valid);
    plx_img_free(&map);
}

void plx_fl_render(const plx_img *gray, plx_sim T, const plx_lattice *L, const float *u, const plx_cg_params *p,
                   plx_cg_view *v) {
    PLX_ASSERT(gray->c == 1);
    v->gray = plx_canon_new(1, p->margin);
    v->valid = plx_canon_new(1, p->margin);
    plx_fl_render_gray(gray, T, L, u, p->margin, p->interp, &v->gray, &v->valid);
    plx_cg_view_finish(p, v);
}

plx_img plx_fl_warp_crop(const plx_img *gray, plx_sim T, const plx_lattice *L, const float *u) {
    plx_img out = plx_canon_new(gray->c, 0);
    plx_fl_render_gray(gray, T, L, u, 0, PLX_INTERP_BILINEAR, &out, NULL);
    return out;
}

/* ---- reference -------------------------------------------------------------------------- */
plx_fl_ref plx_fl_ref_build(const plx_cg_template *t, const plx_fl_params *p, const plx_fl_stage *st) {
    const plx_cg_params *cp = &p->cg;
    PLX_ASSERT(st->level >= 0 && st->level < t->levels && t->K == cp->feat.k && t->margin == cp->margin);
    plx_fl_ref r = {0};
    r.st = *st;
    r.lat = plx_lattice_make(st->step);
    int K = t->K, L = st->level, w = t->mean[L].w, h = t->mean[L].h, N = plx_lat_nodes(&r.lat), dof = 2 * N;
    size_t plane = (size_t)w * h;
    r.K = K;
    r.w = w;
    r.h = h;

    plx_img mu = plx_img_clone(&t->mean[L]), var = plx_img_clone(&t->var[L]);
    if (st->blur > 0) {
        plx_blur(&mu, &mu, st->blur);
        plx_blur(&var, &var, st->blur); /* approximates the variance of blurred features */
    }
    plx_img gx = plx_img_new(w, h, K), gy = plx_img_new(w, h, K);
    plx_gradient_scharr(&mu, &gx, &gy);
    float *om = plx_cg_level_window(w, h, L, cp);
    for (size_t o = 0; o < plane; o++) r.n += om[o] > 0;
    r.eps = plx_cg_var_floor(&var, L, cp);

    r.qx = PLX_NEW(float, r.n);
    r.qy = PLX_NEW(float, r.n);
    r.taps = PLX_NEW(plx_lat_taps, r.n);
    r.omega = PLX_NEW(float, r.n);
    r.mu = PLX_NEW(float, (size_t)r.n * K);
    r.inv = PLX_NEW(float, (size_t)r.n * K);
    r.g = PLX_NEW(float, (size_t)r.n * K * 2);
    r.hdata = PLX_NEW(double, dof);
    plx_mat H = plx_mat_new(dof, dof);
    double lev = ldexp(1.0, L); /* level px per full-res px: gradients divide by it */
    int s = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t o = (size_t)y * w + x;
            if (om[o] <= 0) continue;
            double cx = plx_pyr_coord((float)x, L, 0) - cp->margin, cy = plx_pyr_coord((float)y, L, 0) - cp->margin;
            r.qx[s] = (float)x;
            r.qy[s] = (float)y;
            r.omega[s] = om[o];
            r.omega_sum += om[o];
            plx_lat_taps_at(&r.lat, cx, cy, &r.taps[s]);
            double A[3] = {0}; /* Σ_k w_k ∇μ_k ∇μ_kᵀ: xx, xy, yy */
            for (int k = 0; k < K; k++) {
                double ix = 1.0 / (var.px[k * plane + o] + r.eps);
                double wk = om[o] * (p->weighting == PLX_FL_W_INVVAR ? ix : 1.0);
                double ux = gx.px[k * plane + o] / lev, uy = gy.px[k * plane + o] / lev;
                r.mu[(size_t)s * K + k] = mu.px[k * plane + o];
                r.inv[(size_t)s * K + k] = (float)ix;
                r.g[((size_t)s * K + k) * 2] = (float)(wk * ux);
                r.g[((size_t)s * K + k) * 2 + 1] = (float)(wk * uy);
                A[0] += wk * ux * ux;
                A[1] += wk * ux * uy;
                A[2] += wk * uy * uy;
            }
            /* JᵀWJ with J = ∇μ ⊗ B: block (c, n), (c', m) gets B_n B_m A[c][c'] */
            const plx_lat_taps *tp = &r.taps[s];
            for (int a = 0; a < 4; a++)
                for (int b = 0; b < 4; b++) {
                    double bb = (double)tp->w[a] * tp->w[b];
                    if (bb == 0) continue;
                    int na = tp->n[a], nb = tp->n[b];
                    PLX_M(&H, na, nb) += bb * A[0];
                    PLX_M(&H, na, N + nb) += bb * A[1];
                    PLX_M(&H, N + na, nb) += bb * A[1];
                    PLX_M(&H, N + na, N + nb) += bb * A[2];
                }
            s++;
        }
    free(om);
    plx_img_free(&mu);
    plx_img_free(&var);
    plx_img_free(&gx);
    plx_img_free(&gy);

    r.R = plx_mat_new(dof, dof);
    plx_lat_bending_matrix(&r.lat, &r.R);
    double trH = 0, trR = 0;
    for (int k = 0; k < dof; k++) {
        r.hdata[k] = PLX_M(&H, k, k);
        trH += PLX_M(&H, k, k);
        trR += PLX_M(&r.R, k, k);
    }
    r.lambda_eff = st->lambda * trH / trR;
    r.prox_eff = st->prox * trH / trR;
    r.lambda0_eff = p->lambda0 * trH / dof;
    r.damp_eff = p->damp * trH / dof;
    plx_mat_add(&H, r.lambda_eff + r.prox_eff, &r.R);
    for (int k = 0; k < dof; k++) PLX_M(&H, k, k) += r.lambda0_eff + r.damp_eff;
    if (plx_chol(&H) != PLX_OK)
        plx_die("flow: H not positive definite (step %d, level %d, lambda %g): the data cannot pin the affine part",
                st->step, st->level, st->lambda);
    r.chol = H;
    return r;
}

void plx_fl_ref_free(plx_fl_ref *r) {
    free(r->qx);
    free(r->qy);
    free(r->taps);
    free(r->omega);
    free(r->mu);
    free(r->inv);
    free(r->g);
    free(r->hdata);
    plx_mat_free(&r->R);
    plx_mat_free(&r->chol);
    memset(r, 0, sizeof *r);
}

void plx_fl_data_fraction(const plx_fl_ref *r, float *w) {
    int N = plx_lat_nodes(&r->lat), nx = r->lat.nx;
    double *f = PLX_NEW(double, N);
    for (int n = 0; n < N; n++)
        for (int c = 0; c < 2; c++) {
            int k = c * N + n;
            double hd = r->hdata[k];
            f[n] += 0.5 * hd / (hd + r->lambda_eff * PLX_M(&r->R, k, k) + r->lambda0_eff);
        }
    for (int n = 0; n < N; n++) { /* symmetrise: node (i, j) mirrors to (i, nx - 1 - j) */
        int i = n / nx, j = n % nx;
        w[n] = (float)(0.5 * (f[n] + f[i * nx + nx - 1 - j]));
    }
    free(f);
}

/* ---- solve ------------------------------------------------------------------------------ */
plx_fl_fit plx_fl_solve(const plx_fl_ref *r, const plx_cg_view *v, const plx_fl_params *p, const float *u,
                        float *a, const plx_fl_nodestat *ns) {
    int L = r->st.level, K = r->K, w = r->w, h = r->h, N = plx_lat_nodes(&r->lat), dof = 2 * N;
    PLX_ASSERT(L < v->feat.n);
    const plx_img *F = &v->feat.lev[L], *V = &v->vpyr.lev[L];
    PLX_ASSERT(F->w == w && F->h == h && F->c == K);
    plx_img Fb = {0};
    if (r->st.blur > 0) {
        Fb = plx_img_clone(F);
        plx_blur(&Fb, &Fb, r->st.blur);
        F = &Fb;
    }
    size_t plane = (size_t)w * h;
    double inv_lev = ldexp(1.0, -L);
    double c2 = (double)p->cg.huber * p->cg.huber * K;
    /* a: the image-side increment on top of the render field u (the view is sampled at
     * p + a(p)); utot = the total field it implies; a0 = a at entry, the proximal anchor */
    float a0[MAX_DOF], da[MAX_DOF], delta[MAX_DOF], next[MAX_DOF], utot[MAX_DOF];
    double g[MAX_DOF], Ru[MAX_DOF], Ra[MAX_DOF];
    memcpy(a0, a, sizeof(float) * dof);
    float res[64];
    PLX_ASSERT(K <= 64 && dof <= MAX_DOF);
    plx_fl_fit fit = {0};

    for (int it = 0; it < r->st.iters; it++) {
        memset(g, 0, sizeof(double) * dof);
        double sov = 0, cost = 0, inl = 0;
        for (int s = 0; s < r->n; s++) {
            const plx_lat_taps *tp = &r->taps[s];
            double av[2];
            plx_lat_eval_taps(&r->lat, a, tp, av);
            plx_bil_taps t;
            if (!plx_bil_at(w, h, r->qx[s] + av[0] * inv_lev, r->qy[s] + av[1] * inv_lev, &t)) continue;
            float vv = plx_bil_eval(V->px, &t);
            if (vv <= 0) continue;
            const float *mu = r->mu + (size_t)s * K, *inv = r->inv + (size_t)s * K;
            double e2 = 0;
            for (int k = 0; k < K; k++) {
                float d = plx_bil_eval(F->px + k * plane, &t) - mu[k];
                res[k] = d;
                e2 += (double)d * d * inv[k];
            }
            double hw = e2 <= c2 ? 1.0 : sqrt(c2 / e2), ww = hw * vv;
            const float *gk = r->g + (size_t)s * K * 2;
            double ex = 0, ey = 0;
            for (int k = 0; k < K; k++) {
                ex += res[k] * gk[2 * k];
                ey += res[k] * gk[2 * k + 1];
            }
            ex *= ww;
            ey *= ww;
            for (int q = 0; q < 4; q++) {
                g[tp->n[q]] += tp->w[q] * ex;
                g[N + tp->n[q]] += tp->w[q] * ey;
            }
            double ov = r->omega[s] * vv;
            sov += ov;
            cost += ov * e2;
            inl += ov * hw;
            if (ns && it == 0)
                for (int q = 0; q < 4; q++) {
                    ns->cost[tp->n[q]] += tp->w[q] * ov * e2;
                    ns->mass[tp->n[q]] += tp->w[q] * ov;
                }
        }
        fit.iters = it + 1;
        if (sov < 1e-3 * r->omega_sum) break; /* nothing of the photo left in the window */
        fit.cost = cost / sov;
        fit.inlier = inl / sov;
        fit.valid = sov / r->omega_sum;
        if (it == 0) fit.cost0 = fit.cost;

        /* H d = g + λ R utot + λp R (a - a0) + δ (a - a0) + λ0 utot (H carries λ + λp, δ + λ0);
         * image side Δ = -d */
        plx_lat_compose(&r->lat, a, u, utot);
        for (int k = 0; k < dof; k++) da[k] = a[k] - a0[k];
        plx_lat_bending_apply(&r->lat, utot, Ru);
        plx_lat_bending_apply(&r->lat, da, Ra);
        for (int k = 0; k < dof; k++)
            g[k] += r->lambda_eff * Ru[k] + r->prox_eff * Ra[k] + r->damp_eff * da[k] + r->lambda0_eff * utot[k];
        plx_chol_solve_vec(&r->chol, g);
        double step = 0;
        for (int k = 0; k < dof; k++) delta[k] = (float)-g[k];
        for (int n = 0; n < N; n++) {
            double x = plx_lat_x(&r->lat, n % r->lat.nx), y = plx_lat_y(&r->lat, n / r->lat.nx);
            if (x < PLX_LAT_CORE_X0 || x > PLX_LAT_CORE_X1 || y < PLX_LAT_CORE_Y0 || y > PLX_LAT_CORE_Y1) continue;
            step = fmax(step, hypot(delta[n], delta[N + n]));
        }
        plx_lat_compose(&r->lat, delta, a, next);
        memcpy(a, next, sizeof(float) * dof);
        fit.step = step;
        if (step < p->tol) {
            fit.converged = true;
            break;
        }
    }
    plx_img_free(&Fb);
    return fit;
}

/* ---- model ------------------------------------------------------------------------------ */
int plx_fl_model_save(const plx_fl_model *m, const char *dir) {
    if (plx_mkdir_p(dir) != PLX_OK) return PLX_EIO;
    char path[4096];
    snprintf(path, sizeof path, "%s/model.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) return PLX_EIO;
    const plx_cg_params *c = &m->p.cg;
    fprintf(f, "plx_flow_model 1\n");
    fprintf(f, "feat_k %d\nfeat_signed %d\nfeat_sigma_d %.9g\nfeat_sigma_n %.9g\nfeat_sigma_s %.9g\nfeat_eps %.9g\n",
            c->feat.k, (int)c->feat.signed_, c->feat.sigma_d, c->feat.sigma_n, c->feat.sigma_s, c->feat.eps);
    fprintf(f, "margin %d\nlevels %d\ninterp %d\neps_frac %.9g\nhuber %.9g\nramp %.9g\n", c->margin, c->levels,
            (int)c->interp, c->eps_frac, c->huber, c->ramp);
    fprintf(f, "scale_min %.17g\nscale_max %.17g\nrot_max %.17g\nshift_max %.17g\n", c->scale_min, c->scale_max,
            c->rot_max, c->shift_max);
    fprintf(f, "weighting %d\nlambda0 %.9g\ndamp %.9g\ntol %.9g\n", (int)m->p.weighting, m->p.lambda0, m->p.damp,
            m->p.tol);
    fprintf(f, "stages %d\n", m->n_stages);
    for (int s = 0; s < m->n_stages; s++) {
        const plx_fl_stage *st = &m->stages[s];
        fprintf(f, "stage %d %d %.9g %.9g %.9g %d %d\n", st->step, st->level, st->blur, st->lambda, st->prox, st->iters,
                st->rerender);
    }
    fprintf(f, "template_count %lld\n", (long long)m->tmpl.count);
    int bad = fclose(f) != 0;
    for (int l = 0; l < m->tmpl.levels && !bad; l++)
        for (int which = 0; which < 2 && !bad; which++) {
            const plx_img *im = which ? &m->tmpl.var[l] : &m->tmpl.mean[l];
            snprintf(path, sizeof path, "%s/%s_L%d.plxt", dir, which ? "var" : "mean", l);
            int64_t dim[3] = {im->c, im->h, im->w};
            bad = plx_tensor_save(path, PLX_F32, 3, dim, im->px) != PLX_OK;
        }
    return bad ? PLX_EIO : PLX_OK;
}

int plx_fl_model_load(const char *dir, plx_fl_model *m) {
    memset(m, 0, sizeof *m);
    m->p = plx_fl_default();
    char path[4096], key[64];
    snprintf(path, sizeof path, "%s/model.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) return PLX_EIO;
    int version = 0, ok = fscanf(f, "plx_flow_model %d", &version) == 1 && version == 1;
    plx_cg_params *c = &m->p.cg;
    long long count = 0;
    while (ok && fscanf(f, "%63s", key) == 1) {
        int iv;
        double dv;
        if (!strcmp(key, "stage")) {
            plx_fl_stage st;
            ok = m->n_stages < PLX_FL_MAX_STAGES &&
                 fscanf(f, "%d %d %f %f %f %d %d", &st.step, &st.level, &st.blur, &st.lambda, &st.prox, &st.iters,
                        &st.rerender) == 7;
            if (ok) m->stages[m->n_stages++] = st;
        } else if (!strcmp(key, "template_count")) {
            ok = fscanf(f, "%lld", &count) == 1;
        } else if (!strcmp(key, "feat_k") || !strcmp(key, "feat_signed") || !strcmp(key, "margin") ||
                   !strcmp(key, "levels") || !strcmp(key, "interp") || !strcmp(key, "weighting") ||
                   !strcmp(key, "stages")) {
            ok = fscanf(f, "%d", &iv) == 1;
            if (!strcmp(key, "feat_k")) c->feat.k = iv;
            else if (!strcmp(key, "feat_signed")) c->feat.signed_ = iv != 0;
            else if (!strcmp(key, "margin")) c->margin = iv;
            else if (!strcmp(key, "levels")) c->levels = iv;
            else if (!strcmp(key, "interp")) c->interp = (plx_interp)iv;
            else if (!strcmp(key, "weighting")) m->p.weighting = (plx_fl_weighting)iv;
        } else {
            ok = fscanf(f, "%lf", &dv) == 1;
            if (!strcmp(key, "feat_sigma_d")) c->feat.sigma_d = (float)dv;
            else if (!strcmp(key, "feat_sigma_n")) c->feat.sigma_n = (float)dv;
            else if (!strcmp(key, "feat_sigma_s")) c->feat.sigma_s = (float)dv;
            else if (!strcmp(key, "feat_eps")) c->feat.eps = (float)dv;
            else if (!strcmp(key, "eps_frac")) c->eps_frac = (float)dv;
            else if (!strcmp(key, "huber")) c->huber = (float)dv;
            else if (!strcmp(key, "ramp")) c->ramp = (float)dv;
            else if (!strcmp(key, "scale_min")) c->scale_min = dv;
            else if (!strcmp(key, "scale_max")) c->scale_max = dv;
            else if (!strcmp(key, "rot_max")) c->rot_max = dv;
            else if (!strcmp(key, "shift_max")) c->shift_max = dv;
            else if (!strcmp(key, "lambda0")) m->p.lambda0 = (float)dv;
            else if (!strcmp(key, "damp")) m->p.damp = (float)dv;
            else if (!strcmp(key, "tol")) m->p.tol = (float)dv;
            else ok = 0;
        }
    }
    fclose(f);
    if (!ok || m->n_stages < 1 || c->levels < 1 || c->levels > PLX_CG_MAX_LEVELS) return PLX_EFORMAT;
    plx_cg_template *t = &m->tmpl;
    t->K = c->feat.k;
    t->margin = c->margin;
    t->levels = c->levels;
    t->count = count;
    for (int l = 0; l < t->levels; l++)
        for (int which = 0; which < 2; which++) {
            plx_tensor tt;
            snprintf(path, sizeof path, "%s/%s_L%d.plxt", dir, which ? "var" : "mean", l);
            int st = plx_tensor_load(path, &tt);
            if (st != PLX_OK) {
                plx_fl_model_free(m);
                return st;
            }
            if (tt.dtype != PLX_F32 || tt.ndim != 3 || tt.dim[0] != t->K) {
                plx_tensor_free(&tt);
                plx_fl_model_free(m);
                return PLX_EFORMAT;
            }
            plx_img im = plx_img_new((int)tt.dim[2], (int)tt.dim[1], (int)tt.dim[0]);
            memcpy(im.px, tt.data, tt.nbytes);
            plx_tensor_free(&tt);
            if (which) t->var[l] = im;
            else t->mean[l] = im;
        }
    return PLX_OK;
}

void plx_fl_model_free(plx_fl_model *m) {
    plx_cg_template_free(&m->tmpl);
    memset(m, 0, sizeof *m);
}

/* ---- single image ----------------------------------------------------------------------- */
void plx_fl_fitter_init(plx_fl_fitter *f, const plx_fl_model *m) {
    PLX_ASSERT(m->n_stages >= 1 && m->stages[m->n_stages - 1].step == PLX_FL_STEP);
    f->m = m;
    f->refs = PLX_NEW(plx_fl_ref, m->n_stages);
    for (int s = 0; s < m->n_stages; s++) f->refs[s] = plx_fl_ref_build(&m->tmpl, &m->p, &m->stages[s]);
}

void plx_fl_fitter_free(plx_fl_fitter *f) {
    for (int s = 0; s < f->m->n_stages; s++) plx_fl_ref_free(&f->refs[s]);
    free(f->refs);
    memset(f, 0, sizeof *f);
}

/* Per-image state: the field the photo is rendered through (ub) and the increment sampled on
 * top of it (a), both on the current stage's lattice. The full warp is T(p + compose(a, ub)(p)). */
static void state_prepare(plx_lattice *lat, const plx_fl_stage *st, float *ub, float *a) {
    float tmp[MAX_DOF];
    int dof = plx_lat_dof(lat);
    if (st->rerender || st->step != lat->step) {
        plx_lat_compose(lat, a, ub, tmp);
        memcpy(ub, tmp, sizeof(float) * dof);
        memset(a, 0, sizeof(float) * dof);
    }
    if (st->step != lat->step) {
        plx_lattice nl = plx_lattice_make(st->step);
        plx_lat_resample(lat, ub, &nl, tmp);
        memcpy(ub, tmp, sizeof(float) * plx_lat_dof(&nl));
        memset(a, 0, sizeof(float) * plx_lat_dof(&nl));
        *lat = nl;
    }
}

/* Similarity gauge of the state: S fitted to the total field, T <- T o S, ub pulled back (a is
 * unchanged: (id + ub) o (id + a) pulls back as a whole). */
static void state_sim_gauge(const plx_lattice *lat, plx_sim *T, float *ub, const float *a) {
    float tot[MAX_DOF];
    plx_lat_compose(lat, a, ub, tot);
    plx_sim S = plx_lat_fit_sim(lat, tot);
    plx_lat_csim_pullback(lat, S, ub);
    *T = plx_sim_compose(*T, S);
}

/* out (a new view) = v seen through the increment a: features and validity sampled at p + a(p)
 * on every level, i.e. what the solver compares with the template; gray rendered through the
 * total warp. */
static void warp_level(const plx_img *src, int level, int margin, const plx_lattice *L, const float *a, plx_img *dst) {
    int w = src->w, h = src->h;
    size_t plane = (size_t)w * h;
    double inv = ldexp(1.0, -level);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double v[2];
            plx_lat_eval(L, a, plx_pyr_coord((float)x, level, 0) - margin, plx_pyr_coord((float)y, level, 0) - margin, v);
            double sx = PLX_CLAMP(x + v[0] * inv, 0.0, (double)(w - 1)), sy = PLX_CLAMP(y + v[1] * inv, 0.0, (double)(h - 1));
            plx_bil_taps t;
            if (!plx_bil_at(w, h, sx, sy, &t)) PLX_ASSERT(0);
            for (int k = 0; k < src->c; k++) dst->px[k * plane + (size_t)y * w + x] = plx_bil_eval(src->px + k * plane, &t);
        }
}

static void sampled_view(const plx_img *gray, const plx_cg_view *v, plx_sim T, const plx_lattice *L, const float *ub,
                         const float *a, const plx_cg_params *p, plx_cg_view *out) {
    float tot[MAX_DOF];
    plx_lat_compose(L, a, ub, tot);
    memset(out, 0, sizeof *out);
    out->gray = plx_canon_new(1, p->margin);
    out->valid = plx_canon_new(1, p->margin);
    plx_fl_render_gray(gray, T, L, tot, p->margin, p->interp, &out->gray, &out->valid);
    out->feat.n = v->feat.n;
    out->vpyr.n = v->vpyr.n;
    for (int l = 0; l < v->feat.n; l++) {
        out->feat.lev[l] = plx_img_new(v->feat.lev[l].w, v->feat.lev[l].h, v->feat.lev[l].c);
        out->vpyr.lev[l] = plx_img_new(v->vpyr.lev[l].w, v->vpyr.lev[l].h, 1);
        warp_level(&v->feat.lev[l], l, p->margin, L, a, &out->feat.lev[l]);
        warp_level(&v->vpyr.lev[l], l, p->margin, L, a, &out->vpyr.lev[l]);
    }
}

plx_fl_fit plx_fl_fit_image(const plx_fl_fitter *f, const plx_img *gray, plx_sim T0, plx_sim *T, float *u) {
    const plx_fl_model *m = f->m;
    float ub[MAX_DOF] = {0}, a[MAX_DOF] = {0};
    plx_lattice lat = f->refs[0].lat;
    plx_sim Tc = T0;
    plx_fl_fit fit = {0};
    plx_cg_view v = {0};
    bool have_view = false;
    for (int s = 0; s < m->n_stages; s++) {
        const plx_fl_ref *r = &f->refs[s];
        bool fresh = m->stages[s].rerender || m->stages[s].step != lat.step || !have_view;
        state_prepare(&lat, &m->stages[s], ub, a);
        if (fresh) {
            if (have_view) plx_cg_view_free(&v);
            plx_fl_render(gray, Tc, &lat, ub, &m->p.cg, &v);
            have_view = true;
        }
        fit = plx_fl_solve(r, &v, &m->p, ub, a, NULL);
        /* the gauge leaves the render warp T(p + ub(p)) unchanged, so a kept view stays valid */
        state_sim_gauge(&lat, &Tc, ub, a);
    }
    if (have_view) plx_cg_view_free(&v);
    plx_lat_compose(&lat, a, ub, u);
    *T = Tc;
    return fit;
}

/* ---- population ------------------------------------------------------------------------- */
typedef struct {
    const plx_cg_source *src;
    const plx_fl_params *p;
    const plx_fl_ref    *ref;     /* NULL: accumulate only */
    const plx_lattice   *lat;
    const uint8_t       *trusted;
    plx_sim             *T;
    float               *ub, *a;  /* [n * stride] each: render field and sampled increment */
    int                  stride;
    plx_cg_accum        *acc;     /* [threads], NULL in the evaluation pass */
    float               *cost0, *step;
    uint8_t             *conv;
    plx_fl_quality      *q;       /* evaluation pass only */
    plx_fl_nodestat     *ns;      /* [threads], evaluation pass only (may be NULL) */
} round_ctx;

static bool core_xy(double x, double y) {
    return x >= PLX_LAT_CORE_X0 && x <= PLX_LAT_CORE_X1 && y >= PLX_LAT_CORE_Y0 && y <= PLX_LAT_CORE_Y1;
}

static void round_one(int64_t i, int tid, void *pc) {
    round_ctx *c = pc;
    plx_img g = plx_cg_source_image(c->src, (int)i);
    float *ub = c->ub + (size_t)i * c->stride, *a = c->a + (size_t)i * c->stride;
    int N = plx_lat_nodes(c->lat), dof = 2 * N;
    bool sampled = false;
    for (int k = 0; k < dof && !sampled; k++) sampled = a[k] != 0;
    plx_cg_view v;
    plx_fl_render(&g, c->T[i], c->lat, ub, &c->p->cg, &v);
    if (c->acc && (!c->trusted || c->trusted[i])) {
        if (sampled) { /* the template sees what the solver compares with it */
            plx_cg_view sv;
            sampled_view(&g, &v, c->T[i], c->lat, ub, a, &c->p->cg, &sv);
            plx_cg_accum_add(&c->acc[tid], &sv);
            plx_cg_view_free(&sv);
        } else {
            plx_cg_accum_add(&c->acc[tid], &v);
        }
    }
    if (c->ref) {
        if (c->q) { /* evaluation: measure on a copy, do not move */
            float a2[MAX_DOF], tot[MAX_DOF];
            memcpy(a2, a, sizeof(float) * dof);
            plx_fl_fit f = plx_fl_solve(c->ref, &v, c->p, ub, a2, c->ns ? &c->ns[tid] : NULL);
            plx_lat_compose(c->lat, a, ub, tot);
            float d = 0;
            for (int k = 0; k < N; k++)
                if (core_xy(plx_lat_x(c->lat, k % c->lat->nx), plx_lat_y(c->lat, k / c->lat->nx)))
                    d = fmaxf(d, hypotf(a2[k] - a[k], a2[N + k] - a[N + k]));
            c->q[i].cost = (float)f.cost0;
            c->q[i].inlier = (float)f.inlier;
            c->q[i].valid = (float)f.valid;
            c->q[i].step = d;
            c->q[i].min_jdet = (float)plx_lat_min_jdet(c->lat, tot);
            c->q[i].max_u = (float)plx_lat_max_abs(c->lat, tot);
        } else {
            plx_fl_fit f = plx_fl_solve(c->ref, &v, c->p, ub, a, NULL);
            c->cost0[i] = (float)f.cost0;
            c->step[i] = (float)f.step;
            c->conv[i] = f.converged;
        }
    }
    plx_cg_view_free(&v);
    plx_img_free(&g);
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static float median_f(const float *v, int n) {
    float *c = PLX_NEW(float, n);
    memcpy(c, v, sizeof(float) * n);
    qsort(c, n, sizeof(float), cmp_float);
    float m = c[n / 2];
    free(c);
    return m;
}

/* Similarity gauge per image, then the symmetrised population-mean total field is subtracted
 * (from the increment: to first order the same), then congealing's scale / ty gauge on T.
 * Returns the mean RMS |u| over core nodes. */
static double apply_gauges(const plx_lattice *L, plx_sim *T, float *ub, float *a, int stride, int n,
                           const uint8_t *trusted) {
    int N = plx_lat_nodes(L), dof = 2 * N, nt = 0;
    double mean[MAX_DOF] = {0};
    float tot[MAX_DOF];
    for (int i = 0; i < n; i++) {
        float *ubi = ub + (size_t)i * stride, *ai = a + (size_t)i * stride;
        state_sim_gauge(L, &T[i], ubi, ai);
        if (trusted && !trusted[i]) continue;
        plx_lat_compose(L, ai, ubi, tot);
        for (int k = 0; k < dof; k++) mean[k] += tot[k];
        nt++;
    }
    float mf[MAX_DOF] = {0}, mm[MAX_DOF];
    for (int k = 0; k < dof; k++) mf[k] = (float)(mean[k] / PLX_MAX(nt, 1));
    plx_lat_mirror(L, mf, mm);
    for (int k = 0; k < dof; k++) mf[k] = 0.5f * (mf[k] + mm[k]);
    double rms = 0;
    for (int i = 0; i < n; i++) {
        float *ubi = ub + (size_t)i * stride, *ai = a + (size_t)i * stride;
        for (int k = 0; k < dof; k++) ai[k] -= mf[k];
        plx_lat_compose(L, ai, ubi, tot);
        double s = 0;
        int c = 0;
        for (int k = 0; k < N; k++)
            if (core_xy(plx_lat_x(L, k % L->nx), plx_lat_y(L, k / L->nx)))
                s += (double)tot[k] * tot[k] + (double)tot[N + k] * tot[N + k], c++;
        rms += sqrt(s / c) / n;
    }
    plx_cg_gauge_fix(T, n);
    return rms;
}

void plx_fl_congeal(const plx_cg_source *src, const plx_fl_params *p, const plx_fl_stage *sched, int n_stages,
                    const uint8_t *trusted, plx_sim *T, float *u, plx_cg_template *final_template,
                    plx_fl_round *rounds, plx_fl_quality *q, const plx_fl_nodestat *ns_out) {
    PLX_ASSERT(n_stages >= 1 && sched[n_stages - 1].step == PLX_FL_STEP);
    int n = src->n, nt = plx_num_threads();
    plx_lattice fine = plx_lattice_make(PLX_FL_STEP);
    int stride = plx_lat_dof(&fine);
    plx_lattice lat = plx_lattice_make(sched[0].step);
    float *ub = PLX_NEW(float, (size_t)n * stride), *a = PLX_NEW(float, (size_t)n * stride);
    plx_cg_accum *acc = PLX_NEW(plx_cg_accum, nt);
    float *cost0 = PLX_NEW(float, n), *step = PLX_NEW(float, n);
    uint8_t *conv = PLX_NEW(uint8_t, n);
    plx_cg_template tmpl = {0};

    for (int r = 0; r <= n_stages + 1; r++) {
        double t0 = plx_now();
        const plx_fl_stage *st = r >= 1 && r <= n_stages ? &sched[r - 1] : NULL;
        if (st) {
            plx_lattice nl = lat;
            for (int i = 0; i < n; i++) {
                nl = lat;
                state_prepare(&nl, st, ub + (size_t)i * stride, a + (size_t)i * stride);
            }
            lat = nl;
        }
        plx_fl_ref ref = {0};
        if (st) ref = plx_fl_ref_build(&tmpl, p, st);
        for (int t = 0; t < nt; t++) plx_cg_accum_init(&acc[t], &p->cg);
        round_ctx c = {.src = src, .p = p, .ref = st ? &ref : NULL, .lat = &lat, .trusted = trusted, .T = T, .ub = ub,
                       .a = a, .stride = stride, .acc = acc, .cost0 = cost0, .step = step, .conv = conv};
        plx_parallel_for(n, 8, round_one, &c);
        for (int t = 1; t < nt; t++) plx_cg_accum_merge(&acc[0], &acc[t]);
        plx_cg_template_free(&tmpl);
        plx_fl_round *ri = &rounds[r];
        memset(ri, 0, sizeof *ri);
        plx_cg_accum_finish(&acc[0], &p->cg, &tmpl, &ri->mean_gray);
        for (int t = 0; t < nt; t++) plx_cg_accum_free(&acc[t]);
        plx_cg_core_variance(&tmpl, &ri->core_var, &ri->core_ratio);
        if (st) {
            ri->mean_rms_u = apply_gauges(&lat, T, ub, a, stride, n, trusted);
            double sc = 0;
            for (int i = 0; i < n; i++) {
                sc += cost0[i];
                ri->n_unconverged += !conv[i];
            }
            ri->mean_cost = sc / n;
            ri->median_step = median_f(step, n);
            plx_fl_ref_free(&ref);
        }
        ri->seconds = plx_now() - t0;
        plx_log("flow round %d%s: step %d level %d blur %.2f lambda %.3g%s  core var %.2f (ratio %.4f)  cost %.3f"
                "  median last step %.3f px  rms|u| %.3f px  unconverged %d  (%.1fs)",
                r, st ? "" : " (accumulate)", st ? st->step : 0, st ? st->level : -1, st ? st->blur : 0.0f,
                st ? st->lambda : 0.0f, st && !st->rerender ? " sampled" : "", ri->core_var, ri->core_ratio,
                ri->mean_cost, ri->median_step, ri->mean_rms_u, ri->n_unconverged, ri->seconds);
    }

    /* evaluation pass against the final template with the last stage's setting */
    double t0 = plx_now();
    plx_fl_ref ref = plx_fl_ref_build(&tmpl, p, &sched[n_stages - 1]);
    plx_fl_nodestat *ns = NULL;
    int N = plx_lat_nodes(&fine);
    if (ns_out) {
        ns = PLX_NEW(plx_fl_nodestat, nt);
        for (int t = 0; t < nt; t++) ns[t] = (plx_fl_nodestat){PLX_NEW(double, N), PLX_NEW(double, N)};
    }
    round_ctx c = {.src = src, .p = p, .ref = &ref, .lat = &lat, .T = T, .ub = ub, .a = a, .stride = stride, .q = q,
                   .ns = ns};
    plx_parallel_for(n, 8, round_one, &c);
    for (int i = 0; i < n; i++) {
        q[i].converged = conv[i];
        plx_lat_compose(&lat, a + (size_t)i * stride, ub + (size_t)i * stride, u + (size_t)i * stride);
    }
    if (ns) {
        for (int t = 0; t < nt; t++) {
            for (int k = 0; k < N; k++) ns_out->cost[k] += ns[t].cost[k], ns_out->mass[k] += ns[t].mass[k];
            free(ns[t].cost);
            free(ns[t].mass);
        }
        free(ns);
    }
    plx_fl_ref_free(&ref);
    plx_log("flow evaluation pass (%.1fs)", plx_now() - t0);

    *final_template = tmpl;
    free(ub);
    free(a);
    free(acc);
    free(cost0);
    free(step);
    free(conv);
}
