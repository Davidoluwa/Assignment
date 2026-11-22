/* Depth relief: the population regression of residual horizontal misalignment on yaw, the yaw
 * refinement, and the synthetic known-relief population. See relief.h. */
#include "geom/relief.h"

#include "align/flow.h"
#include "la/la.h"
#include "render/render.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { NLAT = PLX_RL_LAT_NODES, DOF = 2 * NLAT };

plx_rl_params plx_rl_default(const plx_cg_params *cg) {
    plx_rl_params p = {.cg = *cg, .n_stages = 3, .lambda = 0.1f, .eta = 0.8f, .core = plx_core_default()};
    p.stages[0] = (plx_rl_stage){.level = 1, .sigma = 4.0f, .iters = 2};
    p.stages[1] = (plx_rl_stage){.level = 0, .sigma = 2.0f, .iters = 3};
    p.stages[2] = (plx_rl_stage){.level = 0, .sigma = 1.5f, .iters = 3};
    return p;
}

void plx_rl_result_free(plx_rl_result *r) {
    plx_img_free(&r->w);
    plx_img_free(&r->support);
    plx_img_free(&r->se);
    plx_img_free(&r->mean_gray);
    plx_cg_template_free(&r->tmpl);
    memset(r, 0, sizeof *r);
}

/* ---- helpers ----------------------------------------------------------------------------- */
plx_img plx_rl_v_dense(const float *v_lat) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    float f[DOF] = {0};
    memcpy(f, v_lat, sizeof(float) * NLAT);
    plx_img v = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++) {
            double e[2];
            plx_lat_eval(&L, f, x, y, e);
            PLX_PX(&v, x, y, 0) = (float)e[0];
        }
    return v;
}

bool plx_rl_in_core(plx_core_shape cs, int x, int y) {
    double ex = (x - PLX_CANON_CX) / cs.ax, ey = (y - PLX_CANON_CY - cs.yc) / cs.ay;
    return ex * ex + ey * ey <= 1.0;
}

void plx_rl_gauge(plx_img *w, const plx_img *v, const plx_img *wt, plx_core_shape cs, double coef[3]) {
    plx_mat A = plx_mat_new(3, 3);
    double b[3] = {0}, tr = 0;
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++) {
            if (!plx_rl_in_core(cs, x, y)) continue;
            double g = wt ? PLX_PX(wt, x, y, 0) : 1.0, f[3] = {1, y - PLX_CANON_CY, PLX_PX(v, x, y, 0)};
            for (int a = 0; a < 3; a++) {
                b[a] += g * f[a] * PLX_PX(w, x, y, 0);
                for (int c = 0; c < 3; c++) PLX_M(&A, a, c) += g * f[a] * f[c];
            }
        }
    for (int a = 0; a < 3; a++) tr += PLX_M(&A, a, a);
    for (int a = 0; a < 3; a++) PLX_M(&A, a, a) += 1e-12 * tr + 1e-300;
    PLX_ASSERT(plx_chol(&A) == PLX_OK);
    plx_chol_solve_vec(&A, b);
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++)
            PLX_PX(w, x, y, 0) -= (float)(b[0] + b[1] * (y - PLX_CANON_CY) + b[2] * PLX_PX(v, x, y, 0));
    if (coef) memcpy(coef, b, sizeof b);
    plx_mat_free(&A);
}

/* a minus its normalised Gaussian blur inside the core (0 outside). */
static plx_img highpass(const plx_img *a, plx_core_shape cs, float sigma) {
    plx_img m = plx_img_new(a->w, a->h, 1), am = plx_img_new(a->w, a->h, 1), out = plx_img_new(a->w, a->h, 1);
    for (int y = 0; y < a->h; y++)
        for (int x = 0; x < a->w; x++)
            if (plx_rl_in_core(cs, x, y)) PLX_PX(&m, x, y, 0) = 1, PLX_PX(&am, x, y, 0) = PLX_PX(a, x, y, 0);
    plx_blur(&m, &m, sigma);
    plx_blur(&am, &am, sigma);
    for (int y = 0; y < a->h; y++)
        for (int x = 0; x < a->w; x++)
            if (plx_rl_in_core(cs, x, y))
                PLX_PX(&out, x, y, 0) = PLX_PX(a, x, y, 0) - PLX_PX(&am, x, y, 0) / PLX_PX(&m, x, y, 0);
    plx_img_free(&m);
    plx_img_free(&am);
    return out;
}

static double core_corr(const plx_img *a, const plx_img *b, plx_core_shape cs) {
    double n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (int y = 0; y < a->h; y++)
        for (int x = 0; x < a->w; x++) {
            if (!plx_rl_in_core(cs, x, y)) continue;
            double p = PLX_PX(a, x, y, 0), q = PLX_PX(b, x, y, 0);
            n++, sa += p, sb += q, saa += p * p, sbb += q * q, sab += p * q;
        }
    double va = saa - sa * sa / n, vb = sbb - sb * sb / n;
    return va > 0 && vb > 0 ? (sab - sa * sb / n) / sqrt(va * vb) : 0;
}

void plx_rl_compare(const plx_img *a, const plx_img *b, plx_core_shape cs, float hp_sigma, double *corr,
                    double *corr_hp, double *rms) {
    if (corr) *corr = core_corr(a, b, cs);
    if (corr_hp) {
        plx_img ha = highpass(a, cs, hp_sigma), hb = highpass(b, cs, hp_sigma);
        *corr_hp = core_corr(&ha, &hb, cs);
        plx_img_free(&ha);
        plx_img_free(&hb);
    }
    if (rms) {
        double s = 0;
        int n = 0;
        for (int y = 0; y < a->h; y++)
            for (int x = 0; x < a->w; x++)
                if (plx_rl_in_core(cs, x, y)) {
                    double d = PLX_PX(a, x, y, 0) - PLX_PX(b, x, y, 0);
                    s += d * d, n++;
                }
        *rms = sqrt(s / n);
    }
}

static plx_cg_template template_clone(const plx_cg_template *t) {
    plx_cg_template c = *t;
    for (int l = 0; l < t->levels; l++) {
        c.mean[l] = plx_img_clone(&t->mean[l]);
        c.var[l] = plx_img_clone(&t->var[l]);
    }
    return c;
}

/* Ω map -> Ω+ (replicate-extended) -> level L of the pyramid registration (area average). */
static plx_img omega_to_level(const plx_img *a, int margin, int w, int h) {
    plx_img ext = plx_canon_new(1, margin);
    for (int y = 0; y < ext.h; y++)
        for (int x = 0; x < ext.w; x++)
            PLX_PX(&ext, x, y, 0) = PLX_PX(a, PLX_CLAMP(x - margin, 0, PLX_CANON_W - 1), PLX_CLAMP(y - margin, 0, PLX_CANON_H - 1), 0);
    if (ext.w == w && ext.h == h) return ext;
    plx_img out = plx_resize(&ext, w, h);
    plx_img_free(&ext);
    return out;
}

/* level-L Ω+ plane (double) -> Ω, bilinear (the pyramid's pixel-area registration). */
static plx_img level_to_omega(const double *f, int w, int h, int margin) {
    plx_img lv = plx_img_new(w, h, 1);
    for (size_t o = 0; o < (size_t)w * h; o++) lv.px[o] = (float)f[o];
    int W0 = PLX_CANON_W + 2 * margin, H0 = PLX_CANON_H + 2 * margin;
    plx_img full = lv.w == W0 && lv.h == H0 ? plx_img_clone(&lv) : plx_resize(&lv, W0, H0);
    plx_img out = plx_canon_crop(&full, margin);
    plx_img_free(&lv);
    plx_img_free(&full);
    return out;
}

/* ---- one pass ----------------------------------------------------------------------------- */
/* The template prepared for the regression at one pyramid level. */
typedef struct {
    int    L, w, h, K;
    size_t plane;
    double lev;      /* 2^L: level px per full-res px */
    float  sig;      /* window, level px */
    float *mu, *inv; /* [K * plane] planar */
    float *gmu;      /* [K * plane] ∂x μ, full-resolution units */
    float *omega;    /* [plane] congealing's window */
    float *vhr;      /* [plane] v + w (refinement only) */
} level_ref;

static level_ref ref_build(const plx_cg_template *t, const plx_rl_params *p, int L, float sigma, const plx_img *vhr) {
    PLX_ASSERT(L >= 0 && L < t->levels && t->K == p->cg.feat.k && t->margin == p->cg.margin);
    level_ref r = {.L = L, .w = t->mean[L].w, .h = t->mean[L].h, .K = t->K, .lev = ldexp(1.0, L)};
    r.plane = (size_t)r.w * r.h;
    r.sig = (float)(sigma / r.lev);
    r.mu = PLX_NEW(float, r.plane * r.K);
    r.inv = PLX_NEW(float, r.plane * r.K);
    memcpy(r.mu, t->mean[L].px, sizeof(float) * r.plane * r.K);
    double eps = plx_cg_var_floor(&t->var[L], L, &p->cg);
    for (size_t k = 0; k < r.plane * r.K; k++) r.inv[k] = (float)(1.0 / (t->var[L].px[k] + eps));
    plx_img gx = plx_img_new(r.w, r.h, r.K), gy = plx_img_new(r.w, r.h, r.K);
    plx_gradient_scharr(&t->mean[L], &gx, &gy);
    r.gmu = PLX_NEW(float, r.plane * r.K);
    for (size_t k = 0; k < r.plane * r.K; k++) r.gmu[k] = (float)(gx.px[k] / r.lev);
    plx_img_free(&gx);
    plx_img_free(&gy);
    r.omega = plx_cg_level_window(r.w, r.h, L, &p->cg);
    if (vhr) {
        plx_img lv = omega_to_level(vhr, p->cg.margin, r.w, r.h);
        r.vhr = PLX_NEW(float, r.plane);
        memcpy(r.vhr, lv.px, sizeof(float) * r.plane);
        plx_img_free(&lv);
    }
    return r;
}

static void ref_free(level_ref *r) {
    free(r->mu), free(r->inv), free(r->gmu), free(r->omega), free(r->vhr);
    memset(r, 0, sizeof *r);
}

enum { NSTAT = 3 }; /* Σ ω v, Σ ω v e², Σ ω v h */

typedef struct {
    const plx_rl_data   *d;
    const plx_rl_params *p;
    const plx_img       *w;
    const level_ref     *ref;  /* NULL: accumulate only */
    plx_cg_accum        *acc;  /* [threads], NULL: no template */
    double              *N, *D, *Q, *stat; /* [threads * plane] (stat: [threads * NSTAT]) */
    float               *ds;   /* refinement: per-photo step; NULL: relief regression */
} pass_ctx;

/* One photo's terms of steps 2-3 (relief.h): its numerator map n_i into N (and (G * (n_i + n_i∘σ))²
 * into Q), its normaliser into D; or, when refining, its own yaw step. */
static void regress(const pass_ctx *c, int64_t i, int tid, const plx_cg_view *v) {
    const level_ref *r = c->ref;
    int K = r->K, w = r->w, h = r->h;
    size_t plane = r->plane;
    const plx_img *F = &v->feat.lev[r->L], *V = &v->vpyr.lev[r->L];
    PLX_ASSERT(F->w == w && F->h == h && F->c == K);
    double s = c->d->s[i], c2 = (double)c->p->cg.huber * c->p->cg.huber * K, gn = 0, gd = 0;
    double *st = c->stat + (size_t)tid * NSTAT;
    plx_img m = c->ds ? (plx_img){0} : plx_img_new(w, h, 1);
    for (size_t o = 0; o < plane; o++) {
        float vv = V->px[o];
        if (r->omega[o] <= 0 || vv <= 0) continue;
        double e2 = 0, nn = 0, dd = 0;
        for (int k = 0; k < K; k++) {
            double d = F->px[k * plane + o] - r->mu[k * plane + o];
            e2 += d * d * r->inv[k * plane + o];
        }
        double hw = e2 <= c2 ? 1.0 : sqrt(c2 / e2), wt = r->omega[o] * vv;
        for (int k = 0; k < K; k++) { /* the residual enters Huber-clipped; ∂x μ is the instrument */
            double gm = r->gmu[k * plane + o], d = F->px[k * plane + o] - r->mu[k * plane + o];
            nn += r->inv[k * plane + o] * gm * hw * d;
            dd += r->inv[k * plane + o] * gm * gm;
        }
        st[0] += wt, st[1] += wt * e2, st[2] += wt * hw;
        if (c->ds) { /* b = v_hr ∂x μ */
            gn += wt * r->vhr[o] * nn;
            gd += wt * r->vhr[o] * r->vhr[o] * dd;
        } else { /* b = s ∂x μ */
            m.px[o] = (float)(wt * s * nn);
            c->D[(size_t)tid * plane + o] += wt * s * s * dd;
        }
    }
    if (c->ds) {
        c->ds[i] = gd > 0 ? (float)(-gn / gd) : 0.0f;
    } else {
        double *N = c->N + (size_t)tid * plane, *Q = c->Q + (size_t)tid * plane;
        for (size_t o = 0; o < plane; o++) N[o] += m.px[o];
        /* the photo's own contribution to the symmetrised, windowed numerator, squared */
        plx_img ms = plx_img_new(w, h, 1);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) PLX_PX(&ms, x, y, 0) = PLX_PX(&m, x, y, 0) + PLX_PX(&m, w - 1 - x, y, 0);
        plx_blur(&ms, &ms, r->sig);
        for (size_t o = 0; o < plane; o++) Q[o] += (double)ms.px[o] * ms.px[o];
        plx_img_free(&ms);
    }
    plx_img_free(&m);
}

static void pass_one(int64_t i, int tid, void *pc) {
    const pass_ctx *c = pc;
    const plx_rl_data *d = c->d;
    if (d->use && !d->use[i]) return;
    plx_lattice lat = plx_lattice_make(PLX_FL_STEP);
    float u[DOF];
    memcpy(u, d->u + (size_t)i * DOF, sizeof u);
    if (d->s_shift)
        for (int k = 0; k < NLAT; k++) u[k] += d->s_shift[i] * d->v_lat[k];
    plx_img g = plx_cg_source_image(d->src, (int)i);
    plx_cg_view v;
    plx_fl_render_relief(&g, d->T[i], &lat, u, d->s[i], c->w, &c->p->cg, &v);
    if (c->acc) plx_cg_accum_add(&c->acc[tid], &v);
    if (c->ref) regress(c, i, tid, &v);
    plx_cg_view_free(&v);
    plx_img_free(&g);
}

typedef struct {
    plx_cg_template tmpl;      /* when accumulating */
    plx_img         mean_gray;
    double         *N, *D, *Q; /* [plane], merged (relief regression) */
    double          stat[NSTAT];
} pass_out;

static void run_pass(const plx_rl_data *d, const plx_rl_params *p, const plx_img *w, const level_ref *ref,
                     bool accumulate, float *ds, pass_out *po) {
    int nt = plx_num_threads();
    size_t plane = ref ? ref->plane : 0;
    memset(po, 0, sizeof *po);
    pass_ctx c = {.d = d, .p = p, .w = w, .ref = ref, .ds = ds};
    if (accumulate) {
        c.acc = PLX_NEW(plx_cg_accum, nt);
        for (int t = 0; t < nt; t++) plx_cg_accum_init(&c.acc[t], &p->cg);
    }
    if (ref) {
        c.N = PLX_NEW(double, plane * nt);
        c.D = PLX_NEW(double, plane * nt);
        c.Q = PLX_NEW(double, plane * nt);
        c.stat = PLX_NEW(double, (size_t)NSTAT * nt);
    }
    plx_parallel_for(d->src->n, 8, pass_one, &c);
    if (accumulate) {
        for (int t = 1; t < nt; t++) plx_cg_accum_merge(&c.acc[0], &c.acc[t]);
        plx_cg_accum_finish(&c.acc[0], &p->cg, &po->tmpl, &po->mean_gray);
        for (int t = 0; t < nt; t++) plx_cg_accum_free(&c.acc[t]);
        free(c.acc);
    }
    if (ref) {
        for (int t = 1; t < nt; t++) {
            for (size_t o = 0; o < plane; o++) {
                c.N[o] += c.N[t * plane + o];
                c.D[o] += c.D[t * plane + o];
                c.Q[o] += c.Q[t * plane + o];
            }
            for (int k = 0; k < NSTAT; k++) c.stat[k] += c.stat[t * NSTAT + k];
        }
        po->N = c.N, po->D = c.D, po->Q = c.Q;
        memcpy(po->stat, c.stat, sizeof po->stat);
        free(c.stat);
    }
}

/* ---- the update (steps 4-6) ---------------------------------------------------------------- */
static double level_core_mean(const double *f, const level_ref *r, int margin, plx_core_shape cs) {
    double s = 0;
    int n = 0;
    for (int y = 0; y < r->h; y++)
        for (int x = 0; x < r->w; x++) {
            int cx = (int)lround(plx_pyr_coord((float)x, r->L, 0)) - margin, cy = (int)lround(plx_pyr_coord((float)y, r->L, 0)) - margin;
            if (plx_rl_in_core(cs, cx, cy)) s += f[(size_t)y * r->w + x], n++;
        }
    return n ? s / n : 0;
}

/* Δw (Ω), support G*D (Ω) and the standard error (Ω) from a pass's sums at the current w; returns λ.
 * The penalty is on the TOTAL layer, min Σ (r + b Δw)² + λ (w + Δw)², so
 *     Δw = -(G*N + λ w) / (G*D + λ):
 * at the fixed point G*N = -λ w, i.e. w is shrunk towards 0 by (G*D)/(G*D + λ) where the data
 * are weak. (λ on the increment alone, Δw = -G*N/(G*D + λ), would only slow the iteration down:
 * its fixed point is G*N = 0 whatever λ, and unsupported pixels would random-walk.) */
static double solve_step(const pass_out *po, const level_ref *r, const plx_rl_params *p, const plx_img *wcur,
                         plx_img *dw, plx_img *support, plx_img *se) {
    int w = r->w, h = r->h;
    size_t plane = r->plane;
    double *gn = PLX_NEW(double, plane), *gd = PLX_NEW(double, plane), *q = PLX_NEW(double, plane);
    plx_img n = plx_img_new(w, h, 1), dd = plx_img_new(w, h, 1);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t o = (size_t)y * w + x, om = (size_t)y * w + (w - 1 - x);
            n.px[o] = (float)(po->N[o] + po->N[om]); /* the mirror twins (step 4) */
            dd.px[o] = (float)(po->D[o] + po->D[om]);
        }
    plx_blur(&n, &n, r->sig);
    plx_blur(&dd, &dd, r->sig);
    for (size_t o = 0; o < plane; o++) gn[o] = n.px[o], gd[o] = fmax(dd.px[o], 0.0), q[o] = sqrt(po->Q[o]);
    double lam = p->lambda * level_core_mean(gd, r, p->cg.margin, p->core);
    plx_img GN = level_to_omega(gn, w, h, p->cg.margin), Q = level_to_omega(q, w, h, p->cg.margin);
    *support = level_to_omega(gd, w, h, p->cg.margin);
    size_t np = (size_t)PLX_CANON_W * PLX_CANON_H;
    for (size_t k = 0; k < np; k++) support->px[k] = fmaxf(support->px[k], 0.0f);
    /* the prior target: w's support-weighted smooth extension (0 = plain shrinkage to zero) */
    plx_img prior = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    if (p->prior_sigma > 0) {
        plx_img dwt = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1), dd2 = plx_img_clone(support);
        for (size_t k = 0; k < np; k++) dwt.px[k] = support->px[k] * wcur->px[k];
        plx_blur(&dwt, &dwt, p->prior_sigma);
        plx_blur(&dd2, &dd2, p->prior_sigma);
        for (size_t k = 0; k < np; k++) prior.px[k] = (float)(dwt.px[k] / (dd2.px[k] + lam));
        plx_img_free(&dwt), plx_img_free(&dd2);
    }
    *dw = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    *se = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    for (size_t k = 0; k < np; k++) {
        double den = support->px[k] + lam;
        dw->px[k] = (float)(-(GN.px[k] + lam * (wcur->px[k] - prior.px[k])) / den);
        se->px[k] = (float)(Q.px[k] / den);
    }
    plx_img_free(&prior);
    free(gn), free(gd), free(q);
    plx_img_free(&n), plx_img_free(&dd), plx_img_free(&GN), plx_img_free(&Q);
    return lam;
}

static void core_rms(const plx_img *a, plx_core_shape cs, double *rms, double *mx) {
    double s = 0, m = 0;
    int n = 0;
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++)
            if (plx_rl_in_core(cs, x, y)) {
                double v = PLX_PX(a, x, y, 0);
                s += v * v, m = fmax(m, fabs(v)), n++;
            }
    *rms = sqrt(s / n);
    if (mx) *mx = m;
}

static void set_template(plx_rl_result *out, pass_out *po) {
    plx_cg_template_free(&out->tmpl);
    plx_img_free(&out->mean_gray);
    out->tmpl = po->tmpl;
    out->mean_gray = po->mean_gray;
    plx_cg_core_variance(&out->tmpl, &out->core_var, &out->core_ratio);
    out->sharpness = plx_cg_sharpness(&out->mean_gray);
}

void plx_rl_estimate(const plx_rl_data *d, const plx_rl_params *p, const plx_cg_template *init, const plx_img *w0,
                     bool final_accumulate, plx_rl_result *out) {
    memset(out, 0, sizeof *out);
    PLX_ASSERT(p->eta > 0 && p->eta <= 1);
    out->w = w0 ? plx_img_clone(w0) : plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    plx_img v = plx_rl_v_dense(d->v_lat);
    pass_out po;
    if (init) {
        out->tmpl = template_clone(init);
    } else {
        run_pass(d, p, &out->w, NULL, true, NULL, &po);
        set_template(out, &po);
    }
    for (int st = 0; st < p->n_stages; st++)
        for (int it = 0; it < p->stages[st].iters; it++) {
            PLX_ASSERT(out->n_iter < PLX_RL_MAX_ITERS);
            const plx_rl_stage *sg = &p->stages[st];
            plx_rl_iter *ri = &out->iter[out->n_iter++];
            double t0 = plx_now();
            level_ref ref = ref_build(&out->tmpl, p, sg->level, sg->sigma, NULL);
            run_pass(d, p, &out->w, &ref, true, NULL, &po);
            set_template(out, &po);
            plx_img dw;
            plx_img_free(&out->support);
            plx_img_free(&out->se);
            ri->lambda_abs = solve_step(&po, &ref, p, &out->w, &dw, &out->support, &out->se);
            for (size_t k = 0; k < (size_t)dw.w * dw.h; k++) out->w.px[k] += p->eta * dw.px[k];
            {
                plx_img wg = plx_img_clone(&out->w);
                plx_rl_gauge(&wg, &v, &out->support, p->core, ri->gauge);
                plx_img_free(&wg);
            }
            for (size_t k = 0; k < (size_t)dw.w * dw.h; k++) dw.px[k] *= p->eta;
            core_rms(&dw, p->core, &ri->step_rms, &ri->step_max);
            core_rms(&out->w, p->core, &ri->w_rms, NULL);
            ri->stage = st, ri->level = sg->level, ri->sigma = sg->sigma;
            ri->core_var = out->core_var, ri->core_ratio = out->core_ratio, ri->sharpness = out->sharpness;
            ri->cost = po.stat[1] / po.stat[0];
            ri->inlier = po.stat[2] / po.stat[0];
            ri->seconds = plx_now() - t0;
            plx_log("relief pass %d (stage %d, level %d, sigma %.2f): template core var %.2f sharpness %.3f cost %.3f | "
                    "step rms %.4f max %.4f, w rms %.4f (%.1fs)",
                    out->n_iter, st, sg->level, sg->sigma, ri->core_var, ri->sharpness, ri->cost, ri->step_rms,
                    ri->step_max, ri->w_rms, ri->seconds);
            free(po.N), free(po.D), free(po.Q);
            plx_img_free(&dw);
            ref_free(&ref);
        }
    if (final_accumulate) {
        run_pass(d, p, &out->w, NULL, true, NULL, &po);
        set_template(out, &po);
    }
    plx_img_free(&v);
}

void plx_rl_refine_yaw(const plx_rl_data *d, const plx_rl_params *p, const plx_cg_template *t, const plx_img *w,
                       int level, float *ds) {
    plx_img vhr = plx_rl_v_dense(d->v_lat);
    for (size_t k = 0; k < (size_t)vhr.w * vhr.h; k++) vhr.px[k] += w->px[k];
    level_ref ref = ref_build(t, p, level, 1.0f, &vhr);
    memset(ds, 0, sizeof(float) * d->src->n);
    pass_out po;
    run_pass(d, p, w, &ref, false, ds, &po);
    free(po.N), free(po.D), free(po.Q);
    ref_free(&ref);
    plx_img_free(&vhr);
}

/* ---- synthetic population ------------------------------------------------------------------ */
plx_rl_synth_opts plx_rl_synth_defaults(void) {
    return (plx_rl_synth_opts){.n = 600, .yaw_sd_deg = 12, .yaw_clip_deg = 35, .nose_px = 20, .tex_indiv = 1,
                               .seed = 1};
}

/* The smooth initial depth handed to the estimator: plx_synth_depth's half-ellipsoid alone. */
static double dome(double X, double Y) {
    double ex = X / 62, ey = (Y - 4) / 82, e = 1 - ex * ex - ey * ey;
    return 38 * sqrt(e > 0 ? e : 0);
}

/* Smooth unit-variance noise on Ω+ (σ px); mirror-symmetric when sym. */
static plx_img smooth_noise(plx_rng *r, int margin, float sigma, bool sym) {
    plx_img a = plx_canon_new(1, margin);
    for (size_t k = 0; k < (size_t)a.w * a.h; k++) a.px[k] = (float)plx_rng_normal(r);
    plx_blur(&a, &a, sigma);
    if (sym) {
        plx_img f = plx_img_new(a.w, a.h, 1);
        plx_flip_h(&a, &f);
        for (size_t k = 0; k < (size_t)a.w * a.h; k++) a.px[k] += f.px[k];
        plx_img_free(&f);
    }
    double m, var;
    plx_chan_stats(&a, 0, &m, &var);
    plx_img_scale_offset(&a, (float)(1 / sqrt(var)), (float)(-m / sqrt(var)));
    return a;
}

typedef struct {
    const plx_rl_synth_opts *o;
    plx_rl_synth            *s;
    const plx_img           *z;      /* Ω+ true surface */
    const plx_img           *shared; /* Ω+ shared random pattern */
    const double            *yaw;
} synth_ctx;

static void synth_one(int64_t i, int tid, void *pc) {
    const synth_ctx *c = pc;
    const plx_rl_synth_opts *o = c->o;
    plx_rng r;
    plx_rng_seed(&r, o->seed * 7919 + (uint64_t)i + 1);
    int m = PLX_CANON_MARGIN;
    plx_img tex;
    if (o->tex) {
        tex = plx_img_clone(&o->tex[i % o->n_tex]);
    } else {
        /* the population's shared face pattern with a per-photo gain, plus the photo's own texture */
        plx_img own = smooth_noise(&r, m, 2.0f, false);
        tex = plx_canon_new(1, m);
        double a = 1 + 0.2 * plx_rng_normal(&r), sc = 0.12 / sqrt(1 + o->tex_indiv * o->tex_indiv);
        for (size_t k = 0; k < (size_t)tex.w * tex.h; k++)
            tex.px[k] = (float)(0.5 + sc * (a * c->shared->px[k] + o->tex_indiv * own.px[k]));
        plx_img_free(&own);
    }
    PLX_ASSERT(tex.w == c->z->w && tex.h == c->z->h && tex.c == 1);
    plx_mesh mesh = plx_mesh_heightfield(c->z, NULL, &tex);
    double th = c->yaw[i], centre[3] = {PLX_CANON_CX + m, PLX_CANON_CY + m, 0};
    plx_camera cam = plx_camera_make(th, 0, 0, 1, centre, PLX_CANON_IMG_CX, PLX_CANON_IMG_CY);
    plx_raster_opts ro = {.w = PLX_CANON_IMG_SIDE, .h = PLX_CANON_IMG_SIDE, .cull = PLX_CULL_NONE, .bg = 0.5f};
    plx_raster ras;
    plx_render(&mesh, &cam, &ro, &ras);
    uint8_t *px = c->s->px + (size_t)i * PLX_CANON_IMG_SIDE * PLX_CANON_IMG_SIDE;
    for (int k = 0; k < PLX_CANON_IMG_SIDE * PLX_CANON_IMG_SIDE; k++)
        px[k] = (uint8_t)lround(255 * PLX_CLAMP(ras.attr.px[k], 0.0f, 1.0f));
    plx_raster_free(&ras);
    plx_mesh_free(&mesh);
    plx_img_free(&tex);

    /* the flow field the estimator is given: foreshortening + yaw parallax of the dome, plus an
     * optional smooth error (four broad bumps per component) that the photo does not have */
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    float *u = c->s->u + (size_t)i * DOF;
    double bump[2][4][4];
    for (int cc = 0; cc < 2; cc++)
        for (int b = 0; b < 4; b++) {
            bump[cc][b][0] = 40 * (2 * plx_rng_uniform(&r) - 1);
            bump[cc][b][1] = 6 + 50 * (2 * plx_rng_uniform(&r) - 1);
            bump[cc][b][2] = 15 + 10 * plx_rng_uniform(&r);
            bump[cc][b][3] = o->u_noise_px * plx_rng_normal(&r);
        }
    for (int k = 0; k < NLAT; k++) {
        double X = plx_lat_x(&L, k % L.nx) - PLX_CANON_CX, Y = plx_lat_y(&L, k / L.nx) - PLX_CANON_CY, e[2] = {0, 0};
        for (int cc = 0; cc < 2; cc++)
            for (int b = 0; b < 4; b++) {
                double dx = X - bump[cc][b][0], dy = Y - bump[cc][b][1], sg = bump[cc][b][2];
                e[cc] += bump[cc][b][3] * exp(-(dx * dx + dy * dy) / (2 * sg * sg));
            }
        u[k] = (float)((cos(th) - 1) * X + sin(th) * c->s->v_lat[k] + e[0]);
        u[NLAT + k] = (float)e[1];
    }
}

void plx_rl_synth_make(const plx_rl_synth_opts *o, plx_rl_synth *s) {
    memset(s, 0, sizeof *s);
    int n = o->n, m = PLX_CANON_MARGIN;
    s->n = n;
    s->px = PLX_NEW(uint8_t, (size_t)n * PLX_CANON_IMG_SIDE * PLX_CANON_IMG_SIDE);
    s->src = (plx_cg_source){.px = s->px, .n = n, .w = PLX_CANON_IMG_SIDE, .h = PLX_CANON_IMG_SIDE};
    s->T = PLX_NEW(plx_sim, n);
    s->u = PLX_NEW(float, (size_t)n * DOF);
    s->s = PLX_NEW(float, n);
    s->s_true = PLX_NEW(float, n);
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    for (int k = 0; k < NLAT; k++)
        s->v_lat[k] = (float)dome(plx_lat_x(&L, k % L.nx) - PLX_CANON_CX, plx_lat_y(&L, k / L.nx) - PLX_CANON_CY);
    plx_img z = plx_canon_new(1, m);
    for (int y = 0; y < z.h; y++)
        for (int x = 0; x < z.w; x++)
            PLX_PX(&z, x, y, 0) = (float)plx_synth_depth(x - m - PLX_CANON_CX, y - m - PLX_CANON_CY, o->nose_px);
    plx_img v = plx_rl_v_dense(s->v_lat);
    s->w_true = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    for (int y = 0; y < PLX_CANON_H; y++)
        for (int x = 0; x < PLX_CANON_W; x++)
            PLX_PX(&s->w_true, x, y, 0) = PLX_PX(&z, x + m, y + m, 0) - PLX_PX(&v, x, y, 0);
    plx_img_free(&v);

    plx_rng r;
    plx_rng_seed(&r, o->seed);
    double *yaw = PLX_NEW(double, n), m1 = 0, m2 = 0;
    for (int i = 0; i < n; i++) {
        double th;
        do th = o->yaw_sd_deg * plx_rng_normal(&r);
        while (fabs(th) > o->yaw_clip_deg);
        yaw[i] = th * M_PI / 180;
        s->T[i] = plx_sim_identity();
        s->s_true[i] = (float)sin(yaw[i]);
        m1 += s->s_true[i], m2 += (double)s->s_true[i] * s->s_true[i];
    }
    double sd = sqrt(fmax(m2 / n - (m1 / n) * (m1 / n), 0));
    for (int i = 0; i < n; i++) s->s[i] = (float)(s->s_true[i] + o->s_noise * sd * plx_rng_normal(&r));
    plx_img shared = smooth_noise(&r, m, 2.5f, true);
    synth_ctx c = {.o = o, .s = s, .z = &z, .shared = &shared, .yaw = yaw};
    plx_parallel_for(n, 4, synth_one, &c);
    plx_img_free(&shared);
    plx_img_free(&z);
    free(yaw);
}

void plx_rl_synth_free(plx_rl_synth *s) {
    free(s->px), free(s->T), free(s->u), free(s->s), free(s->s_true);
    plx_img_free(&s->w_true);
    memset(s, 0, sizeof *s);
}
