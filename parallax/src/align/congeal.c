/* Similarity congealing: rendering, template statistics, the IC Gauss-Newton solve and the
 * population loop. See congeal.h for the method. */
#include "align/congeal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Face-core box in canonical pixels (the raw LFW mean's face core, docs/DESIGN.md §2). */
enum { CORE_X0 = 24, CORE_X1 = 104, CORE_Y0 = 21, CORE_Y1 = 136 };

plx_cg_params plx_cg_default(void) {
    plx_cg_params p = {
        .feat = plx_orient_default(),
        .margin = PLX_CANON_MARGIN,
        .levels = 4,
        .interp = PLX_INTERP_BILINEAR,
        .eps_frac = 0.5f,
        .huber = 1.5f,
        .ramp = 16.0f,
        .tol = 0.01f,
        .scale_min = 0.6,
        .scale_max = 1.7,
        .rot_max = 40.0 * M_PI / 180.0,
        .shift_max = 45.0,
    };
    return p;
}

/* ---- rendering ------------------------------------------------------------------------ */
void plx_cg_render(const plx_img *gray, plx_sim T, const plx_cg_params *p, plx_cg_view *v) {
    PLX_ASSERT(gray->c == 1 && p->levels >= 1 && p->levels <= PLX_CG_MAX_LEVELS);
    v->gray = plx_canon_new(1, p->margin);
    v->valid = plx_canon_new(1, p->margin);
    plx_canon_warp(gray, T, p->margin, p->interp, &v->gray, &v->valid);
    plx_img f = plx_orient_channels(&v->gray, &p->feat);
    v->feat = plx_pyramid_build(&f, p->levels, 2);
    v->vpyr = plx_pyramid_build(&v->valid, p->levels, 2);
    PLX_ASSERT(v->feat.n == p->levels);
    plx_img_free(&f);
}

void plx_cg_view_free(plx_cg_view *v) {
    plx_img_free(&v->gray);
    plx_img_free(&v->valid);
    plx_pyramid_free(&v->feat);
    plx_pyramid_free(&v->vpyr);
}

/* ---- template statistics -------------------------------------------------------------- */
static int level_size(int n0, int level) {
    for (int l = 0; l < level; l++) n0 = (n0 + 1) / 2;
    return n0;
}

void plx_cg_accum_init(plx_cg_accum *a, const plx_cg_params *p) {
    memset(a, 0, sizeof *a);
    a->K = p->feat.k;
    a->margin = p->margin;
    a->levels = p->levels;
    int w0 = PLX_CANON_W + 2 * p->margin, h0 = PLX_CANON_H + 2 * p->margin;
    for (int l = 0; l < a->levels; l++) {
        size_t n = (size_t)a->K * level_size(w0, l) * level_size(h0, l);
        a->s1[l] = PLX_NEW(double, n);
        a->s2[l] = PLX_NEW(double, n);
    }
    a->gray = PLX_NEW(double, (size_t)w0 * h0);
}

void plx_cg_accum_add(plx_cg_accum *a, const plx_cg_view *v) {
    for (int l = 0; l < a->levels; l++) {
        const plx_img *f = &v->feat.lev[l];
        size_t n = (size_t)f->w * f->h * f->c;
        double *s1 = a->s1[l], *s2 = a->s2[l];
        for (size_t i = 0; i < n; i++) {
            double x = f->px[i];
            s1[i] += x;
            s2[i] += x * x;
        }
    }
    size_t n = (size_t)v->gray.w * v->gray.h;
    for (size_t i = 0; i < n; i++) a->gray[i] += v->gray.px[i];
    a->n++;
}

void plx_cg_accum_merge(plx_cg_accum *dst, const plx_cg_accum *src) {
    int w0 = PLX_CANON_W + 2 * dst->margin, h0 = PLX_CANON_H + 2 * dst->margin;
    for (int l = 0; l < dst->levels; l++) {
        size_t n = (size_t)dst->K * level_size(w0, l) * level_size(h0, l);
        for (size_t i = 0; i < n; i++) {
            dst->s1[l][i] += src->s1[l][i];
            dst->s2[l][i] += src->s2[l][i];
        }
    }
    for (size_t i = 0; i < (size_t)w0 * h0; i++) dst->gray[i] += src->gray[i];
    dst->n += src->n;
}

void plx_cg_accum_finish(const plx_cg_accum *a, const plx_cg_params *p, plx_cg_template *t, plx_img *mean_gray) {
    PLX_ASSERT(a->n > 0 && a->K == p->feat.k);
    memset(t, 0, sizeof *t);
    t->K = a->K;
    t->margin = a->margin;
    t->levels = a->levels;
    t->count = a->n;
    int w0 = PLX_CANON_W + 2 * a->margin, h0 = PLX_CANON_H + 2 * a->margin;
    double inv_n = 1.0 / (double)a->n;
    for (int l = 0; l < a->levels; l++) {
        int w = level_size(w0, l), h = level_size(h0, l);
        size_t n = (size_t)a->K * w * h;
        plx_img m1 = plx_img_new(w, h, a->K), m2 = plx_img_new(w, h, a->K);
        plx_img f1 = plx_img_new(w, h, a->K), f2 = plx_img_new(w, h, a->K);
        for (size_t i = 0; i < n; i++) {
            m1.px[i] = (float)(a->s1[l][i] * inv_n);
            m2.px[i] = (float)(a->s2[l][i] * inv_n);
        }
        /* Symmetrise both moments: the population plus every image's mirror twin. The levels
         * of an even-sized Ω+ are exactly flip-symmetric, so this is exact at every level. */
        plx_orient_mirror(&m1, &p->feat, &f1);
        plx_orient_mirror(&m2, &p->feat, &f2);
        t->mean[l] = plx_img_new(w, h, a->K);
        t->var[l] = plx_img_new(w, h, a->K);
        for (size_t i = 0; i < n; i++) {
            float mu = 0.5f * (m1.px[i] + f1.px[i]), sq = 0.5f * (m2.px[i] + f2.px[i]);
            t->mean[l].px[i] = mu;
            t->var[l].px[i] = fmaxf(sq - mu * mu, 0.0f);
        }
        plx_img_free(&m1);
        plx_img_free(&m2);
        plx_img_free(&f1);
        plx_img_free(&f2);
    }
    if (mean_gray) {
        plx_img ext = plx_img_new(w0, h0, 1);
        for (size_t i = 0; i < (size_t)w0 * h0; i++) ext.px[i] = (float)(a->gray[i] * inv_n);
        *mean_gray = plx_canon_crop(&ext, a->margin);
        plx_img_free(&ext);
    }
}

void plx_cg_accum_free(plx_cg_accum *a) {
    for (int l = 0; l < a->levels; l++) {
        free(a->s1[l]);
        free(a->s2[l]);
    }
    free(a->gray);
    memset(a, 0, sizeof *a);
}

void plx_cg_template_free(plx_cg_template *t) {
    for (int l = 0; l < t->levels; l++) {
        plx_img_free(&t->mean[l]);
        plx_img_free(&t->var[l]);
    }
    memset(t, 0, sizeof *t);
}

void plx_cg_core_variance(const plx_cg_template *t, double *total, double *ratio) {
    const plx_img *m = &t->mean[0], *v = &t->var[0];
    double sv = 0, s2 = 0;
    for (int k = 0; k < t->K; k++)
        for (int y = CORE_Y0; y <= CORE_Y1; y++)
            for (int x = CORE_X0; x <= CORE_X1; x++) {
                double mu = PLX_PX(m, x + t->margin, y + t->margin, k);
                double var = PLX_PX(v, x + t->margin, y + t->margin, k);
                sv += var;
                s2 += var + mu * mu;
            }
    *total = sv;
    *ratio = s2 > 0 ? sv / s2 : 0;
}

/* ---- reference (template prepared for one stage) -------------------------------------- */
/* Raised-cosine window on one axis: 0 at and beyond the Ω edge (x = -0.5 or n - 0.5), 1 once
 * `ramp` pixels inside. Symmetric under x -> n - 1 - x, so the window is σ-symmetric. */
static double window1(double x, int n, double ramp) {
    double d = fmin(x + 0.5, (double)n - 0.5 - x);
    if (d <= 0) return 0;
    if (d >= ramp) return 1;
    return 0.5 * (1.0 - cos(M_PI * d / ramp));
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

plx_cg_ref plx_cg_ref_build(const plx_cg_template *t, const plx_cg_params *p, int level, float blur) {
    PLX_ASSERT(level >= 0 && level < t->levels && t->K == p->feat.k && t->margin == p->margin);
    plx_cg_ref r = {0};
    int K = t->K, w = t->mean[level].w, h = t->mean[level].h;
    size_t plane = (size_t)w * h;
    r.level = level;
    r.K = K;
    r.w = w;
    r.h = h;
    r.blur = blur;
    r.cx = plx_pyr_coord((float)(PLX_CANON_CX + t->margin), 0, level);
    r.cy = plx_pyr_coord((float)(PLX_CANON_CY + t->margin), 0, level);

    plx_img mu = plx_img_clone(&t->mean[level]), var = plx_img_clone(&t->var[level]);
    if (blur > 0) {
        plx_blur(&mu, &mu, blur);
        plx_blur(&var, &var, blur); /* approximates the variance of blurred features */
    }
    plx_img gx = plx_img_new(w, h, K), gy = plx_img_new(w, h, K);
    plx_gradient_scharr(&mu, &gx, &gy);

    /* window in level pixels: evaluate the full-res window at each level pixel's centre */
    float *om = PLX_NEW(float, plane);
    float *vals = PLX_NEW(float, plane * K);
    size_t nv = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double cxp = plx_pyr_coord((float)x, level, 0) - t->margin;
            double cyp = plx_pyr_coord((float)y, level, 0) - t->margin;
            double o = window1(cxp, PLX_CANON_W, p->ramp) * window1(cyp, PLX_CANON_H, p->ramp);
            om[(size_t)y * w + x] = (float)o;
            if (o > 0.5) {
                r.n++;
                for (int k = 0; k < K; k++) vals[nv++] = var.px[k * plane + (size_t)y * w + x];
            } else if (o > 0) {
                r.n++;
            }
        }
    PLX_ASSERT(nv > 0);
    qsort(vals, nv, sizeof(float), cmp_float);
    double med = vals[nv / 2];
    r.eps = fmax(p->eps_frac * med, 1e-12);
    free(vals);

    r.X = PLX_NEW(float, r.n);
    r.Y = PLX_NEW(float, r.n);
    r.omega = PLX_NEW(float, r.n);
    r.mu = PLX_NEW(float, (size_t)r.n * K);
    r.inv = PLX_NEW(float, (size_t)r.n * K);
    r.g = PLX_NEW(float, (size_t)r.n * K * 4);
    r.hp = PLX_NEW(float, (size_t)r.n * 10);
    int s = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t o = (size_t)y * w + x;
            if (om[o] <= 0) continue;
            double X = x - r.cx, Y = y - r.cy;
            r.X[s] = (float)X;
            r.Y[s] = (float)Y;
            r.omega[s] = om[o];
            r.omega_sum += om[o];
            r.rmax = fmax(r.rmax, hypot(X, Y));
            double H[10] = {0};
            for (int k = 0; k < K; k++) {
                double ix = 1.0 / (var.px[k * plane + o] + r.eps), wk = om[o] * ix;
                double ux = gx.px[k * plane + o], uy = gy.px[k * plane + o];
                double J[4] = {ux * X + uy * Y, -ux * Y + uy * X, ux, uy};
                r.mu[(size_t)s * K + k] = mu.px[k * plane + o];
                r.inv[(size_t)s * K + k] = (float)ix;
                for (int j = 0; j < 4; j++) r.g[((size_t)s * K + k) * 4 + j] = (float)(wk * J[j]);
                for (int i = 0, q = 0; i < 4; i++)
                    for (int j = i; j < 4; j++) H[q++] += wk * J[i] * J[j];
            }
            for (int q = 0; q < 10; q++) r.hp[(size_t)s * 10 + q] = (float)H[q];
            s++;
        }
    free(om);
    plx_img_free(&mu);
    plx_img_free(&var);
    plx_img_free(&gx);
    plx_img_free(&gy);
    return r;
}

void plx_cg_ref_free(plx_cg_ref *r) {
    free(r->X);
    free(r->Y);
    free(r->omega);
    free(r->mu);
    free(r->inv);
    free(r->g);
    free(r->hp);
    memset(r, 0, sizeof *r);
}

/* ---- solve ---------------------------------------------------------------------------- */
/* Cholesky solve of the 4x4 SPD system given by its upper triangle (row-major, 10 entries).
 * Returns false if the matrix is not numerically positive definite. */
static bool solve4(const double Hu[10], const double b[4], double x[4]) {
    double A[4][4], L[4][4] = {{0}};
    for (int i = 0, q = 0; i < 4; i++)
        for (int j = i; j < 4; j++, q++) A[i][j] = A[j][i] = Hu[q];
    for (int j = 0; j < 4; j++) {
        double d = A[j][j];
        for (int k = 0; k < j; k++) d -= L[j][k] * L[j][k];
        if (!(d > 1e-12 * fabs(A[j][j])) || d <= 0) return false;
        L[j][j] = sqrt(d);
        for (int i = j + 1; i < 4; i++) {
            double s = A[i][j];
            for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            L[i][j] = s / L[j][j];
        }
    }
    double y[4];
    for (int i = 0; i < 4; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    for (int i = 3; i >= 0; i--) {
        double s = y[i];
        for (int k = i + 1; k < 4; k++) s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
    return true;
}

/* Bilinear taps of a w x h grid (w, h >= 2) at a point inside [0, w-1] x [0, h-1]. */
typedef struct { size_t o; size_t dx, dy; float w00, w10, w01, w11; } bil_taps;
static inline bool bil(int w, int h, double x, double y, bil_taps *t) {
    if (!(x >= 0 && y >= 0 && x <= w - 1 && y <= h - 1)) return false;
    int x0 = (int)x, y0 = (int)y;
    if (x0 > w - 2) x0 = w - 2;
    if (y0 > h - 2) y0 = h - 2;
    float fx = (float)(x - x0), fy = (float)(y - y0);
    t->o = (size_t)y0 * w + x0;
    t->dx = 1;
    t->dy = (size_t)w;
    t->w00 = (1 - fx) * (1 - fy);
    t->w10 = fx * (1 - fy);
    t->w01 = (1 - fx) * fy;
    t->w11 = fx * fy;
    return true;
}
static inline float bil_eval(const float *p, const bil_taps *t) {
    const float *q = p + t->o;
    return t->w00 * q[0] + t->w10 * q[t->dx] + t->w01 * q[t->dy] + t->w11 * q[t->dy + t->dx];
}

double plx_cg_csim_disp(plx_sim S) {
    static const double cx[2] = {CORE_X0, CORE_X1}, cy[2] = {CORE_Y0, CORE_Y1};
    double m = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            double x, y;
            plx_csim_apply(S, cx[i], cy[j], &x, &y);
            m = fmax(m, hypot(x - cx[i], y - cy[j]));
        }
    return m;
}

plx_cg_fit plx_cg_solve(const plx_cg_ref *r, const plx_cg_view *v, const plx_cg_params *p,
                        const plx_cg_stage *st, plx_sim *S) {
    PLX_ASSERT(st->level == r->level && r->level < v->feat.n);
    const plx_img *F = &v->feat.lev[r->level], *V = &v->vpyr.lev[r->level];
    PLX_ASSERT(F->w == r->w && F->h == r->h && F->c == r->K);
    plx_img Fb = {0};
    if (r->blur > 0) {
        Fb = plx_img_clone(F);
        plx_blur(&Fb, &Fb, r->blur);
        F = &Fb;
    }
    int K = r->K, w = r->w, h = r->h;
    size_t plane = (size_t)w * h;
    double lev = ldexp(1.0, r->level);
    /* the solve runs in level pixels; S's translation converts by 2^level (rotation and scale
     * are about c0, which the pyramid registration maps to (r->cx, r->cy)) */
    plx_sim Sl = {S->a, S->b, S->tx / lev, S->ty / lev};
    double c2 = (double)p->huber * p->huber * K;
    float *res = PLX_NEW(float, K);
    plx_cg_fit fit = {0};

    for (int it = 0; it < st->iters; it++) {
        double H[10] = {0}, b[4] = {0}, sov = 0, cost = 0, inl = 0;
        for (int i = 0; i < r->n; i++) {
            double X = r->X[i], Y = r->Y[i];
            double x = r->cx + Sl.a * X - Sl.b * Y + Sl.tx;
            double y = r->cy + Sl.b * X + Sl.a * Y + Sl.ty;
            bil_taps t;
            if (!bil(w, h, x, y, &t)) continue;
            float vv = bil_eval(V->px, &t);
            if (vv <= 0) continue;
            const float *mu = r->mu + (size_t)i * K, *inv = r->inv + (size_t)i * K;
            double e2 = 0;
            for (int k = 0; k < K; k++) {
                float d = bil_eval(F->px + k * plane, &t) - mu[k];
                res[k] = d;
                e2 += (double)d * d * inv[k];
            }
            double hw = e2 <= c2 ? 1.0 : sqrt(c2 / e2), ww = hw * vv;
            const float *g = r->g + (size_t)i * K * 4;
            double b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            for (int k = 0; k < K; k++, g += 4) {
                b0 += res[k] * g[0];
                b1 += res[k] * g[1];
                b2 += res[k] * g[2];
                b3 += res[k] * g[3];
            }
            b[0] += ww * b0;
            b[1] += ww * b1;
            b[2] += ww * b2;
            b[3] += ww * b3;
            const float *hp = r->hp + (size_t)i * 10;
            for (int q = 0; q < 10; q++) H[q] += ww * hp[q];
            double ov = r->omega[i] * vv;
            sov += ov;
            cost += ov * e2;
            inl += ov * hw;
        }
        fit.iters = it + 1;
        if (sov < 1e-3 * r->omega_sum) break; /* nothing of the photo left in the window */
        fit.cost = cost / sov;
        fit.inlier = inl / sov;
        fit.valid = sov / r->omega_sum;
        if (it == 0) fit.cost0 = fit.cost;

        /* Damping towards identity: (H + λD) Δ = b + λ D s, D = diag(H), s = current params
         * (to first order the update maps s to s - Δ). */
        if (st->prior > 0) {
            double s[4] = {Sl.a - 1.0, Sl.b, Sl.tx, Sl.ty};
            static const int diag[4] = {0, 4, 7, 9};
            for (int j = 0; j < 4; j++) {
                double d = st->prior * H[diag[j]];
                b[j] += d * s[j];
                H[diag[j]] += d;
            }
        }
        double D[4];
        if (!solve4(H, b, D)) break;
        plx_sim dS = {1.0 + D[0], D[1], D[2], D[3]};
        Sl = plx_sim_compose(Sl, plx_csim_invert(dS));
        double step = hypot(D[2], D[3]) + hypot(D[0], D[1]) * r->rmax; /* level px */
        fit.step = step * lev;
        if (step < p->tol) {
            fit.converged = true;
            break;
        }
    }
    *S = (plx_sim){Sl.a, Sl.b, Sl.tx * lev, Sl.ty * lev};
    free(res);
    plx_img_free(&Fb);
    return fit;
}

bool plx_cg_clamp(plx_sim *T, const plx_cg_params *p) {
    double s = plx_sim_scale(*T), rho = plx_sim_angle(*T);
    double s2 = PLX_CLAMP(s, p->scale_min, p->scale_max), r2 = PLX_CLAMP(rho, -p->rot_max, p->rot_max);
    double tx = PLX_CLAMP(T->tx, -p->shift_max, p->shift_max), ty = PLX_CLAMP(T->ty, -p->shift_max, p->shift_max);
    bool hit = s2 != s || r2 != rho || tx != T->tx || ty != T->ty || !isfinite(s);
    if (!isfinite(s) || !isfinite(rho)) s2 = 1, r2 = 0;
    if (!isfinite(tx) || !isfinite(ty)) tx = ty = 0;
    if (hit) *T = plx_sim_make(s2, r2, tx, ty);
    return hit;
}

plx_sim plx_cg_align_image(const plx_img *gray, plx_sim T0, const plx_cg_ref *refs, const plx_cg_stage *stages,
                           int n_stages, const plx_cg_params *p, plx_cg_fit *last, bool *bound) {
    plx_sim T = T0;
    bool hit = false;
    plx_cg_fit fit = {0};
    for (int s = 0; s < n_stages; s++) {
        plx_cg_view v;
        plx_cg_render(gray, T, p, &v);
        plx_sim S = plx_sim_identity();
        fit = plx_cg_solve(&refs[s], &v, p, &stages[s], &S);
        T = plx_sim_compose(T, S);
        hit |= plx_cg_clamp(&T, p);
        plx_cg_view_free(&v);
    }
    if (last) *last = fit;
    if (bound) *bound = hit;
    return T;
}

/* ---- population ----------------------------------------------------------------------- */
plx_img plx_cg_source_image(const plx_cg_source *s, int i) {
    PLX_ASSERT(i >= 0 && i < s->n);
    return plx_img_from_u8(s->px + (size_t)i * s->w * s->h, s->w, s->h, 1, 1.0f / 255.0f);
}

typedef struct {
    const plx_cg_source *src;
    const plx_cg_params *p;
    const plx_cg_stage  *stage; /* NULL: accumulate only */
    const plx_cg_ref    *ref;
    plx_sim             *T;
    plx_cg_accum        *acc;   /* [threads], NULL in the evaluation pass */
    float               *cost0, *disp;
    uint8_t             *conv, *bound;
    plx_cg_quality      *q;     /* evaluation pass only */
} round_ctx;

static void round_one(int64_t i, int tid, void *pc) {
    round_ctx *c = pc;
    plx_img g = plx_cg_source_image(c->src, (int)i);
    plx_cg_view v;
    plx_cg_render(&g, c->T[i], c->p, &v);
    if (c->acc) plx_cg_accum_add(&c->acc[tid], &v);
    if (c->stage) {
        plx_sim S = plx_sim_identity();
        plx_cg_fit f = plx_cg_solve(c->ref, &v, c->p, c->stage, &S);
        if (c->q) { /* evaluation: measure, do not move */
            c->q[i].cost = (float)f.cost0;
            c->q[i].inlier = (float)f.inlier;
            c->q[i].valid = (float)f.valid;
            c->q[i].disp = (float)plx_cg_csim_disp(S);
        } else {
            plx_sim T = plx_sim_compose(c->T[i], S);
            if (plx_cg_clamp(&T, c->p)) c->bound[i] = 1;
            c->T[i] = T;
            c->cost0[i] = (float)f.cost0;
            c->disp[i] = (float)plx_cg_csim_disp(S);
            c->conv[i] = f.converged;
        }
    }
    plx_cg_view_free(&v);
    plx_img_free(&g);
}

/* Common canonical-side correction G (T_i <- T_i o G for all i) so that mean log scale = 0 and
 * mean ty = 0. G scales about c0 and then shifts vertically; relative alignment is unchanged. */
static void gauge_fix(plx_sim *T, int n) {
    double ls = 0;
    for (int i = 0; i < n; i++) ls += log(plx_sim_scale(T[i]));
    plx_sim G = {exp(-ls / n), 0, 0, 0};
    double ty = 0, a = 0;
    for (int i = 0; i < n; i++) {
        T[i] = plx_sim_compose(T[i], G);
        ty += T[i].ty;
        a += T[i].a;
    }
    /* t_i += [a -b; b a]_i (0, dy): mean ty moves by mean(a) dy */
    plx_sim D = {1, 0, 0, -ty / a};
    for (int i = 0; i < n; i++) T[i] = plx_sim_compose(T[i], D);
}

static float median_f(const float *v, int n) {
    float *c = PLX_NEW(float, n);
    memcpy(c, v, sizeof(float) * n);
    qsort(c, n, sizeof(float), cmp_float);
    float m = c[n / 2];
    free(c);
    return m;
}

void plx_cg_congeal(const plx_cg_source *src, const plx_cg_params *p, const plx_cg_stage *sched, int n_stages,
                    plx_sim *T, plx_cg_template *final_template, plx_cg_round *rounds, plx_cg_quality *q) {
    int n = src->n, nt = plx_num_threads();
    plx_cg_accum *acc = PLX_NEW(plx_cg_accum, nt);
    float *cost0 = PLX_NEW(float, n), *disp = PLX_NEW(float, n);
    uint8_t *conv = PLX_NEW(uint8_t, n), *bound = PLX_NEW(uint8_t, n);
    plx_cg_template tmpl = {0};

    for (int r = 0; r <= n_stages + 1; r++) {
        double t0 = plx_now();
        const plx_cg_stage *st = r >= 1 && r <= n_stages ? &sched[r - 1] : NULL;
        plx_cg_ref ref = {0};
        if (st) ref = plx_cg_ref_build(&tmpl, p, st->level, st->blur);
        for (int t = 0; t < nt; t++) plx_cg_accum_init(&acc[t], p);
        round_ctx c = {.src = src, .p = p, .stage = st, .ref = &ref, .T = T, .acc = acc,
                       .cost0 = cost0, .disp = disp, .conv = conv, .bound = bound};
        plx_parallel_for(n, 8, round_one, &c);
        for (int t = 1; t < nt; t++) plx_cg_accum_merge(&acc[0], &acc[t]);
        plx_cg_template_free(&tmpl);
        plx_cg_round *ri = &rounds[r];
        memset(ri, 0, sizeof *ri);
        plx_cg_accum_finish(&acc[0], p, &tmpl, &ri->mean_gray);
        for (int t = 0; t < nt; t++) plx_cg_accum_free(&acc[t]);
        plx_cg_core_variance(&tmpl, &ri->core_var, &ri->core_ratio);
        if (st) {
            gauge_fix(T, n);
            double sc = 0;
            for (int i = 0; i < n; i++) {
                sc += cost0[i];
                ri->n_unconverged += !conv[i];
                ri->n_bound += bound[i];
            }
            ri->mean_cost = sc / n;
            ri->median_disp = median_f(disp, n);
            plx_cg_ref_free(&ref);
        }
        ri->seconds = plx_now() - t0;
        plx_log("congeal round %d%s: level %d blur %.2f  core var %.2f (ratio %.4f)  cost %.3f  median update %.3f px"
                "  unconverged %d  bound %d  (%.1fs)",
                r, st ? "" : " (accumulate)", st ? st->level : -1, st ? st->blur : 0.0f, ri->core_var, ri->core_ratio,
                ri->mean_cost, ri->median_disp, ri->n_unconverged, ri->n_bound, ri->seconds);
    }

    /* evaluation pass against the final template at full resolution */
    double t0 = plx_now();
    plx_cg_stage ev = {.level = 0, .blur = 0, .iters = 15, .prior = 0};
    plx_cg_ref ref = plx_cg_ref_build(&tmpl, p, 0, 0);
    round_ctx c = {.src = src, .p = p, .stage = &ev, .ref = &ref, .T = T, .q = q};
    plx_parallel_for(n, 8, round_one, &c);
    for (int i = 0; i < n; i++) {
        q[i].converged = n_stages > 0 ? conv[i] : 0;
        q[i].bound = bound[i];
    }
    plx_cg_ref_free(&ref);
    plx_log("congeal evaluation pass (%.1fs)", plx_now() - t0);

    *final_template = tmpl;
    free(acc);
    free(cost0);
    free(disp);
    free(conv);
    free(bound);
}
