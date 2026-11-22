/* The emerge stage (DESIGN.md §5): depth of the mean face, pose, metric scale, personal depth with
 * uncertainty and the in-plane shape basis, all from the population of gauged flow fields.
 *
 * Steps (numbers as in the stage specification):
 *   1-2  parity split + plane gauge             plx_geom_measure
 *   3    yaw mode v, s_i                         fit_yaw_mode
 *   4    pitch t_i, rigid-body check             rigid_check, gls_pitch
 *   5    metric scale κ                          fit_kappa
 *   6    depth-variation basis B, Σ0, ĉ_i, Λ_i   fit_depth_basis
 *   7    in-plane shape basis                    fit_shape_basis
 *   8    depth tilt (optional)                   fit_tilt */
#include "geom/geom.h"

#include "la/la.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

plx_emerge_opts plx_emerge_defaults(void) {
    return (plx_emerge_opts){.robust_iters = 8, .huber_k = 2.5, .kz_max = 8, .ks_max = 24, .boot = 200,
                             .n_scramble = 200, .seed = 0x9e0, .sigma0_cond = 1e3};
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Weighted median of v over entries with w > 0 (plain median of those entries). */
static double median_pos(const double *v, const double *w, int n) {
    double *t = PLX_NEW(double, n + 1);
    int k = 0;
    for (int i = 0; i < n; i++)
        if (w[i] > 0 && isfinite(v[i])) t[k++] = v[i];
    double m = 0;
    if (k) {
        qsort(t, k, sizeof(double), cmp_double);
        m = k & 1 ? t[k / 2] : 0.5 * (t[k / 2 - 1] + t[k / 2]);
    }
    free(t);
    return m;
}

/* ---- steps 1-2: per-image measurement ---------------------------------------------------------- */
typedef struct {
    double *yE, *yV, *yO, *beta, *gamma, *planeE, *qslope;
} measure_out;

static void measure_full(const plx_gspace *sp, const float *u_i, double *yE, double *yV, double *yO, double *beta,
                         double *gamma, double planeE[2], double *qslope) {
    int P = sp->P;
    double *ux = PLX_NEW(double, 2 * P), *uy = ux + P;
    for (int k = 0; k < 2 * P; k++) ux[k] = u_i[k];
    double c[2], ox = plx_vec_dot(sp->oX, sp->oX, sp->no);
    plx_gspace_even(sp, ux, yE);
    plx_gspace_remove_plane(sp, yE, planeE ? planeE : c);
    plx_gspace_even(sp, uy, yV);
    /* γ = slope of V on Y0 with intercept: exactly the plane coefficient removed from V */
    plx_gspace_remove_plane(sp, yV, c);
    *gamma = c[1];
    plx_gspace_odd(sp, ux, yO);
    /* β = slope of O on X0 through the origin (odd fields have no intercept) */
    *beta = plx_vec_dot(yO, sp->oX, sp->no) / ox;
    if (qslope) {
        double *yQ = PLX_NEW(double, sp->no);
        plx_gspace_odd(sp, uy, yQ);
        *qslope = plx_vec_dot(yQ, sp->oX, sp->no) / ox;
        free(yQ);
    }
    free(ux);
}

void plx_geom_measure(const plx_gspace *sp, const float *u_i, double *yE, double *yV, double *yO, double *beta,
                      double *gamma) {
    measure_full(sp, u_i, yE, yV, yO, beta, gamma, NULL, NULL);
}

typedef struct {
    const plx_gspace *sp;
    const float *u;
    measure_out *mo;
} measure_ctx;

static void measure_task(int64_t i, int tid, void *vc) {
    measure_ctx *c = vc;
    const plx_gspace *sp = c->sp;
    measure_out *m = c->mo;
    measure_full(sp, c->u + (size_t)i * 2 * sp->P, m->yE + i * sp->ne, m->yV + i * sp->ne, m->yO + i * sp->no,
                 m->beta + i, m->gamma + i, m->planeE + 2 * i, m->qslope + i);
}

/* ---- step 3: the yaw mode ----------------------------------------------------------------------
 * E⊥_i = s_i v + residual. The top right singular vector of the √w-weighted E⊥ rows is the top
 * eigenvector of M = Σ w_i y_i y_iᵀ. Robust: images whose residual norm exceeds huber_k times the
 * median get Huber weight k / r (times their flow-quality prior). The median is taken among images
 * of similar s² (BINS equal-count bins): even a perfect fit leaves |s B c|, which grows with yaw, so
 * a global median would down-weight exactly the turned photos that carry the depth and κ signal. */
enum { BINS = 10 };
static void binned_median(const double *key, const double *val, const double *prior, int n, double *med) {
    int *ord = PLX_NEW(int, n), m = 0;
    double *kv = PLX_NEW(double, 2 * (size_t)n), *tmp = PLX_NEW(double, n), *one = PLX_NEW(double, n);
    for (int i = 0; i < n; i++) {
        one[i] = 1;
        if (prior[i] > 0) kv[2 * m] = key[i], kv[2 * m + 1] = i, m++;
        else med[i] = 0;
    }
    qsort(kv, m, 2 * sizeof(double), cmp_double); /* sorts on the key (first of each pair) */
    for (int j = 0; j < m; j++) ord[j] = (int)kv[2 * j + 1];
    for (int b = 0; b < BINS; b++) {
        int lo = (int)((int64_t)m * b / BINS), hi = (int)((int64_t)m * (b + 1) / BINS);
        for (int j = lo; j < hi; j++) tmp[j - lo] = val[ord[j]];
        double md = median_pos(tmp, one, hi - lo);
        for (int j = lo; j < hi; j++) med[ord[j]] = md;
    }
    free(ord), free(kv), free(tmp), free(one);
}
static void weighted_scatter(const double *Y, const double *w, int n, int d, plx_mat *M) {
    plx_mat Yw = plx_mat_new(n, d);
    for (int i = 0; i < n; i++) {
        double sw = sqrt(w[i] > 0 ? w[i] : 0);
        for (int k = 0; k < d; k++) PLX_M(&Yw, i, k) = sw * Y[(size_t)i * d + k];
    }
    plx_syrk(PLX_TRANS, 1.0, &Yw, 0.0, M);
    plx_mat_free(&Yw);
}

/* Top eigenvector u (d) and the leading eigenvalues of a symmetric d x d matrix. */
static void top_eig(const plx_mat *M, double *u, double *eig, int neig, plx_mat *Vout) {
    int d = M->rows;
    double *w = PLX_NEW(double, d);
    plx_mat V = plx_mat_new(d, d);
    PLX_ASSERT(plx_eigh(M, w, &V) == PLX_OK);
    for (int k = 0; k < d; k++) u[k] = PLX_M(&V, k, 0);
    for (int k = 0; k < neig; k++) eig[k] = k < d ? w[k] : 0;
    if (Vout) plx_mat_copy(Vout, &V);
    plx_mat_free(&V);
    free(w);
}

static void fit_yaw_mode(plx_geom_model *m, const measure_out *mo, const double *prior, plx_core_shape cs,
                         const plx_emerge_opts *o, plx_mat *Vall) {
    const plx_gspace *sp = &m->sp;
    int n = m->n, d = sp->ne;
    double *u = PLX_NEW(double, d), *res = PLX_NEW(double, n), *s2 = PLX_NEW(double, n), *med = PLX_NEW(double, n);
    plx_mat M = plx_mat_new(d, d);
    for (int i = 0; i < n; i++) m->w[i] = prior[i];
    for (int it = 0; it <= o->robust_iters; it++) {
        weighted_scatter(mo->yE, m->w, n, d, &M);
        top_eig(&M, u, m->sv2, PLX_GEOM_NSV, Vall);
        for (int i = 0; i < n; i++) {
            const double *y = mo->yE + (size_t)i * d;
            double p = plx_vec_dot(y, u, d), e = plx_vec_dot(y, y, d) - p * p;
            res[i] = sqrt(e > 0 ? e : 0);
        }
        if (it == o->robust_iters) break;
        for (int i = 0; i < n; i++) s2[i] = plx_vec_dot(mo->yE + (size_t)i * d, u, d), s2[i] *= s2[i];
        binned_median(s2, res, prior, n, med);
        for (int i = 0; i < n; i++) {
            double r = med[i] > 0 ? res[i] / med[i] : 0;
            m->w[i] = prior[i] * (r <= o->huber_k ? 1.0 : o->huber_k / r);
        }
    }
    double tot = 0;
    for (int k = 0; k < d; k++) tot += PLX_M(&M, k, k);
    m->sv_total = tot;
    m->sv_frac1 = m->sv2[0] / tot;
    m->sv_ratio12 = sqrt(m->sv2[0] / m->sv2[1]);

    /* Convexity sign: the nose points at the camera, so the face centre is in front of the rest. */
    double ci = 0, cw = 0, ro = 0, rw = 0;
    for (int k = 0; k < d; k++) {
        int node = sp->enode[k];
        double val = u[k] / sp->esw[k], wt = sp->esw[k] * sp->esw[k];
        if (plx_core_centre(&sp->lat, cs, node)) ci += wt * val, cw += wt;
        else ro += wt * val, rw += wt;
    }
    if (cw > 0 && rw > 0 && ci / cw - ro / rw < 0)
        for (int k = 0; k < d; k++) u[k] = -u[k];
    /* unit ω-RMS: |yv|² = Σ ω v² = Σ ω */
    for (int k = 0; k < d; k++) m->yv[k] = sqrt(sp->wsum) * u[k];
    for (int i = 0; i < n; i++) {
        m->s[i] = plx_vec_dot(mo->yE + (size_t)i * d, m->yv, d) / sp->wsum;
        m->t[i] = -plx_vec_dot(mo->yV + (size_t)i * d, m->yv, d) / sp->wsum;
        if (m->w[i] < 0.5 * prior[i] || prior[i] <= 0) m->flags[i] |= PLX_GF_OUTLIER;
    }
    free(u), free(res), free(s2), free(med);
    plx_mat_free(&M);
}

/* v at every lattice node: core nodes from the singular vector, the others by regressing the
 * plane-gauged E_i(node) = E_i - α_i - γ_i Y0 (core-fitted plane) on s_i. */
static void extend_v(plx_geom_model *m, const float *u, const measure_out *mo) {
    const plx_gspace *sp = &m->sp;
    int P = sp->P, n = m->n;
    double *num = PLX_NEW(double, P), den = 0;
    for (int i = 0; i < n; i++) {
        const float *ux = u + (size_t)i * 2 * P;
        double s = m->s[i], w = m->w[i];
        if (w <= 0) continue;
        den += w * s * s;
        for (int k = 0; k < P; k++) {
            double E = 0.5 * ((double)ux[k] + ux[plx_glat_mirror(&sp->lat, k)]);
            E -= mo->planeE[2 * i] + mo->planeE[2 * i + 1] * plx_glat_Y0(&sp->lat, k);
            num[k] += w * s * E;
        }
    }
    for (int k = 0; k < P; k++) m->v[k] = num[k] / den;
    for (int k = 0; k < sp->ne; k++) {
        int node = sp->enode[k];
        m->v[node] = m->v[plx_glat_mirror(&sp->lat, node)] = m->yv[k] / sp->esw[k];
    }
    free(num);
}

/* ---- step 4: rigid-body check -------------------------------------------------------------------
 * Fraction of the V⊥ energy along a unit direction q: qᵀ C q / tr C, C = Σ w yV yVᵀ. Controls: the
 * 2nd..6th E⊥ singular vectors, and v with its node values randomly permuted over the core (then
 * re-gauged and normalised): same spatial statistics, no rigid relation to the vertical channel. */
static double quad(const plx_mat *C, const double *q) {
    int d = C->rows;
    double s = 0;
    for (int a = 0; a < d; a++) {
        double r = 0;
        for (int b = 0; b < d; b++) r += PLX_M(C, a, b) * q[b];
        s += q[a] * r;
    }
    return s;
}

static void rigid_check(plx_geom_model *m, const measure_out *mo, const plx_mat *Vall, const plx_emerge_opts *o) {
    const plx_gspace *sp = &m->sp;
    int d = sp->ne;
    plx_mat C = plx_mat_new(d, d);
    weighted_scatter(mo->yV, m->w, m->n, d, &C);
    double tr = 0, *q = PLX_NEW(double, d), *f = PLX_NEW(double, d);
    for (int k = 0; k < d; k++) tr += PLX_M(&C, k, k);
    double nv = plx_vec_norm(m->yv, d);
    for (int k = 0; k < d; k++) q[k] = m->yv[k] / nv;
    m->rigid_frac = quad(&C, q) / tr;
    double cm = 0, cs2 = 0;
    for (int j = 0; j < PLX_GEOM_NCTRL; j++) {
        for (int k = 0; k < d; k++) q[k] = j + 1 < d ? PLX_M(Vall, k, j + 1) : 0;
        m->rigid_ctrl[j] = quad(&C, q) / tr;
        cm += m->rigid_ctrl[j] / PLX_GEOM_NCTRL;
    }
    for (int j = 0; j < PLX_GEOM_NCTRL; j++) cs2 += (m->rigid_ctrl[j] - cm) * (m->rigid_ctrl[j] - cm);
    m->rigid_z_ctrl = (m->rigid_frac - cm) / sqrt(cs2 / (PLX_GEOM_NCTRL - 1));
    plx_rng r;
    plx_rng_seed(&r, o->seed ^ 0x5c4a);
    int *perm = PLX_NEW(int, d), ns = o->n_scramble > 1 ? o->n_scramble : 2;
    double sm = 0, ss = 0;
    for (int j = 0; j < ns; j++) {
        for (int k = 0; k < d; k++) perm[k] = k;
        plx_rng_shuffle(&r, perm, d);
        for (int k = 0; k < d; k++) f[k] = sp->esw[k] * (m->yv[perm[k]] / sp->esw[perm[k]]);
        plx_gspace_remove_plane(sp, f, NULL);
        double nf = plx_vec_norm(f, d);
        for (int k = 0; k < d; k++) f[k] /= nf;
        double x = quad(&C, f) / tr;
        sm += x, ss += x * x;
    }
    m->rigid_scr_mean = sm / ns;
    m->rigid_scr_sd = sqrt(fmax(ss / ns - m->rigid_scr_mean * m->rigid_scr_mean, 0) * ns / (ns - 1));
    m->rigid_z = (m->rigid_frac - m->rigid_scr_mean) / m->rigid_scr_sd;
    free(perm);
    free(q);
    free(f);
    plx_mat_free(&C);
}

/* ---- step 5: metric scale -------------------------------------------------------------------------
 * Correction to the specified relation sin θ_i = s_i / κ. The similarity gauge removes each photo's
 * isotropic scale, and that scale is itself foreshortened: its least-squares scale is
 * Σg (X0 q_x + Y0 q_y) / Σg (X0² + Y0²) over the gauge points (weights g), so for a photo whose x
 * extent is foreshortened by cos θ and y by cos φ it divides by ρ = fx cos θ + (1 - fx) cos φ,
 * fx = Σg X0² / Σg (X0² + Y0²). E (and V) shrink with it, so E = sin θ z / ρ and
 *     sin θ_i = ρ_i s_i / κ,   sin φ_i cos θ_i = ρ_i t_i / κ,
 * solved per image by fixed-point iteration from ρ = 1 (a contraction: ρ varies slowly). Note that
 * |s| > κ is not by itself a violation: ρ < 1 at large angles. On noiseless synthetic data
 * the plain relation overestimates sin θ by 3.8% on average and up to 9% at 40°, and because that
 * error grows with |θ| exactly like the foreshortening curvature it biased the nonlinear κ ~6% high;
 * with ρ the slope of s on κ sin θ / ρ is 0.997. The anisotropy is unaffected (ρ cancels in A). */
int plx_geom_angles(double fx, double kappa, double s, double t, double *yaw, double *pitch) {
    double rho = 1, st = 0, sp = 0;
    if (fx < 0) { /* the specification's relation, for ablation */
        st = s / kappa, sp = t / kappa;
        int bad = (fabs(st) >= 1) | (fabs(sp) >= 1) << 1;
        if (yaw) *yaw = asin(PLX_CLAMP(st, -0.9999, 0.9999));
        if (pitch) *pitch = asin(PLX_CLAMP(sp, -0.9999, 0.9999));
        return bad;
    }
    int bad = 0;
    for (int it = 0; it < 100; it++) {
        double prev = rho;
        bad = 0;
        st = rho * s / kappa;
        if (fabs(st) >= 1) st = copysign(0.9999, st), bad |= 1;
        double ct = sqrt(1 - st * st);
        sp = rho * t / (kappa * ct);
        if (fabs(sp) >= 1) sp = copysign(0.9999, sp), bad |= 2;
        rho = fx * ct + (1 - fx) * sqrt(1 - sp * sp);
        if (fabs(rho - prev) < 1e-13) break;
    }
    if (yaw) *yaw = asin(st);
    if (pitch) *pitch = asin(sp);
    return bad;
}

/* The fit: log A_i = c + d t̃_i + log cos θ_i(κ) - log cos φ_i(κ), robust (Huber, fixed MAD scale),
 * with c, d profiled out by IRLS and λ = 1/κ² by grid + golden-section search. For small angles this
 * is the specification's c + ½ log(1 - s²/κ²) - ½ log(1 - t²/κ²).
 *
 * Correction to the specified model: the linear term d·t̃. Two effects put a term linear in pitch
 * into log A, both invisible in the specification's table: (i) t is pitch relative to the
 * population mean (the flow stage's mean subtraction removes mean(sin φ cos θ)·z from V), while
 * foreshortening is cos φ of the ABSOLUTE pitch: cos(φ̄ + Δφ) = cos φ̄ - sin φ̄ Δφ - ...; (ii) the
 * true mean depth has a Y-slope c1 (removed from z̄ by the plane gauge but not from the face), and
 * V's -sin φ cos θ z term then contributes -sin φ c1 to the Y0-slope γ: first order in φ, whereas
 * the signal 1 - cos φ is second order. Measured on synthetic data (sweeps.txt, pitch mean -6°,
 * tilt 0.15): d is clearly non-zero but κ barely moves without it, because for a symmetric pitch
 * distribution t is uncorrelated with t² and s². d is kept (and reported) because nothing makes
 * the pitch distribution of real photos symmetric. */
typedef struct {
    int n;
    const double *s, *t, *tl, *L, *w;
    double sigma, fx;
    bool lin;
} kfit_data;

static double huber_w(double r) { return fabs(r) <= 1.345 ? 1.0 : 1.345 / fabs(r); }
static double huber_rho(double r) {
    double a = fabs(r);
    return a <= 1.345 ? 0.5 * r * r : 1.345 * (a - 0.5 * 1.345);
}

/* log A minus the foreshortening predicted at κ */
static double kfit_y(const kfit_data *k, int i, double kappa) {
    double th, ph;
    plx_geom_angles(k->fx, kappa, k->s[i], k->t[i], &th, &ph);
    return k->L[i] - log(cos(th)) + log(cos(ph));
}

/* Profiled robust loss at λ; idx (nullable) is a bootstrap resample. cd receives (c, d). */
static double kfit_loss(const kfit_data *k, double lam, const int *idx, double cd[2]) {
    double c = 0, d = 0, loss = 0, kappa = 1 / sqrt(lam);
    double *y = PLX_NEW(double, k->n);
    for (int j = 0; j < k->n; j++) y[j] = kfit_y(k, idx ? idx[j] : j, kappa);
    for (int it = 0; it < 6; it++) {
        double a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
        for (int j = 0; j < k->n; j++) {
            int i = idx ? idx[j] : j;
            double r = (y[j] - c - d * k->tl[i]) / k->sigma, wt = k->w[i] * (it ? huber_w(r) : 1.0);
            double x = k->lin ? k->tl[i] : 0;
            a00 += wt, a01 += wt * x, a11 += wt * x * x, b0 += wt * y[j], b1 += wt * x * y[j];
        }
        if (k->lin) {
            double det = a00 * a11 - a01 * a01;
            c = (a11 * b0 - a01 * b1) / det, d = (a00 * b1 - a01 * b0) / det;
        } else
            c = b0 / a00, d = 0;
    }
    for (int j = 0; j < k->n; j++) loss += k->w[idx ? idx[j] : j] * huber_rho((y[j] - c - d * k->tl[idx ? idx[j] : j]) / k->sigma);
    free(y);
    if (cd) cd[0] = c, cd[1] = d;
    return loss;
}

/* Minimise the profiled loss over λ in (lo, hi]: grid, then golden section around the best cell. */
static double kfit_search(const kfit_data *k, double lo, double hi, int grid, const int *idx) {
    double best = hi, bl = INFINITY, step = (hi - lo) / grid;
    for (int g = 1; g <= grid; g++) {
        double lam = lo + step * g, l = kfit_loss(k, lam, idx, NULL);
        if (l < bl) bl = l, best = lam;
    }
    double a = fmax(best - step, lo + 1e-12 * hi), b = fmin(best + step, hi);
    const double gr = 0.5 * (sqrt(5.0) - 1);
    double x1 = b - gr * (b - a), x2 = a + gr * (b - a);
    double f1 = kfit_loss(k, x1, idx, NULL), f2 = kfit_loss(k, x2, idx, NULL);
    for (int it = 0; it < 24; it++) { /* bracket shrinks to 1e-5 of a grid cell */
        if (f1 < f2) b = x2, x2 = x1, f2 = f1, x1 = b - gr * (b - a), f1 = kfit_loss(k, x1, idx, NULL);
        else a = x1, x1 = x2, f1 = f2, x2 = a + gr * (b - a), f2 = kfit_loss(k, x2, idx, NULL);
    }
    return 0.5 * (a + b);
}

/* Robust multiple regression y ≈ X β (p columns), Huber IRLS with MAD scale, each round solved as
 * the weighted least-squares problem on √w X (QR; normal equations would square the conditioning). */
static void robust_lin(const double *X, const double *y, const double *w, int n, int p, double *beta) {
    double *r = PLX_NEW(double, n), *wt = PLX_NEW(double, n), *ar = PLX_NEW(double, n);
    for (int i = 0; i < n; i++) wt[i] = w[i];
    plx_mat A = plx_mat_new(n, p), b = plx_mat_new(n, 1), x = plx_mat_new(p, 1);
    for (int it = 0; it < 10; it++) {
        for (int i = 0; i < n; i++) {
            double sw = sqrt(wt[i]);
            for (int a = 0; a < p; a++) PLX_M(&A, i, a) = sw * X[(size_t)i * p + a];
            PLX_M(&b, i, 0) = sw * y[i];
        }
        plx_lstsq(&A, &b, 0, 0, &x);
        for (int a = 0; a < p; a++) beta[a] = PLX_M(&x, a, 0);
        for (int i = 0; i < n; i++) {
            r[i] = y[i];
            for (int a = 0; a < p; a++) r[i] -= X[(size_t)i * p + a] * beta[a];
            ar[i] = fabs(r[i]);
        }
        double sig = 1.4826 * median_pos(ar, w, n);
        for (int i = 0; i < n; i++) wt[i] = w[i] * (sig > 0 ? huber_w(r[i] / sig) : 1.0);
    }
    plx_mat_free(&A), plx_mat_free(&b), plx_mat_free(&x);
    free(r), free(wt), free(ar);
}

typedef struct {
    const kfit_data *k;
    double lo, hi; /* resamples search a window around the full-data λ */
    uint64_t seed;
    double *out;
} boot_ctx;

static void boot_task(int64_t b, int tid, void *vc) {
    boot_ctx *c = vc;
    plx_rng r;
    plx_rng_seed(&r, c->seed + 7919 * (uint64_t)b);
    int n = c->k->n, *idx = PLX_NEW(int, n);
    for (int j = 0; j < n; j++) idx[j] = (int)plx_rng_below(&r, (uint32_t)n);
    c->out[b] = 1 / sqrt(kfit_search(c->k, c->lo, c->hi, 8, idx));
    free(idx);
}

/* ---- step 4b: pitch by shrunk generalised least squares (an addition to the specification) -------
 * The plain t_i = -⟨V⊥_i, v⟩/⟨v, v⟩ also measures cos φ ⟨b_i, v⟩/⟨v, v⟩: the person's own face
 * heights projected on the depth pattern. On synthetic data that is most of the pitch error (corr
 * 0.93 with the truth, rmse 3°) and, through errors-in-variables attenuation of the t² slope, it
 * biased κ_pitch ~15% high. With V⊥ = -t v + b + noise and t independent of b, the best linear
 * unbiased estimate is t = -qᵀV⊥, q = (C_b + σ²)⁻¹ v / vᵀ(C_b + σ²)⁻¹v. C_b cannot be separated from
 * the pitch variance along v, but it need not be: for C = Σ w V⊥ V⊥ᵀ = σ_t² v vᵀ + C_b + σ²,
 * Sherman-Morrison gives (C - σ_t² v vᵀ)⁻¹ v ∝ C⁻¹ v. qᵀv = 1 always.
 * The pure GLS q = C⁻¹v is fragile, though: the pitch signal lies along the TRUE depth pattern, and
 * C⁻¹ amplifies the small error of v̂ in low-variance directions (synthetic: pitch slope 0.90, and
 * worse than the plain projection when identity heights are absent). So C is shrunk towards
 * isotropy, q ∝ (C + μ I)⁻¹ v with μ = mean eigenvalue of C: on synthetic data slope 0.98-1.00,
 * corr 0.98, rmse 1.6° with identity heights, and no loss without them. */
static void gls_direction(const double *Y, const double *w, int n, int d, const double *v, double *q) {
    plx_mat C = plx_mat_new(d, d);
    weighted_scatter(Y, w, n, d, &C);
    double tr = 0;
    for (int k = 0; k < d; k++) tr += PLX_M(&C, k, k);
    for (int k = 0; k < d; k++) PLX_M(&C, k, k) += tr / d;
    memcpy(q, v, sizeof(double) * d);
    PLX_ASSERT(plx_chol(&C) == PLX_OK);
    plx_chol_solve_vec(&C, q);
    double vq = plx_vec_dot(v, q, d);
    for (int k = 0; k < d; k++) q[k] /= vq;
    plx_mat_free(&C);
}

static void gls_pitch(plx_geom_model *m, const measure_out *mo) {
    int d = m->sp.ne;
    double *q = PLX_NEW(double, d);
    gls_direction(mo->yV, m->w, m->n, d, m->yv, q);
    for (int i = 0; i < m->n; i++) m->t[i] = -plx_vec_dot(mo->yV + (size_t)i * d, q, d);
    free(q);
}

static void fit_kappa(plx_geom_model *m, const plx_emerge_opts *o) {
    int n = m->n, *sel = PLX_NEW(int, n), ns = 0;
    for (int i = 0; i < n; i++) {
        double a = 1 + m->beta[i], g = 1 + m->gamma[i];
        m->logA[i] = (a > 0 && g > 0) ? log(a / g) : NAN;
        if (isfinite(m->logA[i]) && m->w[i] >= 0.5) sel[ns++] = i;
    }
    /* The linear regressor is t with its [1, s²] component removed. Residual roll after the
     * similarity gauge is δρ ≈ ½ sin θ c1 (it absorbs E's Y0-slope), and the even part of δρ x' is
     * δρ sin θ z, so t carries a term ∝ c1 s²: a raw d·t would then absorb the foreshortening
     * signal itself (on noiseless synthetic data with no pitch, κ came out 40% high). True pitch is
     * independent of yaw, so the projection removes only that leakage. */
    double *tl = PLX_NEW(double, n);
    {
        double a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
        for (int j = 0; j < ns; j++) {
            int i = sel[j];
            double x = m->s[i] * m->s[i], wt = m->w[i];
            a00 += wt, a01 += wt * x, a11 += wt * x * x, b0 += wt * m->t[i], b1 += wt * x * m->t[i];
        }
        double det = a00 * a11 - a01 * a01, c0 = (a11 * b0 - a01 * b1) / det, c1 = (a00 * b1 - a01 * b0) / det;
        for (int i = 0; i < n; i++) tl[i] = o->no_pitch_lin ? 0 : m->t[i] - c0 - c1 * m->s[i] * m->s[i];
    }
    double *X = PLX_NEW(double, 4 * ns + 4), *y = PLX_NEW(double, ns + 1), *ww = PLX_NEW(double, ns + 1), b[4];
    for (int j = 0; j < ns; j++) {
        int i = sel[j];
        double s = m->s[i], t = m->t[i];
        X[4 * j] = 1, X[4 * j + 1] = tl[i], X[4 * j + 2] = s * s, X[4 * j + 3] = t * t;
        y[j] = m->logA[i], ww[j] = m->w[i];
    }
    /* linearised: slope on s² is -1/(2κ²), slope on t² is +1/(2κ²) */
    robust_lin(X, y, ww, ns, 4, b);
    m->kappa_yaw = b[2] < 0 ? sqrt(-0.5 / b[2]) : NAN;
    m->kappa_pitch = b[3] > 0 ? sqrt(0.5 / b[3]) : NAN;
    /* joint linearised start: one λ shared by both terms */
    for (int j = 0; j < ns; j++) X[4 * j + 2] = 0.5 * (X[4 * j + 3] - X[4 * j + 2]), X[4 * j + 3] = 0;
    double b3[4];
    {
        double *X3 = PLX_NEW(double, 3 * ns + 3);
        for (int j = 0; j < ns; j++)
            for (int a = 0; a < 3; a++) X3[3 * j + a] = X[4 * j + a];
        robust_lin(X3, y, ww, ns, 3, b3);
        free(X3);
    }
    double kap0 = b3[2] > 0 ? 1 / sqrt(b3[2]) : m->kappa_yaw;
    if (!isfinite(kap0)) kap0 = 1e3;
    /* The fixed fit set: images safely inside |s|, |t| < κ0 (images near or beyond κ are model
     * violations: the 1 - s²/κ² term has no real solution there). */
    double *ks = PLX_NEW(double, n), *kt = PLX_NEW(double, n), *tt = PLX_NEW(double, n), *LL = PLX_NEW(double, n),
           *kw = PLX_NEW(double, n);
    int nk = 0;
    double mx = 0;
    for (int j = 0; j < ns; j++) {
        int i = sel[j];
        double a = m->s[i] * m->s[i], c = m->t[i] * m->t[i];
        if (a >= 0.8 * kap0 * kap0 || c >= 0.8 * kap0 * kap0) continue;
        ks[nk] = m->s[i], kt[nk] = m->t[i], tt[nk] = tl[i], LL[nk] = m->logA[i], kw[nk] = m->w[i];
        mx = fmax(mx, fmax(a, c));
        sel[nk++] = i;
    }
    for (int i = 0; i < n; i++) m->flags[i] |= PLX_GF_NOKAPPA;
    for (int j = 0; j < nk; j++) m->flags[sel[j]] &= (uint8_t)~PLX_GF_NOKAPPA;
    kfit_data k = {.n = nk, .s = ks, .t = kt, .tl = tt, .L = LL, .w = kw, .sigma = 1, .fx = m->fx,
                   .lin = !o->no_pitch_lin};
    {
        double cd[2], *ar = PLX_NEW(double, nk + 1);
        kfit_loss(&k, 1 / (kap0 * kap0), NULL, cd);
        for (int j = 0; j < nk; j++) ar[j] = fabs(kfit_y(&k, j, kap0) - cd[0] - cd[1] * tt[j]);
        k.sigma = 1.4826 * median_pos(ar, kw, nk);
        if (!(k.sigma > 0)) k.sigma = 1e-3;
        free(ar);
    }
    /* λ ≤ 0.98 / max(s², t²) keeps every fit image's angles real (ρ ≤ 1) */
    double lmax = 0.98 / mx, lam = kfit_search(&k, 0, lmax, 200, NULL), cd[2];
    kfit_loss(&k, lam, NULL, cd);
    m->kappa = 1 / sqrt(lam);
    m->kappa_c = cd[0], m->kappa_d = cd[1], m->kappa_scale = k.sigma, m->kappa_n = nk;
    if (o->boot > 1) {
        double *bk = PLX_NEW(double, o->boot);
        boot_ctx bc = {.k = &k, .lo = 0.4 * lam, .hi = fmin(1.8 * lam, lmax), .seed = o->seed ^ 0xb007, .out = bk};
        plx_parallel_for(o->boot, 1, boot_task, &bc);
        qsort(bk, o->boot, sizeof(double), cmp_double);
        m->kappa_lo = bk[(int)floor(0.025 * (o->boot - 1))];
        m->kappa_hi = bk[(int)ceil(0.975 * (o->boot - 1))];
        free(bk);
    } else
        m->kappa_lo = m->kappa_hi = NAN;
    for (int i = 0; i < n; i++) {
        int bad = plx_geom_angles(m->fx, m->kappa, m->s[i], m->t[i], NULL, NULL);
        if (bad & 1) m->flags[i] |= PLX_GF_YAWVIOL;
        if (bad & 2) m->flags[i] |= PLX_GF_PITCHVIOL;
    }
    free(sel), free(tl), free(X), free(y), free(ww), free(ks), free(kt), free(tt), free(LL), free(kw);
}

/* ---- step 6: depth variation with uncertainty ------------------------------------------------
 * R_i = E⊥_i - s_i v = s_i B c_i + n_i, so Cov(R | s) = s² B Bᵀ + Σ0: regress R_i R_iᵀ on s_i²
 * (weighted LS with intercept); slope Σ_B, intercept Σ0. Everything lives in an orthonormal basis
 * Uc of the complement of span{1, Y0, v} (R is in it by construction; B must be by the gauges).
 * Weights: robust weight / (1 + τ s²)², τ = tr Σ_B / tr Σ0 from a first pass, because the
 * variance of R Rᵀ grows like (Σ0 + s² Σ_B)². */
static void eig_sym(const plx_mat *A, double *w, plx_mat *V) { PLX_ASSERT(plx_eigh(A, w, V) == PLX_OK); }

static void fit_depth_basis(plx_geom_model *m, const measure_out *mo, const plx_emerge_opts *o) {
    const plx_gspace *sp = &m->sp;
    int n = m->n, d = sp->ne, nr = d - 3;
    PLX_ASSERT(nr >= 1);
    m->nr = nr;
    /* complement basis: eigenvectors of I - Q Qᵀ with eigenvalue 1, Q orthonormal span{1, Y0, v} */
    plx_mat Pm = plx_mat_identity(d), Vd = plx_mat_new(d, d);
    {
        double *q = PLX_NEW(double, 3 * d);
        memcpy(q, sp->e1, sizeof(double) * d);
        memcpy(q + d, sp->eY, sizeof(double) * d);
        memcpy(q + 2 * d, m->yv, sizeof(double) * d);
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < a; b++) plx_vec_axpy(q + a * d, -plx_vec_dot(q + a * d, q + b * d, d), q + b * d, d);
            double nq = plx_vec_norm(q + a * d, d);
            for (int k = 0; k < d; k++) q[a * d + k] /= nq;
        }
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) PLX_M(&Pm, i, j) -= q[a * d + i] * q[a * d + j];
        free(q);
    }
    double *ev = PLX_NEW(double, d);
    eig_sym(&Pm, ev, &Vd);
    m->Uc = PLX_NEW(double, (size_t)d * nr);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < nr; j++) m->Uc[(size_t)i * nr + j] = PLX_M(&Vd, i, j);
    free(ev);
    plx_mat_free(&Pm);
    plx_mat_free(&Vd);

    double *R = PLX_NEW(double, (size_t)n * nr), *y = PLX_NEW(double, d);
    for (int i = 0; i < n; i++) {
        const double *e = mo->yE + (size_t)i * d;
        for (int k = 0; k < d; k++) y[k] = e[k] - m->s[i] * m->yv[k];
        for (int j = 0; j < nr; j++) {
            double acc = 0;
            for (int k = 0; k < d; k++) acc += m->Uc[(size_t)k * nr + j] * y[k];
            R[(size_t)i * nr + j] = acc;
        }
    }
    free(y);
    plx_mat SB = plx_mat_new(nr, nr), S0 = plx_mat_new(nr, nr), Mp = plx_mat_new(nr, nr), Mn = plx_mat_new(nr, nr);
    double *h = PLX_NEW(double, n), *hp = PLX_NEW(double, n), tau = 0;
    for (int pass = 0; pass < 2; pass++) {
        double hs = 0, hm = 0, den = 0;
        for (int i = 0; i < n; i++) {
            double s2 = m->s[i] * m->s[i], q = 1 + tau * s2;
            h[i] = m->w[i] / (q * q);
            hs += h[i], hm += h[i] * s2;
        }
        hm /= hs;
        for (int i = 0; i < n; i++) den += h[i] * (m->s[i] * m->s[i] - hm) * (m->s[i] * m->s[i] - hm);
        /* slope = Σ h (s² - m) R Rᵀ / den, split by sign for the syrk kernel */
        for (int i = 0; i < n; i++) hp[i] = fmax(h[i] * (m->s[i] * m->s[i] - hm), 0);
        weighted_scatter(R, hp, n, nr, &Mp);
        for (int i = 0; i < n; i++) hp[i] = fmax(-h[i] * (m->s[i] * m->s[i] - hm), 0);
        weighted_scatter(R, hp, n, nr, &Mn);
        plx_mat_copy(&SB, &Mp);
        plx_mat_add(&SB, -1.0, &Mn);
        plx_mat_scale(&SB, 1.0 / den);
        /* intercept = weighted mean of R Rᵀ minus m · slope */
        weighted_scatter(R, h, n, nr, &S0);
        plx_mat_scale(&S0, 1.0 / hs);
        plx_mat_add(&S0, -hm, &SB);
        double tb = 0, t0 = 0;
        for (int k = 0; k < nr; k++) tb += PLX_M(&SB, k, k), t0 += PLX_M(&S0, k, k);
        m->sb_trace = tb, m->s0_trace = t0;
        tau = (tb > 0 && t0 > 0) ? tb / t0 : 0;
    }
    /* B: positive part of Σ_B. The negative eigenvalues measure the estimation noise of Σ_B (its
     * true value is PSD), so candidate modes must stand above 2 x |most negative|; among those, K_z
     * is placed at the largest eigenvalue ratio λ_k / λ_k+1. (The positive noise eigenvalues are
     * skewed upwards by second-order pose terms, so the floor alone kept 7 modes for 4 true ones on
     * synthetic data; the gap finds 4.) */
    double *wb = PLX_NEW(double, nr);
    plx_mat Vb = plx_mat_new(nr, nr);
    eig_sym(&SB, wb, &Vb);
    m->zeig = PLX_NEW(double, nr);
    memcpy(m->zeig, wb, sizeof(double) * nr);
    m->zeig_floor = wb[nr - 1] < 0 ? -wb[nr - 1] : 0;
    int kc = 0, kz = 0;
    while (kc < o->kz_max && kc < nr && wb[kc] > 2 * m->zeig_floor && wb[kc] > 0) kc++;
    double best = 0;
    int klast = 0;
    for (int k = 1; k <= kc; k++) {
        double next = k < nr ? fmax(wb[k], m->zeig_floor) : m->zeig_floor;
        double gap = next > 0 ? wb[k - 1] / next : INFINITY;
        if (gap > best) best = gap, kz = k;
        if (gap > 3) klast = k;
    }
    if (klast) kz = klast;
    m->kz = kz;
    m->Br = PLX_NEW(double, (size_t)nr * kz + 1);
    for (int i = 0; i < nr; i++)
        for (int j = 0; j < kz; j++) m->Br[(size_t)i * kz + j] = PLX_M(&Vb, i, j) * sqrt(wb[j]);
    /* Σ_rest: the positive part of Σ_B beyond the kept modes. It is pose-dependent residual
     * variance (depth variation too weak to resolve as modes, second-order pose terms, flow error
     * that grows with yaw) and the posterior must count it as noise: with Σ0 alone the codes were
     * ~20% overconfident on synthetic data (mean z² 1.22); with N_i = Σ0 + s_i² Σ_rest, see below. */
    m->Sr = PLX_NEW(double, (size_t)nr * nr);
    for (int i = 0; i < nr; i++)
        for (int j = 0; j < nr; j++) {
            double acc = 0;
            for (int k = kz; k < nr; k++)
                if (wb[k] > 0) acc += PLX_M(&Vb, i, k) * wb[k] * PLX_M(&Vb, j, k);
            m->Sr[(size_t)i * nr + j] = acc;
        }
    /* Σ0 regularised: eigenvalues floored at λ_max / sigma0_cond (the intercept of a regression is
     * not guaranteed PSD; the floor also bounds the condition number used by the posterior). */
    double *w0 = PLX_NEW(double, nr);
    plx_mat V0 = plx_mat_new(nr, nr);
    eig_sym(&S0, w0, &V0);
    double fl = fmax(w0[0], 1e-12) / o->sigma0_cond;
    m->S0 = PLX_NEW(double, (size_t)nr * nr);
    for (int i = 0; i < nr; i++)
        for (int j = 0; j < nr; j++) {
            double acc = 0;
            for (int k = 0; k < nr; k++) acc += PLX_M(&V0, i, k) * fmax(w0[k], fl) * PLX_M(&V0, j, k);
            m->S0[(size_t)i * nr + j] = acc;
        }
    /* node fields of the modes */
    m->Bnode = PLX_NEW(double, (size_t)kz * sp->P + 1);
    double *yb = PLX_NEW(double, d);
    for (int a = 0; a < kz; a++) {
        for (int k = 0; k < d; k++) {
            double acc = 0;
            for (int j = 0; j < nr; j++) acc += m->Uc[(size_t)k * nr + j] * m->Br[(size_t)j * kz + a];
            yb[k] = acc;
        }
        plx_gspace_even_field(sp, yb, NAN, m->Bnode + (size_t)a * sp->P);
    }
    free(yb);
    m->chat = PLX_NEW(double, (size_t)n * kz + 1);
    m->Lam = PLX_NEW(double, (size_t)n * kz * kz + 1);
    for (int i = 0; i < n; i++)
        plx_geom_depth_code(m, mo->yE + (size_t)i * d, m->s[i], m->chat + (size_t)i * kz,
                            m->Lam + (size_t)i * kz * kz);
    free(R), free(h), free(hp), free(wb), free(w0);
    plx_mat_free(&SB), plx_mat_free(&S0), plx_mat_free(&Mp), plx_mat_free(&Mn), plx_mat_free(&Vb), plx_mat_free(&V0);
}

/* With N = Σ0 + s² Σ_rest: Λ = I + s² Bᵀ N⁻¹ B, ĉ = Λ⁻¹ s Bᵀ N⁻¹ R. */
void plx_geom_depth_code(const plx_geom_model *m, const double *yE, double s, double *c, double *Lam) {
    int kz = m->kz, nr = m->nr, d = m->sp.ne;
    if (!kz) return;
    double *r = PLX_NEW(double, nr), *y = PLX_NEW(double, d);
    for (int k = 0; k < d; k++) y[k] = yE[k] - s * m->yv[k];
    for (int j = 0; j < nr; j++) {
        double acc = 0;
        for (int k = 0; k < d; k++) acc += m->Uc[(size_t)k * nr + j] * y[k];
        r[j] = acc;
    }
    plx_mat N = plx_mat_new(nr, nr), X = plx_mat_from(nr, kz, m->Br), L = plx_mat_new(kz, kz);
    for (int i = 0; i < nr; i++)
        for (int j = 0; j <= i; j++) PLX_M(&N, i, j) = m->S0[(size_t)i * nr + j] + s * s * m->Sr[(size_t)i * nr + j];
    PLX_ASSERT(plx_chol(&N) == PLX_OK);
    plx_chol_solve(&N, &X); /* X = N⁻¹ B */
    for (int a = 0; a < kz; a++) {
        double acc = 0;
        for (int j = 0; j < nr; j++) acc += PLX_M(&X, j, a) * r[j];
        c[a] = s * acc;
        for (int b = 0; b < kz; b++) {
            double g = 0;
            for (int j = 0; j < nr; j++) g += m->Br[(size_t)j * kz + a] * PLX_M(&X, j, b);
            PLX_M(&L, a, b) = Lam[a * kz + b] = (a == b) + s * s * g;
        }
    }
    /* symmetrise exactly (the two triangles differ by rounding) */
    for (int a = 0; a < kz; a++)
        for (int b = 0; b < a; b++)
            Lam[a * kz + b] = Lam[b * kz + a] = PLX_M(&L, a, b) = PLX_M(&L, b, a) = 0.5 * (Lam[a * kz + b] + Lam[b * kz + a]);
    PLX_ASSERT(plx_chol(&L) == PLX_OK);
    plx_chol_solve_vec(&L, c);
    plx_mat_free(&N), plx_mat_free(&X), plx_mat_free(&L);
    free(r), free(y);
}

/* Information form: image i contributes exp(h_iᵀ c - ½ cᵀ P_i c), P_i = Λ_i - I, h_i = Λ_i ĉ_i.
 * Integrating c ~ N(0, I) out: log p = ½ hᵀ(I + ΣP)⁻¹h - ½ log|I + ΣP| for the summed evidence, so
 *   LLR = ½ (log|Λ1| + log|Λ2| - log|Λs|) + ½ (hsᵀ Λs⁻¹ hs - ĉ1ᵀΛ1ĉ1 - ĉ2ᵀΛ2ĉ2),
 * Λs = Λ1 + Λ2 - I, hs = h1 + h2. */
double plx_geom_llr(int kz, const double *c1, const double *L1, const double *c2, const double *L2) {
    if (kz <= 0) return 0;
    plx_mat A = plx_mat_new(kz, kz), B = plx_mat_new(kz, kz), S = plx_mat_new(kz, kz);
    double *hs = PLX_NEW(double, kz), q1 = 0, q2 = 0;
    for (int a = 0; a < kz; a++) {
        double h1 = 0, h2 = 0;
        for (int b = 0; b < kz; b++) {
            PLX_M(&A, a, b) = L1[a * kz + b];
            PLX_M(&B, a, b) = L2[a * kz + b];
            PLX_M(&S, a, b) = L1[a * kz + b] + L2[a * kz + b] - (a == b);
            h1 += L1[a * kz + b] * c1[b];
            h2 += L2[a * kz + b] * c2[b];
        }
        hs[a] = h1 + h2;
        q1 += c1[a] * h1, q2 += c2[a] * h2;
    }
    PLX_ASSERT(plx_chol(&A) == PLX_OK && plx_chol(&B) == PLX_OK && plx_chol(&S) == PLX_OK);
    double ld = plx_chol_logdet(&A) + plx_chol_logdet(&B) - plx_chol_logdet(&S);
    double *x = PLX_NEW(double, kz);
    memcpy(x, hs, sizeof(double) * kz);
    plx_chol_solve_vec(&S, x);
    double qs = plx_vec_dot(hs, x, kz);
    free(x), free(hs);
    plx_mat_free(&A), plx_mat_free(&B), plx_mat_free(&S);
    return 0.5 * ld + 0.5 * (qs - q1 - q2);
}

/* ---- step 7: in-plane shape ---------------------------------------------------------------------
 * x_i = [O_i - β_i X0 ; V⊥_i + t_i v] (foreshortening and pitch parallax removed), weighted PCA.
 * Modes kept: eigenvalues above twice the median eigenvalue (the i.i.d. flow-noise floor). */
static void fit_shape_basis(plx_geom_model *m, const measure_out *mo, const plx_emerge_opts *o) {
    const plx_gspace *sp = &m->sp;
    int n = m->n, no = sp->no, ne = sp->ne, D = no + ne;
    double *X = PLX_NEW(double, (size_t)n * D), *mu = PLX_NEW(double, D), ws = 0;
    for (int i = 0; i < n; i++) {
        double *x = X + (size_t)i * D;
        for (int k = 0; k < no; k++) x[k] = mo->yO[(size_t)i * no + k] - m->beta[i] * sp->oX[k];
        for (int k = 0; k < ne; k++) x[no + k] = mo->yV[(size_t)i * ne + k] + m->t[i] * m->yv[k];
        ws += m->w[i];
        for (int k = 0; k < D; k++) mu[k] += m->w[i] * x[k];
    }
    for (int k = 0; k < D; k++) mu[k] /= ws;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < D; k++) X[(size_t)i * D + k] -= mu[k];
    plx_mat C = plx_mat_new(D, D), V = plx_mat_new(D, D);
    weighted_scatter(X, m->w, n, D, &C);
    plx_mat_scale(&C, 1.0 / ws);
    m->nsh = D;
    m->sheig = PLX_NEW(double, D);
    eig_sym(&C, m->sheig, &V);
    double *srt = PLX_NEW(double, D);
    memcpy(srt, m->sheig, sizeof(double) * D);
    qsort(srt, D, sizeof(double), cmp_double);
    double med = srt[D / 2];
    free(srt);
    int ks = 0;
    while (ks < o->ks_max && ks < D && m->sheig[ks] > 2 * med) ks++;
    m->ks = ks;
    m->shape = PLX_NEW(double, (size_t)ks * 2 * sp->P + 1);
    m->scode = PLX_NEW(double, (size_t)n * ks + 1);
    double *ax = PLX_NEW(double, D);
    for (int a = 0; a < ks; a++) {
        double sd = sqrt(m->sheig[a]);
        for (int k = 0; k < D; k++) ax[k] = PLX_M(&V, k, a) * sd;
        plx_gspace_odd_field(sp, ax, NAN, m->shape + (size_t)a * 2 * sp->P);
        plx_gspace_even_field(sp, ax + no, NAN, m->shape + (size_t)a * 2 * sp->P + sp->P);
        for (int i = 0; i < n; i++) {
            double acc = 0;
            for (int k = 0; k < D; k++) acc += X[(size_t)i * D + k] * PLX_M(&V, k, a);
            m->scode[(size_t)i * ks + a] = acc / sd;
        }
    }
    free(ax), free(X), free(mu);
    plx_mat_free(&C), plx_mat_free(&V);
}

/* ---- step 8: depth tilt ----------------------------------------------------------------------------
 * The plane gauge removes z̄'s Y-slope c1 because roll leaves the same -δρ Y0 in E. But roll also
 * leaves +δρ X0 in Q, while c1 does not: the Y0-slope of E is e = sin θ c1 - δρ and the X0-slope of
 * Q is q = sin φ sin θ + δρ (identity widths ignored), so e + q = sin θ (c1 + sin φ). With
 * sin θ = s/κ and sin φ ≈ t / (κ cos θ): regress e + q - s t / κ² on s / κ (robust, no intercept).
 * Pitch is relative to the population mean, so c1 is the tilt seen from the mean viewpoint. */
static void fit_tilt(plx_geom_model *m, const measure_out *mo) {
    int n = m->n;
    double *X = PLX_NEW(double, n), *y = PLX_NEW(double, n), *w = PLX_NEW(double, n), b;
    for (int i = 0; i < n; i++) {
        double k = m->kappa, sn = m->s[i] / k;
        X[i] = sn;
        y[i] = mo->planeE[2 * i + 1] + mo->qslope[i] - m->s[i] * m->t[i] / (k * k);
        w[i] = (m->flags[i] & (PLX_GF_YAWVIOL | PLX_GF_PITCHVIOL)) ? 0 : m->w[i];
    }
    robust_lin(X, y, w, n, 1, &b);
    double rs = 0, xs = 0, ws = 0;
    for (int i = 0; i < n; i++) {
        double r = y[i] - b * X[i];
        rs += w[i] * r * r, xs += w[i] * X[i] * X[i], ws += w[i];
    }
    m->tilt_c1 = b;
    m->tilt_se = sqrt(rs / ws / xs);
    free(X), free(y), free(w);
}

/* ---- driver ------------------------------------------------------------------------------------- */
int plx_emerge(const plx_glat *L, const float *u, int n, const float *node_weight, const uint8_t *core,
               plx_core_shape cs, const plx_emerge_opts *o, plx_geom_model *m) {
    memset(m, 0, sizeof *m);
    plx_gspace_init(&m->sp, L, node_weight, core);
    plx_gspace *sp = &m->sp;
    int ne = sp->ne, no = sp->no, P = sp->P;
    if (n < 4 * ne || ne < 6) {
        plx_log("emerge: %d images for %d even coordinates: too few", n, ne);
        plx_gspace_free(sp);
        return PLX_EINVAL;
    }
    m->n = n;
    m->v = PLX_NEW(double, P);
    m->yv = PLX_NEW(double, ne);
    m->s = PLX_NEW(double, n);
    m->t = PLX_NEW(double, n);
    m->t_raw = PLX_NEW(double, n);
    m->w = PLX_NEW(double, n);
    m->flags = PLX_NEW(uint8_t, n);
    m->beta = PLX_NEW(double, n);
    m->gamma = PLX_NEW(double, n);
    m->logA = PLX_NEW(double, n);
    m->tilt_c1 = m->tilt_se = NAN;
    {
        double wx = 0, wy = 0;
        for (int k = 0; k < P; k++) {
            double X = plx_glat_X0(L, k), Y = plx_glat_Y0(L, k), g = o->gauge_w ? o->gauge_w[k] : sp->omega[k];
            wx += g * X * X, wy += g * Y * Y;
        }
        m->fx = o->plain_angles ? -1 : wx / (wx + wy);
    }
    measure_out mo = {.yE = PLX_NEW(double, (size_t)n * ne), .yV = PLX_NEW(double, (size_t)n * ne),
                      .yO = PLX_NEW(double, (size_t)n * no), .beta = m->beta, .gamma = m->gamma,
                      .planeE = PLX_NEW(double, 2 * (size_t)n), .qslope = PLX_NEW(double, n)};
    measure_ctx mc = {.sp = sp, .u = u, .mo = &mo};
    plx_parallel_for(n, 64, measure_task, &mc);
    double *prior = PLX_NEW(double, n);
    for (int i = 0; i < n; i++) {
        prior[i] = o->prior_w ? PLX_CLAMP((double)o->prior_w[i], 0.0, 1.0) : 1.0;
        if (prior[i] < 0.5) m->flags[i] |= PLX_GF_PRIOR;
    }
    plx_mat Vall = plx_mat_new(ne, ne);
    fit_yaw_mode(m, &mo, prior, cs, o, &Vall);
    extend_v(m, u, &mo);
    rigid_check(m, &mo, &Vall, o);
    /* step 7 uses the plain t: V⊥ + t v is then exactly V⊥ with its v-component removed */
    fit_shape_basis(m, &mo, o);
    memcpy(m->t_raw, m->t, sizeof(double) * n);
    if (!o->plain_pitch) gls_pitch(m, &mo);
    fit_kappa(m, o);
    fit_depth_basis(m, &mo, o);
    if (o->tilt) fit_tilt(m, &mo);
    plx_mat_free(&Vall);
    free(prior);
    free(mo.yE), free(mo.yV), free(mo.yO), free(mo.planeE), free(mo.qslope);
    return PLX_OK;
}

void plx_geom_model_free(plx_geom_model *m) {
    plx_gspace_free(&m->sp);
    free(m->v), free(m->yv), free(m->s), free(m->t), free(m->t_raw), free(m->w), free(m->flags);
    free(m->beta), free(m->gamma), free(m->logA);
    free(m->zeig), free(m->Uc), free(m->Br), free(m->S0), free(m->Sr), free(m->Bnode);
    free(m->chat), free(m->Lam), free(m->sheig), free(m->shape), free(m->scode);
    memset(m, 0, sizeof *m);
}
