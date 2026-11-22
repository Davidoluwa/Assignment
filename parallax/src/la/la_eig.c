/* Symmetric eigendecomposition: Householder tridiagonalisation + implicit-shift QL.
 *
 * Cost structure for n in the thousands, and how it is handled:
 *   - Tridiagonalisation (4/3 n^3) and forming Q^T (4/3 n^3) are level-2 sweeps over the trailing
 *     matrix; rows are contiguous, so each step is a set of independent row dot/axpy operations,
 *     threaded over rows.
 *   - QL itself is O(n^2) on (d, e). Accumulating its ~n^2 Givens rotations into the eigenvectors
 *     is the 6 n^3 hot spot, and applying each sweep directly would stream the whole n x n matrix
 *     through memory ~2n times. But the rotations never depend on the vectors, so they are recorded
 *     and replayed later on narrow column slices of Z^T that stay in L2: every slice sees the whole
 *     rotation sequence, slices are independent (threaded), and each rotation touches two
 *     contiguous row segments, which vectorises. */
#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

/* ---- tridiagonalisation -------------------------------------------------------------------- */

typedef struct {
    double *W;
    int n, k;
    const double *v; /* length L = n - k - 1, v[0] = 1 */
    double *p, tau;
} tri_job;

static void tri_symv_row(int64_t i, int tid, void *ctx) {
    tri_job *t = ctx;
    const int L = t->n - t->k - 1, o = t->k + 1;
    t->p[i] = t->tau * la_dot(t->W + (size_t)(o + i) * t->n + o, t->v, (size_t)L);
}

static void tri_rank2_row(int64_t i, int tid, void *ctx) { /* p now holds w */
    tri_job *t = ctx;
    const int L = t->n - t->k - 1, o = t->k + 1;
    double *row = t->W + (size_t)(o + i) * t->n + o;
    la_axpy(row, -t->v[i], t->p, (size_t)L);
    la_axpy(row, -t->p[i], t->v, (size_t)L);
}

/* W (n x n, full symmetric) -> d, e (e[i] couples i and i+1, e[n-1] = 0) with Q^T W Q tridiagonal,
 * Q = H_0 ... H_{n-3}. Reflector k is stored in row k, entries k+1..n-1 (leading 1 explicit). */
static void tridiagonalise(double *W, int n, double *d, double *e, double *tau) {
    double *p = plx_malloc(sizeof(double) * n);
    for (int k = 0; k < n - 2; k++) {
        const int L = n - k - 1;
        double *x = W + (size_t)k * n + k + 1;
        d[k] = W[(size_t)k * n + k];
        double xnorm = la_norm_strided(x + 1, 1, L - 1);
        tau[k] = 0;
        if (xnorm == 0) {
            e[k] = x[0];
            x[0] = 1;
            continue;
        }
        double alpha = x[0], beta = -copysign(hypot(alpha, xnorm), alpha);
        double scale = 1.0 / (alpha - beta);
        for (int i = 1; i < L; i++) x[i] *= scale;
        x[0] = 1;
        e[k] = beta;
        tau[k] = (beta - alpha) / beta;
        /* H S H = S - v w^T - w v^T with p = tau S v, w = p - (tau/2)(p.v) v */
        tri_job t = {W, n, k, x, p, tau[k]};
        la_par_for(L, 2.0 * L, tri_symv_row, &t);
        double K = 0.5 * tau[k] * la_dot(p, x, (size_t)L);
        la_axpy(p, -K, x, (size_t)L);
        la_par_for(L, 4.0 * L, tri_rank2_row, &t);
    }
    if (n >= 2) {
        d[n - 2] = W[(size_t)(n - 2) * n + n - 2];
        e[n - 2] = W[(size_t)(n - 1) * n + n - 2];
        tau[n - 2] = 0;
    }
    d[n - 1] = W[(size_t)(n - 1) * n + n - 1];
    e[n - 1] = 0;
    free(p);
}

typedef struct {
    const double *W, *tau;
    double *Zt;
    int n, k;
} formq_job;

static void formq_row(int64_t r, int tid, void *ctx) {
    formq_job *f = ctx;
    const int o = f->k + 1, L = f->n - o;
    const double *v = f->W + (size_t)f->k * f->n + o;
    double *row = f->Zt + (size_t)(o + r) * f->n + o;
    la_axpy(row, -f->tau[f->k] * la_dot(row, v, (size_t)L), v, (size_t)L);
}

/* Zt = Q^T = H_{n-3} ... H_0, accumulated as Zt <- Zt H_k for k descending. Before step k, Zt
 * differs from I only in rows/cols >= k+2, so the step touches rows and columns k+1.. only, and
 * as a right-multiplication it is one dot + axpy per (contiguous) row. */
static void form_qt(const double *W, const double *tau, int n, double *Zt) {
    memset(Zt, 0, sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++) Zt[(size_t)i * n + i] = 1;
    for (int k = n - 3; k >= 0; k--) {
        if (tau[k] == 0) continue;
        formq_job f = {W, tau, Zt, n, k};
        la_par_for(n - k - 1, 4.0 * (n - k - 1), formq_row, &f);
    }
}

/* ---- QL with recorded rotations ------------------------------------------------------------ */

typedef struct {
    double c, s;
    int i; /* rotates rows i and i+1 of Zt */
} rot;

typedef struct {
    rot *r;
    size_t n, cap;
    double *Zt;
    int dim, slice;
} rot_log;

static void rot_apply_slice(int64_t b, int tid, void *ctx) {
    const rot_log *L = ctx;
    const int c0 = (int)b * L->slice, w = PLX_MIN(L->slice, L->dim - c0);
    for (size_t q = 0; q < L->n; q++) {
        double *x = L->Zt + (size_t)L->r[q].i * L->dim + c0;
        la_rot(x, x + L->dim, (size_t)w, L->r[q].c, L->r[q].s);
    }
}

static void rot_flush(rot_log *L) {
    int nb = (L->dim + L->slice - 1) / L->slice;
    la_par_for(nb, 6.0 * (double)L->n * L->slice, rot_apply_slice, L);
    L->n = 0;
}

static void rot_push(rot_log *L, double c, double s, int i) {
    if (L->n == L->cap) rot_flush(L); /* bounded memory: replay what we have so far */
    L->r[L->n++] = (rot){c, s, i};
}

/* Implicit QL with Wilkinson-style shift on the tridiagonal (d, e); e[i] couples d[i], d[i+1].
 * Each sweep chases the bulge from the bottom of the unreduced block [l, m] to its top.
 * Deflation: e[m] is dropped when it is negligible next to its neighbours (the classical local
 * test, which keeps graded matrices accurate) OR next to ||T||. The second test matters for
 * clusters of eigenvalues at rounding level (rank-deficient covariance matrices): the local test
 * would ask for |e| ~ eps * 1e-13 there, below the noise the bulge chase itself injects, and QL
 * would stall. Dropping |e| <= eps ||T|| is a perturbation no larger than the one the Householder
 * reduction already made, so it costs no accuracy we claim. */
static int tridiag_ql(double *d, double *e, int n, rot_log *log) {
    double tnorm = 0;
    for (int i = 0; i < n; i++) tnorm = PLX_MAX(tnorm, fabs(d[i]) + fabs(e[i]) + (i ? fabs(e[i - 1]) : 0.0));
    const double small = DBL_EPSILON * tnorm;
    int budget = 30 * n; /* total sweeps, as in LAPACK: ~2 per eigenvalue are typical */
    for (int l = 0; l < n; l++) {
        for (;;) {
            int m = l;
            for (; m < n - 1; m++)
                if (fabs(e[m]) <= DBL_EPSILON * (fabs(d[m]) + fabs(d[m + 1])) || fabs(e[m]) <= small) break;
            if (m == l) break;
            if (budget-- == 0) return PLX_ENOCONV;
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + copysign(r, g));
            double s = 1, c = 1, p = 0;
            int i = m - 1;
            for (; i >= l; i--) {
                double f = s * e[i], b = c * e[i];
                r = hypot(f, g);
                e[i + 1] = r;
                if (r == 0) { /* exact underflow split: the block decouples at i+1 */
                    d[i + 1] -= p;
                    e[m] = 0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                if (log) rot_push(log, c, s, i);
            }
            if (r == 0 && i >= l) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0;
        }
    }
    return PLX_OK;
}

/* ---- public ------------------------------------------------------------------------------- */

int plx_eigh(const plx_mat *A, double *w, plx_mat *V) {
    const int n = A->rows;
    PLX_ASSERT(A->cols == n && n >= 1);
    PLX_ASSERT(!V || (V->rows == n && V->cols == n));
    double *W = plx_malloc(sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++) /* lower triangle, mirrored */
        for (int j = 0; j <= i; j++) W[(size_t)i * n + j] = W[(size_t)j * n + i] = PLX_M(A, i, j);
    double *d = plx_malloc(sizeof(double) * n), *e = plx_malloc(sizeof(double) * n);
    double *tau = plx_malloc(sizeof(double) * n);
    tridiagonalise(W, n, d, e, tau);

    rot_log log = {0};
    double *Zt = NULL;
    if (V) {
        Zt = plx_malloc(sizeof(double) * (size_t)n * n);
        form_qt(W, tau, n, Zt);
        log.cap = (size_t)1 << 21;
        log.r = plx_malloc(sizeof(rot) * log.cap);
        log.Zt = Zt;
        log.dim = n;
        log.slice = PLX_CLAMP(65536 / n / 8 * 8, 8, 256); /* n x slice doubles ~ 512 KB: L2 resident */
    }
    int rc = tridiag_ql(d, e, n, V ? &log : NULL);
    int *idx = plx_malloc(sizeof(int) * n);
    if (rc == PLX_OK) {
        la_argsort_desc(d, n, idx);
        for (int j = 0; j < n; j++) w[j] = d[idx[j]];
    }
    if (V && rc == PLX_OK) {
        rot_flush(&log);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) PLX_M(V, i, j) = Zt[(size_t)idx[j] * n + i];
        la_fix_signs(V->a, n, n, NULL, 0);
    }
    free(log.r);
    free(Zt);
    free(idx);
    free(tau);
    free(e);
    free(d);
    free(W);
    return rc;
}

static void transpose_square(double *a, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            double t = a[(size_t)i * n + j];
            a[(size_t)i * n + j] = a[(size_t)j * n + i];
            a[(size_t)j * n + i] = t;
        }
}

int plx_eigh_gen(const plx_mat *A, const plx_mat *B, double *w, plx_mat *V) {
    const int n = A->rows;
    PLX_ASSERT(A->cols == n && B->rows == n && B->cols == n);
    PLX_ASSERT(!V || (V->rows == n && V->cols == n && V->a != B->a));
    plx_mat L = plx_mat_clone(B);
    int rc = plx_chol(&L);
    if (rc != PLX_OK) {
        plx_mat_free(&L);
        return rc;
    }
    plx_mat C = plx_mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) PLX_M(&C, i, j) = PLX_M(&C, j, i) = PLX_M(A, i, j);
    /* C = L^-1 A L^-T = L^-1 (L^-1 A)^T, using the symmetry of A: two triangular solves. */
    la_trsm(1, 0, 0, n, L.a, n, C.a, n, n);
    transpose_square(C.a, n);
    la_trsm(1, 0, 0, n, L.a, n, C.a, n, n);
    rc = plx_eigh(&C, w, V ? &C : NULL);
    if (rc == PLX_OK && V) {
        la_trsm(1, 1, 0, n, L.a, n, C.a, n, n); /* v = L^-T y */
        la_fix_signs(C.a, n, n, NULL, 0);
        plx_mat_copy(V, &C);
    }
    plx_mat_free(&C);
    plx_mat_free(&L);
    return rc;
}
