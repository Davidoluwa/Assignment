/* Symmetric eigendecomposition: Householder tridiagonalisation + implicit-shift QL.
 *
 * Cost structure for n in the thousands, and how it is handled:
 *   - Tridiagonalisation (4/3 n^3) is level-2 and memory bound. Rows are contiguous, so each step
 *     is one independent "rank-2 update, then dot with the next reflector" per row, threaded over
 *     rows: fusing step k's update with step k+1's matrix-vector product streams the trailing
 *     matrix once per step instead of twice.
 *   - Forming Q^T (4/3 n^3) groups the reflectors into compact-WY blocks and runs on gemm.
 *   - QL itself is O(n^2) on (d, e). Accumulating its ~n^2 Givens rotations into the eigenvectors
 *     is the 6 n^3 hot spot, and applying each sweep directly would stream the whole n x n matrix
 *     through memory ~2n times. But the rotations never depend on the vectors, so they are recorded
 *     and replayed later on narrow column slices of Z^T that stay in L1: every slice sees the whole
 *     rotation sequence, slices are independent (threaded), and each rotation touches two
 *     contiguous row segments, which vectorises. */
#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

/* ---- tridiagonalisation -------------------------------------------------------------------- */

typedef struct {
    double *W;
    int n, k;
    const double *vp, *wp; /* pending update S -= vp wp^T + wp vp^T, indexed from k (NULL: none) */
    const double *v;       /* reflector k, indexed from k+1, v[0] = 1 */
    double *p, tau;
} tri_job;

/* Row k+1+r: apply step k-1's rank-2 update to columns k+1.., then p_r = tau (row . v). */
static void tri_row(int64_t r, int tid, void *ctx) {
    const tri_job *t = ctx;
    const int L = t->n - t->k - 1, o = t->k + 1;
    double *row = t->W + (size_t)(o + r) * t->n + o;
    if (t->vp) {
        la_axpy(row, -t->vp[r + 1], t->wp + 1, (size_t)L);
        la_axpy(row, -t->wp[r + 1], t->vp + 1, (size_t)L);
    }
    t->p[r] = t->tau == 0 ? 0.0 : t->tau * la_dot(row, t->v, (size_t)L);
}

/* W (n x n, full symmetric) -> d, e (e[i] couples i and i+1, e[n-1] = 0) with Q^T W Q tridiagonal,
 * Q = H_0 ... H_{n-3}. Reflector k is stored in row k, entries k+1..n-1 (leading 1 explicit).
 * Step k: H S H = S - v w^T - w v^T with p = tau S v, w = p - (tau/2)(p.v) v, where S is the
 * trailing block. Step k-1's update is deferred into step k's pass over rows k+1..; only row k,
 * which defines reflector k, gets it up front. */
static void tridiagonalise(double *W, int n, double *d, double *e, double *tau) {
    double *wbuf[2] = {plx_malloc(sizeof(double) * n), plx_malloc(sizeof(double) * n)};
    const double *vp = NULL, *wp = NULL;
    for (int k = 0; k < n - 2; k++) {
        const int L = n - k - 1;
        double *x = W + (size_t)k * n + k + 1;
        if (vp) {
            la_axpy(x - 1, -vp[0], wp, (size_t)L + 1);
            la_axpy(x - 1, -wp[0], vp, (size_t)L + 1);
        }
        d[k] = W[(size_t)k * n + k];
        double xnorm = la_norm_strided(x + 1, 1, L - 1);
        tau[k] = 0;
        e[k] = x[0];
        if (xnorm != 0) {
            double alpha = x[0], beta = -copysign(hypot(alpha, xnorm), alpha);
            double scale = 1.0 / (alpha - beta);
            for (int i = 1; i < L; i++) x[i] *= scale;
            e[k] = beta;
            tau[k] = (beta - alpha) / beta;
        }
        x[0] = 1;
        double *p = wbuf[k & 1];
        tri_job t = {W, n, k, vp, wp, x, p, tau[k]};
        la_par_for(L, 6.0 * L, tri_row, &t);
        la_axpy(p, -0.5 * tau[k] * la_dot(p, x, (size_t)L), x, (size_t)L); /* p <- w */
        vp = x;
        wp = p;
    }
    if (vp) /* the last update reaches only the trailing 2 x 2 block */
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) W[(size_t)(n - 2 + a) * n + n - 2 + b] -= vp[a] * wp[b] + wp[a] * vp[b];
    if (n >= 2) {
        d[n - 2] = W[(size_t)(n - 2) * n + n - 2];
        e[n - 2] = W[(size_t)(n - 1) * n + n - 2];
        tau[n - 2] = 0;
    }
    d[n - 1] = W[(size_t)(n - 1) * n + n - 1];
    e[n - 1] = 0;
    free(wbuf[0]);
    free(wbuf[1]);
}

enum { WY_NB = 32 };

/* Zt = Q^T = H_{n-3} ... H_0, built block by block from the last reflectors to the first. For
 * the block k0..k1-1, H_{k0} ... H_{k1-1} = I - V T V^T, so Zt <- Zt H_{k1-1} ... H_{k0} =
 * Zt (I - V T^T V^T), three gemms. Before the block, Zt differs from I only in rows/cols > k1,
 * so only the trailing (n-k0-1)^2 block (where V is nonzero) is touched. */
static void form_qt(const double *W, const double *tau, int n, double *Zt, int64_t ldz) {
    memset(Zt, 0, sizeof(double) * (size_t)n * ldz);
    for (int i = 0; i < n; i++) Zt[(size_t)i * ldz + i] = 1;
    const int nr = n - 2; /* reflectors 0..n-3 */
    if (nr <= 0) return;
    const int mm_max = n - 1;
    double *V = plx_malloc(sizeof(double) * (size_t)mm_max * WY_NB), *T = plx_malloc(sizeof(double) * WY_NB * WY_NB);
    double *M = plx_malloc(sizeof(double) * (size_t)mm_max * WY_NB), *M2 = plx_malloc(sizeof(double) * (size_t)mm_max * WY_NB);
    for (int k0 = (nr - 1) / WY_NB * WY_NB; k0 >= 0; k0 -= WY_NB) {
        const int jb = PLX_MIN(WY_NB, nr - k0), mm = n - k0 - 1;
        for (int r = 0; r < mm; r++) /* V[r][c] = v_{k0+c} at global index k0+1+r */
            for (int c = 0; c < jb; c++)
                V[r * jb + c] = r < c ? 0.0 : r == c ? 1.0 : W[(size_t)(k0 + c) * n + k0 + 1 + r];
        la_wy_t(V, mm, jb, tau + k0, T, WY_NB);
        double *Z = Zt + (size_t)(k0 + 1) * ldz + k0 + 1;
        la_dgemm(0, 0, mm, jb, mm, 1.0, Z, ldz, V, jb, 0.0, M, jb);
        la_dgemm(0, 1, mm, jb, jb, 1.0, M, jb, T, WY_NB, 0.0, M2, jb);
        la_dgemm(0, 1, mm, mm, jb, -1.0, M2, jb, V, jb, 1.0, Z, ldz);
    }
    free(V);
    free(T);
    free(M);
    free(M2);
}

/* ---- QL with recorded rotations ------------------------------------------------------------ */

typedef struct {
    double c, s;
    int i; /* rotates rows i and i+1 of Zt */
} rot;

/* Slices are one cache line (8 doubles) wide: a slice of Zt is then 8n doubles (16 KB at
 * n = 2000) and stays in L1 while the rotation list streams past it from L2/L3; wider slices
 * measured 1.2-1.8x slower. Zt rows are padded to a multiple of 8 doubles and the array is
 * 64-byte aligned, so every slice is a column of whole cache lines and threads never share one. */
enum { ROT_SLICE = 8 };
typedef struct {
    rot *r;
    size_t n, cap;
    double *Zt;
    int64_t ldz;
    int dim;
} rot_log;

static inline void rot_line(double *restrict x, double *restrict y, double c, double s) {
    for (int j = 0; j < ROT_SLICE; j++) {
        double xj = x[j], yj = y[j];
        x[j] = c * xj - s * yj;
        y[j] = s * xj + c * yj;
    }
}

static void rot_apply_slice(int64_t b, int tid, void *ctx) {
    const rot_log *L = ctx;
    const int c0 = (int)b * ROT_SLICE, w = PLX_MIN(ROT_SLICE, L->dim - c0);
    double *base = L->Zt + c0;
    for (size_t q = 0; q < L->n; q++) {
        double *x = base + (size_t)L->r[q].i * L->ldz;
        if (w == ROT_SLICE) rot_line(x, x + L->ldz, L->r[q].c, L->r[q].s);
        else la_rot(x, x + L->ldz, (size_t)w, L->r[q].c, L->r[q].s);
    }
}

static void rot_flush(rot_log *L) {
    int nb = (L->dim + ROT_SLICE - 1) / ROT_SLICE;
    la_par_for(nb, 6.0 * (double)L->n * ROT_SLICE, rot_apply_slice, L);
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
    const int64_t ldz = (n + 7) / 8 * 8;
    if (V) {
        Zt = aligned_alloc(64, sizeof(double) * (size_t)n * ldz);
        if (!Zt) plx_die("out of memory (%d x %d eigenvectors)", n, n);
        form_qt(W, tau, n, Zt, ldz);
        log.cap = (size_t)1 << 21;
        log.r = plx_malloc(sizeof(rot) * log.cap);
        log.Zt = Zt;
        log.ldz = ldz;
        log.dim = n;
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
            for (int i = 0; i < n; i++) PLX_M(V, i, j) = Zt[(size_t)idx[j] * ldz + i];
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
