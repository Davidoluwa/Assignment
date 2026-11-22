/* Thin SVD by one-sided Jacobi with QR preconditioning (Drmac & Veselic, 2008).
 *
 * For tall A (m >= n):  A = Q R (blocked Householder),  R P = Q2 R2 (column pivoting),  then
 * Hestenes-Jacobi rotations orthogonalise the ROWS of R2 (= columns of R2^T, which converge in a
 * handful of sweeps thanks to the pivoting), accumulating the rotations in J:
 *     J R2 = G with orthogonal rows  =>  A P = (Q Q2 J^T) diag(|g_i|) (g_i / |g_i|)
 * so U = Q Q2 J^T is orthonormal by construction and V = P [g_i / |g_i|]. Working on rows keeps
 * every inner loop contiguous in row-major storage. Each stage is column-wise backward stable and
 * Jacobi computes the singular values of a well-conditioned-after-scaling matrix to high relative
 * accuracy, which is what lets small singular values come out right, not just the leading ones. */
#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

enum { MAX_SWEEPS = 60 };

/* Rows below this norm (after scaling A to max|a| ~ 1) cannot be normalised reliably: their
 * squared norms are in the underflow range, so they are treated as exactly zero. */
#define TINY_SV 1e-140

static void dot3(const double *x, const double *y, int n, double *xx, double *yy, double *xy) {
    double a[4] = {0}, b[4] = {0}, c[4] = {0};
    int i = 0;
    for (; i + 4 <= n; i += 4)
        for (int j = 0; j < 4; j++) {
            a[j] = fma(x[i + j], x[i + j], a[j]);
            b[j] = fma(y[i + j], y[i + j], b[j]);
            c[j] = fma(x[i + j], y[i + j], c[j]);
        }
    double ta = 0, tb = 0, tc = 0;
    for (; i < n; i++) {
        ta = fma(x[i], x[i], ta);
        tb = fma(y[i], y[i], tb);
        tc = fma(x[i], y[i], tc);
    }
    *xx = (a[0] + a[2]) + (a[1] + a[3]) + ta;
    *yy = (b[0] + b[2]) + (b[1] + b[3]) + tb;
    *xy = (c[0] + c[2]) + (c[1] + c[3]) + tc;
}

typedef struct {
    double *G, *J;
    int n, nblk;
    double tol;
    const int *pairs; /* block pairs of the current round */
    unsigned char *hit;
} jac_job;

/* Orthogonalises rows p and r of G (and rotates J alongside); returns 1 if a rotation was done. */
static int jacobi_rotate(const jac_job *jb, int p, int r) {
    const int n = jb->n;
    double *gp = jb->G + (size_t)p * n, *gr = jb->G + (size_t)r * n;
    double alpha, beta, gamma;
    dot3(gp, gr, n, &alpha, &beta, &gamma);
    if (!(fabs(gamma) > jb->tol * sqrt(alpha) * sqrt(beta))) return 0;
    /* Rutishauser's rotation: the smaller root of t^2 + 2 zeta t - 1 = 0 (|angle| <= pi/4) */
    double zeta = (beta - alpha) / (2.0 * gamma);
    double t = copysign(1.0, zeta) / (fabs(zeta) + hypot(1.0, zeta));
    double c = 1.0 / hypot(1.0, t), s = c * t;
    la_rot(gp, gr, (size_t)n, c, s);
    if (jb->J) la_rot(jb->J + (size_t)p * n, jb->J + (size_t)r * n, (size_t)n, c, s);
    return 1;
}

static int blk_lo(const jac_job *jb, int b) { return (int)((int64_t)b * jb->n / jb->nblk); }

static void jacobi_within(int64_t b, int tid, void *ctx) {
    const jac_job *jb = ctx;
    const int lo = blk_lo(jb, (int)b), hi = blk_lo(jb, (int)b + 1);
    int hit = 0;
    for (int p = lo; p < hi; p++)
        for (int r = p + 1; r < hi; r++) hit |= jacobi_rotate(jb, p, r);
    jb->hit[b] = (unsigned char)hit;
}

static void jacobi_cross(int64_t q, int tid, void *ctx) {
    const jac_job *jb = ctx;
    const int a = jb->pairs[2 * q], b = jb->pairs[2 * q + 1];
    int hit = 0;
    for (int p = blk_lo(jb, a); p < blk_lo(jb, a + 1); p++)
        for (int r = blk_lo(jb, b); r < blk_lo(jb, b + 1); r++) hit |= jacobi_rotate(jb, p, r);
    jb->hit[q] = (unsigned char)hit;
}

/* Block-cyclic parallel Jacobi. The rows are split into 2P blocks (P from n alone, so the result
 * does not depend on the thread count). A sweep first does every pair inside each block, then
 * 2P-1 tournament rounds in which P disjoint block pairs do all their cross pairs concurrently:
 * every row pair is visited once per sweep, a block pair (~2n/2P rows of G and J) stays in L2,
 * and there are ~2P parallel calls per sweep instead of the n-1 of a row-level tournament. */
static int jacobi_rows(double *G, int n, double *J) {
    const int P = PLX_CLAMP(n / 48, 1, 16), nblk = 2 * P;
    int *players = plx_malloc(sizeof(int) * nblk), *pairs = plx_malloc(sizeof(int) * nblk);
    unsigned char *hit = plx_malloc((size_t)nblk);
    /* convergence when every pair has |g_p . g_q| <= tol |g_p| |g_q|; sqrt(n) eps is the size of
     * the rounding noise in a length-n dot product, so a tighter tolerance would chase noise. */
    jac_job jb = {G, J, n, nblk, sqrt((double)n) * DBL_EPSILON, pairs, hit};
    const double bs = (double)n / nblk, pair_flops = (J ? 18.0 : 12.0) * n;
    int rc = n <= 1 ? PLX_OK : PLX_ENOCONV;
    for (int sweep = 0; sweep < MAX_SWEEPS && rc != PLX_OK; sweep++) {
        int rotated = 0;
        la_par_for(nblk, 0.5 * bs * bs * pair_flops, jacobi_within, &jb);
        for (int i = 0; i < nblk; i++) rotated |= hit[i];
        for (int i = 0; i < nblk; i++) players[i] = i;
        for (int round = 0; round < nblk - 1; round++) {
            for (int i = 0; i < P; i++) {
                pairs[2 * i] = players[i];
                pairs[2 * i + 1] = players[nblk - 1 - i];
            }
            la_par_for(P, bs * bs * pair_flops, jacobi_cross, &jb);
            for (int i = 0; i < P; i++) rotated |= hit[i];
            int last = players[nblk - 1]; /* keep players[0] fixed, rotate the rest by one */
            memmove(players + 2, players + 1, sizeof(int) * (nblk - 2));
            players[1] = last;
        }
        if (!rotated) rc = PLX_OK;
    }
    free(players);
    free(pairs);
    free(hit);
    return rc;
}

/* Orthonormal completion: columns r..n-1 of V (n x n) become an orthonormal basis of the
 * complement of span(V[:, 0:r]): the trailing columns of the full Householder Q of V[:, 0:r]. */
static void complete_basis(double *V, int n, int r) {
    double *B = plx_malloc(sizeof(double) * (size_t)n * PLX_MAX(r, 1));
    double *E = PLX_NEW(double, (size_t)n * (n - r));
    for (int i = 0; i < n; i++) memcpy(B + (size_t)i * r, V + (size_t)i * n, sizeof(double) * r);
    for (int c = 0; c < n - r; c++) E[(size_t)(r + c) * (n - r) + c] = 1.0;
    la_qr q;
    la_qr_factor(&q, B, n, r, r);
    la_qr_apply(&q, 0, E, n - r, n - r);
    la_qr_free(&q);
    for (int i = 0; i < n; i++) memcpy(V + (size_t)i * n + r, E + (size_t)i * (n - r), sizeof(double) * (n - r));
    free(B);
    free(E);
}

/* m >= n. U (m x n) nullable; V (n x n) required. */
static int svd_tall(double *a, int m, int n, double *U, double *s, double *V) {
    double amax = 0;
    for (size_t i = 0; i < (size_t)m * n; i++) {
        if (!isfinite(a[i])) return PLX_EINVAL;
        amax = PLX_MAX(amax, fabs(a[i]));
    }
    if (amax == 0) {
        memset(s, 0, sizeof(double) * n);
        memset(V, 0, sizeof(double) * (size_t)n * n);
        for (int i = 0; i < n; i++) V[(size_t)i * n + i] = 1;
        if (U) {
            memset(U, 0, sizeof(double) * (size_t)m * n);
            for (int i = 0; i < n; i++) U[(size_t)i * n + i] = 1;
        }
        return PLX_OK;
    }
    int ex;
    frexp(amax, &ex);
    const double scale = ldexp(1.0, -ex); /* power of two: exact, keeps squared norms in range */
    for (size_t i = 0; i < (size_t)m * n; i++) a[i] *= scale;

    la_qr q1 = {0};
    double *R = a;
    if (m > n) {
        la_qr_factor(&q1, a, m, n, n);
        R = PLX_NEW(double, (size_t)n * n);
        for (int i = 0; i < n; i++) memcpy(R + (size_t)i * n + i, a + (size_t)i * n + i, sizeof(double) * (n - i));
    }
    int *perm = plx_malloc(sizeof(int) * n);
    la_qr q2;
    la_qrcp_factor(&q2, R, n, n, n, perm);
    double *G = PLX_NEW(double, (size_t)n * n), *J = NULL;
    for (int i = 0; i < n; i++) memcpy(G + (size_t)i * n + i, R + (size_t)i * n + i, sizeof(double) * (n - i));
    if (U) {
        J = PLX_NEW(double, (size_t)n * n);
        for (int i = 0; i < n; i++) J[(size_t)i * n + i] = 1;
    }
    int rc = jacobi_rows(G, n, J);

    double *sv = plx_malloc(sizeof(double) * n);
    int *order = plx_malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) sv[i] = la_norm_strided(G + (size_t)i * n, 1, n);
    la_argsort_desc(sv, n, order);
    int rank = 0;
    for (int j = 0; j < n; j++) {
        const double sj = sv[order[j]];
        const double *g = G + (size_t)order[j] * n;
        if (sj > TINY_SV) {
            rank = j + 1;
            for (int c = 0; c < n; c++) V[(size_t)perm[c] * n + j] = g[c] / sj;
        }
        s[j] = sj > TINY_SV ? sj / scale : 0.0;
    }
    if (rank < n) complete_basis(V, n, rank);

    if (U) { /* U = Q [Q2 J^T(:, order); 0] */
        double *M = plx_malloc(sizeof(double) * (size_t)n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) M[(size_t)i * n + j] = J[(size_t)order[j] * n + i];
        la_qr_apply(&q2, 0, M, n, n);
        if (m > n) {
            memset(U, 0, sizeof(double) * (size_t)m * n);
            memcpy(U, M, sizeof(double) * (size_t)n * n);
            la_qr_apply(&q1, 0, U, n, n);
        } else {
            memcpy(U, M, sizeof(double) * (size_t)n * n);
        }
        free(M);
    }
    la_fix_signs(V, n, n, U, m);
    la_qr_free(&q2);
    if (m > n) {
        la_qr_free(&q1);
        free(R);
    }
    free(perm);
    free(G);
    free(J);
    free(sv);
    free(order);
    return rc;
}

int la_svd_raw(double *a, int m, int n, double *U, double *s, double *V) {
    if (m >= n) {
        double *Vt = V ? V : plx_malloc(sizeof(double) * (size_t)n * n);
        int rc = svd_tall(a, m, n, U, s, Vt);
        if (!V) free(Vt);
        return rc;
    }
    /* wide: A^T = U' S V'^T, so U = V' and V = U' */
    double *at = plx_malloc(sizeof(double) * (size_t)m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) at[(size_t)j * m + i] = a[(size_t)i * n + j];
    double *Ut = U ? U : plx_malloc(sizeof(double) * (size_t)m * m);
    int rc = svd_tall(at, n, m, V, s, Ut);
    if (!U) free(Ut);
    free(at);
    return rc;
}

int plx_svd(const plx_mat *A, plx_mat *U, double *s, plx_mat *V) {
    const int m = A->rows, n = A->cols, k = PLX_MIN(m, n);
    PLX_ASSERT(m >= 1 && n >= 1);
    PLX_ASSERT(!U || (U->rows == m && U->cols == k));
    PLX_ASSERT(!V || (V->rows == n && V->cols == k));
    double *a = plx_malloc(sizeof(double) * (size_t)m * n);
    memcpy(a, A->a, sizeof(double) * (size_t)m * n);
    int rc = la_svd_raw(a, m, n, U ? U->a : NULL, s, V ? V->a : NULL);
    free(a);
    return rc;
}
