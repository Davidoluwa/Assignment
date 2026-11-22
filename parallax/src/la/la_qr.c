/* Householder QR (blocked, compact WY), column-pivoted QR, least squares. */
#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

enum { NB = LA_QR_NB };

/* Householder reflector for x (n entries, stride inc): on return x[0] = beta, x[1..] = v[1..] with
 * v[0] = 1 implicit, and (I - tau v v^T) x = beta e_0. beta takes the sign opposite to x[0] so
 * that x[0] - beta never cancels. tau = 0 (H = I) when x[1..] is already zero. */
static double house(double *x, int64_t inc, int n) {
    if (n <= 1) return 0;
    double xnorm = la_norm_strided(x + inc, inc, n - 1);
    if (xnorm == 0) return 0;
    double alpha = x[0], beta = -copysign(hypot(alpha, xnorm), alpha);
    double scale = 1.0 / (alpha - beta);
    for (int i = 1; i < n; i++) x[i * inc] *= scale;
    x[0] = beta;
    return (beta - alpha) / beta;
}

/* Applies H = I - tau v v^T (v in column j of a, rows j..m-1, v_j = 1) to columns [c0, c1) of a. */
static void house_apply_cols(double *a, int64_t lda, int m, int j, double tau, int c0, int c1) {
    if (tau == 0 || c1 <= c0) return;
    enum { W = 256 };
    double w[W];
    for (int cb = c0; cb < c1; cb += W) {
        int nw = PLX_MIN(W, c1 - cb);
        memcpy(w, a + j * lda + cb, sizeof(double) * nw);
        for (int i = j + 1; i < m; i++) la_axpy(w, a[i * lda + j], a + i * lda + cb, (size_t)nw);
        la_axpy(a + j * lda + cb, -tau, w, (size_t)nw);
        for (int i = j + 1; i < m; i++) la_axpy(a + i * lda + cb, -tau * a[i * lda + j], w, (size_t)nw);
    }
}

/* Explicit V (mm x jb, unit diagonal, zeros above) for the reflectors j0..j0+jb-1. */
static void block_v(const la_qr *q, int j0, int jb, double *V) {
    int mm = q->m - j0;
    for (int r = 0; r < mm; r++)
        for (int c = 0; c < jb; c++) V[r * jb + c] = r > c ? q->a[(j0 + r) * q->lda + j0 + c] : (r == c ? 1.0 : 0.0);
}

/* Compact WY: H_0 ... H_{jb-1} = I - V T V^T with T upper triangular, built column by column from
 * T[0:j, j] = -tau_j T[0:j, 0:j] (V[:, 0:j]^T v_j). T is NB x NB (leading dim NB). */
static void block_t(const double *V, int mm, int jb, const double *tau, double *T) {
    double *G = plx_malloc(sizeof(double) * jb * jb);
    la_dgemm(1, 0, jb, jb, mm, 1.0, V, jb, V, jb, 0.0, G, jb);
    memset(T, 0, sizeof(double) * NB * NB);
    for (int j = 0; j < jb; j++) {
        T[j * NB + j] = tau[j];
        for (int i = 0; i < j; i++) {
            double s = 0;
            for (int p = i; p < j; p++) s += T[i * NB + p] * G[p * jb + j];
            T[i * NB + j] = -tau[j] * s;
        }
    }
    free(G);
}

/* C (mm x nc) <- (I - V op(T) V^T) C, op(T) = T^T for trans (that is Q_b^T), T otherwise. */
static void block_apply(const double *V, int mm, int jb, const double *T, int trans, double *C, int nc,
                        int64_t ldc, double *W, double *W2) {
    la_dgemm(1, 0, jb, nc, mm, 1.0, V, jb, C, ldc, 0.0, W, nc);
    la_dgemm(trans, 0, jb, nc, jb, 1.0, T, NB, W, nc, 0.0, W2, nc);
    la_dgemm(0, 0, mm, nc, jb, -1.0, V, jb, W2, nc, 1.0, C, ldc);
}

static void qr_alloc(la_qr *q, double *a, int m, int n, int64_t lda) {
    q->m = m;
    q->n = n;
    q->k = PLX_MIN(m, n);
    q->a = a;
    q->lda = lda;
    q->tau = PLX_NEW(double, PLX_MAX(q->k, 1));
    q->t = PLX_NEW(double, (size_t)PLX_MAX((q->k + NB - 1) / NB, 1) * NB * NB);
}

void la_qr_free(la_qr *q) {
    free(q->tau);
    free(q->t);
    memset(q, 0, sizeof *q);
}

/* Blocked right-looking QR: an unblocked panel of NB columns, then the trailing columns get the
 * whole block at once as three gemms. */
void la_qr_factor(la_qr *q, double *a, int m, int n, int64_t lda) {
    qr_alloc(q, a, m, n, lda);
    double *V = plx_malloc(sizeof(double) * (size_t)m * NB);
    double *W = plx_malloc(sizeof(double) * (size_t)NB * PLX_MAX(n, 1)), *W2 = plx_malloc(sizeof(double) * (size_t)NB * PLX_MAX(n, 1));
    for (int j0 = 0; j0 < q->k; j0 += NB) {
        int jb = PLX_MIN(NB, q->k - j0), mm = m - j0;
        for (int j = j0; j < j0 + jb; j++) {
            q->tau[j] = house(a + j * lda + j, lda, m - j);
            house_apply_cols(a, lda, m, j, q->tau[j], j + 1, j0 + jb);
        }
        double *T = q->t + (size_t)(j0 / NB) * NB * NB;
        block_v(q, j0, jb, V);
        block_t(V, mm, jb, q->tau + j0, T);
        if (n > j0 + jb) block_apply(V, mm, jb, T, 1, a + j0 * lda + j0 + jb, n - j0 - jb, lda, W, W2);
    }
    free(V);
    free(W);
    free(W2);
}

typedef struct {
    double *a;
    int64_t lda;
    int m, j, c0, c1;
    double tau;
} hcols_job;

enum { QRCP_COLS = 128 };

static void qrcp_update_task(int64_t b, int tid, void *ctx) {
    const hcols_job *h = ctx;
    int c0 = h->c0 + (int)b * QRCP_COLS;
    house_apply_cols(h->a, h->lda, h->m, h->j, h->tau, c0, PLX_MIN(c0 + QRCP_COLS, h->c1));
}

/* Businger-Golub pivoting: at step j bring the remaining column of largest norm forward. Column
 * norms are downdated (|a_l|^2 -= a_jl^2) and recomputed when the downdate has cancelled more than
 * half the digits (the LAPACK xLAQP2 safeguard), otherwise rank decisions would rest on noise. */
void la_qrcp_factor(la_qr *q, double *a, int m, int n, int64_t lda, int *perm) {
    qr_alloc(q, a, m, n, lda);
    double *vn1 = plx_malloc(sizeof(double) * n), *vn2 = plx_malloc(sizeof(double) * n);
    for (int j = 0; j < n; j++) {
        perm[j] = j;
        vn1[j] = vn2[j] = la_norm_strided(a + j, lda, m);
    }
    const double tol3z = sqrt(DBL_EPSILON);
    for (int j = 0; j < q->k; j++) {
        int p = j;
        for (int l = j + 1; l < n; l++)
            if (vn1[l] > vn1[p]) p = l;
        if (p != j) {
            for (int i = 0; i < m; i++) {
                double t = a[i * lda + j];
                a[i * lda + j] = a[i * lda + p];
                a[i * lda + p] = t;
            }
            int ti = perm[j];
            perm[j] = perm[p];
            perm[p] = ti;
            vn1[p] = vn1[j];
            vn2[p] = vn2[j];
        }
        q->tau[j] = house(a + j * lda + j, lda, m - j);
        hcols_job h = {a, lda, m, j, j + 1, n, q->tau[j]};
        la_par_for((n - j - 1 + QRCP_COLS - 1) / QRCP_COLS, 4.0 * (m - j) * QRCP_COLS, qrcp_update_task, &h);
        for (int l = j + 1; l < n; l++) {
            if (vn1[l] == 0) continue;
            double r = fabs(a[j * lda + l]) / vn1[l];
            double t = PLX_MAX(0.0, (1.0 + r) * (1.0 - r));
            double t2 = t * (vn1[l] / vn2[l]) * (vn1[l] / vn2[l]);
            if (t2 <= tol3z) {
                vn1[l] = j + 1 < m ? la_norm_strided(a + (j + 1) * lda + l, lda, m - j - 1) : 0.0;
                vn2[l] = vn1[l];
            } else {
                vn1[l] *= sqrt(t);
            }
        }
    }
    double *V = plx_malloc(sizeof(double) * (size_t)m * NB);
    for (int j0 = 0; j0 < q->k; j0 += NB) {
        int jb = PLX_MIN(NB, q->k - j0);
        block_v(q, j0, jb, V);
        block_t(V, m - j0, jb, q->tau + j0, q->t + (size_t)(j0 / NB) * NB * NB);
    }
    free(V);
    free(vn1);
    free(vn2);
}

void la_qr_apply(const la_qr *q, int trans, double *C, int nc, int64_t ldc) {
    if (q->k == 0 || nc <= 0) return;
    double *V = plx_malloc(sizeof(double) * (size_t)q->m * NB);
    double *W = plx_malloc(sizeof(double) * (size_t)NB * nc), *W2 = plx_malloc(sizeof(double) * (size_t)NB * nc);
    int nblk = (q->k + NB - 1) / NB;
    for (int s = 0; s < nblk; s++) { /* Q^T = Q_last^T ... Q_0^T applies block 0 first */
        int b = trans ? s : nblk - 1 - s, j0 = b * NB, jb = PLX_MIN(NB, q->k - j0);
        block_v(q, j0, jb, V);
        block_apply(V, q->m - j0, jb, q->t + (size_t)b * NB * NB, trans, C + j0 * ldc, nc, ldc, W, W2);
    }
    free(V);
    free(W);
    free(W2);
}

/* Q [I_k; 0], applying blocks last to first: block b leaves columns < j0 untouched (they are still
 * unit vectors in rows < j0, where its reflectors are zero), so it only updates columns j0..k-1. */
void la_qr_form_q(const la_qr *q, double *Q, int64_t ldq) {
    const int m = q->m, k = q->k;
    for (int i = 0; i < m; i++) {
        memset(Q + i * ldq, 0, sizeof(double) * k);
        if (i < k) Q[i * ldq + i] = 1.0;
    }
    if (k == 0) return;
    double *V = plx_malloc(sizeof(double) * (size_t)m * NB);
    double *W = plx_malloc(sizeof(double) * (size_t)NB * k), *W2 = plx_malloc(sizeof(double) * (size_t)NB * k);
    for (int b = (k + NB - 1) / NB - 1; b >= 0; b--) {
        int j0 = b * NB, jb = PLX_MIN(NB, k - j0);
        block_v(q, j0, jb, V);
        block_apply(V, m - j0, jb, q->t + (size_t)b * NB * NB, 0, Q + j0 * ldq + j0, k - j0, ldq, W, W2);
    }
    free(V);
    free(W);
    free(W2);
}

/* ---- public ------------------------------------------------------------------------------ */

void plx_qr(const plx_mat *A, plx_mat *Q, plx_mat *R) {
    const int m = A->rows, n = A->cols, k = PLX_MIN(m, n);
    PLX_ASSERT(R->rows == k && R->cols == n && R->a != A->a);
    PLX_ASSERT(!Q || (Q->rows == m && Q->cols == k && Q->a != A->a && Q->a != R->a));
    plx_mat W = plx_mat_clone(A);
    la_qr q;
    la_qr_factor(&q, W.a, m, n, n);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++) PLX_M(R, i, j) = j >= i ? PLX_M(&W, i, j) : 0.0;
    if (Q) la_qr_form_q(&q, Q->a, k);
    la_qr_free(&q);
    plx_mat_free(&W);
}

/* min ||M y - C|| over the stacked system M = [A; sqrt(ridge) I]:
 *   1. M = Q1 R (blocked, the expensive part) when M is tall, so the pivoted QR below only sees
 *      an n x n matrix: rank revelation at O(n^3) instead of O(m n^2) level-2 work;
 *   2. R P = Q2 R2 with column pivoting; r = numerical rank from the diagonal of R2;
 *   3. full rank: back substitution. Rank deficient: R2[0:r, :] = S^T W^T (QR of its transpose)
 *      gives the minimum-norm y = W S^-T c, i.e. the complete orthogonal decomposition. */
int plx_lstsq(const plx_mat *A, const plx_mat *B, double ridge, double rcond, plx_mat *X) {
    const int m = A->rows, n = A->cols, nrhs = B->cols;
    PLX_ASSERT(B->rows == m && X->rows == n && X->cols == nrhs && ridge >= 0);
    PLX_ASSERT(X->a != A->a && X->a != B->a);
    const int mm = ridge > 0 ? m + n : m;
    double *M = PLX_NEW(double, (size_t)mm * n), *C = PLX_NEW(double, (size_t)mm * nrhs);
    memcpy(M, A->a, sizeof(double) * (size_t)m * n);
    memcpy(C, B->a, sizeof(double) * (size_t)m * nrhs);
    for (int i = 0; i < mm - m; i++) M[(size_t)(m + i) * n + i] = sqrt(ridge);

    double *R = M;
    int r0 = mm;
    if (mm > n) {
        la_qr q1;
        la_qr_factor(&q1, M, mm, n, n);
        la_qr_apply(&q1, 1, C, nrhs, nrhs);
        la_qr_free(&q1);
        R = PLX_NEW(double, (size_t)n * n);
        for (int i = 0; i < n; i++) memcpy(R + (size_t)i * n + i, M + (size_t)i * n + i, sizeof(double) * (n - i));
        r0 = n;
    }
    int *perm = PLX_NEW(int, n);
    la_qr q2;
    la_qrcp_factor(&q2, R, r0, n, n, perm);
    la_qr_apply(&q2, 1, C, nrhs, nrhs);
    if (rcond <= 0) rcond = PLX_MAX(m, n) * DBL_EPSILON;
    int r = 0;
    while (r < q2.k && fabs(R[(size_t)r * n + r]) > rcond * fabs(R[0])) r++;

    double *Y = PLX_NEW(double, (size_t)n * nrhs);
    memcpy(Y, C, sizeof(double) * (size_t)r * nrhs);
    if (r == n) {
        la_trsm(0, 0, 0, n, R, n, Y, nrhs, nrhs);
    } else if (r > 0) {
        double *Wt = PLX_NEW(double, (size_t)n * r); /* R2[0:r, :]^T, n x r lower trapezoidal */
        for (int i = 0; i < r; i++)
            for (int j = i; j < n; j++) Wt[(size_t)j * r + i] = R[(size_t)i * n + j];
        la_qr q3;
        la_qr_factor(&q3, Wt, n, r, r);
        la_trsm(0, 1, 0, r, Wt, r, Y, nrhs, nrhs); /* S^T z = c, S = upper part of Wt */
        la_qr_apply(&q3, 0, Y, nrhs, nrhs);        /* y = W [z; 0] */
        la_qr_free(&q3);
        free(Wt);
    }
    for (int i = 0; i < n; i++) memcpy(&PLX_M(X, perm[i], 0), Y + (size_t)i * nrhs, sizeof(double) * nrhs);
    la_qr_free(&q2);
    if (R != M) free(R);
    free(perm);
    free(Y);
    free(M);
    free(C);
    return r;
}
