/* Triangular solves, Cholesky, LU with partial pivoting. */
#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

/* ---- triangular solve ------------------------------------------------------------------
 * Row-major T: the loop order is chosen so T is always read along its rows. Without transpose
 * that is the "left-looking" (dot-product) order, B_i -= sum_k T_ik B_k; with transpose it is the
 * "right-looking" (axpy) order, B_i -= T_ki B_k for all i after B_k is final. */
typedef struct {
    int lower, trans, unit, n, nrhs;
    const double *T;
    int64_t ldt, ldb;
    double *B;
} trsm_job;

enum { TRSM_COLS = 64 };

static void trsm_block(int64_t blk, int tid, void *ctx) {
    const trsm_job *j = ctx;
    const int n = j->n, c0 = (int)blk * TRSM_COLS, w = PLX_MIN(TRSM_COLS, j->nrhs - c0);
    const int64_t ldb = j->ldb;
    double *B = j->B + c0;
    if (!j->trans) {
        for (int s = 0; s < n; s++) {
            int i = j->lower ? s : n - 1 - s;
            int k0 = j->lower ? 0 : i + 1, k1 = j->lower ? i : n;
            const double *t = j->T + i * j->ldt;
            double *bi = B + i * ldb;
            if (w == 1) {
                bi[0] -= la_dot_strided(t + k0, 1, B + k0 * ldb, ldb, k1 - k0);
            } else {
                for (int k = k0; k < k1; k++)
                    if (t[k] != 0) la_axpy(bi, -t[k], B + k * ldb, (size_t)w);
            }
            if (!j->unit)
                for (int c = 0; c < w; c++) bi[c] /= t[i];
        }
    } else {
        for (int s = 0; s < n; s++) {
            int k = j->lower ? n - 1 - s : s; /* op(T) = T^T: lower^T is upper -> backward */
            int i0 = j->lower ? 0 : k + 1, i1 = j->lower ? k : n;
            const double *t = j->T + k * j->ldt;
            double *bk = B + k * ldb;
            if (!j->unit)
                for (int c = 0; c < w; c++) bk[c] /= t[k];
            if (w == 1 && ldb == 1) {
                la_axpy(B + i0, -bk[0], t + i0, (size_t)(i1 - i0));
            } else {
                for (int i = i0; i < i1; i++)
                    if (t[i] != 0) la_axpy(B + i * ldb, -t[i], bk, (size_t)w);
            }
        }
    }
}

void la_trsm(int lower, int trans, int unit, int n, const double *T, int64_t ldt, double *B, int nrhs,
             int64_t ldb) {
    if (n <= 0 || nrhs <= 0) return;
    trsm_job j = {lower, trans, unit, n, nrhs, T, ldt, ldb, B};
    la_par_for((nrhs + TRSM_COLS - 1) / TRSM_COLS, (double)n * n * PLX_MIN(nrhs, TRSM_COLS), trsm_block, &j);
}

/* ---- Cholesky ------------------------------------------------------------------------- */

enum { CHOL_NB = 96 };

/* Row-oriented (dot product) Cholesky of an n x n block: L_ij = (A_ij - L_i,0:j . L_j,0:j) / L_jj. */
static int chol_unblocked(double *A, int64_t lda, int n) {
    for (int j = 0; j < n; j++) {
        double *aj = A + j * lda;
        double d = aj[j] - la_dot(aj, aj, (size_t)j);
        if (!(d > 0) || !isfinite(d)) return PLX_ENOTSPD; /* !(d > 0) also catches NaN */
        double l = sqrt(d);
        aj[j] = l;
        for (int i = j + 1; i < n; i++) {
            double *ai = A + i * lda;
            ai[j] = (ai[j] - la_dot(ai, aj, (size_t)j)) / l;
        }
    }
    return PLX_OK;
}

typedef struct {
    double *A21;
    const double *L11;
    int64_t lda;
    int kb;
} panel_job;

/* Row r of A21 <- r L11^-T, i.e. forward substitution with L11 on the row itself. */
static void chol_panel_row(int64_t r, int tid, void *ctx) {
    const panel_job *p = ctx;
    double *x = p->A21 + r * p->lda;
    for (int j = 0; j < p->kb; j++) {
        const double *lj = p->L11 + j * p->lda;
        x[j] = (x[j] - la_dot(lj, x, (size_t)j)) / lj[j];
    }
}

/* Right-looking blocked Cholesky: factor the diagonal block, solve the panel below it, then
 * subtract A21 A21^T from the trailing lower triangle with the packed gemm (lower tiles only). */
int plx_chol(plx_mat *A) {
    const int n = A->rows;
    PLX_ASSERT(A->cols == n);
    double *a = A->a;
    for (int k0 = 0; k0 < n; k0 += CHOL_NB) {
        int kb = PLX_MIN(CHOL_NB, n - k0), n2 = n - k0 - kb;
        if (chol_unblocked(a + (size_t)k0 * n + k0, n, kb) != PLX_OK) return PLX_ENOTSPD;
        if (n2 == 0) break;
        double *A21 = a + (size_t)(k0 + kb) * n + k0;
        panel_job pj = {A21, a + (size_t)k0 * n + k0, n, kb};
        la_par_for(n2, (double)kb * kb, chol_panel_row, &pj);
        la_opnd l = la_opnd_d(A21, n, 0), r = la_opnd_d(A21, n, 1);
        la_gemm(n2, n2, kb, -1.0, &l, &r, 1.0, A21 + kb, n, 1);
    }
    for (int i = 0; i < n; i++) memset(a + (size_t)i * n + i + 1, 0, sizeof(double) * (size_t)(n - 1 - i));
    return PLX_OK;
}

void plx_chol_solve(const plx_mat *L, plx_mat *B) {
    PLX_ASSERT(L->rows == L->cols && B->rows == L->rows);
    la_trsm(1, 0, 0, L->rows, L->a, L->cols, B->a, B->cols, B->cols);
    la_trsm(1, 1, 0, L->rows, L->a, L->cols, B->a, B->cols, B->cols);
}

void plx_chol_solve_vec(const plx_mat *L, double *b) {
    PLX_ASSERT(L->rows == L->cols);
    la_trsm(1, 0, 0, L->rows, L->a, L->cols, b, 1, 1);
    la_trsm(1, 1, 0, L->rows, L->a, L->cols, b, 1, 1);
}

double plx_chol_logdet(const plx_mat *L) {
    PLX_ASSERT(L->rows == L->cols);
    double s = 0;
    for (int i = 0; i < L->rows; i++) s += log(PLX_M(L, i, i));
    return 2.0 * s;
}

int plx_spd_inverse(const plx_mat *A, plx_mat *Ainv) {
    const int n = A->rows;
    PLX_ASSERT(A->cols == n && Ainv->rows == n && Ainv->cols == n && Ainv->a != A->a);
    plx_mat L = plx_mat_clone(A);
    int rc = plx_chol(&L);
    if (rc == PLX_OK) {
        memset(Ainv->a, 0, sizeof(double) * (size_t)n * n);
        for (int i = 0; i < n; i++) PLX_M(Ainv, i, i) = 1.0;
        plx_chol_solve(&L, Ainv);
        for (int i = 0; i < n; i++) /* the two triangles differ by rounding; average them */
            for (int j = 0; j < i; j++) PLX_M(Ainv, i, j) = PLX_M(Ainv, j, i) = 0.5 * (PLX_M(Ainv, i, j) + PLX_M(Ainv, j, i));
    }
    plx_mat_free(&L);
    return rc;
}

/* ---- LU --------------------------------------------------------------------------------- */

enum { LU_NB = 64 };

/* Blocked right-looking LU with partial pivoting. A panel of LU_NB columns is factored with
 * row-major-friendly elimination (whole-row swaps, so the rows left and right of the panel are
 * permuted along, as LAPACK's laswp would); then U12 = L11^-1 A12 (triangular solve) and the
 * trailing A22 -= L21 U12 runs on the packed gemm, which is where the n^3 flops go. */
int plx_lu(plx_mat *A, int *piv) {
    const int n = A->rows;
    PLX_ASSERT(A->cols == n);
    double *a = A->a, amax = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) amax = PLX_MAX(amax, fabs(a[i]));
    const double tol = n * DBL_EPSILON * amax;
    int rc = PLX_OK;
    double *tmp = plx_malloc(sizeof(double) * (size_t)n);
    for (int k0 = 0; k0 < n; k0 += LU_NB) {
        const int kb = PLX_MIN(LU_NB, n - k0), k1 = k0 + kb, n2 = n - k1;
        for (int k = k0; k < k1; k++) {
            int p = k;
            for (int i = k + 1; i < n; i++)
                if (fabs(a[(size_t)i * n + k]) > fabs(a[(size_t)p * n + k])) p = i;
            piv[k] = p;
            if (p != k) {
                memcpy(tmp, a + (size_t)k * n, sizeof(double) * n);
                memcpy(a + (size_t)k * n, a + (size_t)p * n, sizeof(double) * n);
                memcpy(a + (size_t)p * n, tmp, sizeof(double) * n);
            }
            const double *ak = a + (size_t)k * n;
            if (!(fabs(ak[k]) > tol)) rc = PLX_ESINGULAR;
            if (ak[k] == 0) continue; /* the whole column below is zero too: nothing to eliminate */
            for (int i = k + 1; i < n; i++) {
                double *ai = a + (size_t)i * n, l = ai[k] / ak[k];
                ai[k] = l;
                if (l != 0) la_axpy(ai + k + 1, -l, ak + k + 1, (size_t)(k1 - k - 1));
            }
        }
        if (n2 == 0) break;
        double *A11 = a + (size_t)k0 * n + k0;
        la_trsm(1, 0, 1, kb, A11, n, A11 + kb, n2, n);
        la_dgemm(0, 0, n2, n2, kb, -1.0, A11 + (size_t)kb * n, n, A11 + kb, n, 1.0, A11 + (size_t)kb * n + kb, n);
    }
    free(tmp);
    return rc;
}

static void lu_permute(const int *piv, int n, double *B, int nrhs, int64_t ldb) {
    for (int k = 0; k < n; k++)
        if (piv[k] != k)
            for (int c = 0; c < nrhs; c++) {
                double t = B[k * ldb + c];
                B[k * ldb + c] = B[piv[k] * ldb + c];
                B[piv[k] * ldb + c] = t;
            }
}

void plx_lu_solve(const plx_mat *LU, const int *piv, plx_mat *B) {
    const int n = LU->rows;
    PLX_ASSERT(LU->cols == n && B->rows == n);
    lu_permute(piv, n, B->a, B->cols, B->cols);
    la_trsm(1, 0, 1, n, LU->a, n, B->a, B->cols, B->cols);
    la_trsm(0, 0, 0, n, LU->a, n, B->a, B->cols, B->cols);
}

void plx_lu_solve_vec(const plx_mat *LU, const int *piv, double *b) {
    const int n = LU->rows;
    PLX_ASSERT(LU->cols == n);
    lu_permute(piv, n, b, 1, 1);
    la_trsm(1, 0, 1, n, LU->a, n, b, 1, 1);
    la_trsm(0, 0, 0, n, LU->a, n, b, 1, 1);
}

double plx_lu_det(const plx_mat *LU, const int *piv) {
    double d = 1;
    for (int k = 0; k < LU->rows; k++) d *= piv[k] != k ? -PLX_M(LU, k, k) : PLX_M(LU, k, k);
    return d;
}

double plx_det(const plx_mat *A) {
    plx_mat LU = plx_mat_clone(A);
    int *piv = PLX_NEW(int, A->rows);
    plx_lu(&LU, piv);
    double d = plx_lu_det(&LU, piv);
    free(piv);
    plx_mat_free(&LU);
    return d;
}

int plx_solve(const plx_mat *A, plx_mat *B) {
    PLX_ASSERT(A->rows == A->cols && B->rows == A->rows);
    plx_mat LU = plx_mat_clone(A);
    int *piv = PLX_NEW(int, A->rows);
    int rc = plx_lu(&LU, piv);
    if (rc == PLX_OK) plx_lu_solve(&LU, piv, B);
    free(piv);
    plx_mat_free(&LU);
    return rc;
}

int plx_inverse(const plx_mat *A, plx_mat *Ainv) {
    const int n = A->rows;
    PLX_ASSERT(A->cols == n && Ainv->rows == n && Ainv->cols == n && Ainv->a != A->a);
    plx_mat I = plx_mat_identity(n);
    int rc = plx_solve(A, &I);
    if (rc == PLX_OK) plx_mat_copy(Ainv, &I);
    plx_mat_free(&I);
    return rc;
}
