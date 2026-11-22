/* Private to src/la: raw-pointer kernels shared between the la_*.c files. Not a public API. */
#ifndef PLX_LA_INTERNAL_H
#define PLX_LA_INTERNAL_H

#include "la/la.h"

#include <float.h>
#include <math.h>

/* Below this many flops a loop is run on the calling thread: plx_parallel_for spawns threads per
 * call (~10-30 us), so tiny jobs would be dominated by thread start-up. */
#define LA_PAR_MIN_FLOPS 2.0e5

/* Runs fn over [0, n) in parallel if the total work justifies it, else serially. */
void la_par_for(int64_t n, double flops_per_item, plx_par_fn fn, void *ctx);

/* Vectorisable reductions: 8 independent partial sums, so the auto-vectoriser can use SIMD without
 * -ffast-math (a single accumulator is a serial dependency it may not reorder). */
double la_dot(const double *x, const double *y, size_t n);
double la_dot_strided(const double *x, int64_t incx, const double *y, int64_t incy, int n);
double la_norm_strided(const double *x, int64_t inc, int n); /* overflow/underflow safe */
void   la_axpy(double *restrict y, double alpha, const double *restrict x, size_t n);
/* Plane rotation of two distinct rows: x <- c x - s y, y <- s x + c y. */
void   la_rot(double *restrict x, double *restrict y, size_t n, double c, double s);

/* ---- gemm core -------------------------------------------------------------------------
 * An operand is a logical matrix op(X) whose element (i, j) lives at base[i * rs + j * cs], with
 * base either double or float (exactly one non-NULL). sub_row / sub_col (nullable) are subtracted
 * from element (i, j) as sub_row[i] / sub_col[j] while packing: this is how float data matrices
 * are centred and widened to double on the fly. */
typedef struct {
    const double *d;
    const float *f;
    int64_t rs, cs;
    const double *sub_row, *sub_col;
} la_opnd;

static inline la_opnd la_opnd_d(const double *p, int64_t ld, int trans) {
    la_opnd o = {.d = p, .rs = trans ? 1 : ld, .cs = trans ? ld : 1};
    return o;
}

/* C (m x n, leading dim ldc) = alpha * A * B + beta * C with A m x k, B k x n.
 * lower_only: compute only the C tiles that intersect the lower triangle (i >= j); tiles straddling
 * the diagonal also produce some upper entries, the rest of the upper triangle is left untouched. */
void la_gemm(int m, int n, int k, double alpha, const la_opnd *A, const la_opnd *B, double beta,
             double *C, int64_t ldc, int lower_only);

/* Plain double convenience: op(A) is m x k with storage leading dimension lda, etc. */
void la_dgemm(int ta, int tb, int m, int n, int k, double alpha, const double *A, int64_t lda,
              const double *B, int64_t ldb, double beta, double *C, int64_t ldc);

/* Copies the strict lower triangle of n x n C onto the upper one. */
void la_mirror_lower(double *C, int n, int64_t ldc);

/* ---- triangular solve ------------------------------------------------------------------
 * B (n x nrhs) <- op(T)^-1 B where T is n x n lower (lower=1) or upper triangular, op = T or T^T,
 * unit: diagonal taken as 1 (not read). Threaded across right-hand-side column blocks. */
void la_trsm(int lower, int trans, int unit, int n, const double *T, int64_t ldt, double *B, int nrhs,
             int64_t ldb);

/* ---- Householder QR --------------------------------------------------------------------
 * Reflector j (j < k = min(m, n)) is H_j = I - tau_j v_j v_j^T with v_j = e_j below row j stored
 * in the strict lower part of column j of a; Q = H_0 H_1 ... H_{k-1}. t holds the compact-WY
 * triangular factors of each block of LA_QR_NB reflectors (Q_b = I - V_b T_b V_b^T), so applying Q
 * is done with gemm. a is borrowed, tau/t are owned. */
#define LA_QR_NB 32
typedef struct {
    int m, n, k;
    double *a;
    int64_t lda;
    double *tau, *t;
} la_qr;

void la_qr_factor(la_qr *q, double *a, int m, int n, int64_t lda);
/* Column-pivoted (rank-revealing) QR, unblocked: A P = Q R with |R_00| >= |R_11| >= ... ;
 * perm[j] is the original index of column j. */
void la_qrcp_factor(la_qr *q, double *a, int m, int n, int64_t lda, int *perm);
/* C (m x nc) <- Q C (trans = 0) or Q^T C (trans = 1). */
void la_qr_apply(const la_qr *q, int trans, double *C, int nc, int64_t ldc);
/* Q[:, 0:k] explicitly, into an m x k array. */
void la_qr_form_q(const la_qr *q, double *Q, int64_t ldq);
void la_qr_free(la_qr *q);

/* ---- decompositions on raw buffers ------------------------------------------------------ */
/* Thin SVD of the m x n row-major array a (destroyed), k = min(m, n); U (m x k) and V (n x k)
 * nullable, contiguous. Used by plx_svd and the PCA routes. */
int la_svd_raw(double *a, int m, int n, double *U, double *s, double *V);

/* idx[0..n) = indices of keys in descending order (ties by index: deterministic). */
void la_argsort_desc(const double *keys, int n, int *idx);

/* Makes the largest-magnitude entry of each column of the r x c array X positive; when Y is
 * non-NULL (ry x c) its columns are flipped along (paired singular vectors). */
void la_fix_signs(double *X, int r, int c, double *Y, int ry);

#endif
