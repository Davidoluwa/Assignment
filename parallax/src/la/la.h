/* Parallax dense linear algebra: row-major double matrices, BLAS-like products, factorisations,
 * eigen/singular value decompositions, Krylov solves and float32 data-matrix statistics (PCA).
 * Hand-written from the algorithms; libc/libm/pthreads only.
 *
 * Conventions (apply to every function below unless its comment says otherwise):
 *   - Shapes are "rows x cols". A vector argument is a plain double* of the documented length.
 *   - Output matrices are allocated by the caller with the exact documented shape; the shape is
 *     checked with PLX_ASSERT (a mismatch is a programmer error and aborts). Functions whose name
 *     ends in _new, plus clone/identity/from/transpose/matmul, return a fresh matrix to free.
 *   - Outputs must not alias inputs unless the comment says "may alias".
 *   - Recoverable numerical failure (matrix not SPD, singular, no convergence) is returned as a
 *     plx_status-style negative int; PLX_OK (0) on success.
 *   - Symmetric inputs: only the LOWER triangle (i >= j) is read; the upper triangle is ignored.
 *   - Eigen/singular values are sorted DESCENDING; vectors are COLUMNS of the output matrix, with a
 *     deterministic sign: the entry of largest magnitude in each eigenvector (each principal axis
 *     for PCA; see plx_svd for its variant) is positive.
 *   - Heavy routines thread with plx_parallel_for; every result is bitwise independent of the
 *     thread count (each output element is reduced in a fixed order by exactly one task). */
#ifndef PLX_LA_H
#define PLX_LA_H

#include "core/core.h"

/* ---- status --------------------------------------------------------------------------
 * Numerical statuses, extending core's plx_status (core's plx_status_str does not know them;
 * plx_la_status_str knows both). */
enum {
    PLX_ENOTSPD = -16,    /* matrix (numerically) not symmetric positive definite */
    PLX_ESINGULAR = -17,  /* matrix (numerically) singular */
    PLX_ENOCONV = -18     /* iteration did not converge */
};
const char *plx_la_status_str(int status);

/* ---- matrix type & basics ------------------------------------------------------------ */
typedef struct { int rows, cols; double *a; } plx_mat; /* row-major, owns a */
#define PLX_M(m, i, j) ((m)->a[(size_t)(i) * (m)->cols + (j)])

plx_mat plx_mat_new(int rows, int cols);                         /* zeroed; rows, cols >= 0 */
void    plx_mat_free(plx_mat *m);                                /* frees and zeroes *m; NULL-safe a */
plx_mat plx_mat_clone(const plx_mat *a);
plx_mat plx_mat_identity(int n);
plx_mat plx_mat_from(int rows, int cols, const double *data);   /* copies rows*cols row-major values */
plx_mat plx_mat_transpose(const plx_mat *a);                     /* new cols x rows matrix */
void    plx_mat_copy(plx_mat *dst, const plx_mat *src);          /* same shape; may alias */
void    plx_mat_scale(plx_mat *a, double s);                     /* a *= s */
void    plx_mat_add(plx_mat *y, double alpha, const plx_mat *x); /* y += alpha * x, same shape */
double  plx_mat_norm_fro(const plx_mat *a);                      /* overflow-safe */

double plx_vec_dot(const double *x, const double *y, size_t n);
void   plx_vec_axpy(double *y, double alpha, const double *x, size_t n); /* y += alpha * x */
double plx_vec_norm(const double *x, size_t n);                          /* overflow-safe 2-norm */

/* ---- products ----------------------------------------------------------------------- */
typedef enum { PLX_NO_TRANS = 0, PLX_TRANS = 1 } plx_op;

/* C = alpha * op(A) * op(B) + beta * C.  op(A): m x k, op(B): k x n, C: m x n (preallocated).
 * beta == 0 overwrites C without reading it (NaNs in C do not propagate). Cache-blocked, threaded. */
void plx_gemm(plx_op ta, plx_op tb, double alpha, const plx_mat *A, const plx_mat *B, double beta, plx_mat *C);
plx_mat plx_matmul(const plx_mat *A, const plx_mat *B); /* new A * B */

/* y = alpha * op(A) * x + beta * y.  x has cols(op(A)) entries, y has rows(op(A)); beta == 0 ignores
 * y. x and y must not overlap. */
void plx_gemv(plx_op ta, double alpha, const plx_mat *A, const double *x, double beta, double *y);

/* Symmetric rank-k update, both triangles of C written:
 *   PLX_TRANS:    C = alpha * A^T A + beta * C,  C is cols(A) x cols(A)
 *   PLX_NO_TRANS: C = alpha * A A^T + beta * C,  C is rows(A) x rows(A)
 * Only the lower triangle of the incoming C is used when beta != 0. Half the flops of plx_gemm. */
void plx_syrk(plx_op t, double alpha, const plx_mat *A, double beta, plx_mat *C);

/* ---- Cholesky (A = L L^T) ---------------------------------------------------------------
 * plx_chol factors the n x n SPD matrix in place: reads the lower triangle, and on PLX_OK leaves L
 * in the lower triangle with the strict upper triangle zeroed (so A is then exactly L). Returns
 * PLX_ENOTSPD if a pivot is <= 0 or not finite; A's contents are then unspecified (keep a copy if
 * you want to retry with damping). Blocked; the trailing update runs on the threaded gemm. */
int    plx_chol(plx_mat *A);
void   plx_chol_solve(const plx_mat *L, plx_mat *B);   /* B (n x nrhs) <- A^-1 B, from L = plx_chol(A) */
void   plx_chol_solve_vec(const plx_mat *L, double *b); /* b (n) <- A^-1 b; allocation-free */
double plx_chol_logdet(const plx_mat *L);              /* log det A = 2 sum log L_ii */
int    plx_spd_inverse(const plx_mat *A, plx_mat *Ainv); /* Ainv n x n, symmetric; PLX_ENOTSPD */

/* ---- LU with partial pivoting (P A = L U) ----------------------------------------------
 * plx_lu factors the n x n matrix in place (unit-lower L below the diagonal, U on and above) and
 * fills piv[n] (row k was swapped with row piv[k] at step k). It always completes; it returns
 * PLX_ESINGULAR if some |U_kk| <= n * eps * max|A| (singular to working precision: solves are then
 * meaningless), else PLX_OK. Blocked; the trailing update runs on the threaded gemm. */
int    plx_lu(plx_mat *A, int *piv);
void   plx_lu_solve(const plx_mat *LU, const int *piv, plx_mat *B);  /* B (n x nrhs) <- A^-1 B */
void   plx_lu_solve_vec(const plx_mat *LU, const int *piv, double *b);
double plx_lu_det(const plx_mat *LU, const int *piv);
double plx_det(const plx_mat *A);                      /* via LU of a copy; exact 0 only if a pivot is 0 */
/* Convenience wrappers on a copy of A; on PLX_ESINGULAR the output is left unchanged. */
int    plx_solve(const plx_mat *A, plx_mat *B);        /* B (n x nrhs) <- A^-1 B */
int    plx_inverse(const plx_mat *A, plx_mat *Ainv);   /* Ainv n x n */

/* ---- QR and least squares ---------------------------------------------------------------
 * plx_qr: thin Householder QR of m x n A, k = min(m, n): Q is m x k with orthonormal columns
 * (pass NULL to skip forming it), R is k x n upper trapezoidal. Blocked (compact WY), threaded. */
void plx_qr(const plx_mat *A, plx_mat *Q, plx_mat *R);

/* Least squares: X (n x nrhs) minimises ||A X - B||_F^2 + ridge * ||X||_F^2 for any m x n A
 * (tall, square or wide) and B (m x nrhs). ridge >= 0 (Tikhonov; it is applied as the stacked
 * system [A; sqrt(ridge) I], never by forming A^T A, so it does not square the conditioning).
 * Rank handling: after QR, a column-pivoted QR of R reveals the numerical rank r = #{|R_kk| >
 * rcond * |R_00|} (rcond <= 0 selects max(m, n) * eps). If r < n, the directions below the
 * threshold are dropped and X is the MINIMUM-NORM solution of the truncated problem (complete
 * orthogonal decomposition), so rank-deficient and underdetermined systems give the same answer
 * as the pseudo-inverse with that cutoff. Returns r. */
int plx_lstsq(const plx_mat *A, const plx_mat *B, double ridge, double rcond, plx_mat *X);

/* ---- symmetric eigenproblems ----------------------------------------------------------
 * plx_eigh: A (n x n, lower triangle read) = V diag(w) V^T. w[n] descending; V n x n orthogonal
 * with eigenvectors as columns, or NULL for eigenvalues only (much cheaper). V may alias A.
 * Householder tridiagonalisation, then implicit-shift QL; the rotations are recorded and applied
 * to the eigenvector matrix in cache-sized column slices in parallel. Residual and orthogonality
 * are O(n eps ||A||) (tests: <= 0.7 n eps); eigenvalue errors are O(eps ||A||) absolute, so tiny
 * eigenvalues of an ill-conditioned matrix are not relatively accurate (use plx_svd on a factor
 * if that matters). Repeated eigenvalues get an orthonormal basis of their eigenspace.
 * Returns PLX_ENOCONV if QL exceeds 30 n sweeps (not seen in practice). */
int plx_eigh(const plx_mat *A, double *w, plx_mat *V);

/* Generalised A v = lambda B v, A symmetric, B SPD (lower triangles read): w[n] descending, V
 * (nullable) n x n with B-orthonormal columns (V^T B V = I, V^T A V = diag(w)). Via Cholesky
 * B = L L^T and eigh of L^-1 A L^-T. Returns PLX_ENOTSPD if B is not SPD. */
int plx_eigh_gen(const plx_mat *A, const plx_mat *B, double *w, plx_mat *V);

/* ---- SVD ----------------------------------------------------------------------------------
 * Thin SVD of m x n A, k = min(m, n): A = U diag(s) V^T, U m x k and V n x k with orthonormal
 * columns (either may be NULL), s[k] descending, >= 0. Method: Householder QR (if m > n), then
 * column-pivoted QR of R and one-sided Jacobi on the rows of the pivoted R (Drmac-Veselic
 * preconditioning: few sweeps). Because every stage is column-wise backward stable, singular values
 * are computed to high RELATIVE accuracy (roughly eps * cond(A D^-1) for the best column scaling
 * D), not just eps * s_max: e.g. a 40 x 12 matrix B D with B well conditioned and column scales
 * down to 1e-11 gets all s_i to 5e-16 relative error (see tests). On unstructured matrices the
 * small s_i still have absolute error O(eps s_max) like any method. Exactly-zero singular values
 * get an arbitrary orthonormal completion. Sign convention (an exception to the general one): the
 * largest-magnitude entry of each singular vector on the SMALLER side (V when m >= n, else U) is
 * positive, so the result does not depend on which outputs are requested. Returns PLX_ENOCONV if
 * Jacobi fails to converge in 60 sweeps, PLX_EINVAL for non-finite input. */
int plx_svd(const plx_mat *A, plx_mat *U, double *s, plx_mat *V);

/* ---- fixed small sizes (allocation-free; 3x3 arrays are row-major) --------------------------
 * plx_eig2_sym: [[a, b], [b, c]] = V diag(w) V^T, w[0] >= w[1], V rows-major with eigenvectors as
 * columns, det V = +1. Uses the Jacobi rotation formulas, so both eigenvalues are accurate even
 * when they nearly cancel.
 * plx_svd3: A = U diag(s) V^T, s descending >= 0, U and V orthogonal (columns are the vectors).
 * plx_nearest_rotation3: the rotation R (R^T R = I, det R = +1) minimising ||R - A||_F, i.e.
 * U diag(1, 1, det(U V^T)) V^T. Output may alias the input. */
void plx_eig2_sym(double a, double b, double c, double w[2], double V[4]);
void plx_svd3(const double A[9], double U[9], double s[3], double V[9]);
void plx_nearest_rotation3(const double A[9], double R[9]);

/* ---- conjugate gradient -----------------------------------------------------------------
 * Solves A x = b for SPD A given only y = A x through the callback. x holds the initial guess on
 * entry (zeros are fine) and the solution on exit. diag (nullable) is the diagonal of A, used as
 * a Jacobi preconditioner (entries must be > 0). Stops when ||b - A x|| <= tol * ||b||, or after
 * max_iter iterations (max_iter <= 0 selects 2n). Returns PLX_OK, PLX_ENOCONV (iteration cap; x is
 * the last iterate), or PLX_ENOTSPD (p^T A p <= 0 encountered). info (nullable) receives the
 * iteration count and the final true relative residual. */
typedef void (*plx_matvec_fn)(const double *x, double *y, void *ctx); /* y = A x */
typedef struct { int iters; double rel_residual; } plx_cg_info;
int plx_cg(int n, plx_matvec_fn matvec, void *ctx, const double *b, double *x, const double *diag,
           double tol, int max_iter, plx_cg_info *info);

/* ---- float32 data matrices -------------------------------------------------------------
 * X is row-major n x d float (one sample per row: the descriptor format). Accumulation is always
 * in double; centring is done on the fly while the data streams through the gemm kernels, so no
 * centred or double copy of X is ever made (13000 x 40000 floats is 2 GB; a double copy is not).
 * mean arguments are d doubles; NULL means "do not centre". */
void plx_col_mean_f32(const float *X, int n, int d, double *mean);
void plx_scatter_f32(const float *X, int n, int d, const double *mean, plx_mat *S); /* S (d x d) = Xc^T Xc */
void plx_gram_f32(const float *X, int n, int d, const double *mean, plx_mat *G);    /* G (n x n) = Xc Xc^T */
/* Sample covariance C (d x d) = Xc^T Xc / (n - 1) with Xc centred by the column mean, which is
 * written to mean_out if non-NULL. n >= 2. */
void plx_cov_f32(const float *X, int n, int d, double *mean_out, plx_mat *C);

/* Truncated PCA of the centred data: Xc ~= U diag(sv) axes, keeping k <= min(n, d) components.
 * Routes (PLX_PCA_AUTO picks the cheapest by a flop/memory model, see la_pca.c):
 *   PLX_PCA_COV:        eig of the d x d scatter matrix. Exact; needs 8 d^2 bytes.
 *                       sv from eigenvalues, so relative accuracy of sv_i ~ eps (sv_0 / sv_i)^2.
 *   PLX_PCA_GRAM:       eig of the n x n Gram matrix, then a Rayleigh-Ritz SVD of Xc^T Q. Exact.
 *   PLX_PCA_RANDOMIZED: Gaussian test matrix with k + oversample columns, power_iters rounds of
 *                       subspace iteration with Householder re-orthonormalisation after every
 *                       product, then an exact SVD of the small projected matrix (Halko et al.).
 *                       Accurate when the spectrum decays; error ~ (sv_{k+p} / sv_i)^(2q+1).
 * Defaults (field 0): oversample = max(10, k / 2), power_iters = 3, seed = 0x5eed (large
 * oversampling is the cheap way to get the trailing components right on slow spectra).
 * out->axes is k x d with ROWS the principal axes (orthonormal); sv[k] are singular values of Xc
 * (variance of component i = sv_i^2 / (n - 1)); out->mean is the column mean.
 * Returns PLX_OK or PLX_ENOCONV (from the inner eigen/SVD solvers). */
typedef enum { PLX_PCA_AUTO = 0, PLX_PCA_COV, PLX_PCA_GRAM, PLX_PCA_RANDOMIZED } plx_pca_method;
typedef struct {
    plx_pca_method method;
    int oversample, power_iters; /* randomised route only; <= 0 selects the default */
    uint64_t seed;               /* randomised route only; 0 selects the default */
} plx_pca_opts;
typedef struct {
    int n, d, k;
    double *mean;          /* d */
    plx_mat axes;          /* k x d, rows orthonormal */
    double *sv;            /* k, descending */
    plx_pca_method method; /* route actually used */
} plx_pca;
int  plx_pca_f32(const float *X, int n, int d, int k, const plx_pca_opts *opts, plx_pca *out);
void plx_pca_free(plx_pca *p);

/* Y (n x k float) = ((X - mean) basis^T) / scale per column. basis is k x d with rows as the
 * directions; mean and scale (k entries) are nullable. Computed in double, threaded, row-blocked. */
void plx_project_f32(const float *X, int n, int d, const double *mean, const plx_mat *basis,
                     const double *scale, float *Y);
/* Projects onto p's axes; whiten divides component i by its standard deviation sv_i / sqrt(n-1)
 * so whitened training data has unit variance. Components with sv_i == 0 come out as 0. */
void plx_pca_project_f32(const plx_pca *p, const float *X, int n, bool whiten, float *Y);

#endif
