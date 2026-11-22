/* float32 data-matrix statistics and truncated PCA.
 *
 * X stays float and uncentred in memory; every product with the centred Xc = X - 1 mean^T goes
 * through la_gemm with a float operand whose packing routine widens to double and subtracts the
 * mean. Centring inside the packed block is exact in double (no X^T X - n mean mean^T
 * cancellation) and costs no extra pass over the 2+ GB matrix. */
#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

/* ---- statistics ------------------------------------------------------------------------- */

typedef struct {
    const float *X;
    int n, d;
    double *mean;
} mean_job;

enum { MEAN_COLS = 256 };

/* Column blocks rather than row blocks: each column is summed in row order by one task, so the
 * result does not depend on the thread count. */
static void mean_task(int64_t b, int tid, void *ctx) {
    const mean_job *j = ctx;
    const int c0 = (int)b * MEAN_COLS, w = PLX_MIN(MEAN_COLS, j->d - c0);
    double acc[MEAN_COLS] = {0};
    for (int i = 0; i < j->n; i++) {
        const float *x = j->X + (size_t)i * j->d + c0;
        if (w == MEAN_COLS) /* constant trip count: vectorises at -O2 */
            for (int c = 0; c < MEAN_COLS; c++) acc[c] += x[c];
        else
            for (int c = 0; c < w; c++) acc[c] += x[c];
    }
    for (int c = 0; c < w; c++) j->mean[c0 + c] = acc[c] / j->n;
}

void plx_col_mean_f32(const float *X, int n, int d, double *mean) {
    PLX_ASSERT(n >= 1 && d >= 1);
    mean_job j = {X, n, d, mean};
    la_par_for((d + MEAN_COLS - 1) / MEAN_COLS, (double)n * MEAN_COLS, mean_task, &j);
}

/* Logical views of the centred data: Xc (n x d) and Xc^T (d x n). */
static la_opnd xc_view(const float *X, int d, const double *mean) {
    la_opnd o = {.f = X, .rs = d, .cs = 1, .sub_col = mean};
    return o;
}
static la_opnd xct_view(const float *X, int d, const double *mean) {
    la_opnd o = {.f = X, .rs = 1, .cs = d, .sub_row = mean};
    return o;
}

void plx_scatter_f32(const float *X, int n, int d, const double *mean, plx_mat *S) {
    PLX_ASSERT(n >= 1 && d >= 1 && S->rows == d && S->cols == d);
    la_opnd a = xct_view(X, d, mean), b = xc_view(X, d, mean);
    la_gemm(d, d, n, 1.0, &a, &b, 0.0, S->a, d, 1);
    la_mirror_lower(S->a, d, d);
}

void plx_gram_f32(const float *X, int n, int d, const double *mean, plx_mat *G) {
    PLX_ASSERT(n >= 1 && d >= 1 && G->rows == n && G->cols == n);
    la_opnd a = xc_view(X, d, mean), b = xct_view(X, d, mean);
    la_gemm(n, n, d, 1.0, &a, &b, 0.0, G->a, n, 1);
    la_mirror_lower(G->a, n, n);
}

void plx_cov_f32(const float *X, int n, int d, double *mean_out, plx_mat *C) {
    PLX_ASSERT(n >= 2);
    double *mean = mean_out ? mean_out : plx_malloc(sizeof(double) * d);
    plx_col_mean_f32(X, n, d, mean);
    plx_scatter_f32(X, n, d, mean, C);
    plx_mat_scale(C, 1.0 / (n - 1));
    if (!mean_out) free(mean);
}

/* ---- PCA ------------------------------------------------------------------------------------ */

/* Y (n x l) = Xc M with M d x l, or (transposed) Y (d x l) = Xc^T M with M n x l. */
static void xc_mul(const float *X, int n, int d, const double *mean, int trans, const double *M, int l, double *Y) {
    la_opnd a = trans ? xct_view(X, d, mean) : xc_view(X, d, mean), b = la_opnd_d(M, l, 0);
    la_gemm(trans ? d : n, l, trans ? n : d, 1.0, &a, &b, 0.0, Y, l, 0);
}

/* Columns of Y (r x l, r >= l) replaced by an orthonormal basis of their span (Householder, so it
 * stays orthonormal even when Y is numerically rank deficient, as power iterates become). */
static void orthonormalise(double *Y, int r, int l) {
    la_qr q;
    double *Q = plx_malloc(sizeof(double) * (size_t)r * l);
    la_qr_factor(&q, Y, r, l, l);
    la_qr_form_q(&q, Q, l);
    la_qr_free(&q);
    memcpy(Y, Q, sizeof(double) * (size_t)r * l);
    free(Q);
}

/* Rayleigh-Ritz on an orthonormal left basis Q (n x l): Xc ~= Q Q^T Xc and Q^T Xc = (Xc^T Q)^T,
 * so an SVD of Bt = Xc^T Q (d x l) = Ub S Vb^T gives Xc ~= (Q Vb) S Ub^T: axes = Ub. */
static int ritz_from_left_basis(const float *X, int n, int d, const double *mean, const double *Q, int l,
                                plx_pca *out) {
    double *Bt = plx_malloc(sizeof(double) * (size_t)d * l), *Ub = plx_malloc(sizeof(double) * (size_t)d * l);
    double *s = plx_malloc(sizeof(double) * l);
    xc_mul(X, n, d, mean, 1, Q, l, Bt);
    int rc = la_svd_raw(Bt, d, l, Ub, s, NULL);
    for (int i = 0; i < out->k; i++) {
        out->sv[i] = s[i];
        for (int c = 0; c < d; c++) PLX_M(&out->axes, i, c) = Ub[(size_t)c * l + i];
    }
    free(Bt);
    free(Ub);
    free(s);
    return rc;
}

static int pca_cov(const float *X, int n, int d, plx_pca *out) {
    plx_mat S = plx_mat_new(d, d);
    plx_scatter_f32(X, n, d, out->mean, &S);
    double *w = plx_malloc(sizeof(double) * d);
    int rc = plx_eigh(&S, w, &S);
    for (int i = 0; i < out->k; i++) {
        out->sv[i] = sqrt(PLX_MAX(w[i], 0.0));
        for (int c = 0; c < d; c++) PLX_M(&out->axes, i, c) = PLX_M(&S, c, i);
    }
    free(w);
    plx_mat_free(&S);
    return rc;
}

static int pca_gram(const float *X, int n, int d, plx_pca *out) {
    plx_mat G = plx_mat_new(n, n);
    plx_gram_f32(X, n, d, out->mean, &G);
    double *w = plx_malloc(sizeof(double) * n);
    int rc = plx_eigh(&G, w, &G);
    if (rc == PLX_OK) {
        const int k = out->k;
        double *Q = plx_malloc(sizeof(double) * (size_t)n * k);
        for (int i = 0; i < n; i++) memcpy(Q + (size_t)i * k, &PLX_M(&G, i, 0), sizeof(double) * k);
        rc = ritz_from_left_basis(X, n, d, out->mean, Q, k, out);
        free(Q);
    }
    free(w);
    plx_mat_free(&G);
    return rc;
}

/* Randomised range finder with subspace (power) iteration, Halko-Martinsson-Tropp Alg. 4.4:
 * re-orthonormalising after every product keeps the iterates from collapsing onto the leading
 * singular vector in floating point. */
static int pca_randomized(const float *X, int n, int d, int l, int q, uint64_t seed, plx_pca *out) {
    plx_rng rng;
    plx_rng_seed(&rng, seed);
    double *Om = plx_malloc(sizeof(double) * (size_t)d * l);
    for (size_t i = 0; i < (size_t)d * l; i++) Om[i] = plx_rng_normal(&rng);
    double *Y = plx_malloc(sizeof(double) * (size_t)n * l);
    xc_mul(X, n, d, out->mean, 0, Om, l, Y);
    orthonormalise(Y, n, l);
    for (int it = 0; it < q; it++) {
        xc_mul(X, n, d, out->mean, 1, Y, l, Om); /* Om reused as the d x l iterate */
        orthonormalise(Om, d, l);
        xc_mul(X, n, d, out->mean, 0, Om, l, Y);
        orthonormalise(Y, n, l);
    }
    free(Om);
    int rc = ritz_from_left_basis(X, n, d, out->mean, Y, l, out);
    free(Y);
    return rc;
}

/* Rough cost model, in gemm-flop equivalents. EIG_COST: flops-equivalent per n^3 of a full
 * plx_eigh with vectors relative to the threaded gemm (measured: level-2 tridiagonalisation and
 * rotation replay run ~20x below gemm speed). QR_COST likewise for Householder QR + forming Q. */
#define EIG_COST 60.0
#define QR_COST 6.0
#define EXACT_PREFERENCE 2.0 /* randomised must be this much cheaper to be chosen: it is approximate */

static plx_pca_method pick_method(double n, double d, double k, double l, double q) {
    double c_cov = n * d * d + EIG_COST * d * d * d;
    double c_gram = n * n * d + EIG_COST * n * n * n + 2 * n * d * k + QR_COST * d * k * k;
    double c_rand = (2 * q + 2) * 2 * n * d * l + QR_COST * (q + 1) * (n + d) * l * l + QR_COST * d * l * l;
    const double mem_limit = 4e9; /* bytes for the d x d / n x n double matrix + eigenvectors */
    if (16 * d * d > mem_limit) c_cov = INFINITY;
    if (16 * n * n > mem_limit) c_gram = INFINITY;
    if (l >= PLX_MIN(n, d)) c_rand = INFINITY; /* no subspace to sample: exact is as cheap */
    c_rand *= EXACT_PREFERENCE;
    if (c_cov <= c_gram && c_cov <= c_rand) return PLX_PCA_COV;
    if (c_gram <= c_rand) return PLX_PCA_GRAM;
    return PLX_PCA_RANDOMIZED;
}

int plx_pca_f32(const float *X, int n, int d, int k, const plx_pca_opts *opts, plx_pca *out) {
    PLX_ASSERT(n >= 2 && d >= 1 && k >= 1 && k <= PLX_MIN(n, d));
    plx_pca_opts o = opts ? *opts : (plx_pca_opts){0};
    if (o.oversample <= 0) o.oversample = PLX_MAX(10, k / 10);
    if (o.power_iters <= 0) o.power_iters = 4;
    if (o.seed == 0) o.seed = 0x5eed;
    const int l = PLX_MIN(k + o.oversample, PLX_MIN(n, d));
    if (o.method == PLX_PCA_AUTO) o.method = pick_method(n, d, k, l, o.power_iters);

    memset(out, 0, sizeof *out);
    out->n = n;
    out->d = d;
    out->k = k;
    out->method = o.method;
    out->mean = plx_malloc(sizeof(double) * d);
    out->sv = plx_malloc(sizeof(double) * k);
    out->axes = plx_mat_new(k, d);
    plx_col_mean_f32(X, n, d, out->mean);
    int rc;
    switch (o.method) {
    case PLX_PCA_COV: rc = pca_cov(X, n, d, out); break;
    case PLX_PCA_GRAM: rc = pca_gram(X, n, d, out); break;
    case PLX_PCA_RANDOMIZED: rc = pca_randomized(X, n, d, l, o.power_iters, o.seed, out); break;
    default: plx_die("plx_pca_f32: bad method %d", (int)o.method);
    }
    /* sign convention on rows: transpose view as d x k columns */
    double *cols = plx_malloc(sizeof(double) * (size_t)d * k);
    for (int i = 0; i < k; i++)
        for (int c = 0; c < d; c++) cols[(size_t)c * k + i] = PLX_M(&out->axes, i, c);
    la_fix_signs(cols, d, k, NULL, 0);
    for (int i = 0; i < k; i++)
        for (int c = 0; c < d; c++) PLX_M(&out->axes, i, c) = cols[(size_t)c * k + i];
    free(cols);
    return rc;
}

void plx_pca_free(plx_pca *p) {
    free(p->mean);
    free(p->sv);
    plx_mat_free(&p->axes);
    memset(p, 0, sizeof *p);
}

void plx_project_f32(const float *X, int n, int d, const double *mean, const plx_mat *basis,
                     const double *scale, float *Y) {
    const int k = basis->rows;
    PLX_ASSERT(basis->cols == d && n >= 0 && k >= 1);
    enum { ROWS = 4096 }; /* bounds the double scratch to ROWS x k */
    double *buf = plx_malloc(sizeof(double) * (size_t)PLX_MIN(n, ROWS) * k);
    la_opnd b = la_opnd_d(basis->a, d, 1);
    for (int r0 = 0; r0 < n; r0 += ROWS) {
        const int rows = PLX_MIN(ROWS, n - r0);
        la_opnd a = xc_view(X + (size_t)r0 * d, d, mean);
        la_gemm(rows, k, d, 1.0, &a, &b, 0.0, buf, k, 0);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < k; j++) {
                double v = buf[(size_t)i * k + j];
                if (scale) v = scale[j] != 0 ? v / scale[j] : 0.0;
                Y[(size_t)(r0 + i) * k + j] = (float)v;
            }
    }
    free(buf);
}

void plx_pca_project_f32(const plx_pca *p, const float *X, int n, bool whiten, float *Y) {
    double *scale = NULL;
    if (whiten) {
        scale = plx_malloc(sizeof(double) * p->k);
        for (int i = 0; i < p->k; i++) scale[i] = p->sv[i] / sqrt((double)(p->n - 1));
    }
    plx_project_f32(X, n, p->d, p->mean, &p->axes, scale, Y);
    free(scale);
}
