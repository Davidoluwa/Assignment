// plx-deps: la core
#include "test.h"

#include "core/core.h"
#include "la/la.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

/* Property tests for src/la. Every decomposition is checked by what defines it (reconstruction,
 * orthogonality, ordering) and against an independent route; tolerances are stated relative to
 * eps = 2.2e-16 and the matrix size. PLX_SLOW_TESTS=1 adds the scale/speed runs. */

#define EPS DBL_EPSILON

/* ---- helpers ---------------------------------------------------------------------------- */

static plx_mat randn(plx_rng *r, int m, int n) {
    plx_mat A = plx_mat_new(m, n);
    for (size_t i = 0; i < (size_t)m * n; i++) A.a[i] = plx_rng_normal(r);
    return A;
}

static plx_mat rand_sym(plx_rng *r, int n) {
    plx_mat A = plx_mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) PLX_M(&A, i, j) = PLX_M(&A, j, i) = plx_rng_normal(r);
    return A;
}

static plx_mat rand_orth(plx_rng *r, int m, int n) { /* m x n, orthonormal columns */
    plx_mat G = randn(r, m, n), Q = plx_mat_new(m, n), R = plx_mat_new(n, n);
    plx_qr(&G, &Q, &R);
    plx_mat_free(&G);
    plx_mat_free(&R);
    return Q;
}

/* U diag(s) V^T with U m x k, V n x k */
static plx_mat from_svd(const plx_mat *U, const double *s, const plx_mat *V) {
    plx_mat US = plx_mat_clone(U), A = plx_mat_new(U->rows, V->rows);
    for (int i = 0; i < US.rows; i++)
        for (int j = 0; j < US.cols; j++) PLX_M(&US, i, j) *= s[j];
    plx_gemm(PLX_NO_TRANS, PLX_TRANS, 1.0, &US, V, 0.0, &A);
    plx_mat_free(&US);
    return A;
}

static plx_mat rand_spd(plx_rng *r, int n, double cond) { /* eigenvalues log-spaced in [1/cond, 1] */
    plx_mat Q = rand_orth(r, n, n);
    double *s = PLX_NEW(double, n);
    for (int i = 0; i < n; i++) s[i] = pow(cond, -(double)i / PLX_MAX(n - 1, 1));
    plx_mat A = from_svd(&Q, s, &Q);
    for (int i = 0; i < n; i++) /* exact symmetry */
        for (int j = 0; j < i; j++) PLX_M(&A, j, i) = PLX_M(&A, i, j);
    free(s);
    plx_mat_free(&Q);
    return A;
}

static plx_mat hilbert(int n) {
    plx_mat H = plx_mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) PLX_M(&H, i, j) = 1.0 / (i + j + 1);
    return H;
}

static double max_abs(const plx_mat *A) {
    double m = 0;
    for (size_t i = 0; i < (size_t)A->rows * A->cols; i++) m = PLX_MAX(m, fabs(A->a[i]));
    return m;
}

static double max_diff(const plx_mat *A, const plx_mat *B) {
    PLX_ASSERT(A->rows == B->rows && A->cols == B->cols);
    double m = 0;
    for (size_t i = 0; i < (size_t)A->rows * A->cols; i++) m = PLX_MAX(m, fabs(A->a[i] - B->a[i]));
    return m;
}

/* ||A - B||_F / ||B||_F */
static double rel_err(const plx_mat *A, const plx_mat *B) {
    plx_mat D = plx_mat_clone(A);
    plx_mat_add(&D, -1.0, B);
    double e = plx_mat_norm_fro(&D), nb = plx_mat_norm_fro(B);
    plx_mat_free(&D);
    return nb > 0 ? e / nb : e;
}

/* max |Q^T Q - I| */
static double orth_err(const plx_mat *Q) {
    plx_mat G = plx_mat_new(Q->cols, Q->cols);
    plx_syrk(PLX_TRANS, 1.0, Q, 0.0, &G);
    for (int i = 0; i < G.rows; i++) PLX_M(&G, i, i) -= 1.0;
    double e = max_abs(&G);
    plx_mat_free(&G);
    return e;
}

/* op(A) op(B) accumulated in long double: an independent reference for the packed kernel. */
static plx_mat ref_gemm(plx_op ta, plx_op tb, const plx_mat *A, const plx_mat *B) {
    int m = ta ? A->cols : A->rows, k = ta ? A->rows : A->cols, n = tb ? B->rows : B->cols;
    plx_mat C = plx_mat_new(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            long double s = 0;
            for (int p = 0; p < k; p++)
                s += (long double)(ta ? PLX_M(A, p, i) : PLX_M(A, i, p)) * (tb ? PLX_M(B, j, p) : PLX_M(B, p, j));
            PLX_M(&C, i, j) = (double)s;
        }
    return C;
}

/* ||A V - V diag(w)||_F / ||A||_F */
static double eig_resid(const plx_mat *A, const double *w, const plx_mat *V) {
    plx_mat AV = plx_matmul(A, V);
    for (int i = 0; i < AV.rows; i++)
        for (int j = 0; j < AV.cols; j++) PLX_M(&AV, i, j) -= w[j] * PLX_M(V, i, j);
    double e = plx_mat_norm_fro(&AV), na = plx_mat_norm_fro(A);
    plx_mat_free(&AV);
    return na > 0 ? e / na : e;
}

/* Every column's largest-magnitude entry is positive (the documented sign convention). */
static int signs_canonical(const plx_mat *V) {
    for (int j = 0; j < V->cols; j++) {
        int b = 0;
        for (int i = 1; i < V->rows; i++)
            if (fabs(PLX_M(V, i, j)) > fabs(PLX_M(V, b, j))) b = i;
        if (PLX_M(V, b, j) < 0) return 0;
    }
    return 1;
}

static int descending(const double *w, int n) {
    for (int i = 1; i < n; i++)
        if (w[i] > w[i - 1]) return 0;
    return 1;
}

/* Singular values by one-sided Jacobi in long double (64-bit mantissa): an oracle that is ~2000x
 * more precise than double, for checking relative accuracy of tiny singular values. */
static void ref_singular_values(const plx_mat *A, double *s) {
    const int m = A->rows, n = A->cols;
    long double *G = calloc((size_t)m * n, sizeof(long double));
    for (int i = 0; i < m * n; i++) G[i] = A->a[i];
    for (int sweep = 0; sweep < 100; sweep++) {
        int rot = 0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                long double al = 0, be = 0, ga = 0;
                for (int i = 0; i < m; i++) {
                    al += G[i * n + p] * G[i * n + p];
                    be += G[i * n + q] * G[i * n + q];
                    ga += G[i * n + p] * G[i * n + q];
                }
                if (fabsl(ga) <= 1e-19L * sqrtl(al * be)) continue;
                rot = 1;
                long double z = (be - al) / (2 * ga), t = (z >= 0 ? 1 : -1) / (fabsl(z) + sqrtl(1 + z * z));
                long double c = 1 / sqrtl(1 + t * t), sn = c * t;
                for (int i = 0; i < m; i++) {
                    long double x = G[i * n + p], y = G[i * n + q];
                    G[i * n + p] = c * x - sn * y;
                    G[i * n + q] = sn * x + c * y;
                }
            }
        if (!rot) break;
    }
    for (int j = 0; j < n; j++) {
        long double t = 0;
        for (int i = 0; i < m; i++) t += G[i * n + j] * G[i * n + j];
        s[j] = (double)sqrtl(t);
    }
    for (int i = 0; i < n; i++) /* sort descending */
        for (int j = i + 1; j < n; j++)
            if (s[j] > s[i]) { double t = s[i]; s[i] = s[j]; s[j] = t; }
    free(G);
}

/* ---- basics ----------------------------------------------------------------------------- */

TEST(basics) {
    plx_mat Z = plx_mat_new(3, 4);
    CHECK(Z.rows == 3 && Z.cols == 4 && max_abs(&Z) == 0);
    plx_mat E = plx_mat_new(0, 5); /* empty matrices are legal */
    CHECK(plx_mat_norm_fro(&E) == 0);
    plx_mat_free(&E);
    CHECK(E.a == NULL && E.rows == 0);

    double d[6] = {1, 2, 3, 4, 5, 6};
    plx_mat A = plx_mat_from(2, 3, d), T = plx_mat_transpose(&A);
    CHECK(T.rows == 3 && T.cols == 2 && PLX_M(&T, 2, 1) == 6 && PLX_M(&T, 0, 1) == 4);
    plx_mat TT = plx_mat_transpose(&T);
    CHECK(max_diff(&TT, &A) == 0);
    plx_mat C = plx_mat_clone(&A);
    PLX_M(&C, 0, 0) = 99;
    CHECK(PLX_M(&A, 0, 0) == 1); /* deep copy */
    plx_mat_copy(&C, &A);
    CHECK(max_diff(&C, &A) == 0);
    plx_mat_scale(&C, -2);
    plx_mat_add(&C, 2.0, &A); /* -2A + 2A */
    CHECK(max_abs(&C) == 0);
    CHECK_NEAR(plx_mat_norm_fro(&A), sqrt(91.0), 1e-14);
    plx_mat I = plx_mat_identity(4);
    CHECK_NEAR(plx_mat_norm_fro(&I), 2.0, 0);

    plx_rng r;
    plx_rng_seed(&r, 1);
    plx_mat B = randn(&r, 300, 257); /* big transpose crosses tile boundaries */
    plx_mat Bt = plx_mat_transpose(&B);
    int ok = 1;
    for (int i = 0; i < 300; i++)
        for (int j = 0; j < 257; j++) ok &= PLX_M(&B, i, j) == PLX_M(&Bt, j, i);
    CHECK(ok);

    /* vectors: dot/axpy against plain loops; norms survive over/underflow of the squares */
    double x[37], y[37], y0[37], ref = 0;
    for (int i = 0; i < 37; i++) {
        x[i] = plx_rng_normal(&r);
        y[i] = y0[i] = plx_rng_normal(&r);
        ref += x[i] * y[i];
    }
    CHECK_NEAR(plx_vec_dot(x, y, 37), ref, 1e-13);
    plx_vec_axpy(y, 0.5, x, 37);
    int axpy_ok = 1;
    for (int i = 0; i < 37; i++) axpy_ok &= fabs(y[i] - (y0[i] + 0.5 * x[i])) < 1e-15;
    CHECK(axpy_ok);
    double big[3] = {3e200, 4e200, 0}, tiny[2] = {3e-200, 4e-200};
    CHECK_NEAR(plx_vec_norm(big, 3) / 5e200, 1.0, 1e-15);
    CHECK_NEAR(plx_vec_norm(tiny, 2) / 5e-200, 1.0, 1e-15);
    plx_mat Mb = plx_mat_from(1, 3, big);
    CHECK_NEAR(plx_mat_norm_fro(&Mb) / 5e200, 1.0, 1e-15);
    CHECK(plx_vec_norm(x, 0) == 0);
    CHECK(strcmp(plx_la_status_str(PLX_ENOTSPD), "matrix not positive definite") == 0);
    CHECK(strcmp(plx_la_status_str(PLX_EIO), "i/o error") == 0);

    plx_mat_free(&Z); plx_mat_free(&A); plx_mat_free(&T); plx_mat_free(&TT); plx_mat_free(&C);
    plx_mat_free(&I); plx_mat_free(&B); plx_mat_free(&Bt); plx_mat_free(&Mb);
}

/* ---- products --------------------------------------------------------------------------- */

TEST(gemm_matches_reference) {
    plx_rng r;
    plx_rng_seed(&r, 2);
    /* shapes straddle the blocking parameters (MR = NR = 8, MC = 128, NB = 512, KC = 256) and the
     * small-matrix path (m n k <= 4096) */
    static const int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {16, 16, 16}, {17, 9, 33}, {129, 70, 257},
                                    {8, 513, 300}, {200, 1, 260}, {1, 140, 600}};
    double worst = 0;
    for (size_t t = 0; t < sizeof shapes / sizeof shapes[0]; t++)
        for (int ta = 0; ta < 2; ta++)
            for (int tb = 0; tb < 2; tb++) {
                int m = shapes[t][0], n = shapes[t][1], k = shapes[t][2];
                plx_mat A = ta ? randn(&r, k, m) : randn(&r, m, k), B = tb ? randn(&r, n, k) : randn(&r, k, n);
                plx_mat C = plx_mat_new(m, n), R = ref_gemm(ta, tb, &A, &B);
                for (size_t i = 0; i < (size_t)m * n; i++) C.a[i] = NAN; /* beta = 0 must not read C */
                plx_gemm(ta, tb, 1.0, &A, &B, 0.0, &C);
                /* rounding bound: |error_ij| <= k eps sum_p |a_ip||b_pj| ~ k eps * sqrt(k) for N(0,1) */
                double e = max_diff(&C, &R) / (k * EPS * sqrt((double)k));
                worst = PLX_MAX(worst, e);
                CHECK(e < 1.0);
                plx_mat_free(&A); plx_mat_free(&B); plx_mat_free(&C); plx_mat_free(&R);
            }
    printf("  gemm: worst error %.3g of the k*eps*sqrt(k) bound\n", worst);

    /* alpha / beta semantics: C = 2 A B - 3 C */
    plx_mat A = randn(&r, 70, 90), B = randn(&r, 90, 50), C0 = randn(&r, 70, 50);
    plx_mat C = plx_mat_clone(&C0), R = ref_gemm(0, 0, &A, &B);
    plx_gemm(PLX_NO_TRANS, PLX_NO_TRANS, 2.0, &A, &B, -3.0, &C);
    plx_mat_scale(&R, 2.0);
    plx_mat_add(&R, -3.0, &C0);
    CHECK(max_diff(&C, &R) < 1e-12);
    plx_gemm(PLX_NO_TRANS, PLX_NO_TRANS, 0.0, &A, &B, 0.5, &C); /* alpha = 0: pure scaling */
    plx_mat_scale(&R, 0.5);
    CHECK(max_diff(&C, &R) < 1e-12);
    plx_mat M = plx_matmul(&A, &B), R1 = ref_gemm(0, 0, &A, &B);
    CHECK(max_diff(&M, &R1) < 1e-12);

    /* the result is independent of the thread count, bitwise */
    plx_mat P = randn(&r, 300, 400), Q = randn(&r, 400, 350), C4 = plx_mat_new(300, 350), C1 = plx_mat_new(300, 350);
    plx_gemm(PLX_NO_TRANS, PLX_NO_TRANS, 1.0, &P, &Q, 0.0, &C4);
    plx_set_num_threads(1);
    plx_gemm(PLX_NO_TRANS, PLX_NO_TRANS, 1.0, &P, &Q, 0.0, &C1);
    plx_set_num_threads(0);
    CHECK(memcmp(C1.a, C4.a, sizeof(double) * 300 * 350) == 0);

    plx_mat_free(&A); plx_mat_free(&B); plx_mat_free(&C0); plx_mat_free(&C); plx_mat_free(&R);
    plx_mat_free(&M); plx_mat_free(&R1); plx_mat_free(&P); plx_mat_free(&Q); plx_mat_free(&C4); plx_mat_free(&C1);
}

TEST(gemv_and_syrk) {
    plx_rng r;
    plx_rng_seed(&r, 3);
    static const int shapes[][2] = {{1, 1}, {7, 3}, {130, 600}, {600, 130}, {50, 1100}};
    for (size_t t = 0; t < 5; t++) {
        int m = shapes[t][0], n = shapes[t][1];
        plx_mat A = randn(&r, m, n), x = randn(&r, n, 1), xt = randn(&r, m, 1);
        plx_mat y = randn(&r, m, 1), yt = randn(&r, n, 1);
        plx_mat Rn = ref_gemm(0, 0, &A, &x), Rt = ref_gemm(1, 0, &A, &xt);
        plx_mat_scale(&Rn, 1.5);
        plx_mat_add(&Rn, 0.25, &y);
        plx_gemv(PLX_NO_TRANS, 1.5, &A, x.a, 0.25, y.a);
        CHECK(max_diff(&y, &Rn) < 1e-12);
        for (int i = 0; i < n; i++) yt.a[i] = NAN;
        plx_gemv(PLX_TRANS, 1.0, &A, xt.a, 0.0, yt.a);
        CHECK(max_diff(&yt, &Rt) < 1e-12);

        /* syrk both ways against gemm; output exactly symmetric */
        plx_mat S = plx_mat_new(n, n), G = plx_mat_new(n, n), S2 = plx_mat_new(m, m), G2 = plx_mat_new(m, m);
        plx_syrk(PLX_TRANS, 1.0, &A, 0.0, &S);
        plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &A, &A, 0.0, &G);
        plx_syrk(PLX_NO_TRANS, 1.0, &A, 0.0, &S2);
        plx_gemm(PLX_NO_TRANS, PLX_TRANS, 1.0, &A, &A, 0.0, &G2);
        CHECK(max_diff(&S, &G) < 1e-11 * m && max_diff(&S2, &G2) < 1e-11 * n);
        plx_mat St = plx_mat_transpose(&S), S2t = plx_mat_transpose(&S2);
        CHECK(max_diff(&S, &St) == 0 && max_diff(&S2, &S2t) == 0);
        plx_mat_free(&A); plx_mat_free(&x); plx_mat_free(&xt); plx_mat_free(&y); plx_mat_free(&yt);
        plx_mat_free(&Rn); plx_mat_free(&Rt); plx_mat_free(&S); plx_mat_free(&G); plx_mat_free(&S2);
        plx_mat_free(&G2); plx_mat_free(&St); plx_mat_free(&S2t);
    }
    /* syrk with beta: only the lower triangle of the incoming C counts */
    plx_mat A = randn(&r, 40, 20), C = rand_sym(&r, 20), R = ref_gemm(1, 0, &A, &A);
    plx_mat_add(&R, 2.0, &C);
    for (int i = 0; i < 20; i++)
        for (int j = i + 1; j < 20; j++) PLX_M(&C, i, j) = 1e300; /* garbage upper */
    plx_syrk(PLX_TRANS, 1.0, &A, 2.0, &C);
    CHECK(max_diff(&C, &R) < 1e-12);
    plx_mat_free(&A); plx_mat_free(&C); plx_mat_free(&R);
}

/* ---- Cholesky / LU ----------------------------------------------------------------------- */

TEST(cholesky) {
    plx_rng r;
    plx_rng_seed(&r, 4);
    static const int sizes[] = {1, 2, 5, 96, 97, 250};
    for (size_t t = 0; t < 6; t++) {
        int n = sizes[t];
        plx_mat A = rand_spd(&r, n, 1e6), L = plx_mat_clone(&A);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) PLX_M(&L, i, j) = NAN; /* upper triangle must be ignored */
        CHECK(plx_chol(&L) == PLX_OK);
        int upper_zero = 1;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) upper_zero &= PLX_M(&L, i, j) == 0;
        CHECK(upper_zero);
        plx_mat LLt = plx_mat_new(n, n);
        plx_syrk(PLX_NO_TRANS, 1.0, &L, 0.0, &LLt);
        CHECK(rel_err(&LLt, &A) < 10 * n * EPS);

        /* solves: backward error of the residual is ~ eps, forward error ~ cond * eps */
        plx_mat X0 = randn(&r, n, 3), B = plx_matmul(&A, &X0), X = plx_mat_clone(&B);
        plx_chol_solve(&L, &X);
        plx_mat AX = plx_matmul(&A, &X);
        CHECK(rel_err(&AX, &B) < 100 * n * EPS);
        CHECK(rel_err(&X, &X0) < 1e6 * 100 * EPS);
        double *b = PLX_NEW(double, n);
        for (int i = 0; i < n; i++) b[i] = PLX_M(&B, i, 1);
        plx_chol_solve_vec(&L, b);
        double vd = 0;
        for (int i = 0; i < n; i++) vd = PLX_MAX(vd, fabs(b[i] - PLX_M(&X, i, 1)));
        CHECK(vd < 1e-6);

        /* logdet against the product of eigenvalues, inverse against the identity */
        double want = 0;
        for (int i = 0; i < n; i++) want += log(pow(1e6, -(double)i / PLX_MAX(n - 1, 1)));
        CHECK_NEAR(plx_chol_logdet(&L), want, 1e-8 * n);
        plx_mat Ai = plx_mat_new(n, n);
        CHECK(plx_spd_inverse(&A, &Ai) == PLX_OK);
        plx_mat AAi = plx_matmul(&A, &Ai), I = plx_mat_identity(n), Ait = plx_mat_transpose(&Ai);
        CHECK(max_diff(&AAi, &I) < 1e6 * 100 * EPS * n && max_diff(&Ai, &Ait) == 0);
        free(b);
        plx_mat_free(&A); plx_mat_free(&L); plx_mat_free(&LLt); plx_mat_free(&X0); plx_mat_free(&B);
        plx_mat_free(&X); plx_mat_free(&AX); plx_mat_free(&Ai); plx_mat_free(&AAi); plx_mat_free(&I);
        plx_mat_free(&Ait);
    }

    /* Hilbert(10): cond 1.6e13, still SPD in double; solution error tracks cond * eps */
    plx_mat H = hilbert(10), L = plx_mat_clone(&H);
    CHECK(plx_chol(&L) == PLX_OK);
    plx_mat x = plx_mat_new(10, 1), b = plx_mat_new(10, 1);
    for (int i = 0; i < 10; i++) PLX_M(&x, i, 0) = 1.0;
    plx_gemv(PLX_NO_TRANS, 1.0, &H, x.a, 0.0, b.a);
    plx_chol_solve(&L, &b);
    CHECK(rel_err(&b, &x) < 1e-2);

    /* failures: indefinite, semidefinite (exact zero pivot), NaN, a negative 1x1 */
    double indef[4] = {1, 2, 2, 1}, semi[4] = {1, 1, 1, 1}, nanm[4] = {NAN, 0, 0, 1}, neg[1] = {-1};
    plx_mat F1 = plx_mat_from(2, 2, indef), F2 = plx_mat_from(2, 2, semi), F3 = plx_mat_from(2, 2, nanm), F4 = plx_mat_from(1, 1, neg);
    CHECK(plx_chol(&F1) == PLX_ENOTSPD && plx_chol(&F2) == PLX_ENOTSPD);
    CHECK(plx_chol(&F3) == PLX_ENOTSPD && plx_chol(&F4) == PLX_ENOTSPD);
    plx_mat Big = rand_spd(&r, 200, 10); /* failure detected deep inside the blocked path */
    PLX_M(&Big, 150, 150) = -1;
    CHECK(plx_chol(&Big) == PLX_ENOTSPD);
    plx_mat Fi = plx_mat_new(2, 2);
    CHECK(plx_spd_inverse(&F1, &Fi) == PLX_ENOTSPD);
    plx_mat_free(&H); plx_mat_free(&L); plx_mat_free(&x); plx_mat_free(&b); plx_mat_free(&F1);
    plx_mat_free(&F2); plx_mat_free(&F3); plx_mat_free(&F4); plx_mat_free(&Big); plx_mat_free(&Fi);
}

TEST(lu) {
    plx_rng r;
    plx_rng_seed(&r, 5);
    static const int sizes[] = {1, 2, 9, 120};
    for (size_t t = 0; t < 4; t++) {
        int n = sizes[t];
        plx_mat A = randn(&r, n, n), LU = plx_mat_clone(&A);
        int *piv = PLX_NEW(int, n);
        CHECK(plx_lu(&LU, piv) == PLX_OK);
        /* P A = L U, reconstructed explicitly */
        plx_mat L = plx_mat_identity(n), U = plx_mat_new(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                if (j < i) PLX_M(&L, i, j) = PLX_M(&LU, i, j);
                else PLX_M(&U, i, j) = PLX_M(&LU, i, j);
            }
        plx_mat PA = plx_mat_clone(&A), LUp = plx_matmul(&L, &U);
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++) {
                double tmp = PLX_M(&PA, k, j);
                PLX_M(&PA, k, j) = PLX_M(&PA, piv[k], j);
                PLX_M(&PA, piv[k], j) = tmp;
            }
        CHECK(rel_err(&LUp, &PA) < 10 * n * EPS);
        CHECK(max_abs(&L) <= 1.0); /* partial pivoting bounds the multipliers */

        plx_mat X0 = randn(&r, n, 4), B = plx_matmul(&A, &X0), X = plx_mat_clone(&B);
        plx_lu_solve(&LU, piv, &X);
        plx_mat AX = plx_matmul(&A, &X);
        CHECK(rel_err(&AX, &B) < 100 * n * EPS);
        plx_mat Y = plx_mat_clone(&B);
        CHECK(plx_solve(&A, &Y) == PLX_OK && max_diff(&X, &Y) == 0);
        double *b = PLX_NEW(double, n);
        for (int i = 0; i < n; i++) b[i] = PLX_M(&B, i, 2);
        plx_lu_solve_vec(&LU, piv, b);
        double vd = 0;
        for (int i = 0; i < n; i++) vd = PLX_MAX(vd, fabs(b[i] - PLX_M(&X, i, 2)));
        CHECK(vd < 1e-10);

        plx_mat Ai = plx_mat_new(n, n);
        CHECK(plx_inverse(&A, &Ai) == PLX_OK);
        plx_mat AAi = plx_matmul(&A, &Ai), I = plx_mat_identity(n);
        CHECK(max_diff(&AAi, &I) < 1e-9);
        /* det(A B) = det A det B; det A^-1 = 1 / det A */
        plx_mat C = randn(&r, n, n), AC = plx_matmul(&A, &C);
        double dA = plx_lu_det(&LU, piv), dC = plx_det(&C), dAC = plx_det(&AC);
        CHECK(fabs(dAC - dA * dC) <= 1e-9 * fabs(dA * dC));
        CHECK(fabs(plx_det(&Ai) * dA - 1.0) < 1e-9);
        free(piv); free(b);
        plx_mat_free(&A); plx_mat_free(&LU); plx_mat_free(&L); plx_mat_free(&U); plx_mat_free(&PA);
        plx_mat_free(&LUp); plx_mat_free(&X0); plx_mat_free(&B); plx_mat_free(&X); plx_mat_free(&AX);
        plx_mat_free(&Y); plx_mat_free(&Ai); plx_mat_free(&AAi); plx_mat_free(&I); plx_mat_free(&C);
        plx_mat_free(&AC);
    }
    /* known determinants: triangular, permutation sign */
    double tri[9] = {2, 7, 1, 0, 3, 5, 0, 0, -4}, perm[9] = {0, 1, 0, 0, 0, 1, 1, 0, 0}, swap[9] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
    plx_mat T = plx_mat_from(3, 3, tri), P = plx_mat_from(3, 3, perm), S = plx_mat_from(3, 3, swap);
    CHECK_NEAR(plx_det(&T), -24, 1e-13);
    CHECK_NEAR(plx_det(&P), 1, 0);
    CHECK_NEAR(plx_det(&S), -1, 0);

    /* singular: exact, numerically (rank 49 of 50), and zero; B is left untouched on failure */
    double sing[4] = {1, 2, 2, 4};
    plx_mat Sg = plx_mat_from(2, 2, sing), Bs = plx_mat_identity(2), Bs0 = plx_mat_identity(2);
    CHECK(plx_solve(&Sg, &Bs) == PLX_ESINGULAR && max_diff(&Bs, &Bs0) == 0);
    CHECK(plx_det(&Sg) == 0);
    plx_mat G1 = randn(&r, 50, 49), G2 = randn(&r, 49, 50), R49 = plx_matmul(&G1, &G2), Ri = plx_mat_new(50, 50);
    CHECK(plx_inverse(&R49, &Ri) == PLX_ESINGULAR);
    plx_mat Z = plx_mat_new(4, 4);
    int zp[4];
    CHECK(plx_lu(&Z, zp) == PLX_ESINGULAR && plx_lu_det(&Z, zp) == 0);
    plx_mat_free(&T); plx_mat_free(&P); plx_mat_free(&S); plx_mat_free(&Sg); plx_mat_free(&Bs);
    plx_mat_free(&Bs0); plx_mat_free(&G1); plx_mat_free(&G2); plx_mat_free(&R49); plx_mat_free(&Ri); plx_mat_free(&Z);
}

/* ---- QR / least squares ------------------------------------------------------------------- */

TEST(qr) {
    plx_rng r;
    plx_rng_seed(&r, 6);
    static const int shapes[][2] = {{1, 1}, {5, 1}, {1, 5}, {7, 7}, {40, 33}, {33, 40}, {300, 70}, {65, 130}};
    for (size_t t = 0; t < 8; t++) {
        int m = shapes[t][0], n = shapes[t][1], k = PLX_MIN(m, n);
        plx_mat A = randn(&r, m, n);
        if (n > 3) /* zero column and a duplicated one: rank deficiency must not break QR */
            for (int i = 0; i < m; i++) {
                PLX_M(&A, i, 1) = 0;
                PLX_M(&A, i, 3) = PLX_M(&A, i, 2);
            }
        plx_mat Q = plx_mat_new(m, k), R = plx_mat_new(k, n), R2 = plx_mat_new(k, n);
        plx_qr(&A, &Q, &R);
        plx_qr(&A, NULL, &R2);
        CHECK(max_diff(&R, &R2) == 0);
        int upper = 1;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < PLX_MIN(i, n); j++) upper &= PLX_M(&R, i, j) == 0;
        CHECK(upper);
        plx_mat QR = plx_matmul(&Q, &R);
        CHECK(rel_err(&QR, &A) < 10 * m * EPS);
        CHECK(orth_err(&Q) < 10 * m * EPS);
        plx_mat_free(&A); plx_mat_free(&Q); plx_mat_free(&R); plx_mat_free(&R2); plx_mat_free(&QR);
    }
}

TEST(lstsq) {
    plx_rng r;
    plx_rng_seed(&r, 7);
    /* tall full rank: agrees with the normal equations, residual orthogonal to range(A) */
    plx_mat A = randn(&r, 200, 30), B = randn(&r, 200, 3), X = plx_mat_new(30, 3);
    CHECK(plx_lstsq(&A, &B, 0.0, 0.0, &X) == 30);
    plx_mat N = plx_mat_new(30, 30), AtB = plx_mat_new(30, 3);
    plx_syrk(PLX_TRANS, 1.0, &A, 0.0, &N);
    plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &A, &B, 0.0, &AtB);
    plx_mat L = plx_mat_clone(&N), Xn = plx_mat_clone(&AtB);
    CHECK(plx_chol(&L) == PLX_OK);
    plx_chol_solve(&L, &Xn);
    CHECK(rel_err(&X, &Xn) < 1e-12);
    plx_mat Res = plx_matmul(&A, &X), AtR = plx_mat_new(30, 3);
    plx_mat_add(&Res, -1.0, &B);
    plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &A, &Res, 0.0, &AtR);
    CHECK(max_abs(&AtR) < 1e-11);

    /* ridge: equals (A^T A + lambda I)^-1 A^T B */
    plx_mat Xr = plx_mat_new(30, 3), Nr = plx_mat_clone(&N), Xrn = plx_mat_clone(&AtB);
    CHECK(plx_lstsq(&A, &B, 2.5, 0.0, &Xr) == 30);
    for (int i = 0; i < 30; i++) PLX_M(&Nr, i, i) += 2.5;
    CHECK(plx_chol(&Nr) == PLX_OK);
    plx_chol_solve(&Nr, &Xrn);
    CHECK(rel_err(&Xr, &Xrn) < 1e-12);

    /* rank deficient (column 5 = column 2 + column 7, column 9 = 0): rank 28 and the minimum-norm
     * solution, i.e. the pseudo-inverse one, built here from the SVD */
    plx_mat Ad = plx_mat_clone(&A);
    for (int i = 0; i < 200; i++) {
        PLX_M(&Ad, i, 5) = PLX_M(&Ad, i, 2) + PLX_M(&Ad, i, 7);
        PLX_M(&Ad, i, 9) = 0;
    }
    plx_mat Xd = plx_mat_new(30, 3);
    CHECK(plx_lstsq(&Ad, &B, 0.0, 0.0, &Xd) == 28);
    plx_mat U = plx_mat_new(200, 30), V = plx_mat_new(30, 30), UtB = plx_mat_new(30, 3);
    double s[30];
    CHECK(plx_svd(&Ad, &U, s, &V) == PLX_OK);
    CHECK(s[27] > 1e-3 * s[0] && s[28] < 1e-13 * s[0]);
    plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &U, &B, 0.0, &UtB);
    for (int i = 0; i < 30; i++)
        for (int j = 0; j < 3; j++) PLX_M(&UtB, i, j) = i < 28 ? PLX_M(&UtB, i, j) / s[i] : 0.0;
    plx_mat Xp = plx_matmul(&V, &UtB);
    CHECK(rel_err(&Xd, &Xp) < 1e-10);

    /* wide (underdetermined) full row rank: exact fit, and X lies in the row space of A */
    plx_mat W = randn(&r, 8, 20), Bw = randn(&r, 8, 2), Xw = plx_mat_new(20, 2);
    CHECK(plx_lstsq(&W, &Bw, 0.0, 0.0, &Xw) == 8);
    plx_mat WX = plx_matmul(&W, &Xw);
    CHECK(rel_err(&WX, &Bw) < 1e-13);
    plx_mat WWt = plx_mat_new(8, 8), Yw = plx_mat_clone(&Bw); /* min-norm: X = W^T (W W^T)^-1 B */
    plx_syrk(PLX_NO_TRANS, 1.0, &W, 0.0, &WWt);
    CHECK(plx_solve(&WWt, &Yw) == PLX_OK);
    plx_mat Xmn = plx_mat_new(20, 2);
    plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &W, &Yw, 0.0, &Xmn);
    CHECK(rel_err(&Xw, &Xmn) < 1e-12);

    /* zero matrix: rank 0, X = 0; 1x1 */
    plx_mat Z = plx_mat_new(5, 3), Bz = randn(&r, 5, 1), Xz = randn(&r, 3, 1);
    CHECK(plx_lstsq(&Z, &Bz, 0.0, 0.0, &Xz) == 0 && max_abs(&Xz) == 0);
    double a1 = 4, b1 = 2;
    plx_mat A1 = plx_mat_from(1, 1, &a1), B1 = plx_mat_from(1, 1, &b1), X1 = plx_mat_new(1, 1);
    CHECK(plx_lstsq(&A1, &B1, 0.0, 0.0, &X1) == 1);
    CHECK_NEAR(X1.a[0], 0.5, 1e-16);

    /* Hilbert(12), cond ~1.7e16: numerically rank deficient. lstsq still returns a small-norm
     * solution with a tiny residual instead of the garbage a plain solve gives. */
    plx_mat H = hilbert(12), xh = plx_mat_new(12, 1), bh = plx_mat_new(12, 1);
    for (int i = 0; i < 12; i++) PLX_M(&xh, i, 0) = 1;
    plx_gemv(PLX_NO_TRANS, 1.0, &H, xh.a, 0.0, bh.a);
    plx_mat Xh = plx_mat_new(12, 1);
    int rh = plx_lstsq(&H, &bh, 0.0, 0.0, &Xh);
    plx_mat Hx = plx_matmul(&H, &Xh);
    CHECK(rh >= 10 && rh <= 12);
    CHECK(rel_err(&Hx, &bh) < 1e-13 && plx_mat_norm_fro(&Xh) < 2 * plx_mat_norm_fro(&xh));

    plx_mat_free(&A); plx_mat_free(&B); plx_mat_free(&X); plx_mat_free(&N); plx_mat_free(&AtB);
    plx_mat_free(&L); plx_mat_free(&Xn); plx_mat_free(&Res); plx_mat_free(&AtR); plx_mat_free(&Xr);
    plx_mat_free(&Nr); plx_mat_free(&Xrn); plx_mat_free(&Ad); plx_mat_free(&Xd); plx_mat_free(&U);
    plx_mat_free(&V); plx_mat_free(&UtB); plx_mat_free(&Xp); plx_mat_free(&W); plx_mat_free(&Bw);
    plx_mat_free(&Xw); plx_mat_free(&WX); plx_mat_free(&WWt); plx_mat_free(&Yw); plx_mat_free(&Xmn);
    plx_mat_free(&Z); plx_mat_free(&Bz); plx_mat_free(&Xz); plx_mat_free(&A1); plx_mat_free(&B1);
    plx_mat_free(&X1); plx_mat_free(&H); plx_mat_free(&xh); plx_mat_free(&bh); plx_mat_free(&Xh); plx_mat_free(&Hx);
}

/* ---- eigen -------------------------------------------------------------------------------- */

TEST(eigh_random_and_structured) {
    plx_rng r;
    plx_rng_seed(&r, 8);
    static const int sizes[] = {1, 2, 3, 10, 64, 257};
    double worst_res = 0, worst_orth = 0;
    for (size_t t = 0; t < 6; t++) {
        int n = sizes[t];
        plx_mat A = rand_sym(&r, n), V = plx_mat_new(n, n);
        double *w = PLX_NEW(double, n), *w2 = PLX_NEW(double, n), tr = 0, ws = 0;
        CHECK(plx_eigh(&A, w, &V) == PLX_OK);
        double res = eig_resid(&A, w, &V), orth = orth_err(&V);
        worst_res = PLX_MAX(worst_res, res / (n * EPS));
        worst_orth = PLX_MAX(worst_orth, orth / (n * EPS));
        CHECK(res < 10 * n * EPS && orth < 10 * n * EPS);
        CHECK(descending(w, n) && signs_canonical(&V));
        for (int i = 0; i < n; i++) {
            tr += PLX_M(&A, i, i);
            ws += w[i];
        }
        CHECK_NEAR(ws, tr, 1e-12 * n);
        CHECK(plx_eigh(&A, w2, NULL) == PLX_OK); /* values-only path agrees */
        double dv = 0;
        for (int i = 0; i < n; i++) dv = PLX_MAX(dv, fabs(w[i] - w2[i]));
        CHECK(dv < 10 * n * EPS * fabs(w[0]) + 1e-300);
        free(w); free(w2);
        plx_mat_free(&A); plx_mat_free(&V);
    }
    printf("  eigh: residual <= %.2f n eps ||A||, orthogonality <= %.2f n eps\n", worst_res, worst_orth);

    /* known spectrum with repeated eigenvalues and a 1e-12 cluster: Q diag(l) Q^T */
    const int n = 60;
    plx_mat Q = rand_orth(&r, n, n);
    double l[60];
    for (int i = 0; i < n; i++) l[i] = i < 10 ? 5.0 : i < 20 ? 2.0 : i < 25 ? -1.0 + 1e-12 * i : pow(10.0, -(i - 25) / 3.0);
    plx_mat A = from_svd(&Q, l, &Q), V = plx_mat_new(n, n);
    double w[60], ls[60];
    memcpy(ls, l, sizeof l);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (ls[j] > ls[i]) { double t = ls[i]; ls[i] = ls[j]; ls[j] = t; }
    CHECK(plx_eigh(&A, w, &V) == PLX_OK);
    double de = 0;
    for (int i = 0; i < n; i++) de = PLX_MAX(de, fabs(w[i] - ls[i]));
    CHECK(de < 20 * n * EPS * 5.0); /* absolute accuracy eps ||A|| for every eigenvalue */
    CHECK(orth_err(&V) < 10 * n * EPS && eig_resid(&A, w, &V) < 10 * n * EPS);
    /* the invariant subspace of the 10-fold eigenvalue 5 is recovered: ||Q1^T V1|| = I-ish */
    plx_mat Q1 = plx_mat_new(n, 10), V1 = plx_mat_new(n, 10), C = plx_mat_new(10, 10);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 10; j++) {
            PLX_M(&Q1, i, j) = PLX_M(&Q, i, j);
            PLX_M(&V1, i, j) = PLX_M(&V, i, j);
        }
    plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &Q1, &V1, 0.0, &C);
    CHECK(orth_err(&C) < 1e-12); /* C orthogonal <=> span(V1) = span(Q1) */

    /* diagonal, zero, upper-triangle garbage, aliasing V = A */
    double diag[9] = {1, 0, 0, 0, 3, 0, 0, 0, 2};
    plx_mat D = plx_mat_from(3, 3, diag), DV = plx_mat_new(3, 3);
    double dw[3];
    CHECK(plx_eigh(&D, dw, &DV) == PLX_OK && dw[0] == 3 && dw[1] == 2 && dw[2] == 1);
    CHECK(PLX_M(&DV, 1, 0) == 1 && PLX_M(&DV, 2, 1) == 1 && PLX_M(&DV, 0, 2) == 1);
    plx_mat Z = plx_mat_new(5, 5), ZV = plx_mat_new(5, 5);
    double zw[5];
    CHECK(plx_eigh(&Z, zw, &ZV) == PLX_OK && zw[0] == 0 && zw[4] == 0 && orth_err(&ZV) == 0);
    plx_mat G = rand_sym(&r, 30), Gu = plx_mat_clone(&G), GV = plx_mat_new(30, 30);
    for (int i = 0; i < 30; i++)
        for (int j = i + 1; j < 30; j++) PLX_M(&Gu, i, j) = 1e30;
    double gw[30], guw[30];
    CHECK(plx_eigh(&G, gw, &GV) == PLX_OK && plx_eigh(&Gu, guw, &Gu) == PLX_OK);
    CHECK(memcmp(gw, guw, sizeof gw) == 0 && max_diff(&GV, &Gu) == 0);

    /* Wilkinson W21+: pairs of eigenvalues agreeing to ~1e-14 relative; vectors still orthogonal */
    plx_mat Wk = plx_mat_new(21, 21), WV = plx_mat_new(21, 21);
    for (int i = 0; i < 21; i++) {
        PLX_M(&Wk, i, i) = fabs(10.0 - i);
        if (i) PLX_M(&Wk, i, i - 1) = PLX_M(&Wk, i - 1, i) = 1;
    }
    double ww[21];
    CHECK(plx_eigh(&Wk, ww, &WV) == PLX_OK);
    CHECK(orth_err(&WV) < 100 * EPS && eig_resid(&Wk, ww, &WV) < 100 * EPS);
    CHECK_NEAR(ww[0], 10.746194182903393, 1e-13); /* known largest eigenvalue of W21+ */
    CHECK(fabs(ww[0] - ww[1]) < 1e-12);

    /* low rank PSD (a scatter matrix of 12 samples): 138 eigenvalues at rounding level next to 12
     * large ones. Regression: a purely local QL deflation test stalls on this cluster. */
    plx_mat Bl = randn(&r, 150, 12), Pl = plx_mat_new(150, 150), PV = plx_mat_new(150, 150);
    plx_syrk(PLX_NO_TRANS, 1.0, &Bl, 0.0, &Pl);
    double pw[150];
    CHECK(plx_eigh(&Pl, pw, &PV) == PLX_OK);
    CHECK(eig_resid(&Pl, pw, &PV) < 10 * 150 * EPS && orth_err(&PV) < 10 * 150 * EPS);
    CHECK(pw[11] > 1 && fabs(pw[12]) < 150 * EPS * pw[0] && fabs(pw[149]) < 150 * EPS * pw[0]);
    plx_mat_free(&Bl); plx_mat_free(&Pl); plx_mat_free(&PV);

    /* bitwise independent of the thread count */
    plx_mat T = rand_sym(&r, 200), V4 = plx_mat_new(200, 200), V1t = plx_mat_new(200, 200);
    double w4[200], w1[200];
    plx_eigh(&T, w4, &V4);
    plx_set_num_threads(1);
    plx_eigh(&T, w1, &V1t);
    plx_set_num_threads(0);
    CHECK(memcmp(w4, w1, sizeof w4) == 0 && memcmp(V4.a, V1t.a, sizeof(double) * 200 * 200) == 0);

    plx_mat_free(&Q); plx_mat_free(&A); plx_mat_free(&V); plx_mat_free(&Q1); plx_mat_free(&V1);
    plx_mat_free(&C); plx_mat_free(&D); plx_mat_free(&DV); plx_mat_free(&Z); plx_mat_free(&ZV);
    plx_mat_free(&G); plx_mat_free(&Gu); plx_mat_free(&GV); plx_mat_free(&Wk); plx_mat_free(&WV);
    plx_mat_free(&T); plx_mat_free(&V4); plx_mat_free(&V1t);
}

TEST(eigh_generalised) {
    plx_rng r;
    plx_rng_seed(&r, 9);
    const int n = 80;
    plx_mat A = rand_sym(&r, n), B = rand_spd(&r, n, 1e3), V = plx_mat_new(n, n);
    double w[80];
    CHECK(plx_eigh_gen(&A, &B, w, &V) == PLX_OK);
    CHECK(descending(w, n));
    plx_mat AV = plx_matmul(&A, &V), BV = plx_matmul(&B, &V), VtBV = plx_mat_new(n, n), VtAV = plx_mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) PLX_M(&AV, i, j) -= w[j] * PLX_M(&BV, i, j);
    CHECK(plx_mat_norm_fro(&AV) / plx_mat_norm_fro(&A) < 1e3 * n * EPS);
    plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &V, &BV, 0.0, &VtBV);
    plx_gemm(PLX_TRANS, PLX_NO_TRANS, 1.0, &V, &AV, 0.0, &VtAV);
    for (int i = 0; i < n; i++) PLX_M(&VtBV, i, i) -= 1;
    CHECK(max_abs(&VtBV) < 1e3 * n * EPS); /* B-orthonormal */
    double wv[80];
    CHECK(plx_eigh_gen(&A, &B, wv, NULL) == PLX_OK);
    CHECK(fabs(wv[0] - w[0]) < 1e-10 && fabs(wv[n - 1] - w[n - 1]) < 1e-10);

    /* B = I reduces to the standard problem; B = 4 I scales eigenvalues by 1/4 */
    plx_mat I4 = plx_mat_identity(n), Vs = plx_mat_new(n, n);
    plx_mat_scale(&I4, 4.0);
    double ws[80], wg[80];
    CHECK(plx_eigh(&A, ws, &Vs) == PLX_OK && plx_eigh_gen(&A, &I4, wg, NULL) == PLX_OK);
    double dw = 0;
    for (int i = 0; i < n; i++) dw = PLX_MAX(dw, fabs(ws[i] / 4 - wg[i]));
    CHECK(dw < 1e-13);

    plx_mat NB = rand_sym(&r, n); /* indefinite B */
    CHECK(plx_eigh_gen(&A, &NB, w, &V) == PLX_ENOTSPD);
    plx_mat_free(&A); plx_mat_free(&B); plx_mat_free(&V); plx_mat_free(&AV); plx_mat_free(&BV);
    plx_mat_free(&VtBV); plx_mat_free(&VtAV); plx_mat_free(&I4); plx_mat_free(&Vs); plx_mat_free(&NB);
}

/* ---- SVD ---------------------------------------------------------------------------------- */

static void check_svd(const plx_mat *A, double tol_scale, double *worst) {
    const int m = A->rows, n = A->cols, k = PLX_MIN(m, n);
    plx_mat U = plx_mat_new(m, k), V = plx_mat_new(n, k);
    double *s = PLX_NEW(double, k), *s2 = PLX_NEW(double, k);
    CHECK(plx_svd(A, &U, s, &V) == PLX_OK);
    plx_mat R = from_svd(&U, s, &V);
    double re = rel_err(&R, A), ou = orth_err(&U), ov = orth_err(&V), sz = PLX_MAX(m, n) * EPS;
    worst[0] = PLX_MAX(worst[0], re / sz);
    worst[1] = PLX_MAX(worst[1], PLX_MAX(ou, ov) / sz);
    CHECK(re < tol_scale * sz && ou < tol_scale * sz && ov < tol_scale * sz);
    CHECK(descending(s, k) && s[k - 1] >= 0);
    CHECK(signs_canonical(m >= n ? &V : &U));
    CHECK(plx_svd(A, NULL, s2, NULL) == PLX_OK); /* values only: identical */
    CHECK(memcmp(s, s2, sizeof(double) * k) == 0);
    plx_mat V2 = plx_mat_new(n, k);
    CHECK(plx_svd(A, NULL, s2, &V2) == PLX_OK && max_diff(&V, &V2) == 0);
    free(s); free(s2);
    plx_mat_free(&U); plx_mat_free(&V); plx_mat_free(&R); plx_mat_free(&V2);
}

TEST(svd_shapes_and_properties) {
    plx_rng r;
    plx_rng_seed(&r, 10);
    static const int shapes[][2] = {{1, 1}, {1, 6}, {6, 1}, {2, 2}, {9, 4}, {4, 9}, {50, 50}, {200, 31}, {31, 200}, {129, 128}};
    double worst[2] = {0, 0};
    for (size_t t = 0; t < 10; t++) {
        plx_mat A = randn(&r, shapes[t][0], shapes[t][1]);
        check_svd(&A, 20, worst);
        plx_mat_free(&A);
    }
    printf("  svd: reconstruction <= %.2f max(m,n) eps, orthogonality <= %.2f max(m,n) eps\n", worst[0], worst[1]);

    /* agreement with eig of A^T A for the leading part of the spectrum */
    plx_mat A = randn(&r, 120, 40), AtA = plx_mat_new(40, 40), V = plx_mat_new(40, 40), Vs = plx_mat_new(40, 40);
    double s[40], w[40];
    plx_syrk(PLX_TRANS, 1.0, &A, 0.0, &AtA);
    CHECK(plx_svd(&A, NULL, s, &Vs) == PLX_OK && plx_eigh(&AtA, w, &V) == PLX_OK);
    double ds = 0;
    for (int i = 0; i < 40; i++) ds = PLX_MAX(ds, fabs(s[i] * s[i] - w[i]) / w[0]);
    CHECK(ds < 100 * 40 * EPS);
    double align = 1; /* |<v_i(svd), v_i(eig)>| = 1 for the well-separated leading vectors */
    for (int j = 0; j < 5; j++) {
        double d = 0;
        for (int i = 0; i < 40; i++) d += PLX_M(&V, i, j) * PLX_M(&Vs, i, j);
        align = PLX_MIN(align, fabs(d));
    }
    CHECK(align > 1 - 1e-10);

    /* rank 3 (the structure-from-motion shape: 2F x P = (2F x 3)(3 x P)) */
    plx_mat M = randn(&r, 60, 3), S = randn(&r, 3, 45), W = plx_matmul(&M, &S);
    double sw[45];
    plx_mat Uw = plx_mat_new(60, 45), Vw = plx_mat_new(45, 45);
    CHECK(plx_svd(&W, &Uw, sw, &Vw) == PLX_OK);
    CHECK(sw[2] > 1e-2 * sw[0] && sw[3] < 1e-14 * sw[0]);
    CHECK(orth_err(&Uw) < 1e-13 && orth_err(&Vw) < 1e-13);
    plx_mat Rw = from_svd(&Uw, sw, &Vw);
    CHECK(rel_err(&Rw, &W) < 1e-14);

    /* exact zero rows/columns and the zero matrix: completed orthonormal bases */
    plx_mat Zc = randn(&r, 10, 6);
    for (int i = 0; i < 10; i++) PLX_M(&Zc, i, 2) = PLX_M(&Zc, i, 4) = 0;
    for (int j = 0; j < 6; j++) PLX_M(&Zc, 7, j) = 0;
    check_svd(&Zc, 20, worst);
    plx_mat Z = plx_mat_new(7, 4);
    check_svd(&Z, 20, worst);
    plx_mat Zt = plx_mat_new(4, 7);
    check_svd(&Zt, 20, worst);

    /* NaN input is reported, not propagated silently */
    plx_mat Nn = randn(&r, 4, 3);
    PLX_M(&Nn, 2, 1) = NAN;
    CHECK(plx_svd(&Nn, NULL, s, NULL) == PLX_EINVAL);

    /* bitwise independent of the thread count */
    plx_mat T = randn(&r, 300, 150), U4 = plx_mat_new(300, 150), U1 = plx_mat_new(300, 150);
    double s4[150], s1[150];
    plx_svd(&T, &U4, s4, NULL);
    plx_set_num_threads(1);
    plx_svd(&T, &U1, s1, NULL);
    plx_set_num_threads(0);
    CHECK(memcmp(s4, s1, sizeof s4) == 0 && memcmp(U4.a, U1.a, sizeof(double) * 300 * 150) == 0);

    plx_mat_free(&A); plx_mat_free(&AtA); plx_mat_free(&V); plx_mat_free(&Vs); plx_mat_free(&M); plx_mat_free(&S);
    plx_mat_free(&W); plx_mat_free(&Uw); plx_mat_free(&Vw); plx_mat_free(&Rw); plx_mat_free(&Zc); plx_mat_free(&Z);
    plx_mat_free(&Zt); plx_mat_free(&Nn); plx_mat_free(&T); plx_mat_free(&U4); plx_mat_free(&U1);
}

TEST(svd_relative_accuracy) {
    plx_rng r;
    plx_rng_seed(&r, 11);
    /* A = B D: B well conditioned, column scales 1 .. 1e-12. The singular values span 12 decades;
     * an algorithm that is only normwise stable gets the small ones with error eps * s_max, i.e.
     * 100% wrong. Compared against a long-double Jacobi oracle. */
    const int m = 40, n = 12;
    plx_mat A = randn(&r, m, n);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) PLX_M(&A, i, j) *= pow(10.0, -(double)j);
    double s[12], ref[12], worst = 0;
    CHECK(plx_svd(&A, NULL, s, NULL) == PLX_OK);
    ref_singular_values(&A, ref);
    for (int i = 0; i < n; i++) worst = PLX_MAX(worst, fabs(s[i] - ref[i]) / ref[i]);
    printf("  svd: graded 40x12 (s from %.1e to %.1e): max relative error %.2e\n", ref[0], ref[n - 1], worst);
    CHECK(worst < 1e-13);

    /* and the same on the transpose (wide path) and with rows graded instead */
    plx_mat At = plx_mat_transpose(&A);
    CHECK(plx_svd(&At, NULL, s, NULL) == PLX_OK);
    double worst_t = 0;
    for (int i = 0; i < n; i++) worst_t = PLX_MAX(worst_t, fabs(s[i] - ref[i]) / ref[i]);
    CHECK(worst_t < 1e-13);
    plx_mat_free(&A); plx_mat_free(&At);
}

/* ---- small fixed sizes -------------------------------------------------------------------- */

static double frob3(const double *a, const double *b) {
    double s = 0;
    for (int i = 0; i < 9; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(s);
}
static void mul3(const double *a, const double *b, double *c, int tb) { /* c = a * op(b) */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += a[3 * i + k] * (tb ? b[3 * j + k] : b[3 * k + j]);
            c[3 * i + j] = s;
        }
}
static double orth3(const double *U) {
    double G[9], I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double Ut[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Ut[3 * i + j] = U[3 * j + i];
    mul3(Ut, U, G, 0);
    return frob3(G, I);
}
static double det3t(const double *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

TEST(small_fixed_sizes) {
    plx_rng r;
    plx_rng_seed(&r, 12);
    /* 2x2 symmetric: random, diagonal, equal diagonal, and a nearly cancelling small eigenvalue */
    double cases[][3] = {{1, 0, 3}, {2, 0, 2}, {1, 1, 1}, {1e8, 1, 1e-8}, {1, 1e-20, 1}, {-3, 2, 5}};
    for (int t = 0; t < 26; t++) {
        double a, b, c;
        if (t < 6) { a = cases[t][0]; b = cases[t][1]; c = cases[t][2]; }
        else { a = plx_rng_normal(&r); b = plx_rng_normal(&r); c = plx_rng_normal(&r); }
        double w[2], V[4];
        plx_eig2_sym(a, b, c, w, V);
        CHECK(w[0] >= w[1]);
        CHECK_NEAR(V[0] * V[3] - V[1] * V[2], 1.0, 1e-15);
        double nrm = fabs(a) + fabs(b) + fabs(c);
        for (int j = 0; j < 2; j++) {
            double r0 = a * V[j] + b * V[2 + j] - w[j] * V[j], r1 = b * V[j] + c * V[2 + j] - w[j] * V[2 + j];
            CHECK(hypot(r0, r1) <= 4 * EPS * nrm);
        }
    }
    /* [[1e8, 1], [1, 1]]: the small eigenvalue (1e8 - 1) / (1e8 - 1e-8 ...) ~ 1 - 1e-8 must keep
     * full relative accuracy; (a+c)/2 - hypot((a-c)/2, b) would lose 8 digits */
    double w[2], V[4];
    plx_eig2_sym(1e8, 1, 1, w, V);
    CHECK_NEAR(w[0] * w[1], 1e8 - 1, 1e-6); /* product = det */
    CHECK_NEAR(w[1], (1e8 - 1) / w[0], 4 * EPS);

    /* 3x3 SVD and nearest rotation */
    for (int t = 0; t < 200; t++) {
        double A[9], U[9], s[3], Vv[9], R[9];
        for (int i = 0; i < 9; i++) A[i] = plx_rng_normal(&r);
        if (t % 4 == 1) /* rank 2 */
            for (int i = 0; i < 3; i++) A[3 * i + 2] = A[3 * i] - 2 * A[3 * i + 1];
        if (t % 4 == 2) /* rank 1 */
            for (int i = 0; i < 3; i++) A[3 * i + 1] = 3 * A[3 * i], A[3 * i + 2] = -A[3 * i];
        if (t == 3) memset(A, 0, sizeof A);
        plx_svd3(A, U, s, Vv);
        double US[9], Rc[9];
        for (int i = 0; i < 9; i++) US[i] = U[i] * s[i % 3];
        mul3(US, Vv, Rc, 1);
        double na = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        CHECK(frob3(Rc, A) <= 8 * EPS * na);
        CHECK(orth3(U) < 8 * EPS && orth3(Vv) < 8 * EPS);
        CHECK(s[0] >= s[1] && s[1] >= s[2] && s[2] >= 0);
        if (t % 4 == 1) CHECK(s[2] < 1e-14 * s[0]);
        if (t % 4 == 2) CHECK(s[1] < 1e-14 * s[0]);

        plx_nearest_rotation3(A, R);
        CHECK(orth3(R) < 8 * EPS && fabs(det3t(R) - 1) < 8 * EPS);
        if (t % 4 == 0) { /* optimality: R times a small rotation about any axis is farther from A */
            const double best = frob3(R, A);
            for (int k = 0; k < 6; k++) {
                const int ax = k % 3, a = (ax + 1) % 3, b = (ax + 2) % 3;
                const double th = k < 3 ? 1e-3 : -1e-3;
                double Gk[9] = {0}, R2[9];
                Gk[4 * ax] = 1;
                Gk[3 * a + a] = Gk[3 * b + b] = cos(th);
                Gk[3 * a + b] = -sin(th);
                Gk[3 * b + a] = sin(th);
                mul3(R, Gk, R2, 0);
                CHECK(frob3(R2, A) >= best);
            }
        }
        /* agreement with the general SVD */
        plx_mat Am = plx_mat_from(3, 3, A);
        double sg[3];
        CHECK(plx_svd(&Am, NULL, sg, NULL) == PLX_OK);
        CHECK(fabs(sg[0] - s[0]) <= 8 * EPS * na && fabs(sg[2] - s[2]) <= 8 * EPS * na);
        plx_mat_free(&Am);
    }
    /* a noisy rotation snaps back; a reflection becomes the closest proper rotation */
    double Rt[9], Rn[9], Rr[9], ang = 0.7;
    double R0[9] = {cos(ang), -sin(ang), 0, sin(ang), cos(ang), 0, 0, 0, 1};
    for (int i = 0; i < 9; i++) Rt[i] = R0[i] + 1e-9 * plx_rng_normal(&r);
    plx_nearest_rotation3(Rt, Rn);
    CHECK(frob3(Rn, R0) < 3e-9);
    double refl[9] = {1, 0, 0, 0, 1, 0, 0, 0, -1}, want[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double sc[9];
    for (int i = 0; i < 9; i++) sc[i] = refl[i] * (i == 0 ? 3 : i == 4 ? 2 : 1); /* diag(3, 2, -1) */
    plx_nearest_rotation3(sc, Rr);
    CHECK(frob3(Rr, want) < 8 * EPS);
    plx_nearest_rotation3(sc, sc); /* in place */
    CHECK(frob3(sc, want) < 8 * EPS);
}

/* ---- conjugate gradient ------------------------------------------------------------------- */

typedef struct { int g; const double *scale; } lap_ctx;

/* scale_i * (2D 5-point Laplacian + 0.01 I) * scale_i: SPD, badly scaled when scale varies */
static void lap_mv(const double *x, double *y, void *p) {
    const lap_ctx *c = p;
    const int g = c->g;
    for (int i = 0; i < g; i++)
        for (int j = 0; j < g; j++) {
            int k = i * g + j;
            double xs = x[k] * c->scale[k], v = 4.01 * xs;
            if (i > 0) v -= x[k - g] * c->scale[k - g];
            if (i < g - 1) v -= x[k + g] * c->scale[k + g];
            if (j > 0) v -= x[k - 1] * c->scale[k - 1];
            if (j < g - 1) v -= x[k + 1] * c->scale[k + 1];
            y[k] = c->scale[k] * v;
        }
}

static void indef_mv(const double *x, double *y, void *p) {
    int n = *(const int *)p;
    for (int i = 0; i < n; i++) y[i] = (i % 2 ? -1.0 : 1.0) * (i + 1) * x[i];
}

TEST(conjugate_gradient) {
    plx_rng r;
    plx_rng_seed(&r, 13);
    const int g = 20, n = g * g;
    double *scale = PLX_NEW(double, n), *b = PLX_NEW(double, n), *x = PLX_NEW(double, n), *diag = PLX_NEW(double, n);
    for (int i = 0; i < n; i++) scale[i] = 1.0;
    lap_ctx c = {g, scale};
    for (int i = 0; i < n; i++) b[i] = plx_rng_normal(&r);
    plx_cg_info info;
    CHECK(plx_cg(n, lap_mv, &c, b, x, NULL, 1e-10, 0, &info) == PLX_OK);
    CHECK(info.rel_residual < 2e-10 && info.iters > 0);

    /* agrees with a dense Cholesky solve of the same operator */
    plx_mat A = plx_mat_new(n, n), e = plx_mat_new(n, 1), xc = plx_mat_from(n, 1, b);
    for (int j = 0; j < n; j++) {
        double *col = PLX_NEW(double, n), *unit = PLX_NEW(double, n);
        unit[j] = 1;
        lap_mv(unit, col, &c);
        for (int i = 0; i < n; i++) PLX_M(&A, i, j) = col[i];
        free(col); free(unit);
    }
    CHECK(plx_chol(&A) == PLX_OK);
    plx_chol_solve(&A, &xc);
    double dx = 0, nx = 0;
    for (int i = 0; i < n; i++) {
        dx = PLX_MAX(dx, fabs(x[i] - xc.a[i]));
        nx = PLX_MAX(nx, fabs(xc.a[i]));
    }
    CHECK(dx < 1e-7 * nx);

    /* badly scaled operator: Jacobi preconditioning cuts the iteration count substantially */
    for (int i = 0; i < n; i++) scale[i] = pow(10.0, 3.0 * plx_rng_uniform(&r));
    for (int i = 0; i < n; i++) diag[i] = 4.01 * scale[i] * scale[i];
    plx_cg_info plain, pre;
    memset(x, 0, sizeof(double) * n);
    int rc_plain = plx_cg(n, lap_mv, &c, b, x, NULL, 1e-8, 20 * n, &plain);
    memset(x, 0, sizeof(double) * n);
    CHECK(plx_cg(n, lap_mv, &c, b, x, diag, 1e-8, 0, &pre) == PLX_OK);
    CHECK(pre.rel_residual < 1e-7);
    CHECK(rc_plain != PLX_OK || pre.iters * 3 < plain.iters);
    printf("  cg: badly scaled 400-dof Laplacian: %d iterations plain (%s), %d with Jacobi\n", plain.iters,
           plx_la_status_str(rc_plain), pre.iters);

    /* failure statuses: iteration cap, indefinite operator; warm start at the solution */
    memset(x, 0, sizeof(double) * n);
    CHECK(plx_cg(n, lap_mv, &c, b, x, NULL, 1e-12, 3, &info) == PLX_ENOCONV && info.iters == 3);
    int ni = 10;
    double bi[10], xi[10] = {0};
    for (int i = 0; i < ni; i++) bi[i] = 1;
    CHECK(plx_cg(ni, indef_mv, &ni, bi, xi, NULL, 1e-12, 0, NULL) == PLX_ENOTSPD);
    memset(x, 0, sizeof(double) * n);
    plx_cg(n, lap_mv, &c, b, x, diag, 1e-12, 0, NULL);
    CHECK(plx_cg(n, lap_mv, &c, b, x, diag, 1e-6, 0, &info) == PLX_OK && info.iters == 0);
    free(scale); free(b); free(x); free(diag);
    plx_mat_free(&A); plx_mat_free(&e); plx_mat_free(&xc);
}

/* ---- float data matrices / PCA ------------------------------------------------------------- */

/* n x d float rows with a known spectrum: X = mean + U diag(s) V^T (U n x r, V d x r orthonormal,
 * U columns centred so the mean is exact). Returns V for comparison. */
static float *make_data(plx_rng *r, int n, int d, int rank, const double *s, double offset, plx_mat *Vout) {
    plx_mat U = rand_orth(r, n, rank + 1), V = rand_orth(r, d, rank);
    /* make U's columns orthogonal to the all-ones vector: project out and re-orthonormalise */
    plx_mat Uc = plx_mat_new(n, rank);
    for (int j = 0; j < rank; j++) {
        double m = 0;
        for (int i = 0; i < n; i++) m += PLX_M(&U, i, j);
        for (int i = 0; i < n; i++) PLX_M(&Uc, i, j) = PLX_M(&U, i, j) - m / n;
    }
    plx_mat Q = plx_mat_new(n, rank), R = plx_mat_new(rank, rank);
    plx_qr(&Uc, &Q, &R);
    plx_mat X = from_svd(&Q, s, &V);
    float *Xf = plx_malloc(sizeof(float) * (size_t)n * d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) Xf[(size_t)i * d + j] = (float)(PLX_M(&X, i, j) + offset + 0.1 * j);
    *Vout = V;
    plx_mat_free(&U); plx_mat_free(&Uc); plx_mat_free(&Q); plx_mat_free(&R); plx_mat_free(&X);
    return Xf;
}

TEST(data_statistics) {
    plx_rng r;
    plx_rng_seed(&r, 14);
    const int n = 300, d = 270;
    float *X = plx_malloc(sizeof(float) * n * d);
    for (int i = 0; i < n * d; i++) X[i] = (float)(1000.0 + plx_rng_normal(&r)); /* large offset */
    double *mean = PLX_NEW(double, d), *mean2 = PLX_NEW(double, d);
    plx_col_mean_f32(X, n, d, mean);
    plx_mat Xc = plx_mat_new(n, d); /* explicit double centring as the reference */
    for (int j = 0; j < d; j++) {
        long double s = 0;
        for (int i = 0; i < n; i++) s += X[i * d + j];
        CHECK_NEAR(mean[j], (double)(s / n), 1e-12);
        for (int i = 0; i < n; i++) PLX_M(&Xc, i, j) = X[i * d + j] - (double)(s / n);
    }
    plx_mat S = plx_mat_new(d, d), Sr = plx_mat_new(d, d), C = plx_mat_new(d, d);
    plx_scatter_f32(X, n, d, mean, &S);
    plx_syrk(PLX_TRANS, 1.0, &Xc, 0.0, &Sr);
    CHECK(rel_err(&S, &Sr) < 1e-12); /* no X^T X - n m m^T cancellation despite the 1000 offset */
    plx_cov_f32(X, n, d, mean2, &C);
    plx_mat_scale(&Sr, 1.0 / (n - 1));
    CHECK(rel_err(&C, &Sr) < 1e-12 && memcmp(mean, mean2, sizeof(double) * d) == 0);
    plx_mat G = plx_mat_new(n, n), Gr = plx_mat_new(n, n);
    plx_gram_f32(X, n, d, mean, &G);
    plx_syrk(PLX_NO_TRANS, 1.0, &Xc, 0.0, &Gr);
    CHECK(rel_err(&G, &Gr) < 1e-12);
    plx_mat Su = plx_mat_new(d, d); /* uncentred scatter = X^T X */
    plx_scatter_f32(X, n, d, NULL, &Su);
    CHECK_NEAR(PLX_M(&Su, 3, 3) / (n * 1e6), 1.0, 1e-2);

    /* bitwise thread independence of the float paths */
    plx_mat S1 = plx_mat_new(d, d);
    double *m1 = PLX_NEW(double, d);
    plx_set_num_threads(1);
    plx_col_mean_f32(X, n, d, m1);
    plx_scatter_f32(X, n, d, m1, &S1);
    plx_set_num_threads(0);
    CHECK(memcmp(m1, mean, sizeof(double) * d) == 0 && memcmp(S1.a, S.a, sizeof(double) * d * d) == 0);
    free(X); free(mean); free(mean2); free(m1);
    plx_mat_free(&Xc); plx_mat_free(&S); plx_mat_free(&Sr); plx_mat_free(&C); plx_mat_free(&G);
    plx_mat_free(&Gr); plx_mat_free(&Su); plx_mat_free(&S1);
}

TEST(pca_routes_agree) {
    plx_rng r;
    plx_rng_seed(&r, 15);
    const int n = 400, d = 300, rank = 60, k = 20;
    double s[60];
    for (int i = 0; i < rank; i++) s[i] = 100.0 * pow(0.85, i); /* known, decaying spectrum */
    plx_mat Vt;
    float *X = make_data(&r, n, d, rank, s, 50.0, &Vt);
    static const plx_pca_method methods[] = {PLX_PCA_COV, PLX_PCA_GRAM, PLX_PCA_RANDOMIZED};
    plx_pca p[3];
    for (int m = 0; m < 3; m++) {
        plx_pca_opts o = {.method = methods[m]};
        CHECK(plx_pca_f32(X, n, d, k, &o, &p[m]) == PLX_OK);
        CHECK(p[m].method == methods[m] && p[m].k == k && p[m].axes.rows == k && p[m].axes.cols == d);
        plx_mat At = plx_mat_transpose(&p[m].axes);
        CHECK(orth_err(&At) < 1e-12 && signs_canonical(&At) && descending(p[m].sv, k));
        double se = 0, ae = 0;
        for (int i = 0; i < k; i++) {
            se = PLX_MAX(se, fabs(p[m].sv[i] - s[i]) / s[i]);
            double dot = 0;
            for (int c = 0; c < d; c++) dot += PLX_M(&p[m].axes, i, c) * PLX_M(&Vt, c, i);
            ae = PLX_MAX(ae, 1 - fabs(dot));
        }
        /* data were rounded to float (relative 6e-8 of the ~60 entries), which bounds the agreement */
        CHECK(se < 1e-5 && ae < 1e-8);
        printf("  pca %-10s: max rel sv error %.1e, max 1-|cos| axis error %.1e\n",
               m == 0 ? "cov" : m == 1 ? "gram" : "randomized", se, ae);
        plx_mat_free(&At);
    }
    /* all three routes agree with each other tightly (same float data) */
    double cross = 0;
    for (int m = 1; m < 3; m++)
        for (int i = 0; i < k; i++) cross = PLX_MAX(cross, fabs(p[m].sv[i] - p[0].sv[i]) / p[0].sv[i]);
    CHECK(cross < 1e-9);

    /* AUTO: small d -> covariance, small n -> Gram */
    plx_pca q;
    CHECK(plx_pca_f32(X, n, 40, 5, NULL, &q) == PLX_OK && q.method == PLX_PCA_COV);
    plx_pca_free(&q);
    float *Xw = plx_malloc(sizeof(float) * 50 * 1200);
    for (int i = 0; i < 50 * 1200; i++) Xw[i] = (float)plx_rng_normal(&r);
    CHECK(plx_pca_f32(Xw, 50, 1200, 10, NULL, &q) == PLX_OK && q.method == PLX_PCA_GRAM);
    plx_pca qc; /* forced covariance route: a rank-49 1200 x 1200 scatter matrix */
    plx_pca_opts oc = {.method = PLX_PCA_COV};
    CHECK(plx_pca_f32(Xw, 50, 1200, 10, &oc, &qc) == PLX_OK);
    double dq = 0;
    for (int i = 0; i < 10; i++) dq = PLX_MAX(dq, fabs(q.sv[i] - qc.sv[i]) / qc.sv[i]);
    CHECK(dq < 1e-9);
    plx_pca_free(&q);
    plx_pca_free(&qc);
    free(Xw);

    /* projection: training data projects to uncorrelated components with variance sv^2 / (n-1);
     * whitened ones have unit variance */
    float *Y = plx_malloc(sizeof(float) * n * k), *Yw = plx_malloc(sizeof(float) * n * k);
    plx_pca_project_f32(&p[0], X, n, false, Y);
    plx_pca_project_f32(&p[0], X, n, true, Yw);
    double worst_var = 0, worst_wvar = 0, worst_corr = 0;
    for (int i = 0; i < k; i++)
        for (int j = 0; j <= i; j++) {
            double cij = 0, wij = 0;
            for (int t = 0; t < n; t++) {
                cij += (double)Y[t * k + i] * Y[t * k + j];
                wij += (double)Yw[t * k + i] * Yw[t * k + j];
            }
            cij /= n - 1;
            wij /= n - 1;
            if (i == j) {
                worst_var = PLX_MAX(worst_var, fabs(cij / (p[0].sv[i] * p[0].sv[i] / (n - 1)) - 1));
                worst_wvar = PLX_MAX(worst_wvar, fabs(wij - 1));
            } else {
                worst_corr = PLX_MAX(worst_corr, fabs(wij));
            }
        }
    CHECK(worst_var < 1e-5 && worst_wvar < 1e-5 && worst_corr < 1e-5);
    /* generic projection equals the explicit double computation */
    plx_mat B = randn(&r, 7, d);
    double mean[300], scale[7];
    for (int c = 0; c < d; c++) mean[c] = 0.5 * c;
    for (int i = 0; i < 7; i++) scale[i] = i + 1;
    float *P = plx_malloc(sizeof(float) * n * 7);
    plx_project_f32(X, n, d, mean, &B, scale, P);
    double pe = 0;
    for (int t = 0; t < n; t += 37)
        for (int i = 0; i < 7; i++) {
            double v = 0;
            for (int c = 0; c < d; c++) v += (X[t * d + c] - mean[c]) * PLX_M(&B, i, c);
            pe = PLX_MAX(pe, fabs(P[t * 7 + i] - v / scale[i]) / (fabs(v / scale[i]) + 1));
        }
    CHECK(pe < 1e-6);
    for (int m = 0; m < 3; m++) plx_pca_free(&p[m]);
    free(X); free(Y); free(Yw); free(P);
    plx_mat_free(&Vt); plx_mat_free(&B);
}

/* ---- slow: scale and speed (PLX_SLOW_TESTS=1) ------------------------------------------------ */

static void slow_tests(void) {
    plx_rng r;
    plx_rng_seed(&r, 99);
    printf("slow tests (%d threads)\n", plx_num_threads());
    {
        plx_mat A = randn(&r, 1000, 1000), B = randn(&r, 1000, 1000), C = plx_mat_new(1000, 1000);
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            double t = plx_now();
            plx_gemm(PLX_NO_TRANS, PLX_NO_TRANS, 1.0, &A, &B, 0.0, &C);
            best = PLX_MIN(best, plx_now() - t);
        }
        printf("  gemm 1000^3: %.3f s (%.0f GFLOP/s)\n", best, 2e9 / best / 1e9);
        CHECK(best < 1.0);
        plx_mat_free(&A); plx_mat_free(&B); plx_mat_free(&C);
    }
    static const int en[] = {1000, 2000};
    for (int t = 0; t < 2; t++) {
        int n = en[t];
        plx_mat A = rand_sym(&r, n), V = plx_mat_new(n, n);
        double *w = PLX_NEW(double, n);
        double t0 = plx_now();
        CHECK(plx_eigh(&A, w, &V) == PLX_OK);
        double tv = plx_now() - t0;
        t0 = plx_now();
        plx_eigh(&A, w, NULL);
        double tw = plx_now() - t0, res = eig_resid(&A, w, &V), orth = orth_err(&V);
        printf("  eigh n=%d: %.2f s with vectors, %.2f s values only; residual %.1e, orthogonality %.1e\n", n, tv,
               tw, res, orth);
        CHECK(res < 10 * n * EPS && orth < 10 * n * EPS && tv < 60);
        free(w);
        plx_mat_free(&A); plx_mat_free(&V);
    }
    {
        plx_mat A = randn(&r, 2000, 900), U = plx_mat_new(2000, 900), V = plx_mat_new(900, 900);
        double *s = PLX_NEW(double, 900);
        double t0 = plx_now();
        CHECK(plx_svd(&A, &U, s, &V) == PLX_OK);
        double ts = plx_now() - t0;
        plx_mat R = from_svd(&U, s, &V);
        printf("  svd 2000x900: %.2f s; reconstruction %.1e, orthogonality U %.1e V %.1e\n", ts, rel_err(&R, &A),
               orth_err(&U), orth_err(&V));
        CHECK(rel_err(&R, &A) < 1e-13 && orth_err(&U) < 1e-12 && orth_err(&V) < 1e-12);
        free(s);
        plx_mat_free(&A); plx_mat_free(&U); plx_mat_free(&V); plx_mat_free(&R);
    }
    {
        /* 13000 x 20000 float descriptors, k = 300: rank-500 signal with a decaying spectrum plus
         * isotropic noise, built in row blocks straight into float. */
        const int n = 13000, d = 20000, rank = 500, k = 300;
        double t0 = plx_now();
        plx_mat Z = randn(&r, rank, d);
        for (int i = 0; i < rank; i++)
            for (int c = 0; c < d; c++) PLX_M(&Z, i, c) *= 50.0 * pow(0.99, i);
        float *X = plx_malloc(sizeof(float) * (size_t)n * d);
        plx_mat Gb = plx_mat_new(1000, rank), Xb = plx_mat_new(1000, d);
        for (int r0 = 0; r0 < n; r0 += 1000) {
            for (size_t i = 0; i < (size_t)1000 * rank; i++) Gb.a[i] = plx_rng_normal(&r) / sqrt((double)n);
            plx_gemm(PLX_NO_TRANS, PLX_NO_TRANS, 1.0, &Gb, &Z, 0.0, &Xb);
            for (size_t i = 0; i < (size_t)1000 * d; i++)
                X[(size_t)r0 * d + i] = (float)(Xb.a[i] + 3.0 + 0.05 * plx_rng_normal(&r));
        }
        plx_mat_free(&Gb); plx_mat_free(&Xb); plx_mat_free(&Z);
        printf("  pca data 13000x20000 generated in %.1f s\n", plx_now() - t0);
        plx_pca p;
        t0 = plx_now();
        CHECK(plx_pca_f32(X, n, d, k, NULL, &p) == PLX_OK);
        double tp = plx_now() - t0;
        CHECK(p.method == PLX_PCA_RANDOMIZED);
        /* accuracy without an exact reference: each axis must be an eigenvector of Xc^T Xc,
         * ||Xc^T Xc v_i - sv_i^2 v_i|| / sv_i^2 small, evaluated with two products */
        /* Y = Xc V (n x k), then Xc^T Y = X^T Y - mean (1^T Y) via a projection of X^T onto Y^T */
        float *Yf = plx_malloc(sizeof(float) * (size_t)n * k), *Xt = plx_malloc(sizeof(float) * (size_t)n * d);
        float *XtY = plx_malloc(sizeof(float) * (size_t)d * k);
        plx_pca_project_f32(&p, X, n, false, Yf);
        for (int i = 0; i < n; i++)
            for (int c = 0; c < d; c++) Xt[(size_t)c * n + i] = X[(size_t)i * d + c];
        plx_mat Yt = plx_mat_new(k, n);
        double *colsum = PLX_NEW(double, k);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) {
                PLX_M(&Yt, j, i) = Yf[(size_t)i * k + j];
                colsum[j] += Yf[(size_t)i * k + j];
            }
        plx_project_f32(Xt, d, n, NULL, &Yt, NULL, XtY);
        double r_first = 0, r_last = 0;
        for (int j = 0; j < k; j++) {
            double e2 = 0, l = p.sv[j] * p.sv[j];
            for (int c = 0; c < d; c++) {
                double dv = XtY[(size_t)c * k + j] - p.mean[c] * colsum[j] - l * PLX_M(&p.axes, j, c);
                e2 += dv * dv;
            }
            if (j < 10) r_first = PLX_MAX(r_first, sqrt(e2) / l);
            if (j >= k - 10) r_last = PLX_MAX(r_last, sqrt(e2) / l);
        }
        printf("  pca randomized 13000x20000 k=300: %.1f s; eigen-residual %.1e (top 10), %.1e (last 10); sv %.3g..%.3g\n",
               tp, r_first, r_last, p.sv[0], p.sv[k - 1]);
        CHECK(r_first < 1e-5 && r_last < 1e-2);
        free(Yf); free(Xt); free(XtY); free(colsum);
        plx_mat_free(&Yt);
        plx_pca_free(&p);
        free(X);
    }
}

int main(void) {
    RUN(basics);
    RUN(gemm_matches_reference);
    RUN(gemv_and_syrk);
    RUN(cholesky);
    RUN(lu);
    RUN(qr);
    RUN(lstsq);
    RUN(eigh_random_and_structured);
    RUN(eigh_generalised);
    RUN(svd_shapes_and_properties);
    RUN(svd_relative_accuracy);
    RUN(small_fixed_sizes);
    RUN(conjugate_gradient);
    RUN(data_statistics);
    RUN(pca_routes_agree);
    const char *slow = getenv("PLX_SLOW_TESTS");
    if (slow && atoi(slow) > 0) RUN(slow_tests);
    return TEST_REPORT();
}
