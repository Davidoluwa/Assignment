/* Matrix products. One packed, cache-blocked kernel (Goto/BLIS structure) serves gemm, syrk, the
 * blocked factorisations and the float32 data-matrix products:
 *
 *   for each K block of KC:                       (serial: C accumulates across K blocks)
 *     pack op(B)[K block, all columns] into NR-wide panels          (shared, ~KC x n)
 *     parallel over C tiles of MC x NB:
 *       pack op(A)[tile rows, K block] into MR-high panels          (per thread, sits in L2)
 *       for each NR panel of the tile: for each MR panel: 8x8 register micro-kernel
 *
 * The micro-kernel keeps its 8x8 accumulator in 16 AVX registers: GCC vectorises the j loop and
 * the unroll pragma replicates it over i, so the array is promoted to registers. fma() is
 * explicit because -std=c11 disables floating-point contraction. Each C element is reduced in a
 * fixed order by one task, so results do not depend on the thread count. */
#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

enum { MR = 8, NR = 8, KC = 256, MC = 128, NB = 512 };
#define SMALL_FLOPS 4096.0 /* m*n*k at or below this: direct triple loop, packing would dominate */

static inline double opnd_at(const la_opnd *o, int64_t i, int64_t j) {
    double v = o->d ? o->d[i * o->rs + j * o->cs] : (double)o->f[i * o->rs + j * o->cs];
    if (o->sub_row) v -= o->sub_row[i];
    if (o->sub_col) v -= o->sub_col[j];
    return v;
}

/* Packs rows [r0, r0+nrows) x cols [c0, c0+ncols) of M into W-row panels: panel q holds
 * M(r0 + qW + ii, c0 + p) at [p * W + ii]; a short last panel is zero padded so the micro-kernel
 * never needs edge cases. The loop order follows whichever source stride is unit. */
#define PACK_RAW(T, src)                                                                        \
    do {                                                                                       \
        if (M->cs == 1) {                                                                      \
            for (int ii = 0; ii < w; ii++) {                                                   \
                const T *s = (src) + (r0 + pr + ii) * M->rs + c0;                             \
                for (int p = 0; p < ncols; p++) pd[(size_t)p * W + ii] = (double)s[p];         \
            }                                                                                  \
        } else {                                                                               \
            for (int p = 0; p < ncols; p++) {                                                  \
                const T *s = (src) + (r0 + pr) * M->rs + (c0 + p) * M->cs;                  \
                for (int ii = 0; ii < w; ii++) pd[(size_t)p * W + ii] = (double)s[ii * M->rs]; \
            }                                                                                  \
        }                                                                                      \
    } while (0)

static void pack(const la_opnd *M, int64_t r0, int nrows, int64_t c0, int ncols, int W, double *dst) {
    for (int pr = 0; pr < nrows; pr += W) {
        int w = PLX_MIN(W, nrows - pr);
        double *pd = dst + (size_t)pr * ncols;
        if (w < W) memset(pd, 0, sizeof(double) * (size_t)ncols * W);
        if (M->d) PACK_RAW(double, M->d);
        else PACK_RAW(float, M->f);
        if (M->sub_row || M->sub_col)
            for (int p = 0; p < ncols; p++) {
                double sc = M->sub_col ? M->sub_col[c0 + p] : 0.0;
                for (int ii = 0; ii < w; ii++)
                    pd[(size_t)p * W + ii] -= sc + (M->sub_row ? M->sub_row[r0 + pr + ii] : 0.0);
            }
    }
}

/* C[0:mr, 0:nr] += alpha * Apanel (MR x kc) * Bpanel (kc x NR). */
static void micro_kernel(int kc, const double *restrict a, const double *restrict b, double alpha,
                         double *restrict c, int64_t ldc, int mr, int nr) {
    double acc[MR][NR] = {{0}};
    for (int p = 0; p < kc; p++) {
        const double *ap = a + (size_t)p * MR, *bp = b + (size_t)p * NR;
#pragma GCC unroll 8
        for (int i = 0; i < MR; i++) {
            const double ai = ap[i];
            for (int j = 0; j < NR; j++) acc[i][j] = fma(ai, bp[j], acc[i][j]);
        }
    }
    if (mr == MR && nr == NR) {
        for (int i = 0; i < MR; i++)
            for (int j = 0; j < NR; j++) c[i * ldc + j] = fma(alpha, acc[i][j], c[i * ldc + j]);
    } else {
        for (int i = 0; i < mr; i++)
            for (int j = 0; j < nr; j++) c[i * ldc + j] = fma(alpha, acc[i][j], c[i * ldc + j]);
    }
}

static void scale_block(double *C, int64_t ldc, int m, int n, double beta) {
    for (int i = 0; i < m; i++) {
        double *c = C + i * ldc;
        if (beta == 0) memset(c, 0, sizeof(double) * (size_t)n);
        else if (beta != 1)
            for (int j = 0; j < n; j++) c[j] *= beta;
    }
}

typedef struct {
    int m, n, pc, kc, ntn, lower;
    double alpha, beta;
    const la_opnd *A;
    la_opnd Bt; /* op(B)^T, so B packs with the same routine as A */
    double *C;
    int64_t ldc;
    double *bpack, *apack;
    size_t apack_stride;
    const int *tiles;
} gemm_job;

static void pack_b_task(int64_t q, int tid, void *ctx) {
    gemm_job *g = ctx;
    int j0 = (int)q * NR;
    pack(&g->Bt, j0, PLX_MIN(NR, g->n - j0), g->pc, g->kc, NR, g->bpack + (size_t)q * g->kc * NR);
}

static void tile_task(int64_t t, int tid, void *ctx) {
    gemm_job *g = ctx;
    int i0 = (g->tiles[t] / g->ntn) * MC, j0 = (g->tiles[t] % g->ntn) * NB;
    int mc = PLX_MIN(MC, g->m - i0), nb = PLX_MIN(NB, g->n - j0);
    double *C = g->C + i0 * g->ldc + j0;
    if (g->pc == 0) scale_block(C, g->ldc, mc, nb, g->beta);
    double *ap = g->apack + (size_t)tid * g->apack_stride;
    pack(g->A, i0, mc, g->pc, g->kc, MR, ap);
    for (int jr = 0; jr < nb; jr += NR) {
        const double *bp = g->bpack + (size_t)((j0 + jr) / NR) * g->kc * NR;
        for (int ir = 0; ir < mc; ir += MR) {
            if (g->lower && j0 + jr > i0 + ir + MR - 1) continue; /* micro-tile entirely above diagonal */
            micro_kernel(g->kc, ap + (size_t)ir * g->kc, bp, g->alpha, C + ir * g->ldc + jr, g->ldc,
                         PLX_MIN(MR, mc - ir), PLX_MIN(NR, nb - jr));
        }
    }
}

void la_gemm(int m, int n, int k, double alpha, const la_opnd *A, const la_opnd *B, double beta,
             double *C, int64_t ldc, int lower_only) {
    if (m <= 0 || n <= 0) return;
    if (k <= 0 || alpha == 0) {
        scale_block(C, ldc, m, n, beta);
        return;
    }
    if ((double)m * n * k <= SMALL_FLOPS) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= (lower_only ? PLX_MIN(i, n - 1) : n - 1); j++) {
                double s = 0;
                for (int p = 0; p < k; p++) s = fma(opnd_at(A, i, p), opnd_at(B, p, j), s);
                double *c = C + i * ldc + j;
                *c = beta == 0 ? alpha * s : fma(alpha, s, beta * *c);
            }
        return;
    }

    int ntm = (m + MC - 1) / MC, ntn = (n + NB - 1) / NB, nt = 0;
    int *tiles = PLX_NEW(int, (size_t)ntm * ntn);
    for (int ti = 0; ti < ntm; ti++)
        for (int tj = 0; tj < ntn; tj++)
            if (!lower_only || tj * NB <= ti * MC + MC - 1) tiles[nt++] = ti * ntn + tj;

    int npb = (n + NR - 1) / NR, kcmax = PLX_MIN(KC, k);
    int mcmax = PLX_MIN(MC, (m + MR - 1) / MR * MR), nthreads = plx_num_threads();
    gemm_job g = {.m = m, .n = n, .ntn = ntn, .lower = lower_only, .alpha = alpha, .beta = beta, .A = A,
                  .C = C, .ldc = ldc, .tiles = tiles, .apack_stride = (size_t)mcmax * kcmax};
    g.Bt = (la_opnd){.d = B->d, .f = B->f, .rs = B->cs, .cs = B->rs, .sub_row = B->sub_col, .sub_col = B->sub_row};
    g.bpack = plx_malloc(sizeof(double) * (size_t)npb * NR * kcmax);
    g.apack = plx_malloc(sizeof(double) * g.apack_stride * nthreads);
    for (int pc = 0; pc < k; pc += KC) {
        g.pc = pc;
        g.kc = PLX_MIN(KC, k - pc);
        la_par_for(npb, 4.0 * NR * g.kc, pack_b_task, &g);
        la_par_for(nt, 2.0 * MC * NB * g.kc, tile_task, &g);
    }
    free(g.apack);
    free(g.bpack);
    free(tiles);
}

void la_dgemm(int ta, int tb, int m, int n, int k, double alpha, const double *A, int64_t lda,
              const double *B, int64_t ldb, double beta, double *C, int64_t ldc) {
    la_opnd oa = la_opnd_d(A, lda, ta), ob = la_opnd_d(B, ldb, tb);
    la_gemm(m, n, k, alpha, &oa, &ob, beta, C, ldc, 0);
}

void la_mirror_lower(double *C, int n, int64_t ldc) {
    enum { B = 64 }; /* tiled so the column-order reads stay in cache */
    for (int i0 = 0; i0 < n; i0 += B)
        for (int j0 = 0; j0 <= i0; j0 += B)
            for (int i = i0; i < PLX_MIN(i0 + B, n); i++)
                for (int j = j0; j < PLX_MIN(j0 + B, i); j++) C[j * ldc + i] = C[i * ldc + j];
}

/* ---- public products ------------------------------------------------------------------ */

static int op_rows(const plx_mat *A, plx_op t) { return t == PLX_TRANS ? A->cols : A->rows; }
static int op_cols(const plx_mat *A, plx_op t) { return t == PLX_TRANS ? A->rows : A->cols; }

void plx_gemm(plx_op ta, plx_op tb, double alpha, const plx_mat *A, const plx_mat *B, double beta, plx_mat *C) {
    int m = op_rows(A, ta), k = op_cols(A, ta), n = op_cols(B, tb);
    PLX_ASSERT(op_rows(B, tb) == k && C->rows == m && C->cols == n);
    PLX_ASSERT(C->a != A->a && C->a != B->a);
    la_dgemm(ta == PLX_TRANS, tb == PLX_TRANS, m, n, k, alpha, A->a, A->cols, B->a, B->cols, beta, C->a, C->cols);
}

plx_mat plx_matmul(const plx_mat *A, const plx_mat *B) {
    plx_mat C = plx_mat_new(A->rows, B->cols);
    plx_gemm(PLX_NO_TRANS, PLX_NO_TRANS, 1.0, A, B, 0.0, &C);
    return C;
}

void plx_syrk(plx_op t, double alpha, const plx_mat *A, double beta, plx_mat *C) {
    const int tr = t == PLX_TRANS; /* TRANS: C = A^T A; NO_TRANS: C = A A^T */
    const int n = tr ? A->cols : A->rows, k = tr ? A->rows : A->cols;
    PLX_ASSERT(C->rows == n && C->cols == n && C->a != A->a);
    la_opnd l = la_opnd_d(A->a, A->cols, tr), r = la_opnd_d(A->a, A->cols, !tr);
    la_gemm(n, n, k, alpha, &l, &r, beta, C->a, n, 1);
    la_mirror_lower(C->a, n, n);
}

typedef struct {
    const plx_mat *A;
    const double *x;
    double *y, alpha, beta;
} gemv_job;

enum { GEMV_ROWS = 64, GEMV_COLS = 512 };

static void gemv_n_task(int64_t b, int tid, void *ctx) {
    gemv_job *g = ctx;
    int i1 = PLX_MIN((int)(b + 1) * GEMV_ROWS, g->A->rows);
    for (int i = (int)b * GEMV_ROWS; i < i1; i++) {
        double s = la_dot(&PLX_M(g->A, i, 0), g->x, (size_t)g->A->cols);
        g->y[i] = g->beta == 0 ? g->alpha * s : fma(g->alpha, s, g->beta * g->y[i]);
    }
}

/* y = A^T x accumulates rows of A (contiguous) into a column block of y. */
static void gemv_t_task(int64_t b, int tid, void *ctx) {
    gemv_job *g = ctx;
    int j0 = (int)b * GEMV_COLS, w = PLX_MIN(GEMV_COLS, g->A->cols - j0);
    double acc[GEMV_COLS] = {0};
    for (int i = 0; i < g->A->rows; i++) la_axpy(acc, g->x[i], &PLX_M(g->A, i, j0), (size_t)w);
    for (int j = 0; j < w; j++)
        g->y[j0 + j] = g->beta == 0 ? g->alpha * acc[j] : fma(g->alpha, acc[j], g->beta * g->y[j0 + j]);
}

void plx_gemv(plx_op ta, double alpha, const plx_mat *A, const double *x, double beta, double *y) {
    gemv_job g = {A, x, y, alpha, beta};
    if (ta == PLX_TRANS)
        la_par_for((A->cols + GEMV_COLS - 1) / GEMV_COLS, 2.0 * A->rows * GEMV_COLS, gemv_t_task, &g);
    else
        la_par_for((A->rows + GEMV_ROWS - 1) / GEMV_ROWS, 2.0 * A->cols * GEMV_ROWS, gemv_n_task, &g);
}
