#include "la/la_internal.h"

#include <stdlib.h>
#include <string.h>

const char *plx_la_status_str(int status) {
    switch (status) {
    case PLX_ENOTSPD: return "matrix not positive definite";
    case PLX_ESINGULAR: return "matrix singular";
    case PLX_ENOCONV: return "no convergence";
    default: return plx_status_str(status);
    }
}

void la_par_for(int64_t n, double flops_per_item, plx_par_fn fn, void *ctx) {
    if ((double)n * flops_per_item < LA_PAR_MIN_FLOPS || plx_num_threads() <= 1) {
        for (int64_t i = 0; i < n; i++) fn(i, 0, ctx);
        return;
    }
    plx_parallel_for(n, 0, fn, ctx);
}

/* ---- vector kernels ------------------------------------------------------------------- */

double la_dot(const double *x, const double *y, size_t n) {
    double acc[8] = {0};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; j++) acc[j] = fma(x[i + j], y[i + j], acc[j]);
    double tail = 0;
    for (; i < n; i++) tail = fma(x[i], y[i], tail);
    return ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail;
}

double la_dot_strided(const double *x, int64_t incx, const double *y, int64_t incy, int n) {
    if (incx == 1 && incy == 1) return la_dot(x, y, (size_t)n);
    double s0 = 0, s1 = 0;
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        s0 = fma(x[i * incx], y[i * incy], s0);
        s1 = fma(x[(i + 1) * incx], y[(i + 1) * incy], s1);
    }
    if (i < n) s0 = fma(x[i * incx], y[i * incy], s0);
    return s0 + s1;
}

void la_axpy(double *restrict y, double alpha, const double *restrict x, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = fma(alpha, x[i], y[i]);
}

/* Sum of squares is fast but over/underflows outside ~[1e-150, 1e150]; only then do the scaled
 * one-pass recurrence (scale = running max |x|, ssq = sum (x / scale)^2). */
double la_norm_strided(const double *x, int64_t inc, int n) {
    double s = la_dot_strided(x, inc, x, inc, n);
    if (s > 0x1p-900 && s < 0x1p+1000) return sqrt(s);
    double scale = 0, ssq = 1;
    for (int i = 0; i < n; i++) {
        double a = fabs(x[i * inc]);
        if (a == 0) continue;
        if (scale < a) {
            ssq = 1 + ssq * (scale / a) * (scale / a);
            scale = a;
        } else {
            ssq += (a / scale) * (a / scale);
        }
    }
    return scale * sqrt(ssq);
}

double plx_vec_dot(const double *x, const double *y, size_t n) { return la_dot(x, y, n); }
void plx_vec_axpy(double *y, double alpha, const double *x, size_t n) {
    if (y == x) { /* the restrict-qualified kernel must not see aliasing */
        for (size_t i = 0; i < n; i++) y[i] += alpha * y[i];
        return;
    }
    la_axpy(y, alpha, x, n);
}

double plx_vec_norm(const double *x, size_t n) {
    PLX_ASSERT(n <= (size_t)INT32_MAX);
    return la_norm_strided(x, 1, (int)n);
}

/* ---- matrices ------------------------------------------------------------------------ */

plx_mat plx_mat_new(int rows, int cols) {
    PLX_ASSERT(rows >= 0 && cols >= 0);
    plx_mat m = {rows, cols, PLX_NEW(double, (size_t)rows * (size_t)cols)};
    return m;
}

void plx_mat_free(plx_mat *m) {
    free(m->a);
    memset(m, 0, sizeof *m);
}

plx_mat plx_mat_from(int rows, int cols, const double *data) {
    plx_mat m = plx_mat_new(rows, cols);
    memcpy(m.a, data, sizeof(double) * (size_t)rows * (size_t)cols);
    return m;
}

plx_mat plx_mat_clone(const plx_mat *a) { return plx_mat_from(a->rows, a->cols, a->a); }

plx_mat plx_mat_identity(int n) {
    plx_mat m = plx_mat_new(n, n);
    for (int i = 0; i < n; i++) PLX_M(&m, i, i) = 1.0;
    return m;
}

plx_mat plx_mat_transpose(const plx_mat *a) {
    plx_mat t = plx_mat_new(a->cols, a->rows);
    enum { B = 32 }; /* tiles keep both the read and the write side in cache */
    for (int i0 = 0; i0 < a->rows; i0 += B)
        for (int j0 = 0; j0 < a->cols; j0 += B)
            for (int i = i0; i < PLX_MIN(i0 + B, a->rows); i++)
                for (int j = j0; j < PLX_MIN(j0 + B, a->cols); j++) PLX_M(&t, j, i) = PLX_M(a, i, j);
    return t;
}

void plx_mat_copy(plx_mat *dst, const plx_mat *src) {
    PLX_ASSERT(dst->rows == src->rows && dst->cols == src->cols);
    memmove(dst->a, src->a, sizeof(double) * (size_t)src->rows * (size_t)src->cols);
}

void plx_mat_scale(plx_mat *a, double s) {
    size_t n = (size_t)a->rows * (size_t)a->cols;
    for (size_t i = 0; i < n; i++) a->a[i] *= s;
}

void plx_mat_add(plx_mat *y, double alpha, const plx_mat *x) {
    PLX_ASSERT(y->rows == x->rows && y->cols == x->cols);
    plx_vec_axpy(y->a, alpha, x->a, (size_t)y->rows * (size_t)y->cols);
}

double plx_mat_norm_fro(const plx_mat *a) {
    size_t n = (size_t)a->rows * (size_t)a->cols;
    double s = la_dot(a->a, a->a, n);
    if (s > 0x1p-900 && s < 0x1p+1000) return sqrt(s);
    double acc = 0; /* rare path: combine overflow-safe row norms */
    for (int i = 0; i < a->rows; i++) acc = hypot(acc, la_norm_strided(a->a + (size_t)i * a->cols, 1, a->cols));
    return acc;
}

void la_fix_signs(double *X, int r, int c, double *Y, int ry) {
    for (int j = 0; j < c; j++) {
        int best = 0;
        for (int i = 1; i < r; i++)
            if (fabs(X[(size_t)i * c + j]) > fabs(X[(size_t)best * c + j])) best = i;
        if (r == 0 || X[(size_t)best * c + j] >= 0) continue;
        for (int i = 0; i < r; i++) X[(size_t)i * c + j] = -X[(size_t)i * c + j];
        if (Y)
            for (int i = 0; i < ry; i++) Y[(size_t)i * c + j] = -Y[(size_t)i * c + j];
    }
}
