/* Sub-pixel sampling (bilinear / Catmull-Rom) with analytic gradients, and the warps built on it. */
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>

/* Interpolation taps along one axis: value weights w, derivative weights d = dw/dx, and the
 * (already clamped) element offsets of the taps. */
typedef struct {
    size_t off[4];
    float  w[4], d[4];
} axis_taps;

/* Coordinates beyond [-2, n+1] see only edge pixels under either kernel, so clamping there
 * changes nothing but keeps floorf -> int well defined. The negated test also maps NaN to -2. */
static inline float clamp_coord(float x, int n) {
    if (!(x >= -2.0f)) return -2.0f;
    return x > (float)n + 1.0f ? (float)n + 1.0f : x;
}

static inline void taps_bilinear(float x, int n, size_t stride, axis_taps *t) {
    x = clamp_coord(x, n);
    float fl = floorf(x), f = x - fl;
    int i = (int)fl;
    t->off[0] = (size_t)PLX_CLAMP(i, 0, n - 1) * stride;
    t->off[1] = (size_t)PLX_CLAMP(i + 1, 0, n - 1) * stride;
    t->w[0] = 1.0f - f;
    t->w[1] = f;
    t->d[0] = -1.0f;
    t->d[1] = 1.0f;
}

/* Catmull-Rom (Keys, a = -1/2) on taps i-1 .. i+2 at fraction f. */
static inline void taps_cubic(float x, int n, size_t stride, axis_taps *t) {
    x = clamp_coord(x, n);
    float fl = floorf(x), f = x - fl, f2 = f * f, f3 = f2 * f;
    int i = (int)fl;
    for (int q = 0; q < 4; q++) t->off[q] = (size_t)PLX_CLAMP(i - 1 + q, 0, n - 1) * stride;
    t->w[0] = 0.5f * (-f3 + 2.0f * f2 - f);
    t->w[1] = 0.5f * (3.0f * f3 - 5.0f * f2 + 2.0f);
    t->w[2] = 0.5f * (-3.0f * f3 + 4.0f * f2 + f);
    t->w[3] = 0.5f * (f3 - f2);
    t->d[0] = 0.5f * (-3.0f * f2 + 4.0f * f - 1.0f);
    t->d[1] = 0.5f * (9.0f * f2 - 10.0f * f);
    t->d[2] = 0.5f * (-9.0f * f2 + 8.0f * f + 1.0f);
    t->d[3] = 0.5f * (3.0f * f2 - 2.0f * f);
}

/* Pairwise sums: four independent products instead of one serial chain of FMAs, which is
 * what keeps the 4x4 bicubic case from being latency bound. */
static inline float dot(const float *w, const float *s, int n) {
    return n == 2 ? w[0] * s[0] + w[1] * s[1] : (w[0] * s[0] + w[1] * s[1]) + (w[2] * s[2] + w[3] * s[3]);
}

/* Tensor-product evaluation of every channel; n (2 or 4) is a constant at each call site so
 * the loops unroll. The x-pass rows a[] feed both the value and d/dy, the x-derivative rows
 * b[] feed d/dx. */
static inline void eval_channels(const plx_img *im, const axis_taps *tx, const axis_taps *ty, int n,
                                 float *val, float *grad) {
    size_t plane = (size_t)im->w * im->h;
    for (int k = 0; k < im->c; k++) {
        const float *p = im->px + (size_t)k * plane;
        float a[4], b[4];
        for (int r = 0; r < n; r++) {
            const float *row = p + ty->off[r];
            float s[4];
            for (int q = 0; q < n; q++) s[q] = row[tx->off[q]];
            a[r] = dot(tx->w, s, n);
            b[r] = dot(tx->d, s, n);
        }
        val[k] = dot(ty->w, a, n);
        if (grad) {
            grad[2 * k] = dot(ty->w, b, n);
            grad[2 * k + 1] = dot(ty->d, a, n);
        }
    }
}

bool plx_sample(const plx_img *im, plx_interp m, float x, float y, float *val, float *grad) {
    axis_taps tx, ty;
    if (m == PLX_INTERP_BICUBIC) {
        taps_cubic(x, im->w, 1, &tx);
        taps_cubic(y, im->h, (size_t)im->w, &ty);
        eval_channels(im, &tx, &ty, 4, val, grad);
    } else {
        taps_bilinear(x, im->w, 1, &tx);
        taps_bilinear(y, im->h, (size_t)im->w, &ty);
        eval_channels(im, &tx, &ty, 2, val, grad);
    }
    return plx_img_inside(im, x, y);
}

/* Single-channel calls go through a 1-channel view of plane k, so they share every line of
 * the multi-channel code path (and therefore give bit-identical results). */
static inline plx_img chan_view(const plx_img *im, int k) {
    PLX_ASSERT(k >= 0 && k < im->c);
    plx_img v = {im->w, im->h, 1, plx_img_chan(im, k)};
    return v;
}

float plx_sample1(const plx_img *im, int k, plx_interp m, float x, float y) {
    plx_img v = chan_view(im, k);
    float out;
    plx_sample(&v, m, x, y, &out, NULL);
    return out;
}

float plx_sample1_grad(const plx_img *im, int k, plx_interp m, float x, float y, float *dx, float *dy) {
    plx_img v = chan_view(im, k);
    float out, g[2];
    plx_sample(&v, m, x, y, &out, g);
    *dx = g[0];
    *dy = g[1];
    return out;
}

void plx_sample_points(const plx_img *im, plx_interp m, int n, const float *x, const float *y,
                       float *val, float *grad, uint8_t *inside) {
    size_t c = (size_t)im->c;
    for (int i = 0; i < n; i++) {
        bool in = plx_sample(im, m, x[i], y[i], val + i * c, grad ? grad + 2 * i * c : NULL);
        if (inside) inside[i] = in;
    }
}

/* Shared pull loop: src_xy(x, y) gives the source coordinate of destination pixel (x, y). */
typedef void (*coord_fn)(int x, int y, const void *ctx, float *sx, float *sy);

static void warp_pull(const plx_img *src, coord_fn fn, const void *ctx, plx_interp m, plx_img *dst, plx_img *valid) {
    PLX_ASSERT(dst != src && dst->c == src->c);
    if (valid) PLX_ASSERT(valid->c == 1 && valid->w == dst->w && valid->h == dst->h);
    size_t plane = (size_t)dst->w * dst->h;
    float *v = PLX_NEW(float, src->c);
    for (int y = 0; y < dst->h; y++) {
        for (int x = 0; x < dst->w; x++) {
            float sx, sy;
            fn(x, y, ctx, &sx, &sy);
            bool in = plx_sample(src, m, sx, sy, v, NULL);
            size_t o = (size_t)y * dst->w + x;
            for (int k = 0; k < src->c; k++) dst->px[k * plane + o] = v[k];
            if (valid) valid->px[o] = in ? 1.0f : 0.0f;
        }
    }
    free(v);
}

static void affine_coord(int x, int y, const void *ctx, float *sx, float *sy) {
    const float *A = ctx;
    *sx = A[0] * (float)x + A[1] * (float)y + A[2];
    *sy = A[3] * (float)x + A[4] * (float)y + A[5];
}

static void map_coord(int x, int y, const void *ctx, float *sx, float *sy) {
    const plx_img *map = ctx;
    size_t o = (size_t)y * map->w + x;
    *sx = map->px[o];
    *sy = map->px[(size_t)map->w * map->h + o];
}

void plx_warp_affine(const plx_img *src, const float A[6], plx_interp m, plx_img *dst, plx_img *valid) {
    warp_pull(src, affine_coord, A, m, dst, valid);
}

void plx_warp_map(const plx_img *src, const plx_img *map, plx_interp m, plx_img *dst, plx_img *valid) {
    PLX_ASSERT(map->c == 2 && map->w == dst->w && map->h == dst->h);
    warp_pull(src, map_coord, map, m, dst, valid);
}

void plx_affine_compose(const float A[6], const float B[6], float out[6]) {
    float r[6] = {
        A[0] * B[0] + A[1] * B[3], A[0] * B[1] + A[1] * B[4], A[0] * B[2] + A[1] * B[5] + A[2],
        A[3] * B[0] + A[4] * B[3], A[3] * B[1] + A[4] * B[4], A[3] * B[2] + A[4] * B[5] + A[5],
    };
    for (int i = 0; i < 6; i++) out[i] = r[i];
}

bool plx_affine_invert(const float A[6], float out[6]) {
    double det = (double)A[0] * A[4] - (double)A[1] * A[3];
    if (fabs(det) < 1e-12) return false;
    double a = A[4] / det, b = -A[1] / det, d = -A[3] / det, e = A[0] / det;
    float r[6] = {(float)a, (float)b, (float)(-(a * A[2] + b * A[5])),
                  (float)d, (float)e, (float)(-(d * A[2] + e * A[5]))};
    for (int i = 0; i < 6; i++) out[i] = r[i];
    return true;
}
