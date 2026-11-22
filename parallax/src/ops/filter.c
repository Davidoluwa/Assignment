/* Separable linear filters: Gaussian blur and derivative filters, replicate border. */
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int plx_gauss_radius(float sigma) {
    int r = (int)ceilf(3.0f * sigma);
    return r < 1 ? 1 : r;
}

void plx_gauss_kernel(float sigma, float *k) {
    PLX_ASSERT(sigma > 0.0f);
    int r = plx_gauss_radius(sigma);
    double sum = 0.0, inv2s2 = 1.0 / (2.0 * (double)sigma * sigma);
    for (int i = -r; i <= r; i++) sum += exp(-(double)i * i * inv2s2);
    for (int i = -r; i <= r; i++) k[i + r] = (float)(exp(-(double)i * i * inv2s2) / sum);
}

/* Correlates every channel with kx along x then ky along y (replicate border):
 *   dst(x, y) = sum_j ky[j] sum_i kx[i] src(x + i - rx, y + j - ry).
 * Both passes keep the inner loop running along a row so it vectorises; the horizontal result
 * goes to a private plane, which is what makes dst == src safe. */
static void sep_filter(const plx_img *src, plx_img *dst, const float *kx, int rx, const float *ky, int ry) {
    PLX_ASSERT(plx_img_same_shape(src, dst));
    int w = src->w, h = src->h;
    float *tmp = plx_malloc(sizeof(float) * (size_t)w * h);
    float *row = plx_malloc(sizeof(float) * (size_t)(w + 2 * rx));
    for (int k = 0; k < src->c; k++) {
        const float *s = plx_img_chan(src, k);
        for (int y = 0; y < h; y++) {
            const float *sr = s + (size_t)y * w;
            for (int i = 0; i < rx; i++) {
                row[i] = sr[0];
                row[rx + w + i] = sr[w - 1];
            }
            memcpy(row + rx, sr, sizeof(float) * (size_t)w);
            float *t = tmp + (size_t)y * w;
            for (int x = 0; x < w; x++) t[x] = kx[0] * row[x];
            for (int i = 1; i <= 2 * rx; i++) {
                const float *ri = row + i;
                float ki = kx[i];
                for (int x = 0; x < w; x++) t[x] += ki * ri[x];
            }
        }
        float *d = plx_img_chan(dst, k);
        for (int y = 0; y < h; y++) {
            float *dr = d + (size_t)y * w;
            const float *t0 = tmp + (size_t)PLX_CLAMP(y - ry, 0, h - 1) * w;
            for (int x = 0; x < w; x++) dr[x] = ky[0] * t0[x];
            for (int j = 1; j <= 2 * ry; j++) {
                const float *tj = tmp + (size_t)PLX_CLAMP(y + j - ry, 0, h - 1) * w;
                float kj = ky[j];
                for (int x = 0; x < w; x++) dr[x] += kj * tj[x];
            }
        }
    }
    free(row);
    free(tmp);
}

void plx_blur(const plx_img *src, plx_img *dst, float sigma) {
    PLX_ASSERT(plx_img_same_shape(src, dst));
    if (sigma <= 0.0f) {
        if (dst != src) plx_img_copy(dst, src);
        return;
    }
    int r = plx_gauss_radius(sigma);
    float *k = plx_malloc(sizeof(float) * (size_t)(2 * r + 1));
    plx_gauss_kernel(sigma, k);
    sep_filter(src, dst, k, r, k, r);
    free(k);
}

void plx_gradient_gauss(const plx_img *src, float sigma, plx_img *gx, plx_img *gy) {
    PLX_ASSERT(sigma > 0.0f && gx != src && gy != src && gx != gy);
    int r = plx_gauss_radius(sigma);
    float *g = plx_malloc(sizeof(float) * (size_t)(2 * r + 1));
    float *d = plx_malloc(sizeof(float) * (size_t)(2 * r + 1));
    plx_gauss_kernel(sigma, g);
    /* d[j] proportional to j G(j), scaled so sum_j j d[j] = 1: then correlating with d returns
     * exactly the slope of a ramp, and (d antisymmetric) the derivative of any quadratic. */
    double m2 = 0.0;
    for (int j = -r; j <= r; j++) m2 += (double)j * j * g[j + r];
    for (int j = -r; j <= r; j++) d[j + r] = (float)((double)j * g[j + r] / m2);
    sep_filter(src, gx, d, r, g, r);
    sep_filter(src, gy, g, r, d, r);
    free(d);
    free(g);
}

void plx_gradient_scharr(const plx_img *src, plx_img *gx, plx_img *gy) {
    PLX_ASSERT(gx != src && gy != src && gx != gy);
    static const float diff[3] = {-0.5f, 0.0f, 0.5f};
    static const float smooth[3] = {3.0f / 16, 10.0f / 16, 3.0f / 16};
    sep_filter(src, gx, diff, 1, smooth, 1);
    sep_filter(src, gy, smooth, 1, diff, 1);
}
