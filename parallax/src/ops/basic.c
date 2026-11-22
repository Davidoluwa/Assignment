/* Elementwise helpers, flips, per-channel normalisation and integral images. */
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static size_t img_count(const plx_img *im) { return (size_t)im->w * im->h * im->c; }

bool plx_img_same_shape(const plx_img *a, const plx_img *b) {
    return a->w == b->w && a->h == b->h && a->c == b->c;
}

void plx_img_copy(plx_img *dst, const plx_img *src) {
    PLX_ASSERT(plx_img_same_shape(dst, src));
    if (dst->px != src->px) memcpy(dst->px, src->px, sizeof(float) * img_count(src));
}

void plx_img_fill(plx_img *im, float v) {
    size_t n = img_count(im);
    for (size_t i = 0; i < n; i++) im->px[i] = v;
}

void plx_img_scale_offset(plx_img *im, float a, float b) {
    size_t n = img_count(im);
    for (size_t i = 0; i < n; i++) im->px[i] = a * im->px[i] + b;
}

float plx_img_max_abs_diff(const plx_img *a, const plx_img *b) {
    PLX_ASSERT(plx_img_same_shape(a, b));
    size_t n = img_count(a);
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a->px[i] - b->px[i]);
        if (!(d <= m)) m = d; /* also propagates NaN, so a NaN never hides as "no difference" */
    }
    return m;
}

plx_img plx_img_extract(const plx_img *im, int k) {
    PLX_ASSERT(k >= 0 && k < im->c);
    plx_img out = plx_img_new(im->w, im->h, 1);
    memcpy(out.px, plx_img_chan(im, k), sizeof(float) * (size_t)im->w * im->h);
    return out;
}

/* Reads both mirrored partners before writing either, which is what makes dst == src safe. */
static void flip_plane(const float *s, float *d, int w, int h) {
    for (int y = 0; y < h; y++) {
        const float *sr = s + (size_t)y * w;
        float *dr = d + (size_t)y * w;
        for (int x = 0; x < (w + 1) / 2; x++) {
            float a = sr[x], b = sr[w - 1 - x];
            dr[x] = b;
            dr[w - 1 - x] = a;
        }
    }
}

void plx_flip_h(const plx_img *src, plx_img *dst) {
    PLX_ASSERT(plx_img_same_shape(src, dst));
    for (int k = 0; k < src->c; k++) flip_plane(plx_img_chan(src, k), plx_img_chan(dst, k), src->w, src->h);
}

void plx_flip_h_perm(const plx_img *src, const int *perm, plx_img *dst) {
    PLX_ASSERT(plx_img_same_shape(src, dst));
    bool *seen = PLX_NEW(bool, src->c);
    for (int k = 0; k < src->c; k++) {
        PLX_ASSERT(perm[k] >= 0 && perm[k] < src->c && !seen[perm[k]]);
        seen[perm[k]] = true;
    }
    free(seen);
    /* A general permutation cannot be applied plane-by-plane in place; read from a copy then. */
    plx_img tmp = {0};
    const plx_img *from = src;
    if (src->px == dst->px) {
        tmp = plx_img_clone(src);
        from = &tmp;
    }
    for (int k = 0; k < src->c; k++) flip_plane(plx_img_chan(from, perm[k]), plx_img_chan(dst, k), src->w, src->h);
    plx_img_free(&tmp);
}

void plx_chan_stats(const plx_img *im, int k, double *mean, double *var) {
    PLX_ASSERT(k >= 0 && k < im->c);
    const float *p = plx_img_chan(im, k);
    size_t n = (size_t)im->w * im->h;
    double s = 0.0, s2 = 0.0;
    for (size_t i = 0; i < n; i++) s += p[i];
    double mu = s / (double)n;
    for (size_t i = 0; i < n; i++) s2 += ((double)p[i] - mu) * ((double)p[i] - mu); /* two-pass: no cancellation */
    *mean = mu;
    *var = s2 / (double)n;
}

void plx_normalize_meanvar(plx_img *im, float eps) {
    PLX_ASSERT(eps > 0.0f);
    size_t n = (size_t)im->w * im->h;
    for (int k = 0; k < im->c; k++) {
        double mu, var;
        plx_chan_stats(im, k, &mu, &var);
        float m = (float)mu, inv = (float)(1.0 / sqrt(var + eps));
        float *p = plx_img_chan(im, k);
        for (size_t i = 0; i < n; i++) p[i] = (p[i] - m) * inv;
    }
}

plx_integral plx_integral_build(const plx_img *im) {
    int W = im->w + 1, H = im->h + 1;
    plx_integral ii = {im->w, im->h, im->c, PLX_NEW(double, (size_t)W * H * im->c)};
    for (int k = 0; k < im->c; k++) {
        const float *p = plx_img_chan(im, k);
        double *s = ii.s + (size_t)k * W * H; /* row 0 and column 0 stay zero */
        for (int y = 0; y < im->h; y++) {
            double run = 0.0;
            double *above = s + (size_t)y * W, *cur = s + (size_t)(y + 1) * W;
            for (int x = 0; x < im->w; x++) {
                run += p[(size_t)y * im->w + x];
                cur[x + 1] = above[x + 1] + run;
            }
        }
    }
    return ii;
}

void plx_integral_free(plx_integral *ii) {
    free(ii->s);
    memset(ii, 0, sizeof *ii);
}

double plx_integral_box(const plx_integral *ii, int k, int x0, int y0, int x1, int y1) {
    PLX_ASSERT(k >= 0 && k < ii->c);
    x0 = PLX_CLAMP(x0, 0, ii->w);
    x1 = PLX_CLAMP(x1, 0, ii->w);
    y0 = PLX_CLAMP(y0, 0, ii->h);
    y1 = PLX_CLAMP(y1, 0, ii->h);
    if (x1 <= x0 || y1 <= y0) return 0.0;
    size_t W = (size_t)ii->w + 1;
    const double *s = ii->s + (size_t)k * W * (ii->h + 1);
    return s[y1 * W + x1] - s[y0 * W + x1] - s[y1 * W + x0] + s[y0 * W + x0];
}
