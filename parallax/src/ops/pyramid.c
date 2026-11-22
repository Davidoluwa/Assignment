/* 2x downsampling, Gaussian pyramids and general resize, all through one separable
 * "tap table" resampler so the registration of every path is spelled out in one place. */
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>

/* One axis of a separable resampler:
 *   out[i] = sum_{t < ntap} wt[i*ntap + t] * in[clamp(start[i] + t, 0, n_in - 1)]. */
typedef struct {
    int    n_out, ntap;
    int   *start;
    float *wt;
} axis_taps;

static axis_taps taps_new(int n_out, int ntap) {
    axis_taps a = {n_out, ntap, PLX_NEW(int, n_out), PLX_NEW(float, (size_t)n_out * ntap)};
    return a;
}

static void taps_free(axis_taps *a) {
    free(a->start);
    free(a->wt);
}

/* Output i is centred on input 2i + 0.5: taps 2i-2 .. 2i+3 with binomial weights. */
static axis_taps taps_down2(int n_in) {
    static const float binom[6] = {1 / 32.f, 5 / 32.f, 10 / 32.f, 10 / 32.f, 5 / 32.f, 1 / 32.f};
    axis_taps a = taps_new((n_in + 1) / 2, 6);
    for (int i = 0; i < a.n_out; i++) {
        a.start[i] = 2 * i - 2;
        for (int t = 0; t < 6; t++) a.wt[i * 6 + t] = binom[t];
    }
    return a;
}

/* Pixel-area aligned: output i spans input [i*s, (i+1)*s) in edge coordinates (input pixel j
 * spans [j, j+1)), s = n_in / n_out, i.e. its centre is at (i + 0.5) s - 0.5. */
static axis_taps taps_resize(int n_in, int n_out) {
    double s = (double)n_in / n_out;
    if (s <= 1.0) { /* enlarging (or same size): bilinear at the centre */
        axis_taps a = taps_new(n_out, 2);
        for (int i = 0; i < n_out; i++) {
            double c = (i + 0.5) * s - 0.5;
            int j = (int)floor(c);
            float f = (float)(c - j);
            a.start[i] = j;
            a.wt[2 * i] = 1.0f - f;
            a.wt[2 * i + 1] = f;
        }
        return a;
    }
    /* shrinking: exact overlap of the output span with each input pixel, divided by s */
    axis_taps a = taps_new(n_out, (int)ceil(s) + 1);
    for (int i = 0; i < n_out; i++) {
        double lo = i * s, hi = (i + 1) * s;
        int j0 = (int)floor(lo);
        a.start[i] = j0;
        for (int t = 0; t < a.ntap; t++) {
            double ov = fmin(hi, j0 + t + 1.0) - fmax(lo, (double)(j0 + t));
            a.wt[i * a.ntap + t] = ov > 0.0 ? (float)(ov / s) : 0.0f;
        }
    }
    return a;
}

static plx_img resample_sep(const plx_img *src, const axis_taps *ax, const axis_taps *ay) {
    int w = src->w, h = src->h, ow = ax->n_out, oh = ay->n_out;
    plx_img out = plx_img_new(ow, oh, src->c);
    float *tmp = plx_malloc(sizeof(float) * (size_t)ow * h);
    for (int k = 0; k < src->c; k++) {
        const float *s = plx_img_chan(src, k);
        for (int y = 0; y < h; y++) {
            const float *sr = s + (size_t)y * w;
            float *t = tmp + (size_t)y * ow;
            for (int i = 0; i < ow; i++) {
                const float *wt = ax->wt + (size_t)i * ax->ntap;
                float acc = 0.0f;
                for (int q = 0; q < ax->ntap; q++) acc += wt[q] * sr[PLX_CLAMP(ax->start[i] + q, 0, w - 1)];
                t[i] = acc;
            }
        }
        float *d = plx_img_chan(&out, k);
        for (int i = 0; i < oh; i++) {
            float *dr = d + (size_t)i * ow;
            const float *wt = ay->wt + (size_t)i * ay->ntap;
            for (int x = 0; x < ow; x++) dr[x] = 0.0f;
            for (int q = 0; q < ay->ntap; q++) {
                const float *tr = tmp + (size_t)PLX_CLAMP(ay->start[i] + q, 0, h - 1) * ow;
                float wq = wt[q];
                for (int x = 0; x < ow; x++) dr[x] += wq * tr[x];
            }
        }
    }
    free(tmp);
    return out;
}

plx_img plx_downsample2(const plx_img *src) {
    axis_taps ax = taps_down2(src->w), ay = taps_down2(src->h);
    plx_img out = resample_sep(src, &ax, &ay);
    taps_free(&ax);
    taps_free(&ay);
    return out;
}

plx_img plx_resize(const plx_img *src, int w, int h) {
    PLX_ASSERT(w > 0 && h > 0);
    axis_taps ax = taps_resize(src->w, w), ay = taps_resize(src->h, h);
    plx_img out = resample_sep(src, &ax, &ay);
    taps_free(&ax);
    taps_free(&ay);
    return out;
}

plx_pyramid plx_pyramid_build(const plx_img *im, int max_levels, int min_size) {
    PLX_ASSERT(max_levels >= 1 && max_levels <= PLX_PYR_MAX);
    plx_pyramid p = {0};
    p.lev[0] = plx_img_clone(im);
    p.n = 1;
    while (p.n < max_levels) {
        const plx_img *top = &p.lev[p.n - 1];
        if (PLX_MIN((top->w + 1) / 2, (top->h + 1) / 2) < min_size || PLX_MAX(top->w, top->h) < 2) break;
        p.lev[p.n] = plx_downsample2(top);
        p.n++;
    }
    return p;
}

void plx_pyramid_free(plx_pyramid *p) {
    for (int i = 0; i < p->n; i++) plx_img_free(&p->lev[i]);
    p->n = 0;
}
