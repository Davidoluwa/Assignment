/* Contrast-normalised, soft-binned gradient orientation channels. */
#include "feat/feat.h"

#include <math.h>
#include <stdlib.h>

static const float PI_F = 3.14159265358979f;

plx_orient_params plx_orient_default(void) {
    plx_orient_params p = {.k = 8, .signed_ = true, .sigma_d = 1.0f, .sigma_n = 6.0f, .sigma_s = 1.5f, .eps = 0.004f};
    return p;
}

static float orient_period(const plx_orient_params *p) { return p->signed_ ? 2.0f * PI_F : PI_F; }

float plx_orient_bin_center(const plx_orient_params *p, int j) {
    PLX_ASSERT(j >= 0 && j < p->k);
    return 0.5f * PI_F + (float)j * orient_period(p) / (float)p->k;
}

void plx_orient_perm(const plx_orient_params *p, int *perm) {
    for (int j = 0; j < p->k; j++) perm[j] = (p->k - j) % p->k;
}

void plx_orient_mirror(const plx_img *feat, const plx_orient_params *p, plx_img *dst) {
    PLX_ASSERT(feat->c == p->k);
    int *perm = PLX_NEW(int, p->k);
    plx_orient_perm(p, perm);
    plx_flip_h_perm(feat, perm, dst);
    free(perm);
}

plx_img plx_orient_channels(const plx_img *gray, const plx_orient_params *p) {
    PLX_ASSERT(gray->c == 1 && p->k >= 2 && p->sigma_d > 0.0f && p->eps >= 0.0f);
    int w = gray->w, h = gray->h, K = p->k;
    size_t n = (size_t)w * h;
    plx_img gx = plx_img_new(w, h, 1), gy = plx_img_new(w, h, 1), mag = plx_img_new(w, h, 1);
    plx_gradient_gauss(gray, p->sigma_d, &gx, &gy);
    for (size_t i = 0; i < n; i++) mag.px[i] = sqrtf(gx.px[i] * gx.px[i] + gy.px[i] * gy.px[i]);

    /* denominator: local mean magnitude + eps (or 1 when normalisation is off) */
    plx_img den = plx_img_new(w, h, 1);
    if (p->sigma_n > 0.0f) {
        plx_blur(&mag, &den, p->sigma_n);
        for (size_t i = 0; i < n; i++) den.px[i] += p->eps;
    } else {
        plx_img_fill(&den, 1.0f);
    }

    /* Bin coordinate u = (theta - pi/2) K / P lives on the circle [0, K); measuring from pi/2
     * is what makes the mirror theta -> pi - theta the clean reflection u -> -u. */
    plx_img out = plx_img_new(w, h, K);
    float scale = (float)K / orient_period(p);
    for (size_t i = 0; i < n; i++) {
        if (mag.px[i] == 0.0f) continue; /* no direction; contributes nothing */
        float v = den.px[i] > 0.0f ? mag.px[i] / den.px[i] : 0.0f;
        float u = (atan2f(gy.px[i], gx.px[i]) - 0.5f * PI_F) * scale;
        u -= (float)K * floorf(u / (float)K);
        int j0 = (int)u;
        if (j0 >= K) j0 = K - 1; /* u rounded up to K */
        float f = PLX_CLAMP(u - (float)j0, 0.0f, 1.0f);
        int j1 = j0 + 1 == K ? 0 : j0 + 1;
        out.px[(size_t)j0 * n + i] += (1.0f - f) * v;
        out.px[(size_t)j1 * n + i] += f * v;
    }
    if (p->sigma_s > 0.0f) plx_blur(&out, &out, p->sigma_s);

    plx_img_free(&gx);
    plx_img_free(&gy);
    plx_img_free(&mag);
    plx_img_free(&den);
    return out;
}
