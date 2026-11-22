#include "img/image.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

plx_img plx_img_new(int w, int h, int c) {
    PLX_ASSERT(w > 0 && h > 0 && c > 0);
    plx_img im = {w, h, c, PLX_NEW(float, (size_t)w * h * c)};
    return im;
}

void plx_img_free(plx_img *im) {
    free(im->px);
    memset(im, 0, sizeof *im);
}

plx_img plx_img_clone(const plx_img *im) {
    plx_img out = plx_img_new(im->w, im->h, im->c);
    memcpy(out.px, im->px, sizeof(float) * (size_t)im->w * im->h * im->c);
    return out;
}

plx_img plx_img_from_u8(const uint8_t *data, int w, int h, int c, float scale) {
    plx_img im = plx_img_new(w, h, c);
    size_t n = (size_t)w * h;
    for (int k = 0; k < c; k++) {
        float *dst = plx_img_chan(&im, k);
        for (size_t i = 0; i < n; i++) dst[i] = (float)data[i * c + k] * scale;
    }
    return im;
}

void plx_img_to_u8(const plx_img *im, uint8_t *out, float scale) {
    size_t n = (size_t)im->w * im->h;
    float inv = 1.0f / scale;
    for (int k = 0; k < im->c; k++) {
        const float *src = plx_img_chan(im, k);
        for (size_t i = 0; i < n; i++) {
            float v = src[i] * inv;
            out[i * im->c + k] = (uint8_t)lrintf(v != v ? 0.0f : PLX_CLAMP(v, 0.0f, 255.0f)); /* NaN -> 0 */
        }
    }
}

plx_img plx_img_luma(const plx_img *im) {
    if (im->c == 1) return plx_img_clone(im);
    PLX_ASSERT(im->c == 3);
    plx_img out = plx_img_new(im->w, im->h, 1);
    size_t n = (size_t)im->w * im->h;
    const float *r = plx_img_chan(im, 0), *g = plx_img_chan(im, 1), *b = plx_img_chan(im, 2);
    for (size_t i = 0; i < n; i++) out.px[i] = 0.299f * r[i] + 0.587f * g[i] + 0.114f * b[i];
    return out;
}
