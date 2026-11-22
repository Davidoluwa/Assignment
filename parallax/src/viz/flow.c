#include "viz/viz.h"

#include <math.h>
#include <stdlib.h>

void plx_viz_flow_arrows(plx_img *im, const float *dx, const float *dy, int gw, int gh, float x0, float y0,
                         float step, float scale, const float rgb[3]) {
    for (int j = 0; j < gh; j++)
        for (int i = 0; i < gw; i++) {
            float u = dx[(size_t)j * gw + i] * scale, v = dy[(size_t)j * gw + i] * scale;
            if (!isfinite(u) || !isfinite(v)) continue;
            float px = x0 + i * step, py = y0 + j * step, len = hypotf(u, v);
            if (len < 0.5f) {
                plx_draw_disc(im, px, py, 0.8f, rgb, 1.0f);
                continue;
            }
            float tx = px + u, ty = py + v;
            plx_draw_line(im, px, py, tx, ty, 1.0f, rgb, 1.0f);
            /* Head: two barbs at +/-25 degrees, a third of the shaft long but at least 2 px
             * (readable) and at most 6 px (does not swamp neighbouring arrows). */
            float hl = PLX_CLAMP(0.35f * len, 2.0f, 6.0f), ux = -u / len, uy = -v / len;
            const float c = 0.9063078f, s = 0.4226183f; /* cos, sin of 25 degrees */
            plx_draw_line(im, tx, ty, tx + hl * (c * ux - s * uy), ty + hl * (s * ux + c * uy), 1.0f, rgb, 1.0f);
            plx_draw_line(im, tx, ty, tx + hl * (c * ux + s * uy), ty + hl * (-s * ux + c * uy), 1.0f, rgb, 1.0f);
        }
}

/* Hue in degrees [0, 360), full saturation, value v. */
static void hsv(float hue, float v, float *r, float *g, float *b) {
    float h6 = hue / 60.0f;
    int sector = (int)floorf(h6);
    float f = h6 - (float)sector, q = v * (1 - f), t = v * f;
    switch (((sector % 6) + 6) % 6) {
    case 0: *r = v; *g = t; *b = 0; break;
    case 1: *r = q; *g = v; *b = 0; break;
    case 2: *r = 0; *g = v; *b = t; break;
    case 3: *r = 0; *g = q; *b = v; break;
    case 4: *r = t; *g = 0; *b = v; break;
    default: *r = v; *g = 0; *b = q; break;
    }
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

plx_img plx_viz_flow_color(const float *dx, const float *dy, int w, int h, float max_mag) {
    size_t n = (size_t)w * h;
    if (max_mag <= 0) {
        float *m = plx_malloc(sizeof(float) * n);
        size_t k = 0;
        for (size_t i = 0; i < n; i++) {
            float mag = hypotf(dx[i], dy[i]);
            if (isfinite(mag)) m[k++] = mag;
        }
        if (k) {
            qsort(m, k, sizeof(float), cmp_float);
            max_mag = m[(size_t)(0.99 * (double)(k - 1))];
        }
        free(m);
        if (!(max_mag > 0)) max_mag = 1.0f; /* all-zero or no data: any scale shows black */
    }
    plx_img out = plx_img_new(w, h, 3);
    float *R = out.px, *G = out.px + n, *B = out.px + 2 * n;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(dx[i]) || !isfinite(dy[i])) { R[i] = G[i] = B[i] = 0.5f; continue; }
        float hue = atan2f(dy[i], dx[i]) * (180.0f / (float)M_PI);
        if (hue < 0) hue += 360.0f;
        float v = PLX_MIN(hypotf(dx[i], dy[i]) / max_mag, 1.0f);
        hsv(hue, v, &R[i], &G[i], &B[i]);
    }
    return out;
}

plx_img plx_viz_flow_wheel(int size) {
    PLX_ASSERT(size >= 3);
    size_t n = (size_t)size * size;
    float *dx = plx_malloc(sizeof(float) * n), *dy = plx_malloc(sizeof(float) * n), c = 0.5f * (size - 1);
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            size_t i = (size_t)y * size + x;
            dx[i] = (float)x - c;
            dy[i] = (float)y - c;
            if (hypotf(dx[i], dy[i]) > c) dx[i] = dy[i] = NAN; /* outside the disc */
        }
    plx_img out = plx_viz_flow_color(dx, dy, size, size, c);
    free(dx);
    free(dy);
    return out;
}
