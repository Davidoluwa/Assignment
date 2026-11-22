#include "viz/viz.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Control colours in CIELAB (L*, a*, b*), chosen so the whole interpolated path stays inside
 * the sRGB gamut (clamping would bend the lightness ramp); tests check the resulting
 * luminance is monotonic. */
static const float seq_lab[][3] = {
    {16, 34, -40}, /* dark violet */
    {34, 20, -48}, /* blue */
    {51, -10, -26}, /* teal */
    {66, -38, 12}, /* green */
    {80, -20, 50}, /* yellow-green */
    {95, -14, 62}, /* pale yellow */
};
static const float div_lab[][3] = {
    {32, 8, -46}, /* dark blue */
    {62, 0, -33}, /* light blue */
    {97, 0, 0},   /* near-white centre */
    {62, 36, 28}, /* light red */
    {32, 44, 36}, /* dark red */
};

static float srgb_encode(float c) { /* linear -> sRGB transfer curve, IEC 61966-2-1 */
    c = PLX_CLAMP(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

static float lab_finv(float t) {
    const float d = 6.0f / 29.0f;
    return t > d ? t * t * t : 3.0f * d * d * (t - 4.0f / 29.0f);
}

static void lab_to_srgb(const float lab[3], float rgb[3]) {
    float fy = (lab[0] + 16.0f) / 116.0f, fx = fy + lab[1] / 500.0f, fz = fy - lab[2] / 200.0f;
    float X = 0.95047f * lab_finv(fx), Y = lab_finv(fy), Z = 1.08883f * lab_finv(fz); /* D65 white */
    rgb[0] = srgb_encode(3.2406f * X - 1.5372f * Y - 0.4986f * Z);
    rgb[1] = srgb_encode(-0.9689f * X + 1.8758f * Y + 0.0415f * Z);
    rgb[2] = srgb_encode(0.0557f * X - 0.2040f * Y + 1.0570f * Z);
}

static void interp_lab(const float (*cp)[3], int n, float t, float rgb[3]) {
    float u = t * (float)(n - 1);
    int i = PLX_MIN((int)u, n - 2);
    float f = u - (float)i, lab[3];
    for (int k = 0; k < 3; k++) lab[k] = cp[i][k] + f * (cp[i + 1][k] - cp[i][k]);
    lab_to_srgb(lab, rgb);
}

void plx_cmap_eval(plx_cmap map, float t, float rgb[3]) {
    t = t != t ? 0.0f : PLX_CLAMP(t, 0.0f, 1.0f);
    switch (map) {
    case PLX_CMAP_SEQ: interp_lab(seq_lab, (int)(sizeof seq_lab / sizeof seq_lab[0]), t, rgb); return;
    case PLX_CMAP_DIV: interp_lab(div_lab, (int)(sizeof div_lab / sizeof div_lab[0]), t, rgb); return;
    case PLX_CMAP_GRAY: rgb[0] = rgb[1] = rgb[2] = t; return;
    }
    plx_die("bad colormap %d", (int)map);
}

/* A 1024-entry table built per call: the Lab->sRGB conversion costs a few powf per pixel,
 * the table makes a megapixel map cost one lerp per pixel, and quantisation stays far below
 * one 8-bit level. */
#define LUT_N 1024

plx_img plx_viz_colormap(const float *v, int w, int h, float lo, float hi, plx_cmap map) {
    if (map == PLX_CMAP_DIV) {
        float m = PLX_MAX(fabsf(lo), fabsf(hi));
        lo = -m;
        hi = m;
    }
    float scale = hi > lo ? 1.0f / (hi - lo) : 0.0f;
    float (*lut)[3] = plx_malloc(sizeof(float[3]) * LUT_N);
    for (int i = 0; i < LUT_N; i++) plx_cmap_eval(map, (float)i / (LUT_N - 1), lut[i]);
    plx_img out = plx_img_new(w, h, 3);
    size_t n = (size_t)w * h;
    for (size_t i = 0; i < n; i++) {
        if (v[i] != v[i]) continue; /* NaN stays black */
        float t = (v[i] - lo) * scale * (LUT_N - 1);
        t = PLX_CLAMP(t, 0.0f, (float)(LUT_N - 1));
        int j = PLX_MIN((int)t, LUT_N - 2);
        float f = t - (float)j;
        for (int k = 0; k < 3; k++) out.px[k * n + i] = lut[j][k] + f * (lut[j + 1][k] - lut[j][k]);
    }
    free(lut);
    return out;
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

void plx_viz_auto_range(const float *v, size_t n, float plo, float phi, float *lo, float *hi) {
    float *f = plx_malloc(sizeof(float) * (n ? n : 1));
    size_t m = 0;
    for (size_t i = 0; i < n; i++)
        if (isfinite(v[i])) f[m++] = v[i];
    if (m == 0) {
        *lo = 0.0f;
        *hi = 1.0f;
    } else {
        qsort(f, m, sizeof(float), cmp_float);
        plo = PLX_CLAMP(plo, 0.0f, 1.0f);
        phi = PLX_CLAMP(phi, plo, 1.0f);
        *lo = f[(size_t)lrint(plo * (double)(m - 1))];
        *hi = f[(size_t)lrint(phi * (double)(m - 1))];
    }
    free(f);
    if (!(*hi > *lo)) { /* flat data: widen around the value so it maps mid-range */
        float d = *lo != 0.0f ? 0.01f * fabsf(*lo) : 0.5f;
        *lo -= d;
        *hi += d;
    }
}

plx_img plx_viz_scalar(const plx_img *plane, plx_cmap map) {
    float lo, hi;
    size_t n = (size_t)plane->w * plane->h;
    plx_viz_auto_range(plane->px, n, 0.01f, 0.99f, &lo, &hi);
    return plx_viz_colormap(plane->px, plane->w, plane->h, lo, hi, map);
}
