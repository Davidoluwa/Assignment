#include "viz/viz.h"

#include <math.h>

/* All primitives work the same way: walk the bounding box of the shape (clipped to the
 * image) and blend each pixel with alpha * coverage. Coverage is a 1-pixel-wide linear ramp
 * across the shape's boundary, based on the signed distance from the pixel centre: exact
 * area for axis-aligned edges, and visually indistinguishable from it for curved/oblique
 * ones, at a fraction of the cost of supersampling. */

static inline void blend(plx_img *im, int x, int y, const float rgb[3], float a) {
    if (a <= 0.0f) return;
    size_t plane = (size_t)im->w * im->h, i = (size_t)y * im->w + x;
    for (int k = 0; k < 3; k++) {
        float *p = &im->px[k * plane + i];
        *p += a * (rgb[k] - *p);
    }
}

typedef struct { int x0, y0, x1, y1; } box;

static bool clip_box(const plx_img *im, float fx0, float fy0, float fx1, float fy1, box *b) {
    if (!(fx0 <= fx1 && fy0 <= fy1)) return false; /* also rejects NaN coordinates */
    b->x0 = (int)PLX_MAX(floorf(fx0), 0.0f);
    b->y0 = (int)PLX_MAX(floorf(fy0), 0.0f);
    b->x1 = (int)PLX_MIN(ceilf(fx1), (float)(im->w - 1));
    b->y1 = (int)PLX_MIN(ceilf(fy1), (float)(im->h - 1));
    return b->x0 <= b->x1 && b->y0 <= b->y1;
}

static inline float ramp(float d) { return PLX_CLAMP(d + 0.5f, 0.0f, 1.0f); } /* d = signed inside-distance */

void plx_draw_line(plx_img *im, float x0, float y0, float x1, float y1, float width, const float rgb[3], float alpha) {
    PLX_ASSERT(im->c == 3);
    float hw = 0.5f * PLX_MAX(width, 1.0f), m = hw + 1.0f;
    box b;
    if (!clip_box(im, fminf(x0, x1) - m, fminf(y0, y1) - m, fmaxf(x0, x1) + m, fmaxf(y0, y1) + m, &b)) return;
    float dx = x1 - x0, dy = y1 - y0, len2 = dx * dx + dy * dy;
    /* Lines thinner than a pixel are drawn 1 px wide with proportionally reduced alpha, which
     * keeps their apparent weight right instead of letting them break up. */
    float a = alpha * PLX_MIN(width, 1.0f);
    for (int y = b.y0; y <= b.y1; y++)
        for (int x = b.x0; x <= b.x1; x++) {
            float px = (float)x - x0, py = (float)y - y0;
            float t = len2 > 0 ? PLX_CLAMP((px * dx + py * dy) / len2, 0.0f, 1.0f) : 0.0f;
            float ex = px - t * dx, ey = py - t * dy;
            blend(im, x, y, rgb, a * ramp(hw - sqrtf(ex * ex + ey * ey)));
        }
}

void plx_draw_circle(plx_img *im, float cx, float cy, float r, float width, const float rgb[3], float alpha) {
    PLX_ASSERT(im->c == 3);
    float hw = 0.5f * PLX_MAX(width, 1.0f), m = r + hw + 1.0f, a = alpha * PLX_MIN(width, 1.0f);
    box b;
    if (!clip_box(im, cx - m, cy - m, cx + m, cy + m, &b)) return;
    for (int y = b.y0; y <= b.y1; y++)
        for (int x = b.x0; x <= b.x1; x++) {
            float d = hypotf((float)x - cx, (float)y - cy);
            blend(im, x, y, rgb, a * ramp(hw - fabsf(d - r)));
        }
}

void plx_draw_disc(plx_img *im, float cx, float cy, float r, const float rgb[3], float alpha) {
    PLX_ASSERT(im->c == 3);
    float m = r + 1.0f;
    box b;
    if (!clip_box(im, cx - m, cy - m, cx + m, cy + m, &b)) return;
    for (int y = b.y0; y <= b.y1; y++)
        for (int x = b.x0; x <= b.x1; x++)
            blend(im, x, y, rgb, alpha * ramp(r - hypotf((float)x - cx, (float)y - cy)));
}

/* Length of [a, b] inside the pixel extent [p - 0.5, p + 0.5]. */
static inline float overlap(float p, float a, float b) {
    return PLX_MAX(0.0f, fminf(b, p + 0.5f) - fmaxf(a, p - 0.5f));
}

void plx_draw_rect(plx_img *im, float x0, float y0, float x1, float y1, const float rgb[3], float alpha) {
    PLX_ASSERT(im->c == 3);
    float ax = fminf(x0, x1), bx = fmaxf(x0, x1), ay = fminf(y0, y1), by = fmaxf(y0, y1);
    box b;
    if (!clip_box(im, ax - 1, ay - 1, bx + 1, by + 1, &b)) return;
    for (int y = b.y0; y <= b.y1; y++) {
        float cy = overlap((float)y, ay, by);
        for (int x = b.x0; x <= b.x1; x++) blend(im, x, y, rgb, alpha * cy * overlap((float)x, ax, bx));
    }
}

void plx_draw_rect_outline(plx_img *im, float x0, float y0, float x1, float y1, float width, const float rgb[3],
                           float alpha) {
    /* Four bands that tile the outline without overlapping, so corners are not blended twice. */
    float ax = fminf(x0, x1), bx = fmaxf(x0, x1), ay = fminf(y0, y1), by = fmaxf(y0, y1), h = 0.5f * width;
    plx_draw_rect(im, ax - h, ay - h, bx + h, ay + h, rgb, alpha);
    plx_draw_rect(im, ax - h, by - h, bx + h, by + h, rgb, alpha);
    plx_draw_rect(im, ax - h, ay + h, ax + h, by - h, rgb, alpha);
    plx_draw_rect(im, bx - h, ay + h, bx + h, by - h, rgb, alpha);
}

void plx_draw_cross(plx_img *im, float cx, float cy, float size, float width, const float rgb[3], float alpha) {
    float h = 0.5f * width;
    plx_draw_rect(im, cx - size, cy - h, cx + size, cy + h, rgb, alpha);
    plx_draw_rect(im, cx - h, cy - size, cx + h, cy - h, rgb, alpha); /* upper arm */
    plx_draw_rect(im, cx - h, cy + h, cx + h, cy + size, rgb, alpha);         /* lower arm */
}
