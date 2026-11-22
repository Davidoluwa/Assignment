/* Private helpers shared by the align module's solvers (congeal.c, flow.c). Not part of the
 * module's API. */
#ifndef PLX_ALIGN_INTERNAL_H
#define PLX_ALIGN_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

/* Bilinear taps of a w x h grid (w, h >= 2) at a point inside [0, w-1] x [0, h-1]; false (taps
 * untouched) outside. Points on the last row / column use the last cell with weight 1 on its
 * far edge, so the interpolant is continuous up to the border. */
typedef struct { size_t o, dy; float w00, w10, w01, w11; } plx_bil_taps;

static inline bool plx_bil_at(int w, int h, double x, double y, plx_bil_taps *t) {
    if (!(x >= 0 && y >= 0 && x <= w - 1 && y <= h - 1)) return false;
    int x0 = (int)x, y0 = (int)y;
    if (x0 > w - 2) x0 = w - 2;
    if (y0 > h - 2) y0 = h - 2;
    float fx = (float)(x - x0), fy = (float)(y - y0);
    t->o = (size_t)y0 * w + x0;
    t->dy = (size_t)w;
    t->w00 = (1 - fx) * (1 - fy);
    t->w10 = fx * (1 - fy);
    t->w01 = (1 - fx) * fy;
    t->w11 = fx * fy;
    return true;
}

static inline float plx_bil_eval(const float *p, const plx_bil_taps *t) {
    const float *q = p + t->o;
    return t->w00 * q[0] + t->w10 * q[1] + t->w01 * q[t->dy] + t->w11 * q[t->dy + 1];
}

#endif
