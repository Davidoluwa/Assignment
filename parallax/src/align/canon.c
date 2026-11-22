/* Canonical frame geometry: similarity algebra and the warp into Ω+. */
#include "align/canon.h"

#include <math.h>
#include <string.h>

plx_sim plx_sim_identity(void) { return (plx_sim){1.0, 0.0, 0.0, 0.0}; }

plx_sim plx_sim_make(double s, double rho, double tx, double ty) {
    return (plx_sim){s * cos(rho), s * sin(rho), tx, ty};
}

double plx_sim_scale(plx_sim T) { return hypot(T.a, T.b); }
double plx_sim_angle(plx_sim T) { return atan2(T.b, T.a); }

void plx_sim_apply(plx_sim T, double x, double y, double *ox, double *oy) {
    double X = x - PLX_CANON_CX, Y = y - PLX_CANON_CY;
    *ox = T.a * X - T.b * Y + PLX_CANON_IMG_CX + T.tx;
    *oy = T.b * X + T.a * Y + PLX_CANON_IMG_CY + T.ty;
}

void plx_sim_unapply(plx_sim T, double ix, double iy, double *x, double *y) {
    double u = ix - PLX_CANON_IMG_CX - T.tx, v = iy - PLX_CANON_IMG_CY - T.ty;
    double d = T.a * T.a + T.b * T.b;
    *x = (T.a * u + T.b * v) / d + PLX_CANON_CX;
    *y = (-T.b * u + T.a * v) / d + PLX_CANON_CY;
}

void plx_csim_apply(plx_sim S, double x, double y, double *ox, double *oy) {
    double X = x - PLX_CANON_CX, Y = y - PLX_CANON_CY;
    *ox = S.a * X - S.b * Y + PLX_CANON_CX + S.tx;
    *oy = S.b * X + S.a * Y + PLX_CANON_CY + S.ty;
}

plx_sim plx_sim_compose(plx_sim A, plx_sim B) {
    return (plx_sim){A.a * B.a - A.b * B.b, A.a * B.b + A.b * B.a,
                     A.tx + A.a * B.tx - A.b * B.ty, A.ty + A.b * B.tx + A.a * B.ty};
}

plx_sim plx_csim_invert(plx_sim S) {
    /* S(p) = M (p - c0) + c0 + t  =>  S^-1(q) = M^-1 (q - c0) + c0 - M^-1 t */
    double d = S.a * S.a + S.b * S.b, ia = S.a / d, ib = -S.b / d;
    return (plx_sim){ia, ib, -(ia * S.tx - ib * S.ty), -(ib * S.tx + ia * S.ty)};
}

plx_sim plx_sim_twin(plx_sim T) { return (plx_sim){T.a, -T.b, -T.tx, T.ty}; }

void plx_sim_to_affine(plx_sim T, int margin, float A[6]) {
    /* q -> T(q - margin): linear part [a -b; b a], offset T(-margin, -margin) */
    double ox, oy;
    plx_sim_apply(T, -(double)margin, -(double)margin, &ox, &oy);
    A[0] = (float)T.a;  A[1] = (float)-T.b; A[2] = (float)ox;
    A[3] = (float)T.b;  A[4] = (float)T.a;  A[5] = (float)oy;
}

plx_img plx_canon_new(int channels, int margin) {
    PLX_ASSERT(margin >= 0);
    return plx_img_new(PLX_CANON_W + 2 * margin, PLX_CANON_H + 2 * margin, channels);
}

void plx_canon_warp(const plx_img *src, plx_sim T, int margin, plx_interp m, plx_img *dst, plx_img *valid) {
    PLX_ASSERT(dst->w == PLX_CANON_W + 2 * margin && dst->h == PLX_CANON_H + 2 * margin);
    float A[6];
    plx_sim_to_affine(T, margin, A);
    plx_warp_affine(src, A, m, dst, valid);
}

plx_img plx_canon_crop(const plx_img *ext, int margin) {
    PLX_ASSERT(ext->w == PLX_CANON_W + 2 * margin && ext->h == PLX_CANON_H + 2 * margin);
    plx_img out = plx_img_new(PLX_CANON_W, PLX_CANON_H, ext->c);
    for (int k = 0; k < ext->c; k++)
        for (int y = 0; y < PLX_CANON_H; y++)
            memcpy(plx_img_chan(&out, k) + (size_t)y * PLX_CANON_W,
                   plx_img_chan(ext, k) + (size_t)(y + margin) * ext->w + margin, sizeof(float) * PLX_CANON_W);
    return out;
}
