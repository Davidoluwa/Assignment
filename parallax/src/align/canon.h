/* The canonical face frame Ω and the similarity transforms that map it into photos.
 *
 * This is the geometric contract of docs/DESIGN.md §2, in code. Every stage that talks about
 * "where a canonical pixel lands in image i" goes through these definitions.
 *
 * FRAME. Ω is PLX_CANON_W x PLX_CANON_H = 128 x 160 pixels, x right, y down, pixel centres at
 * integers, centre c0 = (63.5, 79.5). The mirror is σ(x, y) = (127 - x, y): the symmetry axis
 * is x = 63.5. An extended window Ω+ pads Ω by `margin` pixels on every side; Ω+ pixel (qx, qy)
 * is canonical point (qx - margin, qy - margin). Ω+ is σ-symmetric for any margin (its flip
 * qx -> W+2m-1-qx is σ), so mirror operations work on Ω+ images unchanged.
 *
 * SIMILARITIES. One storage type, two readings:
 *   image similarity  T (Ω -> image):   T(p) = [a -b; b a](p - c0) + (124.5 + tx, 128.5 + ty)
 *   canonical similarity S (Ω -> Ω):   S(p) = [a -b; b a](p - c0) + c0 + (tx, ty)
 * a = s cos ρ, b = s sin ρ (ρ positive = clockwise on screen, since y is down). Both read the
 * input point relative to c0, so composition has one law for both: (A o B)(p) = A(B(p)) with
 * B canonical and A either kind is
 *     (a, b)_AB = (a, b)_A * (a, b)_B as complex numbers,   t_AB = t_A + [a -b; b a]_A t_B.
 * Identity is (1, 0, 0, 0) under both readings. A canonical similarity has a canonical inverse;
 * an image similarity is inverted pointwise (plx_sim_unapply).
 *
 * MIRROR TWIN. If T aligns image I (I(T(p)) ~ μ(p) for a σ-symmetric template μ), then the
 * horizontally flipped image I'(x, y) = I(249 - x, y) is aligned by
 *     twin(T) = (a, -b, -tx, ty),
 * because flip o T o σ = twin(T) exactly (flip is x -> 249 - x, 124.5 is the crop centre).
 * The twin of a canonical similarity S under σ (σ o S o σ) has the same formula. */
#ifndef PLX_ALIGN_CANON_H
#define PLX_ALIGN_CANON_H

#include "img/image.h"
#include "ops/ops.h"

#define PLX_CANON_W 128
#define PLX_CANON_H 160
#define PLX_CANON_CX 63.5   /* c0 */
#define PLX_CANON_CY 79.5
#define PLX_CANON_IMG_CX 124.5 /* T(c0) at tx = ty = 0: slightly below the 250x250 crop centre */
#define PLX_CANON_IMG_CY 128.5
#define PLX_CANON_IMG_SIDE 250 /* LFW crop size; the twin rule's flip is x -> SIDE - 1 - x */
/* Default Ω+ margin. The congealing stage works on Ω+ with it, and its saved template
 * (congeal/template_mean.plxt, template_var.plxt: f32 [K, 192, 160]) covers Ω+, not Ω:
 * template pixel (qx, qy) is canonical point (qx - 16, qy - 16). */
#define PLX_CANON_MARGIN 16

static inline double plx_canon_mirror_x(double x) { return (double)(PLX_CANON_W - 1) - x; }

typedef struct { double a, b, tx, ty; } plx_sim;

plx_sim plx_sim_identity(void);
/* From scale s, rotation rho (radians, clockwise on screen) and translation. */
plx_sim plx_sim_make(double s, double rho, double tx, double ty);
double  plx_sim_scale(plx_sim T);  /* hypot(a, b) */
double  plx_sim_angle(plx_sim T);  /* atan2(b, a), radians */

/* Image reading: canonical p -> image, and its pointwise inverse. */
void plx_sim_apply(plx_sim T, double x, double y, double *ox, double *oy);
void plx_sim_unapply(plx_sim T, double ix, double iy, double *x, double *y);
/* Canonical reading: Ω -> Ω. */
void plx_csim_apply(plx_sim S, double x, double y, double *ox, double *oy);

plx_sim plx_sim_compose(plx_sim A, plx_sim B);  /* A o B; B must be a canonical similarity */
plx_sim plx_csim_invert(plx_sim S);             /* inverse of a canonical similarity */
plx_sim plx_sim_twin(plx_sim T);                /* (a, -b, -tx, ty), see header comment */

/* The 2x3 affine plx_warp_affine expects for pulling an Ω+ image (with `margin`) through T:
 * destination pixel q maps to image point T(q - margin). */
void plx_sim_to_affine(plx_sim T, int margin, float A[6]);

/* Ω+ image of the given channel count, zero-filled: (128 + 2 margin) x (160 + 2 margin). */
plx_img plx_canon_new(int channels, int margin);
/* dst (from plx_canon_new, any channel count = src's) = src warped through T into Ω+.
 * valid (optional, 1 channel, dst's size): 1 where the source point is inside src. */
void plx_canon_warp(const plx_img *src, plx_sim T, int margin, plx_interp m, plx_img *dst, plx_img *valid);
/* The Ω part (128 x 160) of an Ω+ image, as a new image. */
plx_img plx_canon_crop(const plx_img *ext, int margin);

#endif
