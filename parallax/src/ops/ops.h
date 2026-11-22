/* Generic image operations on plx_img: filtering, pyramids, sub-pixel sampling, warps.
 *
 * Conventions (on top of img/image.h: planar float32, pixel centres at integer coordinates,
 * x right / y down):
 *   - Every operation acts on each channel independently unless stated otherwise.
 *   - Border: everything uses REPLICATE (clamp-to-edge) extension. Pixel (x, y) outside the
 *     image reads pixel (clamp(x, 0, w-1), clamp(y, 0, h-1)). Filters, pyramids, resize and the
 *     samplers all see the same extended image, and that extension is left-right symmetric, so
 *     every op here commutes exactly with plx_flip_h (up to float rounding).
 *   - Same-shape outputs go into a caller-allocated `dst` (shape asserted) so hot loops can
 *     reuse buffers; `dst` may alias `src` wherever that is noted as "alias-safe".
 *     Shape-changing ops (downsample, resize, pyramid) return a new image.
 *   - All functions are thread-safe and reentrant: scratch memory is per call, no globals. */
#ifndef PLX_OPS_H
#define PLX_OPS_H

#include "img/image.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/* ---- elementwise helpers ------------------------------------------------------------ */
bool  plx_img_same_shape(const plx_img *a, const plx_img *b);
void  plx_img_copy(plx_img *dst, const plx_img *src);            /* shapes must match */
void  plx_img_fill(plx_img *im, float v);
void  plx_img_scale_offset(plx_img *im, float a, float b);       /* v <- a*v + b, in place */
float plx_img_max_abs_diff(const plx_img *a, const plx_img *b);  /* shapes must match */
plx_img plx_img_extract(const plx_img *im, int k);               /* channel k as a new 1-channel image */

/* ---- Gaussian blur --------------------------------------------------------------------
 * Sampled Gaussian exp(-i^2 / 2 sigma^2), radius ceil(3 sigma), normalised to sum 1.
 * (Sampled, not pixel-integrated: its variance is sigma^2 to <0.1% for sigma >= 0.8.) */
int  plx_gauss_radius(float sigma);                  /* ceil(3 sigma), >= 1 */
void plx_gauss_kernel(float sigma, float *k);        /* writes 2*radius+1 taps, k[radius] = centre */
/* Separable blur, per channel, replicate border. sigma <= 0 copies. Alias-safe (dst may be src). */
void plx_blur(const plx_img *src, plx_img *dst, float sigma);

/* ---- derivative filters ---------------------------------------------------------------
 * gx = d/dx, gy = d/dy (y down), per channel, replicate border; gx/gy must have src's shape.
 * Both filters are normalised so a linear ramp a*x + b*y gives exactly (a, b) away from the
 * border, and the Gaussian one is exact for any quadratic (its derivative taps are
 * antisymmetric and the smoothing taps symmetric). Alias-safe for gx == src or gy == src
 * is NOT supported: gx, gy and src must be three distinct images. */
void plx_gradient_gauss(const plx_img *src, float sigma, plx_img *gx, plx_img *gy); /* sigma > 0 */
void plx_gradient_scharr(const plx_img *src, plx_img *gx, plx_img *gy);  /* [-1 0 1]/2 x [3 10 3]/16 */

/* ---- pyramid / resize -----------------------------------------------------------------
 * REGISTRATION (pixel-area convention). Pixel i at level L+1 covers pixels 2i and 2i+1 at
 * level L, so its centre sits at 2i + 0.5:
 *     x_{L+1} = (x_L - 0.5) / 2           x_L = 2 x_{L+1} + 0.5
 *     x_L     = (x_0 + 0.5) / 2^L - 0.5   (same for y)
 * The anti-alias filter is the 6-tap binomial [1 5 10 10 5 1]/32 centred on 2i + 0.5
 * (std 1.12 level-L pixels; it is symmetric about the new centre, so a feature's centroid
 * maps exactly by the formula above). Output size is ((w+1)/2, (h+1)/2); for even sizes the
 * level is also exactly flip-symmetric. */
plx_img plx_downsample2(const plx_img *src);

#define PLX_PYR_MAX 16
typedef struct {
    int     n;                 /* number of levels; lev[0] is a copy of the input */
    plx_img lev[PLX_PYR_MAX];
} plx_pyramid;
/* Builds up to max_levels levels (<= PLX_PYR_MAX), stopping before a level whose smaller side
 * would be < min_size. Level 0 is not pre-blurred. */
plx_pyramid plx_pyramid_build(const plx_img *im, int max_levels, int min_size);
void        plx_pyramid_free(plx_pyramid *p);
/* Coordinate of a level-`from` point at level `to` (either direction). */
static inline float plx_pyr_coord(float x, int from, int to) {
    return ldexpf(x + 0.5f, from - to) - 0.5f;
}

/* Resize to (w, h) with pixel areas aligned: x_dst = (x_src + 0.5) * w / src.w - 0.5.
 * Per axis: exact area averaging when shrinking, bilinear (replicate border) when enlarging
 * or keeping the size (identity when the size is unchanged). */
plx_img plx_resize(const plx_img *src, int w, int h);

/* ---- sub-pixel sampling ---------------------------------------------------------------
 * The interpolant is that of the replicate-extended image (indices clamped), so it is defined
 * and smooth for any (x, y): bilinear is C0 (piecewise linear), Catmull-Rom bicubic is C1 and
 * reproduces linear ramps exactly where its 4x4 support is inside the image (and quadratics too).
 * Derivatives are the analytic derivatives of exactly the interpolant whose value is returned.
 * Where the bilinear derivative jumps (integer x or y), the one-sided derivative toward +x / +y
 * is returned. Outside the image the value is the edge value and the derivative across the
 * edge is 0, exactly as the extended interpolant implies.
 * "inside" means 0 <= x <= w-1 and 0 <= y <= h-1 (pixel centres); NaN is outside. Coordinates
 * are evaluated in float; positions up to ~1e6 are fine. */
typedef enum { PLX_INTERP_BILINEAR = 0, PLX_INTERP_BICUBIC = 1 } plx_interp;

static inline bool plx_img_inside(const plx_img *im, float x, float y) {
    return x >= 0.0f && y >= 0.0f && x <= (float)(im->w - 1) && y <= (float)(im->h - 1);
}

/* Channel k at (x, y). The _grad variant also writes d/dx, d/dy (both required). */
float plx_sample1(const plx_img *im, int k, plx_interp m, float x, float y);
float plx_sample1_grad(const plx_img *im, int k, plx_interp m, float x, float y, float *dx, float *dy);

/* All C channels at one point: val[k] for k < C; if grad != NULL, grad[2k] = d/dx and
 * grad[2k+1] = d/dy of channel k. Returns plx_img_inside(im, x, y). */
bool plx_sample(const plx_img *im, plx_interp m, float x, float y, float *val, float *grad);

/* n points (x[i], y[i]): val[i*C + k], grad[(i*C + k)*2 + {0,1}] (grad may be NULL),
 * inside[i] = 0/1 (may be NULL). Same layout as n calls to plx_sample. */
void plx_sample_points(const plx_img *im, plx_interp m, int n, const float *x, const float *y,
                       float *val, float *grad, uint8_t *inside);

/* ---- warps ----------------------------------------------------------------------------
 * Pull-resampling: each destination pixel centre (x, y) reads src at a source coordinate.
 * dst must be preallocated (its size defines the output grid; channel count = src's), must
 * not alias src. If valid != NULL (1 channel, dst's size) it gets 1 where the source
 * coordinate is inside src, else 0; outside pixels still get the edge-extended value. */

/* Affine A (row-major 2x3): sx = A[0] x + A[1] y + A[2],  sy = A[3] x + A[4] y + A[5]. */
void plx_warp_affine(const plx_img *src, const float A[6], plx_interp m, plx_img *dst, plx_img *valid);
/* map: 2 channels of dst's size, channel 0 = source x, channel 1 = source y. */
void plx_warp_map(const plx_img *src, const plx_img *map, plx_interp m, plx_img *dst, plx_img *valid);

/* Affine algebra in the same 2x3 form. compose: out = A o B (apply B first, then A).
 * invert returns false (out untouched) if |det| < 1e-12. out may alias the inputs. */
void plx_affine_compose(const float A[6], const float B[6], float out[6]);
bool plx_affine_invert(const float A[6], float out[6]);

/* ---- flips / normalisation / integral images ----------------------------------------- */
/* dst(x, y, k) = src(w-1-x, y, k). Alias-safe. */
void plx_flip_h(const plx_img *src, plx_img *dst);
/* dst(x, y, k) = src(w-1-x, y, perm[k]): mirror and relabel channels in one pass.
 * perm is a permutation of [0, C). Alias-safe. */
void plx_flip_h_perm(const plx_img *src, const int *perm, plx_img *dst);

/* Per channel: mean and (population) variance, accumulated in double. */
void plx_chan_stats(const plx_img *im, int k, double *mean, double *var);
/* In place, per channel: v <- (v - mean) / sqrt(var + eps). eps > 0 keeps flat channels finite. */
void plx_normalize_meanvar(plx_img *im, float eps);

/* Summed-area table in double (float would lose ~3 digits on a 250x250 image).
 * s[k][(y)*(w+1) + x] = sum of channel k over [0, x) x [0, y). */
typedef struct {
    int     w, h, c;   /* of the source image; each table is (w+1) x (h+1) */
    double *s;
} plx_integral;
plx_integral plx_integral_build(const plx_img *im);
void         plx_integral_free(plx_integral *ii);
/* Sum of channel k over the half-open box [x0, x1) x [y0, y1), clipped to the image
 * (an empty or fully outside box sums to 0). */
double plx_integral_box(const plx_integral *ii, int k, int x0, int y0, int x1, int y1);

#endif
