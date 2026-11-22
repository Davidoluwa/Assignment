/* Features and descriptors: orientation channels (the alignment feature), LBP codes, and
 * weighted cell-histogram descriptors (the recognition feature).
 * Conventions are those of ops/ops.h (planar float, replicate border, thread-safe, no globals). */
#ifndef PLX_FEAT_H
#define PLX_FEAT_H

#include "img/image.h"
#include "ops/ops.h"

#include <stdbool.h>
#include <stdint.h>

/* ---- orientation channels --------------------------------------------------------------
 * Input: 1-channel image (intensities in [0, 1]). Output: K-channel image of the same size.
 *   1. (gx, gy) = Gaussian-derivative gradient at sigma_d; m = |g|; theta = atan2(gy, gx),
 *      i.e. the direction of increasing intensity, measured from +x towards +y (y down).
 *   2. m_n = m / (blur(m, sigma_n) + eps): edge strength relative to its neighbourhood, so a
 *      channel is unchanged by a brightness offset (exactly) and by a contrast scaling (up to
 *      the eps floor, which keeps flat noise from being amplified). sigma_n <= 0: m_n = m.
 *   3. m_n is split between the two circularly-nearest bin centres by linear interpolation.
 *   4. each channel is blurred by sigma_s (sigma_s <= 0: no blur).
 *
 * BIN LAYOUT. Period P = 2 pi (signed) or pi (unsigned: theta and theta + pi coincide, so the
 * feature ignores contrast polarity). Bin j (0 <= j < K) is centred at
 *     c_j = pi/2 + j P / K,
 * so bin 0 is "intensity increasing downward" (a horizontal edge), and bins advance with theta.
 * MIRROR: flipping the image left-right maps gx -> -gx, gy -> gy, i.e. theta -> pi - theta,
 * i.e. (theta - pi/2) -> -(theta - pi/2). The layout is symmetric about pi/2, so c_j maps onto
 * c_{(K-j) mod K} exactly, for any K and both modes:
 *     perm[j] = (K - j) mod K        (an involution; bin 0, and bin K/2 for even K, are fixed)
 * and features(flip_h(I)) == flip_h_perm(features(I), perm) up to float rounding. */
typedef struct {
    int   k;        /* number of bins, >= 2 */
    bool  signed_;  /* true: span 2 pi; false: span pi */
    float sigma_d;  /* derivative scale (pixels) */
    float sigma_n;  /* contrast-normalisation neighbourhood (pixels) */
    float sigma_s;  /* final spatial blur of each channel (pixels) */
    float eps;      /* normalisation floor, in gradient units of [0,1] intensities per pixel */
} plx_orient_params;

/* k = 8, signed, sigma_d = 1.0, sigma_n = 6.0, sigma_s = 1.5, eps = 0.004. For scale: a step
 * of 0.1 smoothed at sigma_d = 1 peaks at |g| = 0.04; on LFW faces the local mean |g| at
 * sigma_n = 6 has median ~0.025 and 5th percentile ~0.01, so this eps stays below real facial
 * structure (halving the contrast changes typical values by ~12%) while still capping the gain
 * on flat, noise-only areas at 1/eps. */
plx_orient_params plx_orient_default(void);
float   plx_orient_bin_center(const plx_orient_params *p, int j); /* c_j in radians */
plx_img plx_orient_channels(const plx_img *gray, const plx_orient_params *p);
void    plx_orient_perm(const plx_orient_params *p, int *perm);  /* perm[0..k) */
/* dst = flip_h of feat with channels relabelled by perm: the feature image of the mirrored
 * input. Alias-safe. */
void    plx_orient_mirror(const plx_img *feat, const plx_orient_params *p, plx_img *dst);

/* ---- LBP -------------------------------------------------------------------------------
 * 8 neighbours g_p of each centre g_c; bit p of the code is set when g_p >= g_c.
 * Neighbour p sits at angle 2 pi p / 8, counter-clockwise as displayed (p = 0 east, 2 north,
 * 4 west, 6 south). radius 1: the 3x3 square ring (diagonals at (+-1, +-1), no interpolation).
 * radius 2: the circle of radius 2, diagonals at (+-sqrt2, +-sqrt2) bilinearly interpolated.
 * Neighbours outside the image read the replicate-extended image. codes[y*w + x]. */
void plx_lbp(const plx_img *gray, int k, int radius, uint8_t *codes);

/* Uniform patterns (<= 2 circular 0/1 transitions): the 58 uniform codes get labels 0..57 in
 * increasing code order, every other code gets 58. Returns the label count, 59. */
#define PLX_LBP_UNIFORM_BINS 59
int plx_lbp_uniform_table(uint8_t table[256]);

/* ---- cell-histogram descriptors ------------------------------------------------------
 * The region [x0, x0 + w) x [y0, y0 + h) is split into cells_x x cells_y cells; pixel (x, y)
 * belongs to cell ((x - x0) * cells_x / w, (y - y0) * cells_y / h) (integer division; hard
 * assignment). Region pixels outside the image contribute nothing.
 * Output out[(cy * cells_x + cx) * bins + b] (cells_x * cells_y * bins floats).
 * weight (optional, NULL = all ones): a w*h plane of the IMAGE (not the region) with the
 * per-pixel confidence; every contribution is multiplied by it. cell_weight (optional) gets
 * the per-cell sum of weights over in-image pixels, so a matcher can discount cells that
 * were mostly invisible. Normalisation is per cell, after accumulation:
 *   L1      v / (sum |v| + eps)
 *   L2      v / (sqrt(sum v^2) + eps)
 *   L1_SQRT sign(v) sqrt(|v| / (sum |v| + eps))   (Hellinger / RootSIFT: unit L2 norm) */
typedef enum { PLX_HNORM_NONE = 0, PLX_HNORM_L1, PLX_HNORM_L2, PLX_HNORM_L1_SQRT } plx_hist_norm;

typedef struct {
    int           x0, y0, w, h;      /* region, in image pixels */
    int           cells_x, cells_y;  /* each <= the region's size on that axis */
    plx_hist_norm norm;
    float         eps;               /* > 0 keeps empty cells at 0 instead of NaN */
} plx_cell_grid;

/* Hard counts of integer labels (each < bins) over a w x h label image. */
void plx_hist_labels(const uint8_t *labels, int w, int h, int bins, const float *weight,
                     const plx_cell_grid *g, float *out, float *cell_weight);
/* Soft histogram: bin b of a cell = sum over its pixels of weight * channel b (bins = chan->c),
 * e.g. the energy of each orientation channel. */
void plx_hist_channels(const plx_img *chan, const float *weight, const plx_cell_grid *g,
                       float *out, float *cell_weight);

#endif
