/* Recognition front half: canonical texture -> photometric normalisation -> cell-histogram
 * descriptor with per-cell confidence.
 *
 * The seam that matters: everything after "where the texture comes from" takes a canonical
 * texture (PLX_REC_TEX_W x PLX_REC_TEX_H, 1 channel, intensities ~[0, 1]) plus a confidence plane of
 * the same size (1 = fully trusted, 0 = no evidence). Today the texture comes from a similarity
 * (plx_rec_texture_sim) and the confidence is the warp's in-image mask; later rungs hand in a
 * densely warped texture and a visibility map through exactly the same plx_desc_compute call.
 *
 * Conventions are those of img/image.h and ops/ops.h; the canonical frame is DESIGN.md section 2. */
#ifndef PLX_REC_H
#define PLX_REC_H

#include "feat/feat.h"
#include "img/image.h"

#include <stdbool.h>
#include <stdint.h>

#define PLX_REC_TEX_W 128
#define PLX_REC_TEX_H 160
#define PLX_REC_C0X 63.5f
#define PLX_REC_C0Y 79.5f

/* ---- texture ------------------------------------------------------------------------- */
#define PLX_REC_FLOW_DOF (2 * 21 * 17) /* one field of flow/u.plxt */
/* 2x3 affine (plx_warp_affine form) of the similarity sim = (a, b, tx, ty):
 * T(p) = [a -b; b a](p - c0) + (124.5 + tx, 128.5 + ty), canonical -> image pixels. */
void plx_rec_sim_affine(const float sim[4], float A[6]);

/* Canonical texture of a 250x250 gray image under sim. tex and conf must be preallocated
 * PLX_REC_TEX_W x PLX_REC_TEX_H x 1. Bicubic resampling; when the similarity shrinks the image
 * (scale |(a, b)| > 1) the source is pre-blurred so the texture is not aliased. conf gets the
 * in-image mask (1 where the source point lies inside the photo, else 0). */
void plx_rec_texture_sim(const plx_img *gray, const float sim[4], plx_img *tex, plx_img *conf);

/* Canonical texture through the dense flow, W(p) = T(p + alpha u(p)) (align/flow.h's rendering
 * path, bicubic). u: one field of flow/u.plxt, 2 x 21 x 17 floats on the 8-px lattice
 * (component, row, column). alpha = 1 is the full frontalising warp, alpha = 0 the similarity
 * alone; in between keeps part of the person's own in-plane shape. conf as above. */
void plx_rec_texture_flow(const plx_img *gray, const float sim[4], const float *u, float alpha, plx_img *tex,
                          plx_img *conf);

/* Flow-only confidence. jdet (PLX_REC_TEX_W x PLX_REC_TEX_H) gets det(I + alpha grad u) at every
 * canonical pixel: the photo area behind one canonical pixel relative to the similarity's. Below
 * 1 the warp is stretching few photo pixels over many canonical ones (the far side of a turned
 * face), so the texture there is interpolated rather than observed.
 * plx_rec_conf_from_jdet multiplies conf in place by clamp(jdet, 0, 1)^power. */
void plx_rec_flow_jdet(const float *u, float alpha, plx_img *jdet);
void plx_rec_conf_from_jdet(const plx_img *jdet, float power, plx_img *conf);

/* ---- photometric normalisation (Tan & Triggs chain) -----------------------------------
 *   1. gamma:  v <- max(v, 0)^gamma              (gamma <= 0 or == 1: skipped)
 *   2. DoG:    v <- blur(v, s0) - blur(v, s1)     (s1 <= 0: skipped; s0 <= 0 means no inner blur)
 *   3. contrast equalisation, two robust passes over confidence-weighted means:
 *        v <- v / mean(|v|^alpha)^(1/alpha);  v <- v / mean(min(tau, |v|)^alpha)^(1/alpha)
 *   4. v <- tau tanh(v / tau)                     (tau <= 0: steps 3-4 skipped)
 * After steps 2-4 the result is invariant to a contrast gain on the input exactly (up to float
 * rounding), and, with gamma skipped, to a brightness offset too. conf (nullable) weights the
 * means in step 3 so pixels with no evidence do not set the contrast scale. In place. */
typedef struct {
    float gamma;
    float dog_s0, dog_s1;
    float ce_alpha, ce_tau;
} plx_photo_params;
plx_photo_params plx_photo_default(void); /* gamma 0.2, DoG 1 / 2, alpha 0.1, tau 10 */
plx_photo_params plx_photo_none(void);    /* every step skipped */
void plx_photo_normalize(plx_img *tex, const plx_img *conf, const plx_photo_params *p);

/* ---- descriptor -----------------------------------------------------------------------
 * A descriptor is a list of blocks. Each block computes one per-pixel feature on the texture at
 * pyramid level `level` (0 = full resolution, 1 = half, via plx_downsample2) and histograms it
 * over a grid of square-ish cells of `cell` level-pixels covering the face region:
 *   LBP1 / LBP2  uniform LBP(8, 1) / LBP(8, 2) codes (59 bins) of the photometrically
 *                normalised texture;
 *   ORI          energy of each orientation channel (feat's plx_orient_channels, K bins) of the
 *                raw texture: those channels carry their own local contrast normalisation.
 * Every pixel's contribution is multiplied by its confidence. Each cell is L1-normalised, then
 * optionally square-rooted (Hellinger mapping, so the dot product of two cells is their
 * Bhattacharyya coefficient). Per-cell weights are the cell's mean confidence in [0, 1].
 * With `mirror`, each cell's L1 histogram is averaged with that of the mirrored texture (same
 * cell index = mirror-image location) before the sqrt: the descriptor of a face and its mirror
 * twin are then identical. With `mirror_weighted` the two are mixed in proportion to their cell
 * confidences (and the pooled cell weight is the larger of the two), so a stretched or hidden
 * half-face is filled from its visible mirror partner.
 * Descriptor layout: blocks in order, within a block plx_cell_grid's (cell, bin) order. */
typedef enum { PLX_BLOCK_LBP1 = 0, PLX_BLOCK_LBP2, PLX_BLOCK_ORI } plx_block_kind;

#define PLX_DESC_MAX_BLOCKS 8
typedef struct {
    plx_block_kind kind;
    int level; /* 0 or 1 */
    int cell;  /* cell side in level pixels */
} plx_block_spec;

typedef struct {
    int x0, y0, w, h;           /* face region in canonical level-0 pixels; even values */
    int n_blocks;
    plx_block_spec block[PLX_DESC_MAX_BLOCKS];
    bool sqrt_hist;             /* Hellinger mapping of each L1-normalised cell */
    bool mirror;                /* pool each cell with its mirror-twin cell (needs 2 x0 + w = 128) */
    bool mirror_weighted;       /* pool by the two cells' confidences instead of averaging */
    plx_photo_params photo;     /* applied to the texture the LBP blocks see */
    plx_orient_params orient;   /* ORI blocks */
} plx_desc_params;

/* Parse a block list "lbp1:8,lbp2@1:8,ori:10" (kind[@level]:cell) into p. PLX_EINVAL on a
 * malformed spec (the log says why). */
int plx_desc_parse_blocks(const char *spec, plx_desc_params *p);
/* Canonical one-line description of p (blocks, region, photometric and orientation settings):
 * equal strings mean equal descriptors, so it keys the descriptor cache. */
void plx_desc_describe(const plx_desc_params *p, char *buf, size_t cap);

/* Shapes derived from params. */
typedef struct {
    int n_blocks;
    int d, n_cells;                       /* total dims and total cells */
    int off[PLX_DESC_MAX_BLOCKS];         /* first dim of each block */
    int dim[PLX_DESC_MAX_BLOCKS];         /* dims of each block */
    int cell_off[PLX_DESC_MAX_BLOCKS];    /* first cell of each block */
    int cells_x[PLX_DESC_MAX_BLOCKS], cells_y[PLX_DESC_MAX_BLOCKS];
    int bins[PLX_DESC_MAX_BLOCKS];        /* dims per cell */
} plx_desc_layout;
plx_desc_layout plx_desc_layout_of(const plx_desc_params *p);
/* Seam for non-texture codes (shape / depth coefficients from the geometry stage): appends a
 * pseudo-block of `dim` dimensions with one cell, so the caller can append those columns to each
 * descriptor row (and one weight to its cell weights: the code's confidence) and every matcher
 * treats the code as one more block (its own PCA-whitening group and vote, the supervised stage
 * sees it alongside the texture). plx_desc_compute never touches it. */
void plx_desc_layout_append_code(plx_desc_layout *L, int dim);

/* Descriptor of one canonical texture. tex: PLX_REC_TEX_W x PLX_REC_TEX_H x 1 (not modified).
 * conf: same shape, values in [0, 1], or NULL for uniform confidence. desc gets layout.d floats,
 * cell_w (nullable) layout.n_cells floats. Thread-safe; allocates its own scratch. */
void plx_desc_compute(const plx_desc_params *p, const plx_img *tex, const plx_img *conf, float *desc,
                      float *cell_w);

#endif
