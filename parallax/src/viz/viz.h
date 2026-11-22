/* Looking at results: colour maps, contact sheets, drawing, text, flow fields.
 *
 * Every output is an ordinary plx_img with 3 channels (sRGB-encoded values in [0, 1], the
 * same encoding the JPEG loader produces), so it can be saved as .plxt with plx_viz_save and
 * turned into a PNG with tools/oracle/plxt.py. Coordinates follow image.h: x right, y down,
 * pixel centres at integer coordinates. Drawing functions clip silently at the borders. */
#ifndef PLX_VIZ_H
#define PLX_VIZ_H

#include "img/image.h"

/* ---- colour maps ---------------------------------------------------------------------
 * Built at run time from a few control colours given in CIELAB, interpolated linearly in
 * Lab (so lightness L* changes linearly along each map) and converted to sRGB.
 *   SEQ  sequential, dark violet -> blue -> teal -> green -> yellow, L* 16 -> 95
 *        (monotonically increasing luminance: readable in greyscale, ordered for the eye).
 *   DIV  diverging, blue -> near-white -> red, L* 32 -> 97 -> 32, centred on zero: the range
 *        used is [-m, m] with m = max(|lo|, |hi|), so 0 is always the neutral colour.
 *   GRAY linear grey ramp.
 * NaN values are painted black (no map reaches pure black, so "no data" stays visible). */
typedef enum { PLX_CMAP_SEQ = 0, PLX_CMAP_DIV = 1, PLX_CMAP_GRAY = 2 } plx_cmap;

void    plx_cmap_eval(plx_cmap map, float t, float rgb[3]);   /* t in [0, 1], clamped */
plx_img plx_viz_colormap(const float *v, int w, int h, float lo, float hi, plx_cmap map);
/* Robust range: the plo / phi quantiles (e.g. 0.01, 0.99) of the finite values.
 * No finite values -> [0, 1]; a degenerate range is widened so lo < hi. */
void    plx_viz_auto_range(const float *v, size_t n, float plo, float phi, float *lo, float *hi);
/* Convenience: channel 0 of `plane` through `map` with the 1%..99% auto range. */
plx_img plx_viz_scalar(const plx_img *plane, plx_cmap map);

/* ---- drawing on a 3-channel image ----------------------------------------------------
 * Anti-aliased by analytic coverage: a pixel is blended with weight alpha * coverage, where
 * coverage is (approximately) the fraction of the pixel square inside the shape. */
void plx_draw_line(plx_img *im, float x0, float y0, float x1, float y1, float width, const float rgb[3], float alpha);
void plx_draw_circle(plx_img *im, float cx, float cy, float r, float width, const float rgb[3], float alpha);
void plx_draw_disc(plx_img *im, float cx, float cy, float r, const float rgb[3], float alpha);
/* Axis-aligned rectangle with corners at (x0, y0) and (x1, y1); filled, or an outline of `width`. */
void plx_draw_rect(plx_img *im, float x0, float y0, float x1, float y1, const float rgb[3], float alpha);
void plx_draw_rect_outline(plx_img *im, float x0, float y0, float x1, float y1, float width, const float rgb[3],
                           float alpha);
/* "+" marker with arms of half-length `size`. */
void plx_draw_cross(plx_img *im, float cx, float cy, float size, float width, const float rgb[3], float alpha);

/* ---- bitmap text ---------------------------------------------------------------------
 * Built-in 5x7 font (plus one descender row) covering printable ASCII 32..126; any other
 * byte draws as '?'. Glyph cell: 6 x 9 pixels at scale 1 (1 px spacing, 1 px line gap),
 * each font pixel becomes a scale x scale block. (x, y) is the top-left of the first cell.
 * '\n' starts a new line. Not anti-aliased: pixels are either set to rgb or untouched. */
#define PLX_FONT_W 5
#define PLX_FONT_H 8
#define PLX_FONT_ADV 6
#define PLX_FONT_LINE 9
void plx_draw_text(plx_img *im, int x, int y, const char *s, int scale, const float rgb[3]);
void plx_text_size(const char *s, int scale, int *w, int *h);
bool plx_font_pixel(char ch, int col, int row); /* glyph bitmap lookup, for tests */

/* ---- montage -------------------------------------------------------------------------- */
typedef struct {
    int   cols;       /* <= 0: ceil(sqrt(n)) */
    int   pad;        /* pixels between cells and around the border */
    float bg[3];      /* background */
    float fg[3];      /* label colour */
    int   text_scale; /* label glyph scale (>= 1) */
} plx_montage_opts;
plx_montage_opts plx_montage_defaults(void); /* auto cols, pad 4, dark grey bg, white labels, scale 1 */
/* Grid of tiles. Every cell is as large as the largest tile; tiles are centred in their cell.
 * Tiles must have 1 or 3 channels (1 is promoted to grey RGB). labels may be NULL, and any
 * entry may be NULL; when any label is present each cell gets a text strip above its tile. */
plx_img plx_viz_montage(const plx_img *tiles, int n, const char *const *labels, const plx_montage_opts *o);

/* ---- flow fields ---------------------------------------------------------------------
 * Arrows: grid sample (i, j) of the gw x gh planes dx, dy sits at image position
 * (x0 + i * step, y0 + j * step) and is drawn as an arrow to that point + scale * (dx, dy).
 * Non-finite samples are skipped; arrows shorter than half a pixel are drawn as a dot. */
void plx_viz_flow_arrows(plx_img *im, const float *dx, const float *dy, int gw, int gh, float x0, float y0,
                         float step, float scale, const float rgb[3]);
/* Dense colour coding: hue = direction, brightness = magnitude / max_mag (clamped).
 * Direction is measured in image coordinates (y down): +x (right) = red, +y (down) =
 * yellow-green (90 deg), -x (left) = cyan, -y (up) = violet (270 deg); zero flow is black.
 * max_mag <= 0 picks the 99th percentile of the finite magnitudes. NaN -> mid grey. */
plx_img plx_viz_flow_color(const float *dx, const float *dy, int w, int h, float max_mag);
plx_img plx_viz_flow_wheel(int size); /* legend for plx_viz_flow_color (max_mag = radius) */

/* ---- misc ----------------------------------------------------------------------------- */
typedef enum { PLX_UP_NEAREST = 0, PLX_UP_BILINEAR = 1 } plx_upscale_mode;
/* Integer-factor enlargement. Bilinear samples the source at ((X + 0.5) / f - 0.5, ...),
 * i.e. keeps pixel centres aligned, clamping at the borders. */
plx_img plx_viz_upscale(const plx_img *im, int factor, plx_upscale_mode mode);
plx_img plx_viz_to_rgb(const plx_img *im); /* 1 -> 3 channel promotion (3 is cloned) */
/* F32 .plxt: [H, W] for one channel, else [C, H, W]. View with tools/oracle/plxt.py. */
int plx_viz_save(const char *path, const plx_img *im);

#endif
