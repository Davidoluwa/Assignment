/* The image container shared by every module.
 *
 * Planar float32: channel k, row y, column x lives at px[(k*h + y)*w + x].
 * Planar (not interleaved) because most of this system treats an image as a stack of
 * independent scalar fields - feature channels, depth, flow components - not as RGB.
 *
 * Conventions used everywhere:
 *   - intensities are in [0, 1] once loaded (u8 / 255)
 *   - x grows right, y grows down, pixel centres sit at integer coordinates
 *   - a plx_img owns px; pass by pointer, free with plx_img_free */
#ifndef PLX_IMAGE_H
#define PLX_IMAGE_H

#include "core/core.h"

typedef struct {
    int    w, h, c;
    float *px;
} plx_img;

plx_img plx_img_new(int w, int h, int c); /* zero-filled */
void    plx_img_free(plx_img *im);        /* safe on a zeroed struct; zeroes it afterwards */
plx_img plx_img_clone(const plx_img *im);

static inline float *plx_img_chan(const plx_img *im, int k) { return im->px + (size_t)k * im->h * im->w; }
#define PLX_PX(im, x, y, k) ((im)->px[((size_t)(k) * (im)->h + (y)) * (im)->w + (x)])

/* u8 interleaved (RGBRGB...) <-> planar float. value = u8 * scale; back: round(clamp(v / scale)). */
plx_img plx_img_from_u8(const uint8_t *data, int w, int h, int c, float scale);
void    plx_img_to_u8(const plx_img *im, uint8_t *out, float scale);

/* Rec.601 luma from a 3-channel image; a 1-channel input is cloned. */
plx_img plx_img_luma(const plx_img *im);

#endif
