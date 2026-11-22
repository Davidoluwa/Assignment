/* Netpbm binary graymap/pixmap (P5/P6) and Portable Float Map (Pf/PF).
 *
 * Reading tolerates comments ('#' to end of line) and any whitespace between header fields and
 * maxval 1..65535 (two big-endian bytes per sample above 255); samples are scaled to [0, 1].
 * PFM is read in either byte order (sign of the scale field) and returned unscaled. PFM stores
 * rows bottom-to-top; images here are always top row first, the flip happens in this module.
 * Writing produces P5/P6 at maxval 255 and little-endian PFM. */
#ifndef PLX_PNM_H
#define PLX_PNM_H

#include "img/image.h"

int plx_pnm_decode(const uint8_t *data, size_t len, plx_img *out);
int plx_pnm_read(const char *path, plx_img *out);

/* Interleaved 8-bit pixels; c = 1 writes P5, c = 3 writes P6. */
int plx_pnm_write_u8(const char *path, const uint8_t *pixels, int w, int h, int c);
/* c = 1 writes Pf, c = 3 writes PF. */
int plx_pfm_write(const char *path, const plx_img *im);

#endif
