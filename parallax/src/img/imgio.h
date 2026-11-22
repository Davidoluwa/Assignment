/* Format-agnostic image load/save.
 *
 * Loading sniffs the content, not the name: JPEG (FF D8), P5/P6 and Pf/PF. 8-bit sources load
 * in [0, 1]; PFM loads unscaled. PNG is write-only here and loads as PLX_EUNSUPPORTED.
 * Saving picks the format from the extension (case-insensitive): .png (1-4 channels), .pgm
 * (1), .ppm (3), .pfm (1 or 3). 8-bit formats take [0, 1] and clamp; a channel count the
 * format cannot hold is PLX_EINVAL, an unknown extension PLX_EUNSUPPORTED. */
#ifndef PLX_IMGIO_H
#define PLX_IMGIO_H

#include "img/image.h"

int plx_img_decode(const uint8_t *data, size_t len, plx_img *out); /* from memory, e.g. a webcam frame */
int plx_img_load(const char *path, plx_img *out);
int plx_img_save(const char *path, const plx_img *im);

#endif
