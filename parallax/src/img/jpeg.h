/* JPEG decoder: baseline / extended-sequential / progressive Huffman, 8-bit, 1 or 3 components.
 *
 * Output aims to be bit-identical to libjpeg's default decompression (what PIL returns):
 * accurate integer ("islow") IDCT, "fancy" triangle-filter chroma upsampling for the 2x1, 1x2 and
 * 2x2 ratios, box replication for every other ratio, and libjpeg's fixed-point YCbCr->RGB.
 *
 * Colour space follows libjpeg's rules: a JFIF APP0 means YCbCr, else an Adobe APP14 transform
 * flag decides (0 = RGB, otherwise YCbCr), else component ids 'R','G','B' mean RGB, else YCbCr.
 * Missing DHT tables 0/1 fall back to the Annex K tables, as libjpeg-turbo does for Motion-JPEG.
 *
 * Errors: PLX_EFORMAT for malformed input, including entropy data that runs out or hits a marker
 * before the scan's last MCU (truncation is never silently zero-filled). A progressive file that
 * ends cleanly between scans without EOI decodes to whatever the received scans define.
 * PLX_EUNSUPPORTED for arithmetic coding, lossless/hierarchical, 12-bit, 2 or 4 components (CMYK),
 * DNL-defined height, or more than PLX_JPEG_MAX_PIXELS pixels. */
#ifndef PLX_JPEG_H
#define PLX_JPEG_H

#include "img/image.h"

#define PLX_JPEG_MAX_PIXELS ((int64_t)1 << 26)

/* Header only: w, h and output channel count c (1 = gray, 3 = RGB). */
int plx_jpeg_info(const uint8_t *data, size_t len, int *w, int *h, int *c);

/* Interleaved 8-bit pixels (c = 1 or 3); *pixels is caller-freed, NULL on failure. */
int plx_jpeg_decode_u8(const uint8_t *data, size_t len, int *w, int *h, int *c, uint8_t **pixels);

/* Planar float in [0, 1] (u8 / 255). */
int plx_jpeg_decode(const uint8_t *data, size_t len, plx_img *out);

#endif
