/* PNG encoder, with its own zlib/deflate (RFC 1950/1951) and CRC-32.
 *
 * Each row gets the PNG filter (None/Sub/Up/Average/Paeth) with the smallest sum of absolute
 * signed residuals; the filtered stream is LZ77-matched with hash chains and lazy evaluation and
 * every block is emitted as whichever of dynamic Huffman, fixed Huffman or stored is smallest. */
#ifndef PLX_PNG_H
#define PLX_PNG_H

#include "core/core.h"

/* Interleaved 8-bit pixels; c = 1 (gray), 2 (gray+alpha), 3 (RGB) or 4 (RGBA). */
int plx_png_write_u8(const char *path, const uint8_t *pixels, int w, int h, int c);
/* 16-bit grayscale, native-endian input samples. */
int plx_png_write_gray16(const char *path, const uint16_t *pixels, int w, int h);

/* The building blocks, exposed for tests. Running checksums: start from crc = 0, adler = 1. */
uint32_t plx_crc32(uint32_t crc, const void *data, size_t len);
uint32_t plx_adler32(uint32_t adler, const void *data, size_t len);
/* Complete zlib stream of src[0..n); returns its length, *out is caller-freed. */
size_t plx_zlib_compress(const uint8_t *src, size_t n, uint8_t **out);

#endif
