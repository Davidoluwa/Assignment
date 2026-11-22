#include "img/imgio.h"

#include "img/jpeg.h"
#include "img/png.h"
#include "img/pnm.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>

int plx_img_decode(const uint8_t *data, size_t len, plx_img *out) {
    if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) return plx_jpeg_decode(data, len, out);
    if (len >= 2 && data[0] == 'P' && data[1] && strchr("56fF", data[1])) return plx_pnm_decode(data, len, out);
    return PLX_EUNSUPPORTED;
}

int plx_img_load(const char *path, plx_img *out) {
    uint8_t *data;
    size_t len;
    int st = plx_read_file(path, &data, &len);
    if (st < 0) return st;
    st = plx_img_decode(data, len, out);
    free(data);
    return st;
}

int plx_img_save(const char *path, const plx_img *im) {
    const char *dot = strrchr(path, '.');
    if (!dot || strchr(dot, '/')) return PLX_EUNSUPPORTED;
    if (strcasecmp(dot, ".pfm") == 0) return plx_pfm_write(path, im);
    bool png = strcasecmp(dot, ".png") == 0, pgm = strcasecmp(dot, ".pgm") == 0, ppm = strcasecmp(dot, ".ppm") == 0;
    if (!png && !pgm && !ppm) return PLX_EUNSUPPORTED;
    if ((png && im->c > 4) || (pgm && im->c != 1) || (ppm && im->c != 3)) return PLX_EINVAL;
    uint8_t *px = plx_malloc((size_t)im->w * im->h * im->c);
    plx_img_to_u8(im, px, 1.0f / 255.0f);
    int st = png ? plx_png_write_u8(path, px, im->w, im->h, im->c) : plx_pnm_write_u8(path, px, im->w, im->h, im->c);
    free(px);
    return st;
}
