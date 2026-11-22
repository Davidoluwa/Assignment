/* Canonical texture from a similarity alignment. */
#include "rec/rec.h"

#include "ops/ops.h"

#include <math.h>

void plx_rec_sim_affine(const float sim[4], float A[6]) {
    float a = sim[0], b = sim[1];
    A[0] = a;
    A[1] = -b;
    A[2] = -a * PLX_CANON_CX + b * PLX_CANON_CY + 124.5f + sim[2];
    A[3] = b;
    A[4] = a;
    A[5] = -b * PLX_CANON_CX - a * PLX_CANON_CY + 128.5f + sim[3];
}

void plx_rec_texture_sim(const plx_img *gray, const float sim[4], plx_img *tex, plx_img *conf) {
    PLX_ASSERT(gray->c == 1);
    PLX_ASSERT(tex->w == PLX_CANON_W && tex->h == PLX_CANON_H && tex->c == 1);
    PLX_ASSERT(conf->w == PLX_CANON_W && conf->h == PLX_CANON_H && conf->c == 1);
    float A[6];
    plx_rec_sim_affine(sim, A);
    /* One canonical pixel spans s image pixels. Resampling a band-limited-to-1-pixel image on
     * an s-times coarser grid aliases unless it is first blurred to the new Nyquist limit; the
     * extra blur needed on top of the ~0.5 px the camera already gives is 0.5 sqrt(s^2 - 1). */
    double s = hypot(sim[0], sim[1]);
    float sigma = s > 1.0 ? (float)(0.5 * sqrt(s * s - 1.0)) : 0.0f;
    if (sigma > 0.25f) {
        plx_img blurred = plx_img_new(gray->w, gray->h, 1);
        plx_blur(gray, &blurred, sigma);
        plx_warp_affine(&blurred, A, PLX_INTERP_BICUBIC, tex, conf);
        plx_img_free(&blurred);
    } else {
        plx_warp_affine(gray, A, PLX_INTERP_BICUBIC, tex, conf);
    }
}
