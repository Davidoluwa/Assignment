/* Canonical textures: from a similarity (L0 / L1) or through the dense flow (L2), and the
 * flow-derived confidence plane. */
#include "rec/rec.h"

#include "align/flow.h"
#include "align/lattice.h"
#include "ops/ops.h"

#include <stdlib.h>

#include <math.h>

void plx_rec_sim_affine(const float sim[4], float A[6]) {
    float a = sim[0], b = sim[1];
    A[0] = a;
    A[1] = -b;
    A[2] = -a * PLX_REC_C0X + b * PLX_REC_C0Y + 124.5f + sim[2];
    A[3] = b;
    A[4] = a;
    A[5] = -b * PLX_REC_C0X - a * PLX_REC_C0Y + 128.5f + sim[3];
}

/* One canonical pixel spans s image pixels. Resampling a band-limited-to-1-pixel image on an
 * s-times coarser grid aliases unless it is first blurred to the new Nyquist limit; the extra
 * blur needed on top of the ~0.5 px the camera already gives is 0.5 sqrt(s^2 - 1). Returns the
 * image to sample (gray itself, or *tmp holding the blurred copy). */
static const plx_img *antialiased(const plx_img *gray, const float sim[4], plx_img *tmp) {
    double s = hypot(sim[0], sim[1]);
    float sigma = s > 1.0 ? (float)(0.5 * sqrt(s * s - 1.0)) : 0.0f;
    if (sigma <= 0.25f) return gray;
    *tmp = plx_img_new(gray->w, gray->h, 1);
    plx_blur(gray, tmp, sigma);
    return tmp;
}

static void check_planes(const plx_img *gray, const plx_img *tex, const plx_img *conf) {
    PLX_ASSERT(gray->c == 1);
    PLX_ASSERT(tex->w == PLX_REC_TEX_W && tex->h == PLX_REC_TEX_H && tex->c == 1);
    PLX_ASSERT(conf->w == PLX_REC_TEX_W && conf->h == PLX_REC_TEX_H && conf->c == 1);
}

void plx_rec_texture_sim(const plx_img *gray, const float sim[4], plx_img *tex, plx_img *conf) {
    check_planes(gray, tex, conf);
    float A[6];
    plx_rec_sim_affine(sim, A);
    plx_img tmp = {0};
    plx_warp_affine(antialiased(gray, sim, &tmp), A, PLX_INTERP_BICUBIC, tex, conf);
    plx_img_free(&tmp);
}

void plx_rec_texture_flow(const plx_img *gray, const float sim[4], const float *u, float alpha, plx_img *tex,
                          plx_img *conf) {
    check_planes(gray, tex, conf);
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    int dof = plx_lat_dof(&L);
    float *ua = plx_malloc(sizeof(float) * (size_t)dof);
    for (int k = 0; k < dof; k++) ua[k] = alpha * u[k];
    plx_img tmp = {0};
    /* the anti-alias blur follows the similarity's scale only: the flow's own strain is local
     * (det within ~[0.3, 1.5]) and blurring by it would smear exactly where texture is scarce */
    plx_fl_render_gray(antialiased(gray, sim, &tmp), (plx_sim){sim[0], sim[1], sim[2], sim[3]}, &L, ua, 0,
                       PLX_INTERP_BICUBIC, tex, conf);
    plx_img_free(&tmp);
    free(ua);
}

void plx_rec_flow_jdet(const float *u, float alpha, plx_img *jdet) {
    PLX_ASSERT(jdet->w == PLX_REC_TEX_W && jdet->h == PLX_REC_TEX_H && jdet->c == 1);
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    /* u is bilinear inside each lattice cell; a central difference of half a pixel is exact
     * there and averages the two one-sided slopes across a cell edge */
    const double h = 0.5;
    for (int y = 0; y < PLX_REC_TEX_H; y++)
        for (int x = 0; x < PLX_REC_TEX_W; x++) {
            double xp[2], xm[2], yp[2], ym[2];
            plx_lat_eval(&L, u, x + h, y, xp);
            plx_lat_eval(&L, u, x - h, y, xm);
            plx_lat_eval(&L, u, x, y + h, yp);
            plx_lat_eval(&L, u, x, y - h, ym);
            double a = 1.0 + alpha * (xp[0] - xm[0]) / (2 * h), b = alpha * (yp[0] - ym[0]) / (2 * h);
            double c = alpha * (xp[1] - xm[1]) / (2 * h), d = 1.0 + alpha * (yp[1] - ym[1]) / (2 * h);
            PLX_PX(jdet, x, y, 0) = (float)(a * d - b * c);
        }
}

void plx_rec_conf_from_jdet(const plx_img *jdet, float power, plx_img *conf) {
    PLX_ASSERT(plx_img_same_shape(jdet, conf) && conf->c == 1);
    size_t n = (size_t)conf->w * conf->h;
    for (size_t i = 0; i < n; i++) {
        float j = PLX_CLAMP(jdet->px[i], 0.0f, 1.0f);
        conf->px[i] *= power == 1.0f ? j : powf(j, power);
    }
}

void plx_rec_jdet_mirror_ratio(plx_img *jdet) {
    PLX_ASSERT(jdet->w == PLX_REC_TEX_W && jdet->h == PLX_REC_TEX_H && jdet->c == 1);
    plx_img f = plx_img_new(jdet->w, jdet->h, 1);
    plx_flip_h(jdet, &f);
    for (size_t i = 0; i < (size_t)jdet->w * jdet->h; i++)
        jdet->px[i] = f.px[i] > 0.0f ? jdet->px[i] / f.px[i] : 0.0f;
    plx_img_free(&f);
}
