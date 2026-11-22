/* Illumination normalisation: gamma, difference of Gaussians, robust contrast equalisation
 * (Tan & Triggs, "Enhanced local texture feature sets for face recognition under difficult
 * lighting conditions", 2010). */
#include "rec/rec.h"

#include "ops/ops.h"

#include <math.h>

plx_photo_params plx_photo_default(void) {
    return (plx_photo_params){.gamma = 0.2f, .dog_s0 = 1.0f, .dog_s1 = 2.0f, .ce_alpha = 0.1f, .ce_tau = 10.0f};
}

plx_photo_params plx_photo_none(void) {
    return (plx_photo_params){.gamma = 1.0f, .dog_s0 = 0.0f, .dog_s1 = 0.0f, .ce_alpha = 0.1f, .ce_tau = 0.0f};
}

/* (sum_i w_i f(|v_i|)^alpha / sum_i w_i)^(1/alpha), f = identity or min(tau, .). The power mean
 * with a small alpha sits between the geometric mean and the median of |v|: it measures the
 * contrast scale without being dominated by the few saturated speculars or shadow edges. */
static double power_mean(const float *v, const float *w, size_t n, double alpha, double tau) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double wi = w ? w[i] : 1.0;
        if (wi <= 0.0) continue;
        double a = fabs((double)v[i]);
        if (tau > 0.0 && a > tau) a = tau;
        num += wi * pow(a, alpha);
        den += wi;
    }
    if (den <= 0.0 || num <= 0.0) return 1.0;
    return pow(num / den, 1.0 / alpha);
}

void plx_photo_normalize(plx_img *tex, const plx_img *conf, const plx_photo_params *p) {
    PLX_ASSERT(tex->c == 1);
    PLX_ASSERT(!conf || (conf->w == tex->w && conf->h == tex->h && conf->c == 1));
    size_t n = (size_t)tex->w * tex->h;
    float *v = tex->px;
    if (p->gamma > 0.0f && p->gamma != 1.0f)
        for (size_t i = 0; i < n; i++) v[i] = powf(fmaxf(v[i], 0.0f), p->gamma);
    if (p->dog_s1 > 0.0f) {
        plx_img outer = plx_img_new(tex->w, tex->h, 1);
        plx_blur(tex, &outer, p->dog_s1);
        if (p->dog_s0 > 0.0f) plx_blur(tex, tex, p->dog_s0);
        for (size_t i = 0; i < n; i++) v[i] -= outer.px[i];
        plx_img_free(&outer);
    }
    if (p->ce_tau > 0.0f) {
        const float *w = conf ? conf->px : NULL;
        /* the two passes: first a plain robust scale, then one that ignores how far beyond tau
         * the outliers go, so a few extreme pixels cannot shrink everything else */
        double s1 = power_mean(v, w, n, p->ce_alpha, 0.0);
        for (size_t i = 0; i < n; i++) v[i] = (float)(v[i] / s1);
        double s2 = power_mean(v, w, n, p->ce_alpha, p->ce_tau);
        float k = (float)(1.0 / s2), tau = p->ce_tau;
        for (size_t i = 0; i < n; i++) v[i] = tau * tanhf(v[i] * k / tau);
    }
}
