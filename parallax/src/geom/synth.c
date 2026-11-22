/* Synthetic people-photos with known geometry: the ground truth every estimate in emerge.c is
 * checked against. Shapes are exactly mirror-symmetric, projection is the exact weak-perspective
 * camera of DESIGN.md §2 (no small-angle approximation), and the flow stage's two gauges are
 * applied exactly as that stage applies them. */
#include "geom/geom.h"

#include "la/la.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DEG (M_PI / 180.0)

plx_synth_opts plx_synth_defaults(void) {
    return (plx_synth_opts){.n = 4000, .per_person = 1, .yaw_sd_deg = 15, .yaw_clip_deg = 45, .pitch_sd_deg = 8,
                            .pitch_mean_deg = 0, .pitch_clip_deg = 30, .roll_sd_deg = 5, .scale_sd = 0.05,
                            .trans_sd_px = 3, .noise_px = 0.3, .noise_yaw = 0, .nuis_px = 0.5, .nuis_rank = 4,
                            .a_px = 2.5, .b_px = 2.5, .g_px = 1.2, .ka = 6, .kb = 6, .kg = 4, .ar_sd = 0.04,
                            .nose_px = 20, .depth_tilt = 0, .twins = false, .seed = 1};
}

static double gauss2(double x, double y, double sx, double sy) {
    return exp(-0.5 * (x * x / (sx * sx) + y * y / (sy * sy)));
}

/* The nose alone: a ridge from the brows down to the tip at Y0 = +10, then a quick drop. */
static double nose_shape(double X, double Y) {
    double h = Y <= 10 ? exp(-((10 - Y) / 26) * ((10 - Y) / 26)) : exp(-((Y - 10) / 6) * ((Y - 10) / 6));
    return exp(-0.5 * X * X / 49.0) * h;
}

/* Face-like mean depth, px, Z towards the viewer: a half-ellipsoid (62 x 82 px semi-axes, 38 px
 * deep) plus nose, eye-socket dents, brow ridge, lips and chin. With nose_px = 20 the nose tip sits
 * ~25 px above the cheeks (X0 = ±30) on a face ~100 px wide. */
double plx_synth_depth(double X, double Y, double nose_px) {
    double ex = X / 62, ey = (Y - 4) / 82, e = 1 - ex * ex - ey * ey;
    double z = 38 * sqrt(e > 0 ? e : 0);
    z += nose_px * nose_shape(X, Y);
    z -= 7 * (gauss2(X - 19, Y + 15, 8, 8) + gauss2(X + 19, Y + 15, 8, 8));
    z += 3 * gauss2(X, Y + 28, 25, 5);
    z += 3 * gauss2(X, Y - 30, 12, 5);
    z += 4 * gauss2(X, Y - 48, 10, 7);
    return z;
}

/* Smooth random field: 8 Gaussian bumps inside the (slightly enlarged) core ellipse, projected to
 * the requested parity (+1 even, -1 odd, 0 none) and scaled to unit ω-RMS. */
static void random_field(const plx_glat *L, const double *omega, plx_rng *r, int parity, double *f) {
    int P = plx_glat_n(L);
    double cx[8], cy[8], sg[8], am[8];
    for (int k = 0; k < 8; k++) {
        double rad = 1.1 * sqrt(plx_rng_uniform(r)), ang = 2 * M_PI * plx_rng_uniform(r);
        cx[k] = 44 * rad * cos(ang);
        cy[k] = 6 + 54 * rad * sin(ang);
        sg[k] = 10 + 15 * plx_rng_uniform(r);
        am[k] = plx_rng_normal(r);
    }
    double *g = PLX_NEW(double, P);
    for (int n = 0; n < P; n++) {
        double X = plx_glat_X0(L, n), Y = plx_glat_Y0(L, n);
        for (int k = 0; k < 8; k++) g[n] += am[k] * gauss2(X - cx[k], Y - cy[k], sg[k], sg[k]);
    }
    double ss = 0, ws = 0;
    for (int n = 0; n < P; n++) {
        double gm = g[plx_glat_mirror(L, n)];
        f[n] = parity > 0 ? 0.5 * (g[n] + gm) : parity < 0 ? 0.5 * (g[n] - gm) : g[n];
    }
    for (int n = 0; n < P; n++) ss += omega[n] * f[n] * f[n], ws += omega[n];
    double sc = ss > 0 ? 1 / sqrt(ss / ws) : 0;
    for (int n = 0; n < P; n++) f[n] *= sc;
    free(g);
}

/* k modes with amplitudes ∝ 1 / (1 + j), total RMS `rms`. */
static double *random_basis(const plx_glat *L, const double *omega, plx_rng *r, int parity, int k, double rms) {
    int P = plx_glat_n(L);
    double *B = PLX_NEW(double, (size_t)k * P + 1), norm = 0;
    for (int j = 0; j < k; j++) norm += 1.0 / ((1.0 + j) * (1.0 + j));
    for (int j = 0; j < k; j++) {
        random_field(L, omega, r, parity, B + (size_t)j * P);
        double a = rms / (1.0 + j) / sqrt(norm);
        for (int n = 0; n < P; n++) B[(size_t)j * P + n] *= a;
    }
    return B;
}

static double clipped_normal(plx_rng *r, double mean, double sd, double clip) {
    for (;;) {
        double x = mean + sd * plx_rng_normal(r);
        if (fabs(x) <= clip) return x;
    }
}

void plx_synth_make(const plx_glat *L, const float *node_weight, const uint8_t *core, const double *gauge,
                    const plx_synth_opts *o, plx_synth *out) {
    PLX_ASSERT(o->n >= 1 && o->per_person >= 1 && o->kg >= 1);
    int P = plx_glat_n(L), n0 = o->n, n = o->twins ? 2 * n0 : n0, kg = o->kg;
    memset(out, 0, sizeof *out);
    out->n = n, out->P = P, out->kg = kg;
    double *omega = PLX_NEW(double, P);
    for (int k = 0; k < P; k++) omega[k] = core[k] ? (node_weight ? node_weight[k] : 1.0) : 0.0;

    plx_rng r;
    plx_rng_seed(&r, o->seed);
    out->zbar = PLX_NEW(double, P);
    for (int k = 0; k < P; k++) {
        double Y = plx_glat_Y0(L, k);
        out->zbar[k] = plx_synth_depth(plx_glat_X0(L, k), Y, o->nose_px) + o->depth_tilt * Y;
    }
    double *A = random_basis(L, omega, &r, -1, o->ka, o->a_px);
    double *Bh = random_basis(L, omega, &r, +1, o->kb, o->b_px);
    /* depth variation: mode 0 is the nose's own size, ±15% per unit coefficient (a realistic and
     * partly bas-relief mode: its projection on z̄ scales s_i); the rest are smooth random even
     * fields with total RMS g_px */
    out->G = PLX_NEW(double, (size_t)kg * P);
    if (kg > 1) {
        double *Gr = random_basis(L, omega, &r, +1, kg - 1, o->g_px);
        memcpy(out->G + P, Gr, sizeof(double) * (kg - 1) * P);
        free(Gr);
    }
    for (int k = 0; k < P; k++) out->G[k] = 0.15 * o->nose_px * nose_shape(plx_glat_X0(L, k), plx_glat_Y0(L, k));
    int nr = o->nuis_rank;
    double *Nx = random_basis(L, omega, &r, 0, nr, o->nuis_px / sqrt(2.0));
    double *Ny = random_basis(L, omega, &r, 0, nr, o->nuis_px / sqrt(2.0));

    out->u = PLX_NEW(float, (size_t)n * 2 * P);
    out->gc = PLX_NEW(double, (size_t)n * kg);
    out->yaw = PLX_NEW(double, n);
    out->pitch = PLX_NEW(double, n);
    out->roll = PLX_NEW(double, n);
    out->person = PLX_NEW(int, n);
    double *ca = PLX_NEW(double, o->ka + 1), *cb = PLX_NEW(double, o->kb + 1), *cg = PLX_NEW(double, kg);
    double *cn = PLX_NEW(double, 2 * nr + 1), arw = 0, arh = 0;
    for (int i = 0; i < n0; i++) {
        if (i % o->per_person == 0) { /* new person */
            for (int j = 0; j < o->ka; j++) ca[j] = plx_rng_normal(&r);
            for (int j = 0; j < o->kb; j++) cb[j] = plx_rng_normal(&r);
            for (int j = 0; j < kg; j++) cg[j] = plx_rng_normal(&r);
            arw = o->ar_sd * plx_rng_normal(&r);
            arh = o->ar_sd * plx_rng_normal(&r);
        }
        out->person[i] = i / o->per_person;
        memcpy(out->gc + (size_t)i * kg, cg, sizeof(double) * kg);
        double th = clipped_normal(&r, 0, o->yaw_sd_deg, o->yaw_clip_deg) * DEG;
        double ph = clipped_normal(&r, o->pitch_mean_deg, o->pitch_sd_deg, o->pitch_clip_deg) * DEG;
        double rh = o->roll_sd_deg * plx_rng_normal(&r) * DEG;
        double sc = exp(o->scale_sd * plx_rng_normal(&r));
        double tx = o->trans_sd_px * plx_rng_normal(&r), ty = o->trans_sd_px * plx_rng_normal(&r);
        out->yaw[i] = th, out->pitch[i] = ph, out->roll[i] = rh;
        for (int j = 0; j < 2 * nr; j++) cn[j] = plx_rng_normal(&r);
        double sd = o->noise_px * (1 + o->noise_yaw * fabs(sin(th)));
        double ct = cos(th), st = sin(th), cp = cos(ph), sp = sin(ph), cr = cos(rh), sr = sin(rh);
        float *ux = out->u + (size_t)i * 2 * P, *uy = ux + P;
        for (int k = 0; k < P; k++) {
            double X0 = plx_glat_X0(L, k), Y0 = plx_glat_Y0(L, k);
            double X = X0 + arw * X0, Y = Y0 + arh * Y0, Z = out->zbar[k];
            for (int j = 0; j < o->ka; j++) X += ca[j] * A[(size_t)j * P + k];
            for (int j = 0; j < o->kb; j++) Y += cb[j] * Bh[(size_t)j * P + k];
            for (int j = 0; j < kg; j++) Z += cg[j] * out->G[(size_t)j * P + k];
            /* R = Rz(ρ) Rx(φ) Ry(θ), exactly */
            double x1 = ct * X + st * Z, z1 = -st * X + ct * Z;
            double y2 = cp * Y - sp * z1;
            double px = sc * (cr * x1 - sr * y2) + tx, py = sc * (sr * x1 + cr * y2) + ty;
            double dx = 0, dy = 0;
            for (int j = 0; j < nr; j++) {
                dx += cn[j] * Nx[(size_t)j * P + k];
                dy += cn[nr + j] * Ny[(size_t)j * P + k];
            }
            ux[k] = (float)(px - X0 + dx + sd * plx_rng_normal(&r));
            uy[k] = (float)(py - Y0 + dy + sd * plx_rng_normal(&r));
        }
    }
    /* Mirror twin of photo i: the flipped photo's field is σ*u∘σ, i.e. the same symmetric person
     * at yaw -θ, roll -ρ, with mirrored nuisance. */
    for (int i = 0; i < n - n0; i++) {
        const float *s = out->u + (size_t)i * 2 * P;
        float *d = out->u + (size_t)(n0 + i) * 2 * P;
        for (int k = 0; k < P; k++) {
            int m = plx_glat_mirror(L, k);
            d[k] = -s[m];
            d[P + k] = s[P + m];
        }
        out->yaw[n0 + i] = -out->yaw[i];
        out->pitch[n0 + i] = out->pitch[i];
        out->roll[n0 + i] = -out->roll[i];
        out->person[n0 + i] = out->person[i];
        memcpy(out->gc + (size_t)(n0 + i) * kg, out->gc + (size_t)i * kg, sizeof(double) * kg);
    }
    for (int i = 0; i < n; i++) plx_gauge_similarity(L, gauge, out->u + (size_t)i * 2 * P);
    plx_gauge_mean(L, out->u, n);
    free(omega);
    free(A);
    free(Bh);
    free(Nx);
    free(Ny);
    free(ca);
    free(cb);
    free(cg);
    free(cn);
}

void plx_synth_free(plx_synth *s) {
    free(s->u);
    free(s->zbar);
    free(s->G);
    free(s->gc);
    free(s->yaw);
    free(s->pitch);
    free(s->roll);
    free(s->person);
    memset(s, 0, sizeof *s);
}

/* ---- scoring against the truth --------------------------------------------------------------- */
static void lin_fit(const double *x, const double *y, const double *w, int n, double *slope, double *corr) {
    double sw = 0, mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; i++) sw += w[i], mx += w[i] * x[i], my += w[i] * y[i];
    mx /= sw, my /= sw;
    for (int i = 0; i < n; i++) {
        double a = x[i] - mx, b = y[i] - my;
        sxx += w[i] * a * a, syy += w[i] * b * b, sxy += w[i] * a * b;
    }
    *slope = sxy / sxx;
    *corr = sxy / sqrt(sxx * syy);
}

/* Orthonormal basis (columns, in place) of the n x k matrix A by modified Gram-Schmidt. */
static void orthonormalise(double *A, int n, int k) {
    for (int a = 0; a < k; a++) {
        for (int b = 0; b < a; b++) {
            double d = 0;
            for (int i = 0; i < n; i++) d += A[i * k + a] * A[i * k + b];
            for (int i = 0; i < n; i++) A[i * k + a] -= d * A[i * k + b];
        }
        double nn = 0;
        for (int i = 0; i < n; i++) nn += A[i * k + a] * A[i * k + a];
        nn = nn > 0 ? 1 / sqrt(nn) : 0; /* a null column (e.g. no depth variation) stays null */
        for (int i = 0; i < n; i++) A[i * k + a] *= nn;
    }
}

void plx_synth_score(const plx_synth *s, const plx_geom_model *m, plx_synth_report *r) {
    const plx_gspace *sp = &m->sp;
    int n = s->n, d = sp->ne, nr = m->nr, kz = m->kz, kg = s->kg;
    memset(r, 0, sizeof *r);
    double *zt = PLX_NEW(double, d);
    plx_gspace_even(sp, s->zbar, zt);
    plx_gspace_remove_plane(sp, zt, NULL);
    double zz = plx_vec_dot(zt, zt, d), vv = plx_vec_dot(m->yv, m->yv, d);
    r->v_corr = plx_vec_dot(zt, m->yv, d) / sqrt(zz * vv); /* both have zero ω-mean */
    r->kappa_true = sqrt(zz / sp->wsum);
    r->kappa_ratio = m->kappa / r->kappa_true;

    double *x = PLX_NEW(double, n), *y = PLX_NEW(double, n), *w = PLX_NEW(double, n), *ph = PLX_NEW(double, n);
    double mp = 0, se = 0, ws = 0;
    for (int i = 0; i < n; i++) mp += s->pitch[i] / n;
    for (int i = 0; i < n; i++) {
        plx_geom_angles(m->fx, m->kappa, m->s[i], m->t[i], y + i, ph + i);
        x[i] = s->yaw[i];
        w[i] = m->w[i];
        se += w[i] * (y[i] - x[i]) * (y[i] - x[i]), ws += w[i];
    }
    lin_fit(x, y, w, n, &r->yaw_slope, &r->yaw_corr);
    r->yaw_rmse = sqrt(se / ws) / DEG;
    se = 0;
    for (int i = 0; i < n; i++) {
        x[i] = s->pitch[i] - mp;
        se += w[i] * (ph[i] - x[i]) * (ph[i] - x[i]);
    }
    lin_fit(x, ph, w, n, &r->pitch_slope, &r->pitch_corr);
    r->pitch_rmse = sqrt(se / ws) / DEG;
    free(ph);

    /* true depth variation in the complement coordinates, in units of v */
    double *Gc = PLX_NEW(double, (size_t)nr * kg), *yg = PLX_NEW(double, d);
    for (int j = 0; j < kg; j++) {
        plx_gspace_even(sp, s->G + (size_t)j * s->P, yg);
        for (int a = 0; a < nr; a++) {
            double acc = 0;
            for (int k = 0; k < d; k++) acc += m->Uc[(size_t)k * nr + a] * yg[k];
            Gc[(size_t)a * kg + j] = acc / r->kappa_true;
        }
    }
    for (int b = 0; b <= PLX_SYNTH_NBIN; b++) r->bin_edge[b] = (double[]){0, 10, 20, 30, 90}[b];
    if (kz > 0) {
        double *Qt = PLX_NEW(double, (size_t)nr * kg), *Qb = PLX_NEW(double, (size_t)nr * kz);
        memcpy(Qt, Gc, sizeof(double) * nr * kg);
        memcpy(Qb, m->Br, sizeof(double) * nr * kz);
        orthonormalise(Qt, nr, kg);
        orthonormalise(Qb, nr, kz);
        plx_mat C = plx_mat_new(kz, kg);
        for (int a = 0; a < kz; a++)
            for (int b = 0; b < kg; b++) {
                double acc = 0;
                for (int i = 0; i < nr; i++) acc += Qb[(size_t)i * kz + a] * Qt[(size_t)i * kg + b];
                PLX_M(&C, a, b) = acc;
            }
        r->nangle = kz < kg ? kz : kg;
        if (r->nangle > 8) r->nangle = 8;
        double sv[64];
        PLX_ASSERT(plx_svd(&C, NULL, sv, NULL) == PLX_OK);
        for (int a = 0; a < r->nangle; a++) r->angle_cos[a] = sv[a];
        plx_mat_free(&C);
        free(Qt), free(Qb);

        /* code accuracy and calibration */
        plx_mat BtB = plx_mat_new(kz, kz);
        for (int a = 0; a < kz; a++)
            for (int b = 0; b < kz; b++) {
                double acc = 0;
                for (int i = 0; i < nr; i++) acc += m->Br[(size_t)i * kz + a] * m->Br[(size_t)i * kz + b];
                PLX_M(&BtB, a, b) = acc;
            }
        PLX_ASSERT(plx_chol(&BtB) == PLX_OK);
        double num[PLX_SYNTH_NBIN] = {0}, den[PLX_SYNTH_NBIN] = {0}, z2 = 0, zc[8] = {0};
        int nz = 0;
        double *dt = PLX_NEW(double, nr), *ct = PLX_NEW(double, kz), *e = PLX_NEW(double, kz);
        plx_mat L = plx_mat_new(kz, kz);
        for (int i = 0; i < n; i++) {
            const double *gc = s->gc + (size_t)i * kg, *ch = m->chat + (size_t)i * kz, *La = m->Lam + (size_t)i * kz * kz;
            for (int a = 0; a < nr; a++) {
                dt[a] = 0;
                for (int j = 0; j < kg; j++) dt[a] += Gc[(size_t)a * kg + j] * gc[j];
            }
            double en = 0, dn = 0;
            for (int a = 0; a < nr; a++) {
                double bc = 0;
                for (int k = 0; k < kz; k++) bc += m->Br[(size_t)a * kz + k] * ch[k];
                en += (bc - dt[a]) * (bc - dt[a]), dn += dt[a] * dt[a];
            }
            double ay = fabs(s->yaw[i]) / DEG;
            int b = 0;
            while (b + 1 < PLX_SYNTH_NBIN && ay >= r->bin_edge[b + 1]) b++;
            num[b] += en, den[b] += dn;
            if (ay <= 10 || m->w[i] < 0.5) continue;
            for (int k = 0; k < kz; k++) {
                ct[k] = 0;
                for (int a = 0; a < nr; a++) ct[k] += m->Br[(size_t)a * kz + k] * dt[a];
            }
            plx_chol_solve_vec(&BtB, ct);
            for (int k = 0; k < kz; k++) e[k] = ch[k] - ct[k];
            double q = 0;
            for (int a = 0; a < kz; a++)
                for (int k = 0; k < kz; k++) {
                    q += e[a] * La[a * kz + k] * e[k];
                    PLX_M(&L, a, k) = La[a * kz + k];
                }
            z2 += q / kz;
            /* posterior sd of component k: sqrt((Λ⁻¹)_kk) */
            plx_mat Li = plx_mat_new(kz, kz);
            PLX_ASSERT(plx_spd_inverse(&L, &Li) == PLX_OK);
            for (int k = 0; k < kz && k < 8; k++) zc[k] += e[k] * e[k] / PLX_M(&Li, k, k);
            plx_mat_free(&Li);
            nz++;
        }
        for (int b = 0; b < PLX_SYNTH_NBIN; b++) r->code_r2[b] = den[b] > 0 ? 1 - num[b] / den[b] : NAN;
        r->calib_z2 = nz ? z2 / nz : NAN;
        for (int k = 0; k < kz && k < 8; k++) r->calib_sd[k] = nz ? sqrt(zc[k] / nz) : NAN;
        plx_mat_free(&BtB), plx_mat_free(&L);
        free(dt), free(ct), free(e);
    }
    free(Gc), free(yg), free(zt), free(x), free(y), free(w);
}
