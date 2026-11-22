// plx-deps: geom la core
#include "geom/geom.h"
#include "la/la.h"
#include "test.h"

#include <stdlib.h>
#include <string.h>

/* Common setup: a lattice, its default core, unit node weights and the flow stage's gauge box. */
typedef struct {
    plx_glat L;
    plx_core_shape cs;
    int P;
    uint8_t *core;
    double *gauge;
} setup;

static setup make_setup(plx_glat L, plx_core_shape cs) {
    setup s = {.L = L, .cs = cs, .P = plx_glat_n(&L)};
    s.core = PLX_NEW(uint8_t, s.P);
    s.gauge = PLX_NEW(double, s.P);
    plx_core_mask(&L, cs, s.core);
    /* the flow box, rescaled with the face for other lattices */
    double fx = cs.ax / 40.0, fy = cs.ay / 50.0;
    plx_gauge_box(&L, L.cx - 40 * fx, L.cx + 40 * fx, L.cy - 56 * fy, L.cy + 56 * fy, s.gauge);
    return s;
}
static void free_setup(setup *s) { free(s->core), free(s->gauge); }

static void random_field(plx_rng *r, int P, double *f) {
    for (int k = 0; k < P; k++) f[k] = plx_rng_normal(r);
}

TEST(parity_split_identities) {
    plx_glat L = plx_glat_flow();
    int P = plx_glat_n(&L);
    double *ux = PLX_NEW(double, P), *uy = PLX_NEW(double, P), *E = PLX_NEW(double, P), *O = PLX_NEW(double, P),
           *V = PLX_NEW(double, P), *Q = PLX_NEW(double, P);
    plx_rng r;
    plx_rng_seed(&r, 1);
    random_field(&r, P, ux);
    random_field(&r, P, uy);
    plx_parity_split(&L, ux, uy, E, O, V, Q);
    double worst = 0;
    for (int n = 0; n < P; n++) {
        int m = plx_glat_mirror(&L, n);
        worst = fmax(worst, fabs(E[n] + O[n] - ux[n]) + fabs(V[n] + Q[n] - uy[n]));
        worst = fmax(worst, fabs(E[n] - E[m]) + fabs(O[n] + O[m]) + fabs(V[n] - V[m]) + fabs(Q[n] + Q[m]));
    }
    CHECK_NEAR(worst, 0, 1e-15);
    /* the axis column of odd parts is zero; the mirror is an involution */
    for (int row = 0; row < L.rows; row++) CHECK(O[row * L.cols + L.cols / 2] == 0);
    for (int n = 0; n < P; n++) CHECK(plx_glat_mirror(&L, plx_glat_mirror(&L, n)) == n);
    free(ux), free(uy), free(E), free(O), free(V), free(Q);
}

/* ⟨f, g⟩ over the full lattice equals the dot product of the coordinates, for even and odd parts. */
TEST(coordinates_are_isometric) {
    setup S = make_setup(plx_glat_flow(), plx_core_default());
    float *nw = PLX_NEW(float, S.P);
    plx_rng r;
    plx_rng_seed(&r, 2);
    for (int n = 0; n < S.P; n++) nw[n] = 0.5f + (float)plx_rng_uniform(&r);
    for (int n = 0; n < S.P; n++) nw[n] = nw[plx_glat_mirror(&S.L, n)] = 0.5f * (nw[n] + nw[plx_glat_mirror(&S.L, n)]);
    plx_gspace sp;
    plx_gspace_init(&sp, &S.L, nw, S.core);
    double *f = PLX_NEW(double, S.P), *g = PLX_NEW(double, S.P), *fe = PLX_NEW(double, S.P), *ge = PLX_NEW(double, S.P),
           *fo = PLX_NEW(double, S.P), *go = PLX_NEW(double, S.P);
    random_field(&r, S.P, f);
    random_field(&r, S.P, g);
    plx_parity_split(&S.L, f, g, fe, fo, ge, go); /* reuse: fe = even f, fo = odd f, ge = even g, go = odd g */
    double de = 0, dod = 0;
    for (int n = 0; n < S.P; n++) de += sp.omega[n] * fe[n] * ge[n], dod += sp.omega[n] * fo[n] * go[n];
    double *y1 = PLX_NEW(double, sp.ne), *y2 = PLX_NEW(double, sp.ne), *o1 = PLX_NEW(double, sp.no),
           *o2 = PLX_NEW(double, sp.no);
    plx_gspace_even(&sp, f, y1);
    plx_gspace_even(&sp, g, y2);
    plx_gspace_odd(&sp, f, o1);
    plx_gspace_odd(&sp, g, o2);
    CHECK_NEAR(plx_vec_dot(y1, y2, sp.ne), de, 1e-12 * fabs(de) + 1e-12);
    CHECK_NEAR(plx_vec_dot(o1, o2, sp.no), dod, 1e-12 * fabs(dod) + 1e-12);
    /* round trip to node values */
    double *back = PLX_NEW(double, S.P), worst = 0;
    plx_gspace_even_field(&sp, y1, NAN, back);
    for (int n = 0; n < S.P; n++)
        if (sp.omega[n] > 0) worst = fmax(worst, fabs(back[n] - fe[n]));
    plx_gspace_odd_field(&sp, o1, NAN, back);
    for (int n = 0; n < S.P; n++)
        if (sp.omega[n] > 0) worst = fmax(worst, fabs(back[n] - fo[n]));
    CHECK_NEAR(worst, 0, 1e-12);
    free(f), free(g), free(fe), free(ge), free(fo), free(go), free(y1), free(y2), free(o1), free(o2), free(back),
        free(nw);
    plx_gspace_free(&sp);
    free_setup(&S);
}

TEST(plane_gauge_projection) {
    setup S = make_setup(plx_glat_flow(), plx_core_default());
    plx_gspace sp;
    plx_gspace_init(&sp, &S.L, NULL, S.core);
    plx_rng r;
    plx_rng_seed(&r, 3);
    double *h = PLX_NEW(double, S.P), *f = PLX_NEW(double, S.P), *yh = PLX_NEW(double, sp.ne),
           *yf = PLX_NEW(double, sp.ne), c[2];
    random_field(&r, S.P, h);
    for (int n = 0; n < S.P; n++) h[n] = h[plx_glat_mirror(&S.L, n)];
    for (int n = 0; n < S.P; n++) f[n] = h[n] + 3.25 - 0.4 * plx_glat_Y0(&S.L, n);
    plx_gspace_even(&sp, h, yh);
    plx_gspace_even(&sp, f, yf);
    double ch[2];
    plx_gspace_remove_plane(&sp, yh, ch);
    plx_gspace_remove_plane(&sp, yf, c);
    /* exactly the plane is removed: same residual, coefficients differ by the added plane */
    double worst = 0;
    for (int k = 0; k < sp.ne; k++) worst = fmax(worst, fabs(yh[k] - yf[k]));
    CHECK_NEAR(worst, 0, 1e-12);
    CHECK_NEAR(c[0] - ch[0], 3.25, 1e-12);
    CHECK_NEAR(c[1] - ch[1], -0.4, 1e-13);
    CHECK_NEAR(plx_vec_dot(yf, sp.e1, sp.ne), 0, 1e-10);
    CHECK_NEAR(plx_vec_dot(yf, sp.eY, sp.ne), 0, 1e-8);
    /* idempotent */
    plx_gspace_remove_plane(&sp, yf, c);
    CHECK_NEAR(fabs(c[0]) + fabs(c[1]), 0, 1e-12);
    free(h), free(f), free(yh), free(yf);
    plx_gspace_free(&sp);
    free_setup(&S);
}

/* For a mirror-symmetric shape viewed with no residual similarity, every entry of the parity table
 * (DESIGN.md §5.2) holds to rounding with the EXACT camera:
 *   E = sin θ z,  O = (cos θ - 1) X0 + cos θ a,  V = (cos φ - 1) Y0 + cos φ b - sin φ cos θ z,
 *   Q = sin φ sin θ (X0 + a). */
TEST(exact_model_parity_table) {
    plx_glat L = plx_glat_flow();
    int P = plx_glat_n(&L);
    double *a = PLX_NEW(double, P), *b = PLX_NEW(double, P), *z = PLX_NEW(double, P);
    double *ux = PLX_NEW(double, P), *uy = PLX_NEW(double, P), *E = PLX_NEW(double, P), *O = PLX_NEW(double, P),
           *V = PLX_NEW(double, P), *Q = PLX_NEW(double, P);
    plx_rng r;
    plx_rng_seed(&r, 4);
    for (int n = 0; n < P; n++) a[n] = 3 * plx_rng_normal(&r), b[n] = 3 * plx_rng_normal(&r);
    for (int n = 0; n < P; n++) {
        int m = plx_glat_mirror(&L, n);
        if (m < n) continue;
        a[m] = -a[n], b[m] = b[n];
        if (m == n) a[n] = 0;
        z[n] = z[m] = plx_synth_depth(plx_glat_X0(&L, n), plx_glat_Y0(&L, n), 20);
    }
    double poses[3][2] = {{0.3, -0.2}, {-0.7, 0.45}, {0.05, 0.6}}, worst = 0;
    for (int p = 0; p < 3; p++) {
        double th = poses[p][0], ph = poses[p][1], ct = cos(th), st = sin(th), cp = cos(ph), sp = sin(ph);
        for (int n = 0; n < P; n++) {
            double X = plx_glat_X0(&L, n) + a[n], Y = plx_glat_Y0(&L, n) + b[n];
            double x1 = ct * X + st * z[n], z1 = -st * X + ct * z[n];
            ux[n] = x1 - plx_glat_X0(&L, n);
            uy[n] = cp * Y - sp * z1 - plx_glat_Y0(&L, n);
        }
        plx_parity_split(&L, ux, uy, E, O, V, Q);
        for (int n = 0; n < P; n++) {
            double X0 = plx_glat_X0(&L, n), Y0 = plx_glat_Y0(&L, n);
            worst = fmax(worst, fabs(E[n] - st * z[n]));
            worst = fmax(worst, fabs(O[n] - ((ct - 1) * X0 + ct * a[n])));
            worst = fmax(worst, fabs(V[n] - ((cp - 1) * Y0 + cp * b[n] - sp * ct * z[n])));
            worst = fmax(worst, fabs(Q[n] - sp * st * (X0 + a[n])));
        }
    }
    CHECK_NEAR(worst, 0, 1e-12);
    free(a), free(b), free(z), free(ux), free(uy), free(E), free(O), free(V), free(Q);
}

/* Gauge (1) removes a known similarity exactly and leaves the least-squares similarity of the
 * remainder at the identity; gauge (2) leaves a zero mirror-symmetrised mean. */
TEST(flow_gauges) {
    setup S = make_setup(plx_glat_flow(), plx_core_default());
    int P = S.P, n = 6;
    float *u = PLX_NEW(float, (size_t)n * 2 * P), *ref = PLX_NEW(float, 2 * P);
    plx_rng r;
    plx_rng_seed(&r, 5);
    for (int k = 0; k < 2 * P; k++) ref[k] = (float)(2 * plx_rng_normal(&r));
    plx_gauge_similarity(&S.L, S.gauge, ref);
    double worst = 0;
    for (int i = 0; i < n; i++) {
        double sc = 1 + 0.1 * plx_rng_normal(&r), rho = 0.2 * plx_rng_normal(&r), tx = 5 * plx_rng_normal(&r),
               ty = 5 * plx_rng_normal(&r), A = sc * cos(rho), B = sc * sin(rho);
        float *ui = u + (size_t)i * 2 * P;
        for (int k = 0; k < P; k++) {
            double X = plx_glat_X0(&S.L, k) + ref[k], Y = plx_glat_Y0(&S.L, k) + ref[P + k];
            ui[k] = (float)(A * X - B * Y + tx - plx_glat_X0(&S.L, k));
            ui[P + k] = (float)(B * X + A * Y + ty - plx_glat_Y0(&S.L, k));
        }
        plx_gauge_similarity(&S.L, S.gauge, ui);
        for (int k = 0; k < 2 * P; k++) worst = fmax(worst, fabs(ui[k] - ref[k]));
    }
    CHECK_NEAR(worst, 0, 2e-4); /* float storage of ~100 px positions */
    plx_gauge_mean(&S.L, u, n);
    double mx = 0;
    for (int k = 0; k < P; k++) {
        int m = plx_glat_mirror(&S.L, k);
        double sx = 0, sy = 0;
        for (int i = 0; i < n; i++) {
            const float *ui = u + (size_t)i * 2 * P;
            sx += 0.5 * (ui[k] - ui[m]), sy += 0.5 * (ui[P + k] + ui[P + m]);
        }
        mx = fmax(mx, fabs(sx / n) + fabs(sy / n));
    }
    CHECK_NEAR(mx, 0, 1e-5);
    free(u), free(ref);
    free_setup(&S);
}

TEST(angles_roundtrip) {
    double fx = 0.35, kap = 5.5, worst = 0;
    for (int a = -4; a <= 4; a++)
        for (int b = -3; b <= 3; b++) {
            double th = 0.2 * a, ph = 0.1 * b, rho = fx * cos(th) + (1 - fx) * cos(ph);
            double s = kap * sin(th) / rho, t = kap * sin(ph) * cos(th) / rho, th2, ph2;
            CHECK(plx_geom_angles(fx, kap, s, t, &th2, &ph2) == 0);
            worst = fmax(worst, fabs(th2 - th) + fabs(ph2 - ph));
        }
    CHECK_NEAR(worst, 0, 1e-12);
    double th, ph;
    CHECK(plx_geom_angles(fx, kap, 2 * kap, 0, &th, &ph) & 1); /* ρ >= 1 - fx: no real yaw */
}

/* One synthetic population, fitted; returns the score. */
static void run_synth(const setup *S, plx_synth_opts so, plx_synth_report *rep, plx_geom_model *keep) {
    plx_synth s;
    plx_synth_make(&S->L, NULL, S->core, S->gauge, &so, &s);
    plx_emerge_opts o = plx_emerge_defaults();
    o.boot = 0;
    o.n_scramble = 20;
    o.gauge_w = S->gauge;
    plx_geom_model m;
    CHECK(plx_emerge(&S->L, s.u, s.n, NULL, S->core, S->cs, &o, &m) == PLX_OK);
    plx_synth_score(&s, &m, rep);
    if (keep) *keep = m;
    else plx_geom_model_free(&m);
    plx_synth_free(&s);
}

static plx_synth_opts quiet(int n) {
    plx_synth_opts so = plx_synth_defaults();
    so.n = n, so.noise_px = 0, so.nuis_px = 0, so.g_px = 0, so.kg = 1, so.ar_sd = 0, so.a_px = so.b_px = 0;
    return so;
}

TEST(rank1_recovery_and_pose) {
    setup S = make_setup(plx_glat_flow(), plx_core_default());
    plx_synth_report r;
    run_synth(&S, quiet(3000), &r, NULL);
    CHECK(r.v_corr > 0.9995);
    CHECK(fabs(r.yaw_slope - 1) < 0.04 && r.yaw_corr > 0.995);
    CHECK(fabs(r.pitch_slope - 1) < 0.04 && r.pitch_corr > 0.99);
    plx_synth_opts so = plx_synth_defaults();
    so.n = 3000;
    run_synth(&S, so, &r, NULL);
    CHECK(r.v_corr > 0.999);
    CHECK(r.yaw_corr > 0.99 && r.pitch_corr > 0.96);
    free_setup(&S);
}

/* κ: noiseless (only the irreducible bas-relief nose-size variation) and realistic noise. The
 * remaining positive bias is errors-in-variables from person-specific depth scale along v (see
 * emerge.c); the bounds encode the measured behaviour with margin. */
TEST(kappa_recovery) {
    setup S = make_setup(plx_glat_flow(), plx_core_default());
    plx_synth_report r;
    run_synth(&S, quiet(4000), &r, NULL);
    CHECK(fabs(r.kappa_ratio - 1) < 0.03);
    plx_synth_opts so = plx_synth_defaults();
    so.n = 4000;
    run_synth(&S, so, &r, NULL);
    CHECK(fabs(r.kappa_ratio - 1) < 0.08);
    free_setup(&S);
}

/* Covariance regression recovers the depth-variation subspace and calibrated codes. */
TEST(depth_basis_and_posterior) {
    setup S = make_setup(plx_glat_flow(), plx_core_default());
    plx_synth_opts so = plx_synth_defaults();
    so.n = 6000, so.kg = 3, so.g_px = 2.0;
    plx_synth_report r;
    plx_geom_model m;
    run_synth(&S, so, &r, &m);
    CHECK(m.kz == 3);
    for (int k = 0; k < r.nangle; k++) CHECK(r.angle_cos[k] > 0.98);
    CHECK(r.code_r2[3] > 0.7 && r.code_r2[0] < r.code_r2[3]); /* turned faces carry more depth */
    CHECK(r.calib_z2 > 0.7 && r.calib_z2 < 1.5);
    /* a frontal photo returns the prior: ĉ = 0, Λ = I */
    double *c = PLX_NEW(double, m.kz), *Lam = PLX_NEW(double, m.kz * m.kz), *y = PLX_NEW(double, m.sp.ne);
    for (int k = 0; k < m.sp.ne; k++) y[k] = 0.1 * k;
    plx_geom_depth_code(&m, y, 0.0, c, Lam);
    double dev = 0;
    for (int a = 0; a < m.kz; a++) {
        dev += fabs(c[a]);
        for (int b = 0; b < m.kz; b++) dev += fabs(Lam[a * m.kz + b] - (a == b));
    }
    CHECK_NEAR(dev, 0, 1e-12);
    free(c), free(Lam), free(y);
    plx_geom_model_free(&m);
    free_setup(&S);
}

TEST(llr_properties) {
    int kz = 2;
    double I[4] = {1, 0, 0, 1}, c0[2] = {0, 0}, L1[4] = {5, 1, 1, 4}, L2[4] = {3, -0.5, -0.5, 6};
    double c1[2] = {0.8, -0.3}, c2[2] = {0.7, -0.2}, c3[2] = {-0.9, 0.5};
    CHECK_NEAR(plx_geom_llr(kz, c0, I, c1, L1), 0, 1e-12); /* frontal: no evidence either way */
    CHECK_NEAR(plx_geom_llr(kz, c1, L1, c2, L2), plx_geom_llr(kz, c2, L2, c1, L1), 1e-12);
    CHECK(plx_geom_llr(kz, c1, L1, c2, L2) > plx_geom_llr(kz, c1, L1, c3, L2));
    /* kz = 1 closed form: s = shared, d = independent Gaussian evidence */
    double a = 4, b = 9, ca = 0.6, cb = 0.5;
    double h1 = a * ca, h2 = b * cb, want = 0.5 * (log(a) + log(b) - log(a + b - 1)) +
                                            0.5 * ((h1 + h2) * (h1 + h2) / (a + b - 1) - h1 * ca - h2 * cb);
    CHECK_NEAR(plx_geom_llr(1, &ca, &a, &cb, &b), want, 1e-12);
}

/* Nothing depends on the 17 x 21 / 8 px lattice: a 13 x 17 lattice at 10 px spacing on a frame of
 * another size recovers depth and pose just as well. */
TEST(lattice_independence) {
    plx_glat L = {.cols = 13, .rows = 17, .spacing = 10, .x0 = 0, .y0 = 0, .cx = 60, .cy = 80};
    setup S = make_setup(L, (plx_core_shape){.ax = 40, .ay = 50, .yc = 6});
    plx_synth_opts so = plx_synth_defaults();
    so.n = 3000;
    plx_synth_report r;
    run_synth(&S, so, &r, NULL);
    CHECK(r.v_corr > 0.99);
    CHECK(r.yaw_corr > 0.98);
    CHECK(fabs(r.kappa_ratio - 1) < 0.1);
    free_setup(&S);
}

int main(void) {
    RUN(parity_split_identities);
    RUN(coordinates_are_isometric);
    RUN(plane_gauge_projection);
    RUN(exact_model_parity_table);
    RUN(flow_gauges);
    RUN(angles_roundtrip);
    RUN(rank1_recovery_and_pose);
    RUN(kappa_recovery);
    RUN(depth_basis_and_posterior);
    RUN(llr_properties);
    RUN(lattice_independence);
    return TEST_REPORT();
}
