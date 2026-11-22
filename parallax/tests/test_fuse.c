// plx-deps: fuse align render viz feat ops img la core
/* Multi-frame fusion: container, factorisation (exact recovery, noise and missing data,
 * bas-relief degeneracy and its fallback, mirror sign, Procrustes gauge, uncertainty
 * calibration), visibility and localisation of a pasted face.
 *
 * Scratch files go to $PLX_TEST_TMP (default /tmp). The localisation test needs the frozen flow
 * model (runs/main/flow/model) and an LFW photo; it is skipped when they are absent. */
#include "test.h"

#include "align/flow.h"
#include "core/core.h"
#include "fuse/fuse.h"
#include "img/imgio.h"
#include "la/la.h"
#include "ops/ops.h"
#include "render/render.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DEG (M_PI / 180.0)

static char g_dir[512];

/* ---- container ------------------------------------------------------------------------------ */
TEST(container_roundtrip) {
    plx_fu_clip c = {0};
    uint8_t a[5] = {1, 2, 3, 4, 5}, b[1] = {0xff};
    plx_fu_clip_append(&c, 0.0, a, 5);
    plx_fu_clip_append(&c, 33.25, b, 1);
    plx_fu_clip_append(&c, 66.5, a, 0);
    char path[700];
    snprintf(path, sizeof path, "%s/c.plxv", g_dir);
    CHECK(plx_fu_clip_write(path, &c) == PLX_OK);
    plx_fu_clip d;
    CHECK(plx_fu_clip_read(path, &d) == PLX_OK);
    CHECK(d.n == 3);
    if (d.n == 3) {
        CHECK(d.len[0] == 5 && memcmp(d.data[0], a, 5) == 0);
        CHECK(d.len[1] == 1 && d.data[1][0] == 0xff && d.len[2] == 0);
        CHECK(d.t_ms[1] == 33.25 && d.t_ms[2] == 66.5);
    }
    plx_fu_clip_free(&d);
    /* truncated file is a format error, not a short clip */
    uint8_t *buf;
    size_t len;
    CHECK(plx_read_file(path, &buf, &len) == PLX_OK);
    CHECK(plx_write_file(path, buf, len - 1) == PLX_OK);
    CHECK(plx_fu_clip_read(path, &d) == PLX_EFORMAT);
    free(buf);
    plx_fu_clip_free(&c);

    /* a directory of numbered frames, ordered by number (not by name) */
    char dir[700], fp[800];
    snprintf(dir, sizeof dir, "%s/frames", g_dir);
    plx_mkdir_p(dir);
    for (int k = 0; k < 3; k++) {
        int num[3] = {10, 2, 7};
        plx_img im = plx_img_new(4, 3, 1);
        plx_img_fill(&im, num[k] / 20.0f);
        snprintf(fp, sizeof fp, "%s/frame_%d.pgm", dir, num[k]);
        CHECK(plx_img_save(fp, &im) == PLX_OK);
        plx_img_free(&im);
    }
    CHECK(plx_fu_clip_read(dir, &d) == PLX_OK);
    CHECK(d.n == 3);
    float want[3] = {2 / 20.0f, 7 / 20.0f, 10 / 20.0f};
    for (int k = 0; k < d.n && k < 3; k++) {
        plx_img im;
        CHECK(plx_fu_clip_decode(&d, k, &im) == PLX_OK);
        CHECK(im.w == 4 && im.h == 3);
        CHECK_NEAR(im.px[0], want[k], 0.5 / 255);
        plx_img_free(&im);
    }
    plx_fu_clip_free(&d);
}

/* ---- synthetic tracks -------------------------------------------------------------------------- */
enum { GX = 9, GY = 11, GP = GX * GY };
typedef struct {
    double X0[GP], Y0[GP], S[3 * GP], zp[GP];
} face_pts;

/* A face-like surface on a 9 x 11 grid, 10 px apart: dome + nose + small identity bump. */
static void make_face(face_pts *fp, double nose, double prior_sign) {
    for (int i = 0; i < GY; i++)
        for (int j = 0; j < GX; j++) {
            int n = i * GX + j;
            double X = (j - 4) * 10.0, Y = (i - 5) * 10.0;
            double dome = 30 * sqrt(fmax(1 - X * X / 3600 - Y * Y / 6400, 0));
            double z = dome + nose * exp(-X * X / 98 - (Y - 5) * (Y - 5) / 800) + 3 * exp(-(X - 15) * (X - 15) / 200);
            fp->X0[n] = X, fp->Y0[n] = Y;
            fp->S[n] = X + 0.05 * Y, fp->S[GP + n] = Y + 0.03 * X * X / 40, fp->S[2 * GP + n] = z;
            fp->zp[n] = prior_sign * (dome + 0.8 * nose * exp(-X * X / 98 - (Y - 5) * (Y - 5) / 800));
        }
}

typedef struct {
    int F;
    double *x, *w, R[64][9], s[64];
} tracks;

/* F frames: yaw sweep ±yaw, nod ±pitch (two cycles), roll ±3°, scale 1 ± 5 %, translation. */
static void make_tracks(const face_pts *fp, int F, double yaw, double pitch, double noise, double missing,
                        uint64_t seed, tracks *t) {
    plx_rng r;
    plx_rng_seed(&r, seed);
    t->F = F;
    t->x = PLX_NEW(double, (size_t)F * 2 * GP);
    t->w = PLX_NEW(double, (size_t)F * GP);
    for (int f = 0; f < F; f++) {
        double ph = 2 * M_PI * f / F;
        plx_rot_ypr(yaw * DEG * sin(ph), pitch * DEG * sin(2 * ph + 0.7), 3 * DEG * cos(ph), t->R[f]);
        t->s[f] = 1 + 0.05 * sin(3 * ph);
        for (int n = 0; n < GP; n++) {
            const double *R = t->R[f];
            double v[3] = {fp->S[n], fp->S[GP + n], fp->S[2 * GP + n]};
            for (int c = 0; c < 2; c++)
                t->x[((size_t)f * 2 + c) * GP + n] =
                    t->s[f] * (R[c * 3] * v[0] + R[c * 3 + 1] * v[1] + R[c * 3 + 2] * v[2]) + (c ? 128 : 125) +
                    3 * sin(ph + c) + noise * plx_rng_normal(&r);
            t->w[(size_t)f * GP + n] = plx_rng_uniform(&r) < missing ? 0 : 1;
        }
    }
}

static void tracks_free(tracks *t) {
    free(t->x);
    free(t->w);
}

static plx_fu_tracks as_tracks(const face_pts *fp, const tracks *t) {
    return (plx_fu_tracks){.F = t->F, .P = GP, .nx = GX, .ny = GY, .img_w = 250, .img_h = 250, .x = t->x, .w = t->w,
                           .X0 = fp->X0, .Y0 = fp->Y0, .zp = fp->zp};
}

/* RMS distance of the recovered shape to the truth after similarity alignment, and the
 * alignment (truth = s R rec + t). */
static double shape_err(const face_pts *fp, const plx_fu_shape *sh, double *s, double R[9], double t[3]) {
    double w[GP];
    for (int n = 0; n < GP; n++) w[n] = sh->used[n];
    double S[3 * GP];
    for (int k = 0; k < 3 * GP; k++) S[k] = isfinite(sh->S[k]) ? sh->S[k] : 0;
    return plx_fu_procrustes(GP, S, fp->S, w, s, R, t);
}

static void mat3_mul(const double *A, const double *B, double *C) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
}

static plx_fu_factor_opts plain_opts(void) {
    plx_fu_factor_opts o = plx_fu_factor_defaults();
    o.rounds = 1;
    o.boot = 0;
    o.prior_scale = 0;
    return o;
}

/* ---- factorisation ------------------------------------------------------------------------------- */
TEST(procrustes_and_angles) {
    plx_rng r;
    plx_rng_seed(&r, 3);
    double A[3 * 40], B[3 * 40], Rt[9];
    plx_rot_ypr(0.4, -0.3, 0.2, Rt);
    for (int i = 0; i < 40; i++)
        for (int k = 0; k < 3; k++) A[k * 40 + i] = 20 * plx_rng_normal(&r);
    for (int i = 0; i < 40; i++)
        for (int k = 0; k < 3; k++)
            B[k * 40 + i] = 1.7 * (Rt[k * 3] * A[i] + Rt[k * 3 + 1] * A[40 + i] + Rt[k * 3 + 2] * A[80 + i]) + (k + 1) * 5;
    double s, R[9], t[3];
    double e = plx_fu_procrustes(40, A, B, NULL, &s, R, t);
    CHECK(e < 1e-10);
    CHECK_NEAR(s, 1.7, 1e-12);
    for (int k = 0; k < 9; k++) CHECK_NEAR(R[k], Rt[k], 1e-12);
    CHECK_NEAR(t[2], 15, 1e-10);
    /* a reflection is never returned */
    for (int i = 0; i < 40; i++) B[2 * 40 + i] = -B[2 * 40 + i];
    plx_fu_procrustes(40, A, B, NULL, &s, R, t);
    double det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
    CHECK_NEAR(det, 1, 1e-12);
    /* angles invert plx_rot_ypr */
    for (int k = 0; k < 20; k++) {
        double y = (plx_rng_uniform(&r) - 0.5) * 2.0, p = (plx_rng_uniform(&r) - 0.5) * 2.0,
               ro = (plx_rng_uniform(&r) - 0.5) * 2.0, Q[9], y2, p2, r2;
        plx_rot_ypr(y, p, ro, Q);
        plx_fu_angles(Q, &y2, &p2, &r2);
        CHECK_NEAR(y2, y, 1e-12);
        CHECK_NEAR(p2, p, 1e-12);
        CHECK_NEAR(r2, ro, 1e-12);
    }
}

TEST(factor_exact) {
    face_pts fp;
    make_face(&fp, 22, 1);
    tracks t;
    make_tracks(&fp, 12, 25, 12, 0, 0, 1, &t);
    plx_fu_tracks tr = as_tracks(&fp, &t);
    plx_fu_factor_opts o = plain_opts();
    o.visibility = false;
    plx_fu_shape sh;
    CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
    double s, R[9], tt[3];
    double e = shape_err(&fp, &sh, &s, R, tt);
    CHECK(e < 1e-7);
    CHECK(sh.rms < 1e-7);
    CHECK(!sh.degenerate && sh.corr > 0.9 && sh.corr_mirror < sh.corr);
    /* rotations: R_rec = R_true Ralign, scales proportional */
    double maxr = 0, ratio0 = sh.scale[0] / t.s[0], maxs = 0;
    for (int f = 0; f < t.F; f++) {
        double Q[9];
        mat3_mul(t.R[f], R, Q);
        for (int k = 0; k < 9; k++) maxr = fmax(maxr, fabs(Q[k] - sh.R[f * 9 + k]));
        maxs = fmax(maxs, fabs(sh.scale[f] / t.s[f] - ratio0));
    }
    CHECK(maxr < 1e-8);
    CHECK(maxs < 1e-8);
    /* the canonical gauge is near the layout: the alignment to truth is close to identity */
    CHECK(fabs(s - 1) < 0.05);
    plx_fu_shape_free(&sh);
    tracks_free(&t);
}

TEST(factor_noise_and_missing) {
    face_pts fp;
    make_face(&fp, 22, 1);
    double err[3];
    double noise[3] = {0.1, 0.3, 1.0};
    for (int k = 0; k < 3; k++) {
        tracks t;
        make_tracks(&fp, 30, 25, 12, noise[k], 0.25, 10 + k, &t);
        /* an occluded block: the image-right cheek in the frames turned furthest away */
        for (int f = 0; f < t.F; f++)
            if (sin(2 * M_PI * f / t.F) > 0.7)
                for (int n = 0; n < GP; n++)
                    if (n % GX >= 7) t.w[(size_t)f * GP + n] = 0;
        plx_fu_tracks tr = as_tracks(&fp, &t);
        plx_fu_factor_opts o = plain_opts();
        o.visibility = false;
        plx_fu_shape sh;
        CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
        double s, R[9], tt[3];
        err[k] = shape_err(&fp, &sh, &s, R, tt);
        plx_fu_shape_free(&sh);
        tracks_free(&t);
    }
    printf("  shape error vs noise 0.1 / 0.3 / 1.0 px: %.3f %.3f %.3f px\n", err[0], err[1], err[2]);
    CHECK(err[0] < err[1] && err[1] < err[2]);
    CHECK(err[0] < 0.5 && err[2] < 4);

    /* robust rounds reject gross outliers (10 % of entries off by 15 px) */
    tracks t;
    make_tracks(&fp, 30, 25, 12, 0.2, 0, 21, &t);
    plx_rng r;
    plx_rng_seed(&r, 5);
    for (size_t k = 0; k < (size_t)t.F * GP; k++)
        if (plx_rng_uniform(&r) < 0.1) t.x[(k / GP) * 2 * GP + k % GP] += 15;
    plx_fu_tracks tr = as_tracks(&fp, &t);
    plx_fu_factor_opts o = plain_opts();
    o.visibility = false;
    plx_fu_shape a, b;
    CHECK(plx_fu_factorise(&tr, &o, &a) == PLX_OK);
    o.rounds = 4;
    CHECK(plx_fu_factorise(&tr, &o, &b) == PLX_OK);
    double s, R[9], tt[3], ea = shape_err(&fp, &a, &s, R, tt), eb = shape_err(&fp, &b, &s, R, tt);
    printf("  10%% outliers: shape error %.3f px plain, %.3f px robust\n", ea, eb);
    CHECK(eb < 0.5 * ea && eb < 0.6);
    plx_fu_shape_free(&a), plx_fu_shape_free(&b);
    tracks_free(&t);
}

/* Centred, unweighted regression ratio ⟨z, zp⟩ / ⟨zp, zp⟩ over used nodes. */
static double prior_ratio(const face_pts *fp, const plx_fu_shape *sh) {
    double mz = 0, mp = 0, ab = 0, bb = 0;
    int n = 0;
    for (int k = 0; k < GP; k++)
        if (sh->used[k]) mz += sh->S[2 * GP + k], mp += fp->zp[k], n++;
    mz /= n, mp /= n;
    for (int k = 0; k < GP; k++)
        if (sh->used[k]) ab += (sh->S[2 * GP + k] - mz) * (fp->zp[k] - mp), bb += (fp->zp[k] - mp) * (fp->zp[k] - mp);
    return ab / bb;
}

/* Single-axis motion: yaw sweep ±amp only (no nod, no roll), scale drift kept. */
static void yaw_only_tracks(const face_pts *fp, int F, double amp, double noise, uint64_t seed, tracks *t) {
    make_tracks(fp, F, amp, 0, noise, 0, seed, t);
    plx_rng r;
    plx_rng_seed(&r, seed + 1);
    for (int f = 0; f < F; f++) {
        plx_rot_ypr(amp * DEG * sin(2 * M_PI * f / F), 0, 0, t->R[f]);
        for (int n = 0; n < GP; n++)
            for (int c = 0; c < 2; c++) {
                const double *R = t->R[f];
                t->x[((size_t)f * 2 + c) * GP + n] =
                    t->s[f] * (R[c * 3] * fp->S[n] + R[c * 3 + 1] * fp->S[GP + n] + R[c * 3 + 2] * fp->S[2 * GP + n]) +
                    125 + noise * plx_rng_normal(&r);
            }
    }
}

TEST(degenerate_motion) {
    face_pts fp;
    make_face(&fp, 22, 1);
    plx_fu_factor_opts o = plain_opts();
    o.prior_scale = -1;
    o.visibility = false;
    o.boot = 0; /* the depth-scale decision runs its own bootstrap */
    /* a tiny single-axis turn: depth scale and rotation amplitude trade off (bas-relief); the
     * metric solution's bootstrap says so and the prior takes over */
    tracks t;
    yaw_only_tracks(&fp, 30, 3, 0.3, 31, &t);
    plx_fu_tracks tr = as_tracks(&fp, &t);
    plx_fu_shape sh;
    CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
    printf("  yaw +-3 only: spreads %.2f %.2f deg, gap %.2e, metric scale sd %.3f -> prior weight %.3f, degenerate %d\n",
           sh.spread_deg[0], sh.spread_deg[1], sh.gap, sh.scale_sd_metric, sh.prior_weight, sh.degenerate);
    CHECK(sh.degenerate && sh.prior_weight > 0.5);
    /* the fusion's defining property: the person's projection on the prior pattern is the metric
     * one raised to (1 - prior weight); 1 would be the pure prior */
    CHECK_NEAR(prior_ratio(&fp, &sh), pow(sh.depth_ratio, 1 - sh.prior_weight), 1e-6);
    plx_fu_shape_free(&sh);
    /* the prior alone, forced: the projection equals the prior's exactly */
    o.prior_scale = 1;
    CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
    CHECK_NEAR(prior_ratio(&fp, &sh), 1.0, 1e-9);
    plx_fu_shape_free(&sh);
    tracks_free(&t);
    /* a wide single-axis sweep is NOT degenerate: foreshortening (cos θ) pins the scale */
    o.prior_scale = -1;
    yaw_only_tracks(&fp, 30, 30, 0.3, 32, &t);
    tr = as_tracks(&fp, &t);
    CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
    double s, R[9], tt[3], e = shape_err(&fp, &sh, &s, R, tt);
    printf("  yaw +-30 only: spreads %.2f %.2f deg, metric scale sd %.3f -> prior weight %.3f; shape error %.3f px\n",
           sh.spread_deg[0], sh.spread_deg[1], sh.scale_sd_metric, sh.prior_weight, e);
    CHECK(!sh.degenerate && sh.prior_weight < 0.2 && e < 0.6);
    plx_fu_shape_free(&sh);
    tracks_free(&t);
    /* yaw + nod: well posed */
    make_tracks(&fp, 30, 25, 12, 0.3, 0, 33, &t);
    tr = as_tracks(&fp, &t);
    CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
    printf("  yaw +-25 nod +-12: spreads %.2f %.2f deg, metric scale sd %.3f -> prior weight %.3f\n", sh.spread_deg[0],
           sh.spread_deg[1], sh.scale_sd_metric, sh.prior_weight);
    CHECK(!sh.degenerate && sh.prior_weight < 0.1);
    plx_fu_shape_free(&sh);
    tracks_free(&t);
}

TEST(mirror_sign) {
    /* the prior's sign decides between the two mirror solutions */
    for (int sign = -1; sign <= 1; sign += 2) {
        face_pts fp;
        make_face(&fp, 22, sign);
        tracks t;
        make_tracks(&fp, 16, 25, 12, 0.2, 0, 40 + sign, &t);
        plx_fu_tracks tr = as_tracks(&fp, &t);
        plx_fu_factor_opts o = plain_opts();
        o.visibility = false;
        plx_fu_shape sh;
        CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
        double mz = 0, mt = 0, c = 0;
        for (int n = 0; n < GP; n++) mz += sh.S[2 * GP + n] / GP, mt += fp.S[2 * GP + n] / GP;
        for (int n = 0; n < GP; n++) c += (sh.S[2 * GP + n] - mz) * (fp.S[2 * GP + n] - mt);
        /* true depth (nose out) vs recovered: agree iff the prior says nose out */
        CHECK(sign > 0 ? c > 0 : c < 0);
        CHECK(sh.corr > 0.9 && sh.corr_mirror < 0);
        plx_fu_shape_free(&sh);
        tracks_free(&t);
    }
}

TEST(uncertainty_calibration) {
    face_pts fp;
    make_face(&fp, 22, 1);
    /* truth in the canonical gauge: the same Procrustes onto (X0, Y0, zp) the estimator uses */
    double B[3 * GP], s, R[9], t3[3], Tc[3 * GP];
    for (int n = 0; n < GP; n++) B[n] = fp.X0[n], B[GP + n] = fp.Y0[n], B[2 * GP + n] = fp.zp[n];
    plx_fu_procrustes(GP, fp.S, B, NULL, &s, R, t3);
    for (int n = 0; n < GP; n++)
        for (int r = 0; r < 3; r++)
            Tc[r * GP + n] = s * (R[r * 3] * fp.S[n] + R[r * 3 + 1] * fp.S[GP + n] + R[r * 3 + 2] * fp.S[2 * GP + n]) + t3[r];
    double z2 = 0, sdm[2] = {0};
    int nz = 0;
    for (int rep = 0; rep < 24; rep++) {
        int F = rep % 2 ? 8 : 24; /* fewer frames must give larger sd */
        tracks t;
        make_tracks(&fp, F, 20, 10, 0.5, 0.1, 100 + rep, &t);
        plx_fu_tracks tr = as_tracks(&fp, &t);
        plx_fu_factor_opts o = plain_opts();
        o.visibility = false;
        o.boot = 30;
        o.seed = 7 + rep;
        plx_fu_shape sh;
        CHECK(plx_fu_factorise(&tr, &o, &sh) == PLX_OK);
        for (int n = 0; n < GP; n++) {
            if (!sh.used[n]) continue;
            double zs = (sh.S[2 * GP + n] - Tc[2 * GP + n]) / sh.sd[2 * GP + n];
            z2 += zs * zs, nz++;
            sdm[rep % 2] += sh.sd[2 * GP + n] / (GP * 12.0);
        }
        plx_fu_shape_free(&sh);
        tracks_free(&t);
    }
    double zsd = sqrt(z2 / nz);
    printf("  depth z-score sd %.3f over %d node estimates; mean depth sd %.3f px (24 frames) vs %.3f px (8)\n", zsd, nz,
           sdm[0], sdm[1]);
    CHECK(zsd > 0.8 && zsd < 1.25);
    CHECK(sdm[1] > 1.3 * sdm[0]);
}

TEST(visibility) {
    /* a frontal plane is fully visible; a plane turned 75° only faintly; a bump hides what is behind */
    enum { NX = 11, NY = 11, N = NX * NY };
    double S[3 * N], vis[N], R[9];
    for (int i = 0; i < NY; i++)
        for (int j = 0; j < NX; j++) {
            int n = i * NX + j;
            S[n] = (j - 5) * 5.0, S[N + n] = (i - 5) * 5.0, S[2 * N + n] = 0;
        }
    plx_rot_ypr(0, 0, 0, R);
    plx_fu_grid_visibility(S, NULL, NX, NY, R, 1, 60, 60, 120, 120, vis);
    CHECK_NEAR(vis[60], 1, 1e-12);
    CHECK_NEAR(vis[0], 1, 1e-12);
    plx_rot_ypr(75 * DEG, 0, 0, R);
    plx_fu_grid_visibility(S, NULL, NX, NY, R, 1, 60, 60, 120, 120, vis);
    CHECK(vis[60] > 0.4 && vis[60] < 0.6); /* cos 75° / 0.5 */
    /* a tall ridge along the middle column hides the far side at 60° yaw */
    for (int i = 0; i < NY; i++) S[2 * N + i * NX + 5] = 30, S[2 * N + i * NX + 4] = 15, S[2 * N + i * NX + 6] = 15;
    plx_rot_ypr(60 * DEG, 0, 0, R);
    plx_fu_grid_visibility(S, NULL, NX, NY, R, 1, 60, 60, 120, 120, vis);
    CHECK(vis[5 * NX + 8] == 0);  /* behind the ridge (positive yaw turns +x away) */
    CHECK(vis[5 * NX + 1] > 0.5); /* the near side */
}

/* An LFW photo pasted at scale 1.3 into a cluttered 360 x 300 frame is found: the face centre
 * within a tenth of a face, the scale within 10 %. Truth = the photo's flow-stage similarity
 * (runs/main/flow/sim.plxt, row 0 = AJ_Cook_0001) carried through the paste. */
TEST(locate_pasted_face) {
    const char *jpg = "data/lfw/AJ_Cook/AJ_Cook_0001.jpg", *simp = "runs/main/flow/sim.plxt", *model = "runs/main/flow/model";
    if (access(jpg, R_OK) || access(simp, R_OK) || access(model, R_OK)) {
        printf("  skipped: needs %s, %s and %s\n", jpg, simp, model);
        return;
    }
    plx_img rgb, gray;
    plx_tensor sim;
    plx_fl_model m;
    CHECK(plx_img_load(jpg, &rgb) == PLX_OK);
    CHECK(plx_tensor_load(simp, &sim) == PLX_OK);
    CHECK(plx_fl_model_load(model, &m) == PLX_OK);
    gray = plx_img_luma(&rgb);
    const float *s = sim.data;
    plx_sim T = {s[0], s[1], s[2], s[3]};
    plx_fu_synth_opts so = plx_fu_synth_defaults();
    so.frame_w = 360, so.frame_h = 300, so.clutter = true, so.seed = 9;
    plx_img frame = plx_fu_synth_background(&so, 1);
    double g = 1.3, ox = 140, oy = -20, fcx, fcy;
    plx_sim_apply(T, PLX_CANON_CX, PLX_CANON_CY, &fcx, &fcy);
    for (int y = 0; y < frame.h; y++)
        for (int x = 0; x < frame.w; x++) {
            double qx = (x - ox) / g, qy = (y - oy) / g, ex = (qx - fcx) / 75, ey = (qy - fcy) / 95;
            double al = PLX_CLAMP((1.15 - sqrt(ex * ex + ey * ey)) / 0.15, 0.0, 1.0);
            if (al > 0 && qx >= 0 && qy >= 0 && qx <= 249 && qy <= 249)
                PLX_PX(&frame, x, y, 0) = (float)((1 - al) * PLX_PX(&frame, x, y, 0) +
                                                  al * plx_sample1(&gray, 0, PLX_INTERP_BILINEAR, (float)qx, (float)qy));
        }
    plx_fu_locator L;
    plx_fu_locator_init(&L, &m);
    plx_fu_locate_opts lo = plx_fu_locate_defaults();
    plx_fu_located r = plx_fu_locate(&L, &frame, &lo);
    double ex, ey;
    plx_sim_apply(r.T, PLX_CANON_CX, PLX_CANON_CY, &ex, &ey);
    double err = hypot(ex - (g * fcx + ox), ey - (g * fcy + oy)) / (100 * g), sr = plx_sim_scale(r.T) / (g * plx_sim_scale(T));
    printf("  pasted face: centre error %.3f faces, scale ratio %.3f, score %.3f (runner-up %.3f)\n", err, sr, r.score,
           r.runner_score);
    CHECK(err < 0.1);
    CHECK(fabs(log(sr)) < 0.1);
    CHECK(r.score > r.runner_score);
    /* the window built from it is the LFW layout: the face's canonical centre at (124.5, 128.5) */
    plx_fu_window w = plx_fu_window_for(r.T);
    plx_sim C = plx_fu_sim_to_crop(w, r.T);
    double cx, cy;
    plx_sim_apply(C, PLX_CANON_CX, PLX_CANON_CY, &cx, &cy);
    CHECK_NEAR(cx, PLX_CANON_IMG_CX, 1e-9);
    CHECK_NEAR(cy, PLX_CANON_IMG_CY, 1e-9);
    CHECK_NEAR(plx_sim_scale(C), 1, 1e-12);
    plx_sim back = plx_fu_sim_to_frame(w, C);
    CHECK_NEAR(back.tx, r.T.tx, 1e-9);
    CHECK_NEAR(back.ty, r.T.ty, 1e-9);
    plx_fu_locator_free(&L);
    plx_fl_model_free(&m);
    plx_img_free(&rgb), plx_img_free(&gray), plx_img_free(&frame);
    plx_tensor_free(&sim);
}

int main(void) {
    const char *base = getenv("PLX_TEST_TMP");
    snprintf(g_dir, sizeof g_dir, "%s/plx_test_fuse_%d", base ? base : "/tmp", (int)getpid());
    plx_mkdir_p(g_dir);
    RUN(container_roundtrip);
    RUN(procrustes_and_angles);
    RUN(factor_exact);
    RUN(factor_noise_and_missing);
    RUN(degenerate_motion);
    RUN(mirror_sign);
    RUN(uncertainty_calibration);
    RUN(visibility);
    RUN(locate_pasted_face);
    return TEST_REPORT();
}
