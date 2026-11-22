// plx-deps: geom render viz align ops eval data img la core
/* plx_emerge: the geometry-emergence stage (docs/DESIGN.md §5). From the population of gauged
 * flow fields it recovers, without supervision, the depth of the mean face, every photo's pose,
 * the metric scale of depth, a person-specific depth basis with per-photo uncertainty and the
 * in-plane shape basis (geom/geom.h has the method and every correction to the specification).
 *
 *   plx_emerge --flow runs/main/flow --pack runs/main/pack --out runs/main/geom
 *   plx_emerge --synth [--out runs/main/geom_synth] [--n 13233] [--no-sweeps]
 *
 * Inputs (real mode): <flow>/u.plxt f32 [N, 2, 21, 17], node_weight.plxt f32 [21, 17],
 * sim.plxt f32 [N, 4], mean_face.plxt f32 [160, 128], quality.plxt f32 [N, Q] (optional; columns
 * as align/flow.h plx_fl_quality: cost, inlier, valid, step, min_jdet, max_u, converged);
 * <pack>/gray.plxt u8 [N, 250, 250]. Lattice geometry and gauge box are options, not constants:
 * --cols 17 --rows 21 --spacing 8 --x0 -0.5 --y0 -0.5 --gauge-box 23.5,103.5,23.5,135.5,
 * core ellipse --core 40,50,6 (ax, ay, yc).
 *
 * Writes into <out>:
 *   zbar.plxt         f32 [rows, cols]   mean depth κ v, px (plane gauge: zero ω-mean and Y-slope)
 *   zbar_dense.plxt   f32 [160, 128]     bicubic upsampling, NaN outside the core
 *   pose.plxt         f32 [N, 9]         POSE_COLUMNS below
 *   kappa.txt                            κ, 95% bootstrap CI, κ_yaw, κ_pitch, fit constants
 *   depth_basis.plxt  f32 [Kz, rows, cols] κ B (px per unit code), NaN outside the core
 *   sigma0.plxt       f32 [P, P]         Σ0 in node units (px², even residual field, core only)
 *   sigma_rest.plxt   f32 [P, P]         Σ_rest likewise (per unit s²)
 *   depth_code.plxt   f32 [N, Kz + Kz²]  ĉ_i then Λ_i row-major
 *   shape_basis.plxt  f32 [Ks, 2, rows, cols] 1-σ in-plane modes, px (x odd, y even)
 *   shape_code.plxt   f32 [N, Ks]        whitened codes
 *   summary.txt, diag/<n>_<name>.png
 * Synthetic mode adds recovery.txt (estimates vs ground truth) and sweeps.txt. */
#include "align/canon.h"
#include "core/core.h"
#include "data/lfw.h"
#include "eval/verif.h"
#include "geom/geom.h"
#include "img/image.h"
#include "img/imgio.h"
#include "la/la.h"
#include "render/render.h"
#include "viz/viz.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define DEG (M_PI / 180.0)
enum { PCOLS = 9 };
static const char *const POSE_COLUMNS[PCOLS] = {"s", "t", "yaw", "pitch", "beta", "gamma", "logA", "weight", "flags"};
static const float WHITE[3] = {1, 1, 1}, RED[3] = {1, 0.3f, 0.2f}, CYAN[3] = {0.3f, 0.85f, 1}, GREY[3] = {0.6f, 0.6f, 0.6f},
                   YELLOW[3] = {1, 0.85f, 0.2f};

/* ---- small utilities ------------------------------------------------------------------------- */
static FILE *g_sum; /* summary.txt; every report line also goes to stdout */
static void say(const char *fmt, ...) PLX_PRINTF(1, 2);
static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (g_sum) {
        va_start(ap, fmt);
        vfprintf(g_sum, fmt, ap);
        va_end(ap);
    }
}

static bool file_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* q-quantiles of the finite v[i] with w[i] > 0 (w nullable). */
static void quantiles(const double *v, const double *w, int n, const double *q, int nq, double *out) {
    double *t = PLX_NEW(double, n + 1);
    int k = 0;
    for (int i = 0; i < n; i++)
        if ((!w || w[i] > 0) && isfinite(v[i])) t[k++] = v[i];
    qsort(t, k, sizeof(double), cmp_double);
    for (int j = 0; j < nq; j++) {
        double pos = q[j] * (k - 1);
        int i = (int)pos;
        out[j] = k == 0 ? NAN : i + 1 < k ? t[i] + (pos - i) * (t[i + 1] - t[i]) : t[k - 1];
    }
    free(t);
}

static void save_f32(const char *dir, const char *name, int ndim, const int64_t *dim, const double *v) {
    int64_t n = 1;
    for (int k = 0; k < ndim; k++) n *= dim[k];
    float *f = PLX_NEW(float, n + 1);
    for (int64_t i = 0; i < n; i++) f[i] = (float)v[i];
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plx_tensor_save(path, PLX_F32, ndim, dim, f) != PLX_OK) plx_die("cannot write %s", path);
    free(f);
}

static void save_png(const char *dir, const char *name, const plx_img *im) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plx_img_save(path, im) != PLX_OK) plx_die("cannot write %s", path);
}

static void parse_list(const char *s, double *v, int n, const char *what) {
    for (int k = 0; k < n; k++) {
        char *end;
        v[k] = strtod(s, &end);
        if (end == s) plx_die("--%s: want %d comma-separated numbers", what, n);
        s = *end == ',' ? end + 1 : end;
    }
}

/* ---- lattice fields to dense images -------------------------------------------------------------- */
/* NaN nodes filled by repeated averaging of finite 4-neighbours (a harmonic-like extension), so the
 * bicubic interpolation near the core border has values to lean on. */
static void fill_nan(const plx_glat *L, const double *f, double *g) {
    int P = plx_glat_n(L);
    double *h = PLX_NEW(double, P);
    uint8_t *fixed = PLX_NEW(uint8_t, P);
    memcpy(g, f, sizeof(double) * P);
    for (int k = 0; k < P; k++) fixed[k] = isfinite(f[k]);
    for (int it = 0; it < 4 * (L->cols + L->rows); it++) {
        memcpy(h, g, sizeof(double) * P);
        for (int k = 0; k < P; k++) {
            if (fixed[k]) continue;
            int r = k / L->cols, c = k % L->cols, nb[4][2] = {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
            double s = 0;
            int m = 0;
            for (int j = 0; j < 4; j++)
                if (nb[j][0] >= 0 && nb[j][0] < L->rows && nb[j][1] >= 0 && nb[j][1] < L->cols &&
                    isfinite(h[nb[j][0] * L->cols + nb[j][1]]))
                    s += h[nb[j][0] * L->cols + nb[j][1]], m++;
            g[k] = m ? s / m : NAN;
        }
    }
    for (int k = 0; k < P; k++)
        if (!isfinite(g[k])) g[k] = 0;
    free(h), free(fixed);
}

static double catmull(double p0, double p1, double p2, double p3, double t) {
    return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
}

/* Field on the canonical frame (W x H = 128 x 160 times `factor`), Catmull-Rom over the nodes; NaN
 * outside the core ellipse when masked. Values are multiplied by `vscale`. */
static plx_img dense(const plx_glat *L, const double *f, plx_core_shape cs, int factor, bool mask, double vscale) {
    int W = PLX_CANON_W * factor, H = PLX_CANON_H * factor, P = plx_glat_n(L);
    double *g = PLX_NEW(double, P);
    fill_nan(L, f, g);
    plx_img im = plx_img_new(W, H, 1);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double cx = (x + 0.5) / factor - 0.5, cy = (y + 0.5) / factor - 0.5;
            double ex = (cx - L->cx) / cs.ax, ey = (cy - L->cy - cs.yc) / cs.ay;
            if (mask && ex * ex + ey * ey > 1) {
                PLX_PX(&im, x, y, 0) = NAN;
                continue;
            }
            double gx = (cx - L->x0) / L->spacing, gy = (cy - L->y0) / L->spacing;
            int ix = (int)floor(gx), iy = (int)floor(gy);
            double tx = gx - ix, ty = gy - iy, row[4];
            for (int j = 0; j < 4; j++) {
                int r = PLX_CLAMP(iy - 1 + j, 0, L->rows - 1);
                double p[4];
                for (int i = 0; i < 4; i++) p[i] = g[r * L->cols + PLX_CLAMP(ix - 1 + i, 0, L->cols - 1)];
                row[j] = catmull(p[0], p[1], p[2], p[3], tx);
            }
            PLX_PX(&im, x, y, 0) = (float)(vscale * catmull(row[0], row[1], row[2], row[3], ty));
        }
    free(g);
    return im;
}

/* ---- drawing helpers ------------------------------------------------------------------------------ */
static plx_img canvas(int w, int h) {
    plx_img im = plx_img_new(w, h, 3);
    for (size_t k = 0; k < (size_t)3 * w * h; k++) im.px[k] = 0.1f;
    return im;
}

static plx_img vstack(const plx_img *im, const char *const *titles, int n) {
    int W = 0, H = 0, pad = 6, th = 14;
    for (int i = 0; i < n; i++) W = PLX_MAX(W, im[i].w), H += im[i].h + pad + (titles ? th : 0);
    plx_img out = canvas(W + 2 * pad, H + pad);
    int y = pad;
    for (int i = 0; i < n; i++) {
        if (titles && titles[i]) plx_draw_text(&out, pad, y, titles[i], 1, WHITE), y += th;
        plx_img rgb = plx_viz_to_rgb(&im[i]);
        for (int c = 0; c < 3; c++)
            for (int yy = 0; yy < rgb.h; yy++)
                for (int xx = 0; xx < rgb.w; xx++) PLX_PX(&out, pad + xx, y + yy, c) = PLX_PX(&rgb, xx, yy, c);
        plx_img_free(&rgb);
        y += im[i].h + pad;
    }
    return out;
}

typedef struct { double x0, x1, y0, y1; int L, R, T, B, w, h; } plot_frame;
static float px_x(const plot_frame *f, double x) { return (float)(f->L + (x - f->x0) / (f->x1 - f->x0) * (f->w - f->L - f->R)); }
static float px_y(const plot_frame *f, double y) { return (float)(f->h - f->B - (y - f->y0) / (f->y1 - f->y0) * (f->h - f->T - f->B)); }

static plx_img plot_axes(plot_frame *f, const char *title, const char *xl, const char *yl) {
    plx_img im = canvas(f->w, f->h);
    plx_draw_line(&im, (float)f->L, (float)(f->h - f->B), (float)(f->w - f->R), (float)(f->h - f->B), 1, GREY, 1);
    plx_draw_line(&im, (float)f->L, (float)f->T, (float)f->L, (float)(f->h - f->B), 1, GREY, 1);
    char buf[64];
    for (int k = 0; k <= 4; k++) {
        double x = f->x0 + k * (f->x1 - f->x0) / 4, y = f->y0 + k * (f->y1 - f->y0) / 4;
        snprintf(buf, sizeof buf, "%.3g", x);
        plx_draw_text(&im, (int)px_x(f, x) - 10, f->h - f->B + 6, buf, 1, GREY);
        snprintf(buf, sizeof buf, "%.3g", y);
        plx_draw_text(&im, 4, (int)px_y(f, y) - 4, buf, 1, GREY);
    }
    plx_draw_text(&im, f->L, 6, title, 1, WHITE);
    plx_draw_text(&im, f->w / 2 - 40, f->h - 14, xl, 1, WHITE);
    plx_draw_text(&im, 4, f->T - 14, yl, 1, WHITE);
    return im;
}

static plx_img hist_plot(const char *title, const double *v, const double *w, int n, double x0, double x1, int nb,
                         const float *col) {
    double *cnt = PLX_NEW(double, nb), mx = 0;
    for (int i = 0; i < n; i++) {
        if ((w && w[i] <= 0) || !isfinite(v[i])) continue;
        int b = (int)floor((v[i] - x0) / (x1 - x0) * nb);
        if (b >= 0 && b < nb) cnt[b]++;
    }
    for (int b = 0; b < nb; b++) mx = fmax(mx, cnt[b]);
    plot_frame f = {.x0 = x0, .x1 = x1, .y0 = 0, .y1 = mx * 1.05, .L = 60, .R = 20, .T = 30, .B = 36, .w = 820, .h = 260};
    plx_img im = plot_axes(&f, title, "degrees", "images");
    for (int b = 0; b < nb; b++) {
        double xa = x0 + b * (x1 - x0) / nb, xb = xa + (x1 - x0) / nb;
        plx_draw_rect(&im, px_x(&f, xa) + 0.5f, px_y(&f, cnt[b]), px_x(&f, xb) - 0.5f, px_y(&f, 0), col, 0.9f);
    }
    free(cnt);
    return im;
}

/* ---- outputs shared by both modes ------------------------------------------------------------------ */
typedef struct {
    plx_glat L;
    plx_core_shape cs;
    uint8_t *core;
    float *node_weight; /* nullable */
    double *gauge;
    plx_img tex;        /* mean face, 128 x 160 (w = 0 when absent) */
} context;

/* z̄ with the optional tilt c1 Y0 (renders only; the saved zbar keeps the plane gauge). */
static void zbar_nodes(const context *cx, const plx_geom_model *m, bool tilt, double *z) {
    for (int k = 0; k < m->sp.P; k++) {
        z[k] = m->kappa * m->v[k];
        if (tilt && isfinite(m->tilt_c1)) z[k] += m->tilt_c1 * plx_glat_Y0(&cx->L, k);
    }
}

static void node_cov(const plx_geom_model *m, const double *Sr, double *out) {
    /* Σ_nodes = W⁻¹ Uc S Ucᵀ W⁻¹ on the even core coordinates, mirror-duplicated */
    const plx_gspace *sp = &m->sp;
    int P = sp->P, d = sp->ne, nr = m->nr;
    double *T = PLX_NEW(double, (size_t)d * nr);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < nr; j++) {
            double acc = 0;
            for (int k = 0; k < nr; k++) acc += m->Uc[(size_t)i * nr + k] * Sr[(size_t)k * nr + j];
            T[(size_t)i * nr + j] = acc;
        }
    memset(out, 0, sizeof(double) * P * P);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            double acc = 0;
            for (int k = 0; k < nr; k++) acc += T[(size_t)i * nr + k] * m->Uc[(size_t)j * nr + k];
            acc /= sp->esw[i] * sp->esw[j];
            int a = sp->enode[i], b = sp->enode[j], am = plx_glat_mirror(&sp->lat, a), bm = plx_glat_mirror(&sp->lat, b);
            out[(size_t)a * P + b] = out[(size_t)am * P + b] = out[(size_t)a * P + bm] = out[(size_t)am * P + bm] = acc;
        }
    free(T);
}

static void write_outputs(const char *dir, const context *cx, const plx_geom_model *m) {
    const plx_glat *L = &cx->L;
    int n = m->n, P = m->sp.P, kz = m->kz, ks = m->ks;
    double *z = PLX_NEW(double, P);
    zbar_nodes(cx, m, false, z);
    save_f32(dir, "zbar.plxt", 2, (int64_t[]){L->rows, L->cols}, z);
    plx_img zd = dense(L, z, cx->cs, 1, true, 1.0);
    {
        double *dd = PLX_NEW(double, (size_t)zd.w * zd.h);
        for (int k = 0; k < zd.w * zd.h; k++) dd[k] = zd.px[k];
        save_f32(dir, "zbar_dense.plxt", 2, (int64_t[]){zd.h, zd.w}, dd);
        free(dd);
    }
    plx_img_free(&zd);
    double *pose = PLX_NEW(double, (size_t)n * PCOLS);
    for (int i = 0; i < n; i++) {
        double th, ph, *p = pose + (size_t)i * PCOLS;
        plx_geom_angles(m->fx, m->kappa, m->s[i], m->t[i], &th, &ph);
        p[0] = m->s[i], p[1] = m->t[i], p[2] = th, p[3] = ph, p[4] = m->beta[i], p[5] = m->gamma[i];
        p[6] = m->logA[i], p[7] = m->w[i], p[8] = m->flags[i];
    }
    save_f32(dir, "pose.plxt", 2, (int64_t[]){n, PCOLS}, pose);
    free(pose);
    char path[4096];
    snprintf(path, sizeof path, "%s/kappa.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) plx_die("cannot write %s", path);
    fprintf(f, "kappa %.6g\nkappa_ci95 %.6g %.6g\nkappa_yaw %.6g\nkappa_pitch %.6g\nfit_c %.6g\nfit_d %.6g\n"
               "fit_scale %.6g\nfit_n %d\nfx %.6g\ntilt_c1 %.6g\ntilt_se %.6g\n",
            m->kappa, m->kappa_lo, m->kappa_hi, m->kappa_yaw, m->kappa_pitch, m->kappa_c, m->kappa_d, m->kappa_scale,
            m->kappa_n, m->fx, m->tilt_c1, m->tilt_se);
    fclose(f);
    double *B = PLX_NEW(double, (size_t)(kz + 1) * P);
    for (int a = 0; a < kz; a++)
        for (int k = 0; k < P; k++) B[(size_t)a * P + k] = m->kappa * m->Bnode[(size_t)a * P + k];
    save_f32(dir, "depth_basis.plxt", 3, (int64_t[]){kz, L->rows, L->cols}, B);
    free(B);
    double *C = PLX_NEW(double, (size_t)P * P);
    node_cov(m, m->S0, C);
    save_f32(dir, "sigma0.plxt", 2, (int64_t[]){P, P}, C);
    node_cov(m, m->Sr, C);
    save_f32(dir, "sigma_rest.plxt", 2, (int64_t[]){P, P}, C);
    free(C);
    int dc = kz + kz * kz;
    double *code = PLX_NEW(double, (size_t)n * dc + 1);
    for (int i = 0; i < n; i++) {
        memcpy(code + (size_t)i * dc, m->chat + (size_t)i * kz, sizeof(double) * kz);
        memcpy(code + (size_t)i * dc + kz, m->Lam + (size_t)i * kz * kz, sizeof(double) * kz * kz);
    }
    save_f32(dir, "depth_code.plxt", 2, (int64_t[]){n, dc}, code);
    free(code);
    save_f32(dir, "shape_basis.plxt", 4, (int64_t[]){ks, 2, L->rows, L->cols}, m->shape);
    save_f32(dir, "shape_code.plxt", 2, (int64_t[]){n, ks}, m->scode);
    free(z);
}

/* ---- the numbers ----------------------------------------------------------------------------------- */
typedef struct {
    double nose_px, cheek_px, protrusion, eye_dist, face_w;
} depth_facts;

/* Nose protrusion: max of z̄ over the face-centre nodes minus the mean over cheek nodes
 * (|X0| in [20, 36], Y0 in [-4, 20]); eye distance from the mean face (darkest blob per side). */
static depth_facts facts(const context *cx, const plx_geom_model *m) {
    depth_facts d = {.nose_px = -INFINITY};
    double cs = 0;
    int nc = 0;
    for (int k = 0; k < m->sp.P; k++) {
        if (m->sp.omega[k] <= 0) continue;
        double X = plx_glat_X0(&cx->L, k), Y = plx_glat_Y0(&cx->L, k), z = m->kappa * m->v[k];
        if (plx_core_centre(&cx->L, cx->cs, k)) d.nose_px = fmax(d.nose_px, z);
        if (fabs(X) >= 20 && fabs(X) <= 36 && Y >= -4 && Y <= 20) cs += z, nc++;
    }
    d.cheek_px = nc ? cs / nc : NAN;
    d.protrusion = d.nose_px - d.cheek_px;
    d.face_w = 2 * cx->cs.ax;
    d.eye_dist = 38; /* the synthetic face's sockets at X0 = ±19 */
    if (cx->tex.w) {
        double best = INFINITY;
        int bx = 0;
        for (int y = 52; y <= 72; y++)
            for (int x = 66; x <= 104; x++) {
                double s = 0;
                for (int dy = -3; dy <= 3; dy++)
                    for (int dx = -3; dx <= 3; dx++) s += PLX_PX(&cx->tex, x + dx, y + dy, 0);
                if (s < best) best = s, bx = x;
            }
        d.eye_dist = 2 * (bx - PLX_CANON_CX);
    }
    return d;
}

static void report(const context *cx, const plx_geom_model *m) {
    int n = m->n, nw = 0, nviol = 0, nsk = 0, nflag[5] = {0};
    for (int i = 0; i < n; i++) {
        nw += m->w[i] >= 0.5;
        nviol += (m->flags[i] & PLX_GF_YAWVIOL) != 0;
        nsk += fabs(m->s[i]) / m->kappa > 1;
        for (int b = 0; b < 5; b++) nflag[b] += (m->flags[i] >> b) & 1;
    }
    say("pose.plxt columns:");
    for (int k = 0; k < PCOLS; k++) say(" %d=%s", k, POSE_COLUMNS[k]);
    say(" (yaw, pitch in radians; flags: 1 prior, 2 outlier, 4 yaw violation, 8 pitch violation, 16 not in kappa fit)\n");
    say("images %d (robust weight >= 0.5: %d); core nodes: %d even + %d odd coordinates; gauge fx %.4f\n", n, nw,
        m->sp.ne, m->sp.no, m->fx);
    say("flags: prior %d, outlier %d, yaw-violation %d, pitch-violation %d, not-in-kappa-fit %d\n", nflag[0], nflag[1],
        nflag[2], nflag[3], nflag[4]);
    say("\n[3] yaw mode: sigma1^2/sum sigma^2 = %.4f, sigma1/sigma2 = %.2f; leading sigma_k^2 shares:", m->sv_frac1,
        m->sv_ratio12);
    for (int k = 0; k < 6; k++) say(" %.4f", m->sv2[k] / m->sv_total);
    say("\n[4] rigid-body check: V-perp energy along v = %.4f; E-singular-vector controls 2..6 =", m->rigid_frac);
    for (int k = 0; k < PLX_GEOM_NCTRL; k++) say(" %.4f", m->rigid_ctrl[k]);
    say("\n    scrambled v: %.4f +- %.4f  ->  z = %.1f (vs singular-vector controls: z = %.1f)\n", m->rigid_scr_mean,
        m->rigid_scr_sd, m->rigid_z, m->rigid_z_ctrl);
    say("[5] kappa = %.4f px  (95%% bootstrap CI %.4f .. %.4f, n = %d); linearised kappa_yaw = %.4f, kappa_pitch = "
        "%.4f\n    fit: c = %.4f, d (linear pitch term) = %.4f, robust scale %.4f\n",
        m->kappa, m->kappa_lo, m->kappa_hi, m->kappa_n, m->kappa_yaw, m->kappa_pitch, m->kappa_c, m->kappa_d,
        m->kappa_scale);
    say("    |s|/kappa > 1: %d images (%.2f%%); no real yaw (rho |s|/kappa >= 1): %d (%.2f%%)\n", nsk, 100.0 * nsk / n,
        nviol, 100.0 * nviol / n);
    if (isfinite(m->tilt_c1))
        say("[8] depth tilt c1 = %.4f +- %.4f px/px (Y-slope of z-bar, relative to the population-mean pitch)\n",
            m->tilt_c1, m->tilt_se);
    double *yaw = PLX_NEW(double, n), *pit = PLX_NEW(double, n), *w = PLX_NEW(double, n);
    for (int i = 0; i < n; i++) {
        plx_geom_angles(m->fx, m->kappa, m->s[i], m->t[i], yaw + i, pit + i);
        yaw[i] /= DEG, pit[i] /= DEG;
        w[i] = m->w[i] >= 0.5 && !(m->flags[i] & (PLX_GF_YAWVIOL | PLX_GF_PITCHVIOL));
    }
    double q[7] = {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99}, qy[7], qp[7], sy = 0, sp2 = 0, sw = 0, my = 0, mp = 0;
    quantiles(yaw, w, n, q, 7, qy);
    quantiles(pit, w, n, q, 7, qp);
    for (int i = 0; i < n; i++) sw += w[i], my += w[i] * yaw[i], mp += w[i] * pit[i];
    my /= sw, mp /= sw;
    for (int i = 0; i < n; i++) sy += w[i] * (yaw[i] - my) * (yaw[i] - my), sp2 += w[i] * (pit[i] - mp) * (pit[i] - mp);
    say("    implied yaw (deg): mean %+.2f std %.2f | p1 %+.1f p5 %+.1f p25 %+.1f p50 %+.1f p75 %+.1f p95 %+.1f p99 %+.1f\n",
        my, sqrt(sy / sw), qy[0], qy[1], qy[2], qy[3], qy[4], qy[5], qy[6]);
    say("    implied pitch (deg, relative to the population mean): mean %+.2f std %.2f | p1 %+.1f p5 %+.1f p25 %+.1f "
        "p50 %+.1f p75 %+.1f p95 %+.1f p99 %+.1f\n",
        mp, sqrt(sp2 / sw), qp[0], qp[1], qp[2], qp[3], qp[4], qp[5], qp[6]);
    free(yaw), free(pit), free(w);
    depth_facts d = facts(cx, m);
    say("    depth: nose %.2f px above the cheeks (nose max %.2f, cheek mean %.2f); = %.3f of the inter-eye distance "
        "(%.1f px), %.3f of the core width (%.0f px)\n",
        d.protrusion, d.nose_px, d.cheek_px, d.protrusion / d.eye_dist, d.eye_dist, d.protrusion / d.face_w, d.face_w);
    say("[6] depth variation: K_z = %d of %d; Sigma_B eigenvalues:", m->kz, m->nr);
    for (int k = 0; k < 10 && k < m->nr; k++) say(" %.4g", m->zeig[k]);
    say("\n    noise floor (|most negative|) %.4g; tr Sigma_B %.4g, tr Sigma_0 %.4g (complement coordinates)\n",
        m->zeig_floor, m->sb_trace, m->s0_trace);
    double tot = 0, kept = 0;
    for (int k = 0; k < m->nsh; k++) tot += m->sheig[k];
    for (int k = 0; k < m->ks; k++) kept += m->sheig[k];
    say("[7] in-plane shape: K_s = %d of %d (%.1f%% of variance); leading variance shares:", m->ks, m->nsh,
        100 * kept / tot);
    for (int k = 0; k < 10 && k < m->nsh; k++) say(" %.4f", m->sheig[k] / tot);
    say("\n");
}

/* ---- verification by the depth codes ------------------------------------------------------------------ */
typedef struct { double auc[3]; int n[3]; } verif_out;

static void verify_pairs(const plx_geom_model *m, const plx_pair *p, int np, verif_out *out) {
    double *sc = PLX_NEW(double, np);
    int *same = PLX_NEW(int, np), kz = m->kz;
    for (int j = 0; j < np; j++) {
        int a = p[j].a, b = p[j].b;
        sc[j] = plx_geom_llr(kz, m->chat + (size_t)a * kz, m->Lam + (size_t)a * kz * kz, m->chat + (size_t)b * kz,
                             m->Lam + (size_t)b * kz * kz);
        same[j] = p[j].same;
    }
    double lim[3] = {-1, 10, 20};
    double *ss = PLX_NEW(double, np);
    int *sm = PLX_NEW(int, np);
    for (int c = 0; c < 3; c++) {
        int k = 0, ns = 0;
        for (int j = 0; j < np; j++) {
            double ta, tb;
            plx_geom_angles(m->fx, m->kappa, m->s[p[j].a], m->t[p[j].a], &ta, NULL);
            plx_geom_angles(m->fx, m->kappa, m->s[p[j].b], m->t[p[j].b], &tb, NULL);
            if (fabs(ta) / DEG > lim[c] && fabs(tb) / DEG > lim[c]) ss[k] = sc[j], sm[k] = same[j], ns += same[j], k++;
        }
        out->n[c] = k;
        out->auc[c] = (ns > 0 && ns < k) ? plx_verif_auc(ss, sm, k) : NAN;
    }
    free(sc), free(same), free(ss), free(sm);
}

/* ---- diagnostics ---------------------------------------------------------------------------------------- */
static plx_img render_pose_set(const context *cx, const double *z, bool textured, const plx_pose *poses, int k) {
    const int f = 2;
    plx_img zd = dense(&cx->L, z, cx->cs, f, true, f); /* heights in enlarged-grid pixels */
    plx_img tex = {0};
    if (textured && cx->tex.w) tex = plx_viz_upscale(&cx->tex, f, PLX_UP_BILINEAR);
    plx_img out = plx_render_views(&zd, NULL, tex.w ? &tex : NULL, poses, k);
    plx_img_free(&zd), plx_img_free(&tex);
    return out;
}

static void diag_depth(const char *dir, const context *cx, const plx_geom_model *m, const double *ztrue) {
    int P = m->sp.P;
    double *z = PLX_NEW(double, P);
    zbar_nodes(cx, m, true, z);
    /* 1. depth map next to the texture (and the truth, synthetic mode) */
    {
        plx_img zd = dense(&cx->L, z, cx->cs, 2, true, 1.0);
        plx_img tiles[3];
        const char *lab[3];
        int nt = 0;
        float lo, hi;
        plx_viz_auto_range(zd.px, (size_t)zd.w * zd.h, 0.0f, 1.0f, &lo, &hi);
        tiles[nt] = plx_viz_colormap(zd.px, zd.w, zd.h, lo, hi, PLX_CMAP_SEQ), lab[nt++] = "recovered z-bar (px)";
        if (ztrue) {
            plx_img zt = dense(&cx->L, ztrue, cx->cs, 2, true, 1.0);
            /* same plane gauge for the comparison: match mean and Y-slope to the recovered map */
            double a[3] = {0}, b[2] = {0};
            for (int y = 0; y < zt.h; y++)
                for (int x = 0; x < zt.w; x++) {
                    double p = PLX_PX(&zt, x, y, 0), q = PLX_PX(&zd, x, y, 0);
                    if (!isfinite(p)) continue;
                    a[0] += 1, a[1] += y, a[2] += (double)y * y, b[0] += q - p, b[1] += (q - p) * y;
                }
            double det = a[0] * a[2] - a[1] * a[1], c0 = (a[2] * b[0] - a[1] * b[1]) / det,
                   c1 = (a[0] * b[1] - a[1] * b[0]) / det;
            for (int y = 0; y < zt.h; y++)
                for (int x = 0; x < zt.w; x++) PLX_PX(&zt, x, y, 0) += (float)(c0 + c1 * y);
            tiles[nt] = plx_viz_colormap(zt.px, zt.w, zt.h, lo, hi, PLX_CMAP_SEQ), lab[nt++] = "true z-bar (same gauge)";
            plx_img_free(&zt);
        }
        if (cx->tex.w) tiles[nt] = plx_viz_upscale(&cx->tex, 2, PLX_UP_BILINEAR), lab[nt++] = "mean face";
        plx_montage_opts mo = plx_montage_defaults();
        mo.cols = nt;
        plx_img im = plx_viz_montage(tiles, nt, lab, &mo);
        save_png(dir, "1_depth_map.png", &im);
        for (int k = 0; k < nt; k++) plx_img_free(&tiles[k]);
        plx_img_free(&im), plx_img_free(&zd);
    }
    /* 2. the mean face as a surface, turning */
    {
        plx_pose yaws[9], pitches[5] = {{0, -20 * DEG, 0}, {0, -10 * DEG, 0}, {0, 10 * DEG, 0}, {0, 20 * DEG, 0},
                                        {30 * DEG, 15 * DEG, 0}};
        for (int k = 0; k < 9; k++) yaws[k] = (plx_pose){(-40 + 10 * k) * DEG, 0, 0};
        plx_img a = render_pose_set(cx, z, true, yaws, 9);
        save_png(dir, "2_render_yaw.png", &a);
        plx_img b = render_pose_set(cx, z, true, pitches, 5);
        save_png(dir, "2_render_pitch.png", &b);
        plx_img_free(&a), plx_img_free(&b);
        if (ztrue) {
            plx_img t = render_pose_set(cx, ztrue, false, yaws, 9);
            save_png(dir, "2_render_yaw_truth.png", &t);
            plx_img_free(&t);
        }
    }
    /* 5a. depth-variation modes as ±2σ surfaces */
    if (m->kz > 0) {
        int nk = PLX_MIN(m->kz, 4), ni = 0;
        plx_img *ims = PLX_NEW(plx_img, 2 * nk);
        char **tit = PLX_NEW(char *, 2 * nk);
        plx_pose views[3] = {{-30 * DEG, 0, 0}, {0, 0, 0}, {30 * DEG, 0, 0}};
        double *zk = PLX_NEW(double, P);
        for (int a = 0; a < nk; a++)
            for (int sg = -1; sg <= 1; sg += 2) {
                for (int k = 0; k < P; k++) {
                    double b = m->Bnode[(size_t)a * P + k];
                    zk[k] = z[k] + sg * 2 * m->kappa * (isfinite(b) ? b : 0);
                }
                ims[ni] = render_pose_set(cx, zk, false, views, 3);
                char buf[96];
                snprintf(buf, sizeof buf, "depth mode %d, %s2 sigma (Sigma_B eigenvalue %.3g)", a + 1, sg < 0 ? "-" : "+",
                         m->zeig[a]);
                tit[ni++] = plx_strdup(buf);
            }
        plx_img im = vstack(ims, (const char *const *)tit, ni);
        save_png(dir, "5_depth_modes.png", &im);
        for (int k = 0; k < ni; k++) plx_img_free(&ims[k]), free(tit[k]);
        plx_img_free(&im);
        free(ims), free(tit), free(zk);
    }
    free(z);
}

static void diag_shape_modes(const char *dir, const context *cx, const plx_geom_model *m) {
    int nk = PLX_MIN(m->ks, 6), P = m->sp.P;
    if (!nk) return;
    plx_img *tiles = PLX_NEW(plx_img, nk);
    char **lab = PLX_NEW(char *, nk);
    double tot = 0;
    for (int k = 0; k < m->nsh; k++) tot += m->sheig[k];
    float *dx = PLX_NEW(float, P), *dy = PLX_NEW(float, P);
    for (int a = 0; a < nk; a++) {
        plx_img bg = cx->tex.w ? plx_viz_upscale(&cx->tex, 2, PLX_UP_BILINEAR) : plx_img_new(256, 320, 1);
        for (size_t k = 0; k < (size_t)bg.w * bg.h; k++) bg.px[k] *= 0.6f;
        tiles[a] = plx_viz_to_rgb(&bg);
        plx_img_free(&bg);
        for (int k = 0; k < P; k++) dx[k] = (float)m->shape[(size_t)a * 2 * P + k], dy[k] = (float)m->shape[(size_t)a * 2 * P + P + k];
        /* 3σ arrows, drawn at 2x */
        plx_viz_flow_arrows(&tiles[a], dx, dy, cx->L.cols, cx->L.rows, (float)(2 * (cx->L.x0 + 0.5) - 0.5),
                            (float)(2 * (cx->L.y0 + 0.5) - 0.5), (float)(2 * cx->L.spacing), 6.0f, YELLOW);
        char buf[64];
        snprintf(buf, sizeof buf, "shape %d (%.1f%%), 3 sigma", a + 1, 100 * m->sheig[a] / tot);
        lab[a] = plx_strdup(buf);
    }
    plx_montage_opts mo = plx_montage_defaults();
    mo.cols = nk;
    plx_img im = plx_viz_montage(tiles, nk, (const char *const *)lab, &mo);
    save_png(dir, "5_shape_modes.png", &im);
    for (int a = 0; a < nk; a++) plx_img_free(&tiles[a]), free(lab[a]);
    free(tiles), free(lab), free(dx), free(dy);
    plx_img_free(&im);
}

/* 4. log A against s with the fitted foreshortening curve; 6. yaw / pitch histograms. */
static void diag_plots(const char *dir, const plx_geom_model *m, const double *true_yaw) {
    int n = m->n;
    double *x = PLX_NEW(double, n), *y = PLX_NEW(double, n), *w = PLX_NEW(double, n), q[2] = {0.002, 0.998}, qx[2], qy[2];
    for (int i = 0; i < n; i++) {
        x[i] = m->s[i], y[i] = m->logA[i];
        w[i] = !(m->flags[i] & PLX_GF_NOKAPPA);
    }
    quantiles(x, w, n, q, 2, qx);
    quantiles(y, w, n, q, 2, qy);
    double xm = fmax(fabs(qx[0]), fabs(qx[1]));
    plot_frame f = {.x0 = -xm, .x1 = xm, .y0 = qy[0], .y1 = qy[1], .L = 60, .R = 20, .T = 36, .B = 40, .w = 820, .h = 520};
    plx_img im = plot_axes(&f, "foreshortening: log A = log((1+beta)/(1+gamma)) vs yaw parallax s; red: fit at t = 0; "
                               "cyan: binned medians",
                           "s (px)", "log A");
    for (int i = 0; i < n; i++)
        if (w[i] && isfinite(y[i]) && y[i] >= f.y0 && y[i] <= f.y1)
            plx_draw_disc(&im, px_x(&f, x[i]), px_y(&f, y[i]), 1.2f, GREY, 0.25f);
    int nb = 24;
    for (int b = 0; b < nb; b++) {
        double a0 = -xm + 2 * xm * b / nb, a1 = a0 + 2 * xm / nb, *tmp = PLX_NEW(double, n), med;
        int k = 0;
        for (int i = 0; i < n; i++)
            if (w[i] && isfinite(y[i]) && x[i] >= a0 && x[i] < a1) tmp[k++] = y[i];
        if (k >= 10) {
            double one = 0.5;
            quantiles(tmp, NULL, k, &one, 1, &med);
            plx_draw_disc(&im, px_x(&f, 0.5 * (a0 + a1)), px_y(&f, med), 3.5f, CYAN, 1);
        }
        free(tmp);
    }
    double px0 = 0, py0 = 0;
    for (int g = 0; g <= 200; g++) {
        double s = -xm + 2 * xm * g / 200, th;
        plx_geom_angles(m->fx, m->kappa, s, 0, &th, NULL);
        double ly = m->kappa_c + log(cos(th));
        if (g) plx_draw_line(&im, (float)px0, (float)py0, px_x(&f, s), px_y(&f, ly), 2, RED, 1);
        px0 = px_x(&f, s), py0 = px_y(&f, ly);
    }
    char buf[128];
    snprintf(buf, sizeof buf, "kappa = %.3f px [%.3f, %.3f]", m->kappa, m->kappa_lo, m->kappa_hi);
    plx_draw_text(&im, f.L + 10, f.T + 6, buf, 1, RED);
    save_png(dir, "4_foreshortening.png", &im);
    plx_img_free(&im);

    /* histograms */
    plx_img hs[3];
    const char *ht[3];
    int nh = 0;
    for (int i = 0; i < n; i++) {
        plx_geom_angles(m->fx, m->kappa, m->s[i], m->t[i], x + i, y + i);
        x[i] /= DEG, y[i] /= DEG;
        w[i] = m->w[i] >= 0.5;
    }
    hs[nh] = hist_plot("implied yaw (deg)", x, w, n, -60, 60, 96, YELLOW), ht[nh++] = NULL;
    hs[nh] = hist_plot("implied pitch (deg), relative to the population mean", y, w, n, -60, 60, 96, CYAN),
    ht[nh++] = NULL;
    if (true_yaw) {
        plot_frame fs = {.x0 = -50, .x1 = 50, .y0 = -50, .y1 = 50, .L = 60, .R = 20, .T = 30, .B = 36, .w = 820, .h = 520};
        plx_img sc = plot_axes(&fs, "recovered yaw vs true yaw (deg)", "true", "recovered");
        plx_draw_line(&sc, px_x(&fs, -50), px_y(&fs, -50), px_x(&fs, 50), px_y(&fs, 50), 1, RED, 0.8f);
        for (int i = 0; i < n; i++) plx_draw_disc(&sc, px_x(&fs, true_yaw[i] / DEG), px_y(&fs, x[i]), 1.2f, GREY, 0.3f);
        hs[nh] = sc, ht[nh++] = NULL;
    }
    plx_img all = vstack(hs, ht, nh);
    save_png(dir, "6_pose_histograms.png", &all);
    for (int k = 0; k < nh; k++) plx_img_free(&hs[k]);
    plx_img_free(&all);
    free(x), free(y), free(w);
}

/* 3. pose-sorted montages of the real photos (similarity-aligned crops). */
static void diag_montage(const char *dir, const char *name, const plx_geom_model *m, const plx_tensor *gray,
                         const float *sim, bool by_pitch, uint64_t seed) {
    const int K = 14;
    int n = m->n, *ord = PLX_NEW(int, n), ng = 0;
    double *key = PLX_NEW(double, 2 * (size_t)n);
    for (int i = 0; i < n; i++) {
        if (m->w[i] < 0.5 || (m->flags[i] & (PLX_GF_YAWVIOL | PLX_GF_PITCHVIOL | PLX_GF_PRIOR))) continue;
        double th, ph;
        plx_geom_angles(m->fx, m->kappa, m->s[i], m->t[i], &th, &ph);
        key[2 * ng] = by_pitch ? ph : th, key[2 * ng + 1] = i, ng++;
    }
    qsort(key, ng, 2 * sizeof(double), cmp_double);
    for (int j = 0; j < ng; j++) ord[j] = (int)key[2 * j + 1];
    plx_rng r;
    plx_rng_seed(&r, seed);
    plx_img tiles[2 * K];
    char *lab[2 * K];
    int side = (int)gray->dim[1];
    for (int row = 0; row < 2; row++)
        for (int k = 0; k < K; k++) {
            int lo = (int)((int64_t)ng * k / K), hi = (int)((int64_t)ng * (k + 1) / K);
            int j = row == 0 ? (int)((int64_t)ng * (2 * k + 1) / (2 * K)) : lo + (int)plx_rng_below(&r, (uint32_t)(hi - lo));
            int i = ord[j];
            plx_img src = plx_img_from_u8((const uint8_t *)gray->data + (size_t)i * side * side, side, side, 1, 1 / 255.0f);
            plx_img dst = plx_canon_new(1, 0);
            plx_sim T = {sim[4 * i], sim[4 * i + 1], sim[4 * i + 2], sim[4 * i + 3]};
            plx_canon_warp(&src, T, 0, PLX_INTERP_BILINEAR, &dst, NULL);
            tiles[row * K + k] = dst;
            plx_img_free(&src);
            double th, ph;
            plx_geom_angles(m->fx, m->kappa, m->s[i], m->t[i], &th, &ph);
            char buf[48];
            snprintf(buf, sizeof buf, "%s %+.0f #%d", by_pitch ? "pitch" : "yaw", (by_pitch ? ph : th) / DEG, i);
            lab[row * K + k] = plx_strdup(buf);
        }
    plx_montage_opts mo = plx_montage_defaults();
    mo.cols = K;
    plx_img im = plx_viz_montage(tiles, 2 * K, (const char *const *)lab, &mo);
    save_png(dir, name, &im);
    for (int k = 0; k < 2 * K; k++) plx_img_free(&tiles[k]), free(lab[k]);
    plx_img_free(&im);
    free(ord), free(key);
}

/* ---- options -------------------------------------------------------------------------------------------- */
static context make_context(const plx_args *a) {
    context cx = {0};
    cx.L = (plx_glat){.cols = (int)plx_arg_int(a, "cols", 17), .rows = (int)plx_arg_int(a, "rows", 21),
                      .spacing = plx_arg_num(a, "spacing", 8), .x0 = plx_arg_num(a, "x0", -0.5),
                      .y0 = plx_arg_num(a, "y0", -0.5), .cx = PLX_CANON_CX, .cy = PLX_CANON_CY};
    double c[3], g[4];
    parse_list(plx_arg_str(a, "core", "40,50,6"), c, 3, "core");
    parse_list(plx_arg_str(a, "gauge-box", "23.5,103.5,23.5,135.5"), g, 4, "gauge-box");
    cx.cs = (plx_core_shape){c[0], c[1], c[2]};
    int P = plx_glat_n(&cx.L);
    cx.core = PLX_NEW(uint8_t, P);
    cx.gauge = PLX_NEW(double, P);
    plx_core_mask(&cx.L, cx.cs, cx.core);
    plx_gauge_box(&cx.L, g[0], g[1], g[2], g[3], cx.gauge);
    return cx;
}

static plx_emerge_opts make_opts(const plx_args *a, const context *cx) {
    plx_emerge_opts o = plx_emerge_defaults();
    o.boot = (int)plx_arg_int(a, "boot", o.boot);
    o.kz_max = (int)plx_arg_int(a, "kz-max", o.kz_max);
    o.ks_max = (int)plx_arg_int(a, "ks-max", o.ks_max);
    o.seed = (uint64_t)plx_arg_int(a, "seed", (int64_t)o.seed);
    o.tilt = !plx_arg_flag(a, "no-tilt");
    o.gauge_w = cx->gauge;
    return o;
}

static void load_tex(context *cx, const char *path) {
    plx_tensor t;
    if (!file_exists(path) || plx_tensor_load(path, &t) != PLX_OK) return;
    if (t.dtype == PLX_F32 && t.ndim == 2 && t.dim[0] == PLX_CANON_H && t.dim[1] == PLX_CANON_W) {
        cx->tex = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
        memcpy(cx->tex.px, t.data, sizeof(float) * PLX_CANON_W * PLX_CANON_H);
    }
    plx_tensor_free(&t);
}

static void open_summary(const char *out) {
    char path[4096];
    snprintf(path, sizeof path, "%s/summary.txt", out);
    g_sum = fopen(path, "w");
    if (!g_sum) plx_die("cannot write %s", path);
}

/* ---- synthetic mode ------------------------------------------------------------------------------------- */
typedef struct {
    plx_synth_report r;
    verif_out v;
    int kz;
    double kappa_lo, kappa_hi, kyaw, kpitch, seconds;
} synth_result;

/* Pairs for synthetic verification: photos 2j, 2j+1 share a person (per_person = 2). */
static void synth_pairs(const plx_synth *s, plx_rng *r, plx_pair **out, int *np) {
    int npers = s->n / 2, n = 2 * npers;
    plx_pair *p = PLX_NEW(plx_pair, n);
    for (int j = 0; j < npers; j++) {
        p[2 * j] = (plx_pair){2 * j, 2 * j + 1, 1, 0};
        int k;
        do k = (int)plx_rng_below(r, (uint32_t)npers);
        while (k == j);
        p[2 * j + 1] = (plx_pair){2 * j, 2 * k + 1, 0, 0};
    }
    *out = p, *np = n;
}

static synth_result synth_run(const context *cx, const plx_synth_opts *so, const plx_emerge_opts *o, plx_synth *keep_s,
                              plx_geom_model *keep_m) {
    synth_result res = {.v = {.auc = {NAN, NAN, NAN}}};
    plx_synth s;
    double t0 = plx_now();
    plx_synth_make(&cx->L, NULL, cx->core, cx->gauge, so, &s);
    plx_geom_model m;
    if (plx_emerge(&cx->L, s.u, s.n, NULL, cx->core, cx->cs, o, &m) != PLX_OK) plx_die("emerge failed");
    res.seconds = plx_now() - t0;
    plx_synth_score(&s, &m, &res.r);
    res.kz = m.kz, res.kappa_lo = m.kappa_lo, res.kappa_hi = m.kappa_hi, res.kyaw = m.kappa_yaw, res.kpitch = m.kappa_pitch;
    if (so->per_person == 2) {
        plx_rng r;
        plx_rng_seed(&r, so->seed ^ 0xfa1);
        plx_pair *p;
        int np;
        synth_pairs(&s, &r, &p, &np);
        verify_pairs(&m, p, np, &res.v);
        free(p);
    }
    if (keep_s) *keep_s = s;
    else plx_synth_free(&s);
    if (keep_m) *keep_m = m;
    else plx_geom_model_free(&m);
    return res;
}

static void sweep_header(FILE *f) {
    fprintf(f, "%-22s %7s %7s %13s %6s %6s %6s %6s %6s %6s %6s %3s %6s %6s %6s %6s %6s\n", "setting", "v_corr", "k/k*",
            "k_CI/k*", "ky/k*", "kp/k*", "yawSl", "yawR", "yawE", "pitSl", "pitR", "Kz", "minCos", "r2>30", "z2", "AUC",
            "AUC20");
}

static void sweep_line(FILE *f, const char *name, const synth_result *x) {
    const plx_synth_report *r = &x->r;
    double mc = 1;
    for (int k = 0; k < r->nangle; k++) mc = fmin(mc, r->angle_cos[k]);
    char ci[32];
    snprintf(ci, sizeof ci, "%.3f-%.3f", x->kappa_lo / r->kappa_true, x->kappa_hi / r->kappa_true);
    fprintf(f, "%-22s %7.4f %7.4f %13s %6.3f %6.3f %6.3f %6.4f %6.2f %6.3f %6.4f %3d %6.4f %6.3f %6.3f %6.3f %6.3f\n",
            name, r->v_corr, r->kappa_ratio, isfinite(x->kappa_lo) ? ci : "-", x->kyaw / r->kappa_true,
            x->kpitch / r->kappa_true, r->yaw_slope, r->yaw_corr, r->yaw_rmse, r->pitch_slope, r->pitch_corr, x->kz,
            r->nangle ? mc : NAN, r->code_r2[3], r->calib_z2, x->v.auc[0], x->v.auc[2]);
    fflush(f);
}

static int run_synth_mode(const plx_args *a) {
    const char *out = plx_arg_str(a, "out", "runs/main/geom_synth");
    char diag[4096];
    snprintf(diag, sizeof diag, "%s/diag", out);
    if (plx_mkdir_p(diag) != PLX_OK) plx_die("cannot create %s", diag);
    context cx = make_context(a);
    load_tex(&cx, plx_arg_str(a, "tex", "runs/main/congeal/mean_face.plxt"));
    plx_emerge_opts o = make_opts(a, &cx);
    plx_synth_opts so = plx_synth_defaults();
    so.n = (int)plx_arg_int(a, "n", 13233);
    so.per_person = 2;
    so.seed = (uint64_t)plx_arg_int(a, "seed", 1);
    open_summary(out);
    say("plx_emerge --synth: %d synthetic photos (2 per person), yaw N(0, %.0f deg) clipped +-%.0f, pitch N(%.0f, %.0f), "
        "roll N(0, %.0f), node noise %.2f px, nuisance %.2f px rank %d, identity a %.1f b %.1f px, depth: nose +-15%% + "
        "%d modes %.1f px, aspect sd %.2f. No mirror twins: yaw is sampled symmetrically and the flow stage's "
        "symmetrised-mean gauge supplies the twins' effect.\n\n",
        so.n, so.yaw_sd_deg, so.yaw_clip_deg, so.pitch_mean_deg, so.pitch_sd_deg, so.roll_sd_deg, so.noise_px, so.nuis_px,
        so.nuis_rank, so.a_px, so.b_px, so.kg - 1, so.g_px, so.ar_sd);
    plx_synth s;
    plx_geom_model m;
    synth_result res = synth_run(&cx, &so, &o, &s, &m);
    report(&cx, &m);
    const plx_synth_report *r = &res.r;
    say("\nRECOVERY vs GROUND TRUTH (N = %d, %.1f s)\n", s.n, res.seconds);
    say("  v vs true z-bar (plane gauge): correlation %.5f\n", r->v_corr);
    say("  kappa %.4f vs true %.4f: ratio %.4f (CI ratio %.4f..%.4f); kappa_yaw ratio %.4f, kappa_pitch ratio %.4f\n",
        m.kappa, r->kappa_true, r->kappa_ratio, m.kappa_lo / r->kappa_true, m.kappa_hi / r->kappa_true,
        m.kappa_yaw / r->kappa_true, m.kappa_pitch / r->kappa_true);
    say("  yaw: slope %.4f, corr %.5f, rmse %.2f deg\n", r->yaw_slope, r->yaw_corr, r->yaw_rmse);
    say("  pitch (relative to the mean): slope %.4f, corr %.4f, rmse %.2f deg\n", r->pitch_slope, r->pitch_corr,
        r->pitch_rmse);
    say("  depth basis: K_z = %d (true 4); cos of principal angles to the true subspace:", m.kz);
    for (int k = 0; k < r->nangle; k++) say(" %.4f", r->angle_cos[k]);
    say("\n  personal depth recovered (1 - |B c-hat - d|^2/|d|^2) by |yaw|:");
    for (int b = 0; b < PLX_SYNTH_NBIN; b++) say(" [%g,%g): %.3f", r->bin_edge[b], r->bin_edge[b + 1], r->code_r2[b]);
    say("\n  posterior calibration (|yaw| > 10): mean z^2 %.3f (1 = calibrated); per-component sd of standardised "
        "errors:",
        r->calib_z2);
    for (int k = 0; k < m.kz && k < 8; k++) say(" %.3f", r->calib_sd[k]);
    say("\n  verification by depth codes (LLR): AUC %.4f (%d pairs), both |yaw| > 10: %.4f (%d), > 20: %.4f (%d)\n",
        res.v.auc[0], res.v.n[0], res.v.auc[1], res.v.n[1], res.v.auc[2], res.v.n[2]);
    if (isfinite(m.tilt_c1)) {
        /* the true tilt: ω-weighted slope of the true z̄ on Y0, with intercept */
        double sw = 0, sy = 0, sz = 0, syy = 0, syz = 0;
        for (int k = 0; k < m.sp.P; k++) {
            double Y = plx_glat_Y0(&cx.L, k), w = m.sp.omega[k];
            sw += w, sy += w * Y, sz += w * s.zbar[k], syy += w * Y * Y, syz += w * Y * s.zbar[k];
        }
        say("  depth tilt c1: estimated %.4f +- %.4f, true %.4f\n", m.tilt_c1, m.tilt_se,
            (sw * syz - sy * sz) / (sw * syy - sy * sy));
    }
    write_outputs(out, &cx, &m);
    diag_depth(diag, &cx, &m, s.zbar);
    diag_shape_modes(diag, &cx, &m);
    diag_plots(diag, &m, s.yaw);
    plx_geom_model_free(&m);
    plx_synth_free(&s);

    if (!plx_arg_flag(a, "no-sweeps")) {
        char path[4096];
        snprintf(path, sizeof path, "%s/sweeps.txt", out);
        FILE *f = fopen(path, "w");
        if (!f) plx_die("cannot write %s", path);
        fprintf(f, "Synthetic sweeps (N = 13233 unless stated; one seed per line; columns: v correlation, kappa/true, "
                   "bootstrap CI/true, linearised kappa_yaw and kappa_pitch /true, yaw slope / corr / rmse deg, pitch "
                   "slope / corr, K_z, min principal-angle cosine of B, personal-depth r2 at |yaw| > 30, calibration "
                   "mean z^2, depth-code AUC all / both |yaw| > 20)\n");
        plx_emerge_opts os = o;
        os.boot = 50, os.n_scramble = 50;
        const double noise[] = {0, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0}, yaw[] = {3, 5, 8, 10, 15, 20, 25};
        const int ns[] = {250, 500, 1000, 2000, 4000, 8000, 13233};
        char name[64];
        fprintf(f, "\n-- node noise (nuisance scaled with it: 5/3 x noise) --\n");
        sweep_header(f);
        for (size_t k = 0; k < sizeof noise / sizeof *noise; k++) {
            plx_synth_opts x = so;
            x.noise_px = noise[k], x.nuis_px = noise[k] * 5 / 3;
            snprintf(name, sizeof name, "noise %.2f nuis %.2f", x.noise_px, x.nuis_px);
            synth_result rr = synth_run(&cx, &x, &os, NULL, NULL);
            sweep_line(f, name, &rr);
            plx_log("sweep %s done", name);
        }
        fprintf(f, "\n-- yaw spread (sd, deg) --\n");
        sweep_header(f);
        for (size_t k = 0; k < sizeof yaw / sizeof *yaw; k++) {
            plx_synth_opts x = so;
            x.yaw_sd_deg = yaw[k];
            snprintf(name, sizeof name, "yaw sd %.0f", yaw[k]);
            synth_result rr = synth_run(&cx, &x, &os, NULL, NULL);
            sweep_line(f, name, &rr);
            plx_log("sweep %s done", name);
        }
        fprintf(f, "\n-- N --\n");
        sweep_header(f);
        for (size_t k = 0; k < sizeof ns / sizeof *ns; k++) {
            plx_synth_opts x = so;
            x.n = ns[k];
            snprintf(name, sizeof name, "N %d", ns[k]);
            synth_result rr = synth_run(&cx, &x, &os, NULL, NULL);
            sweep_line(f, name, &rr);
            plx_log("sweep %s done", name);
        }
        fprintf(f, "\n-- seeds (default setting) --\n");
        sweep_header(f);
        for (int k = 2; k <= 5; k++) {
            plx_synth_opts x = so;
            x.seed = (uint64_t)k;
            snprintf(name, sizeof name, "seed %d", k);
            synth_result rr = synth_run(&cx, &x, &os, NULL, NULL);
            sweep_line(f, name, &rr);
        }
        fprintf(f, "\n-- corrections: each row disables one (or stresses the case it was made for) --\n");
        sweep_header(f);
        struct { const char *name; int which; double pmean, tilt; } ab[] = {
            {"all corrections", 0, 0, 0},       {"plain angles (no rho)", 1, 0, 0}, {"plain pitch (no GLS)", 2, 0, 0},
            {"no linear t term", 4, 0, 0},      {"pitch mean -6", 0, -6, 0},
            {"pitch mean -6, no lin", 4, -6, 0}, {"tilt .15", 0, 0, 0.15},          {"tilt .15, no lin", 4, 0, 0.15},
            {"mirror twins added", 5, 0, 0}};
        for (size_t k = 0; k < sizeof ab / sizeof *ab; k++) {
            plx_synth_opts x = so;
            plx_emerge_opts oo = os;
            x.pitch_mean_deg = ab[k].pmean, x.depth_tilt = ab[k].tilt;
            oo.plain_angles = ab[k].which == 1, oo.plain_pitch = ab[k].which == 2;
            oo.no_pitch_lin = ab[k].which == 4;
            if (ab[k].which == 5) x.twins = true, x.n = so.n / 2, x.per_person = 1;
            synth_result rr = synth_run(&cx, &x, &oo, NULL, NULL);
            sweep_line(f, ab[k].name, &rr);
            plx_log("ablation %s done", ab[k].name);
        }
        fclose(f);
        say("\nsweeps written to %s\n", path);
    }
    fclose(g_sum);
    free(cx.core), free(cx.gauge), plx_img_free(&cx.tex);
    return 0;
}

/* ---- real mode ------------------------------------------------------------------------------------------- */
static int run_real_mode(const plx_args *a) {
    const char *flow = plx_arg_str(a, "flow", "runs/main/flow"), *pack = plx_arg_str(a, "pack", "runs/main/pack");
    const char *out = plx_arg_str(a, "out", "runs/main/geom");
    char path[4096], diag[4096];
    context cx = make_context(a);
    int P = plx_glat_n(&cx.L);
    plx_tensor u, nw = {0}, sim, q = {0}, gray;
    snprintf(path, sizeof path, "%s/u.plxt", flow);
    if (plx_tensor_map(path, &u) != PLX_OK) plx_die("cannot read %s", path);
    if (u.dtype != PLX_F32 || u.ndim != 4 || u.dim[1] != 2 || u.dim[2] != cx.L.rows || u.dim[3] != cx.L.cols)
        plx_die("%s: want f32 [N, 2, %d, %d]", path, cx.L.rows, cx.L.cols);
    int n = (int)u.dim[0];
    snprintf(path, sizeof path, "%s/node_weight.plxt", flow);
    if (file_exists(path)) {
        if (plx_tensor_load(path, &nw) != PLX_OK || plx_tensor_count(&nw) != P || nw.dtype != PLX_F32)
            plx_die("%s: want f32 [%d, %d]", path, cx.L.rows, cx.L.cols);
        cx.node_weight = nw.data;
    } else
        plx_log("no %s: unit node weights", path);
    snprintf(path, sizeof path, "%s/sim.plxt", flow);
    if (plx_tensor_load(path, &sim) != PLX_OK || sim.dtype != PLX_F32 || sim.dim[0] != n) plx_die("cannot read %s", path);
    snprintf(path, sizeof path, "%s/mean_face.plxt", flow);
    load_tex(&cx, path);
    snprintf(path, sizeof path, "%s/gray.plxt", pack);
    if (plx_tensor_map(path, &gray) != PLX_OK || gray.dim[0] != n) plx_die("cannot read %s (N = %d)", path, n);

    /* flow quality -> prior weights: a non-converged fit (column --q-converged, default 6) or a
     * folded field (min det(I + grad u) <= --q-jdet-min in column --q-jdet, default 4) is untrusted */
    float *prior = PLX_NEW(float, n);
    for (int i = 0; i < n; i++) prior[i] = 1;
    snprintf(path, sizeof path, "%s/quality.plxt", flow);
    int qc = (int)plx_arg_int(a, "q-converged", 6), qj = (int)plx_arg_int(a, "q-jdet", 4), nbad = 0;
    double jmin = plx_arg_num(a, "q-jdet-min", 0.2);
    if (file_exists(path) && plx_tensor_load(path, &q) == PLX_OK && q.dtype == PLX_F32 && q.ndim == 2 && q.dim[0] == n) {
        int Q = (int)q.dim[1];
        const float *qv = q.data;
        for (int i = 0; i < n; i++) {
            if (qc >= 0 && qc < Q && qv[(size_t)i * Q + qc] < 0.5f) prior[i] = 0;
            if (qj >= 0 && qj < Q && qv[(size_t)i * Q + qj] <= jmin) prior[i] = 0;
            nbad += prior[i] == 0;
        }
        plx_log("quality: %d of %d images untrusted", nbad, n);
    } else
        plx_log("no usable %s: every image trusted", path);

    snprintf(diag, sizeof diag, "%s/diag", out);
    if (plx_mkdir_p(diag) != PLX_OK) plx_die("cannot create %s", diag);
    open_summary(out);
    plx_emerge_opts o = make_opts(a, &cx);
    o.prior_w = prior;
    plx_geom_model m;
    double t0 = plx_now();
    if (plx_emerge(&cx.L, u.data, n, cx.node_weight, cx.core, cx.cs, &o, &m) != PLX_OK) plx_die("emerge failed");
    say("plx_emerge --flow %s: %d images, %.1f s; %d untrusted by flow quality\n", flow, n, plx_now() - t0, nbad);
    report(&cx, &m);

    /* LFW View 1 verification by the depth codes (labels for evaluation only) */
    plx_lfw db;
    plx_pairs pl;
    const char *root = plx_arg_str(a, "lfw", "data/lfw"), *pairs = plx_arg_str(a, "pairs", "data/protocol/pairsDevTrain.txt");
    if (plx_lfw_open(&db, root) == PLX_OK) {
        if (db.n_images == n && plx_pairs_load(&pl, pairs, &db) == PLX_OK) {
            verif_out v;
            verify_pairs(&m, pl.p, pl.n, &v);
            say("[6] depth-code verification, %s: AUC %.4f (%d pairs); both |yaw| > 10: %.4f (%d); > 20: %.4f (%d)\n",
                pairs, v.auc[0], v.n[0], v.auc[1], v.n[1], v.auc[2], v.n[2]);
            plx_pairs_free(&pl);
        } else
            plx_log("LFW index (%d images) does not match the flow (%d) or pairs unreadable: no verification",
                    db.n_images, n);
        plx_lfw_free(&db);
    }
    write_outputs(out, &cx, &m);
    diag_depth(diag, &cx, &m, NULL);
    diag_shape_modes(diag, &cx, &m);
    diag_plots(diag, &m, NULL);
    diag_montage(diag, "3_montage_yaw.png", &m, &gray, sim.data, false, 11);
    diag_montage(diag, "3_montage_pitch.png", &m, &gray, sim.data, true, 12);
    say("outputs in %s, diagnostics in %s\n", out, diag);
    fclose(g_sum);
    plx_geom_model_free(&m);
    plx_tensor_free(&u), plx_tensor_free(&nw), plx_tensor_free(&sim), plx_tensor_free(&q), plx_tensor_free(&gray);
    free(prior), free(cx.core), free(cx.gauge), plx_img_free(&cx.tex);
    return 0;
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    return plx_arg_flag(&a, "synth") ? run_synth_mode(&a) : run_real_mode(&a);
}
