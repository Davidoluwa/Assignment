// plx-deps: align feat la ops viz img core
/* plx_flow: stage 2 of the pipeline. Estimates, for every congealed face, the smooth dense
 * displacement field u_i from the canonical template to the photo (align/flow.h), then measures
 * how far the fields can be trusted.
 *
 *   plx_flow --pack runs/main/pack --congeal runs/main/congeal --out runs/main/flow [options]
 *
 * Writes into <out>:
 *   u.plxt            f32 [N, 2, 21, 17]  field (component x/y, lattice row, column), canonical px,
 *                     in the frame after the similarity gauge (W_i(p) = T_i(p + u_i(p)))
 *   sim.plxt          f32 [N, 4]   (a, b, tx, ty) of the updated T_i (align/canon.h convention)
 *   node_weight.plxt  f32 [21, 17] per-node reliability in [0, 1], σ-symmetric (see below)
 *   quality.plxt      f32 [N, 7]   columns (QUALITY_COLUMNS below):
 *                     0 cost      robust cost Σωve²/Σωv against the final template (≈ K typical)
 *                     1 inlier    Huber weight mass in [0, 1]
 *                     2 valid     window fraction backed by real photo pixels
 *                     3 min_jdet  min over the face core of det(I + ∇u) (Jacobian of W relative to
 *                                 its similarity; <= 0 means the warp folds)
 *                     4 max_u     largest |u| at a core node, px
 *                     5 step      px a fresh solve would still move the field (max over core nodes)
 *                     6 flags     bit 0 untrusted by congealing (bound hit or disp > 1 px),
 *                                 bit 1 fold (min_jdet <= 0), bit 2 last round not converged,
 *                                 bit 3 valid < 0.9, bit 4 step > 1 px
 *   mean_face.plxt    f32 [160, 128] mean of the fully warped gray faces
 *   model/            the frozen model for fitting new images (plx_fl_model_load)
 *   diag/             PNG diagnostics and summary.txt with every acceptance number
 *
 * Options: --schedule "step:level:blur:lambda:prox:iters,..." --plain (weights ω instead of
 * ω/(σ²+ε)) --eps-frac --huber --lambda0 --tol --n (first n images, for experiments) --eval-n
 * (sample size of the evaluations, default 400) --seed --skip-eval. */
#include "align/canon.h"
#include "align/congeal.h"
#include "align/flow.h"
#include "align/lattice.h"
#include "core/core.h"
#include "img/image.h"
#include "img/imgio.h"
#include "la/la.h"
#include "ops/ops.h"
#include "viz/viz.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SCHEDULE                                                                                             \
    "32:2:1:0.1:0.3:8,32:2:0.5:0.1:0.3:8,16:1:0.5:0.1:0.3:8,16:1:0:0.1:0.3:8,16:0:0:0.1:1:8,16:0:0:0.1:1:8,"        \
    "8:0:0:0.3:1:8,8:0:0:0.3:1:8,8:0:0:0.3:1:8,8:0:0:0.3:1:8"
enum { QCOLS = 7, DOF = 2 * 17 * 21, NODES = 17 * 21, NX = 17, NY = 21 };
static const char *const QUALITY_COLUMNS[QCOLS] = {"cost", "inlier", "valid", "min_jdet", "max_u", "step", "flags"};
static const float WHITE[3] = {1, 1, 1}, ARROW[3] = {1.0f, 0.85f, 0.1f}, GUIDE[3] = {1.0f, 0.35f, 0.2f};

/* ---- small helpers ------------------------------------------------------------------------ */
static int parse_schedule(const char *s, plx_fl_stage *st) {
    int n = 0;
    while (*s) {
        if (n == PLX_FL_MAX_STAGES) plx_die("schedule: more than %d stages", PLX_FL_MAX_STAGES);
        int used = 0;
        if (sscanf(s, "%d:%d:%f:%f:%f:%d%n", &st[n].step, &st[n].level, &st[n].blur, &st[n].lambda, &st[n].prox,
                   &st[n].iters, &used) != 6)
            plx_die("schedule: cannot parse '%s' (want step:level:blur:lambda:prox:iters,...)", s);
        n++;
        s += used;
        if (*s == ',') s++;
    }
    return n;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* q-quantile of v[0..n) (sorts v in place). */
static double quantile(double *v, int n, double q) {
    if (n <= 0) return NAN;
    qsort(v, n, sizeof(double), cmp_double);
    double pos = q * (n - 1);
    int i = (int)pos;
    return i + 1 < n ? v[i] + (pos - i) * (v[i + 1] - v[i]) : v[n - 1];
}

static void describe(FILE *f, const char *name, const double *v0, int n) {
    double *v = PLX_NEW(double, n), m = 0, s = 0;
    memcpy(v, v0, sizeof(double) * n);
    for (int i = 0; i < n; i++) m += v[i] / n;
    for (int i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m) / n;
    double p[7] = {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99};
    fprintf(f, "  %-10s mean %8.4f  std %7.4f |", name, m, sqrt(s));
    for (int j = 0; j < 7; j++) fprintf(f, " p%02d %8.4f", (int)lrint(100 * p[j]), quantile(v, n, p[j]));
    fprintf(f, "\n");
    free(v);
}

static void save_png(const char *dir, const char *name, const plx_img *im) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plx_img_save(path, im) != PLX_OK) plx_die("cannot write %s", path);
    plx_log("wrote %s", path);
}

static bool core_node(int n) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    double x = plx_lat_x(&L, n % NX), y = plx_lat_y(&L, n / NX);
    return x >= PLX_LAT_CORE_X0 && x <= PLX_LAT_CORE_X1 && y >= PLX_LAT_CORE_Y0 && y <= PLX_LAT_CORE_Y1;
}

/* ---- comparing two full warps of the same photo ------------------------------------------ */
/* Per core node: |W_a(p) - W_b(p)| in canonical px (through T_ref⁻¹), and the even part (under σ)
 * of the x-component of u_a - u_b (both fields in the similarity gauge: the quantity depth is
 * read from). Appends to the arrays (core-node count entries each). */
typedef struct { double *warp, *field, *even; int n, cap; } errs;

static void errs_init(errs *e, int cap) {
    e->warp = PLX_NEW(double, cap);
    e->field = PLX_NEW(double, cap);
    e->even = PLX_NEW(double, cap);
    e->n = 0;
    e->cap = cap;
}
static void errs_free(errs *e) {
    free(e->warp);
    free(e->field);
    free(e->even);
}

/* Gauge-projects (T, u) in place and returns it (the pair as plx_flow stores it). */
static void gauge(plx_sim *T, float *u) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    *T = plx_sim_compose(*T, plx_lat_sim_gauge(&L, u));
}

static void compare(plx_sim Ta, const float *ua, plx_sim Tb, const float *ub, plx_sim Tref, double *warp, double *field,
                    double *even) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    float d[DOF];
    for (int k = 0; k < DOF; k++) d[k] = ua[k] - ub[k];
    int c = 0;
    for (int n = 0; n < NODES; n++) {
        if (!core_node(n)) continue;
        double x = plx_lat_x(&L, n % NX), y = plx_lat_y(&L, n / NX), ax, ay, bx, by, cax, cay, cbx, cby;
        plx_sim_apply(Ta, x + ua[n], y + ua[NODES + n], &ax, &ay);
        plx_sim_apply(Tb, x + ub[n], y + ub[NODES + n], &bx, &by);
        plx_sim_unapply(Tref, ax, ay, &cax, &cay);
        plx_sim_unapply(Tref, bx, by, &cbx, &cby);
        int m = (n / NX) * NX + (NX - 1 - n % NX); /* mirror node */
        warp[c] = hypot(cax - cbx, cay - cby);
        field[c] = hypot(d[n], d[NODES + n]);
        even[c] = fabs(0.5 * (d[n] + d[m]));
        c++;
    }
}

static int n_core_nodes(void) {
    int c = 0;
    for (int n = 0; n < NODES; n++) c += core_node(n);
    return c;
}

static void report_errs(FILE *f, const char *title, errs *e) {
    char line[1024];
    snprintf(line, sizeof line,
             "%s (%d node samples)\n  full warp |W - W'| px:    median %.3f  p95 %.3f\n"
             "  field |u - u'| px (gauge):  median %.3f  p95 %.3f\n"
             "  even part of u_x error px:  median %.3f  p95 %.3f\n",
             title, e->n, quantile(e->warp, e->n, 0.5), quantile(e->warp, e->n, 0.95), quantile(e->field, e->n, 0.5),
             quantile(e->field, e->n, 0.95), quantile(e->even, e->n, 0.5), quantile(e->even, e->n, 0.95));
    fputs(line, f);
    fputs(line, stderr);
}

/* ---- synthetic deformations for the known-warp recovery test ------------------------------ */
/* A smooth field d on the 8-px lattice. type 0: random bumps (σ 20-40 px) plus a random affine
 * part; type 1: "head turn": x-displacement exp(-X0²/2σx² - Y0²/2σy²), even in x and largest at the
 * face centre, plus a small y bump. Scaled so the RMS of its non-similarity part over the core
 * nodes is `rms` px. */
static void make_deformation(plx_rng *r, int type, double rms, float *d) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    double b[4][5], aff[4];
    for (int k = 0; k < 4; k++) {
        b[k][0] = 63.5 + 40 * (2 * plx_rng_uniform(r) - 1);
        b[k][1] = 79.5 + 50 * (2 * plx_rng_uniform(r) - 1);
        b[k][2] = 20 + 20 * plx_rng_uniform(r);
        b[k][3] = 2 * plx_rng_uniform(r) - 1;
        b[k][4] = 2 * plx_rng_uniform(r) - 1;
    }
    for (int k = 0; k < 4; k++) aff[k] = 0.02 * (2 * plx_rng_uniform(r) - 1);
    double sx = 22 + 10 * plx_rng_uniform(r), sy = 35 + 10 * plx_rng_uniform(r), dyb = 0.3 * (2 * plx_rng_uniform(r) - 1);
    for (int n = 0; n < NODES; n++) {
        double x = plx_lat_x(&L, n % NX), y = plx_lat_y(&L, n / NX), X0 = x - PLX_CANON_CX, Y0 = y - PLX_CANON_CY;
        double vx = 0, vy = 0;
        if (type == 0) {
            vx = aff[0] * X0 + aff[1] * Y0;
            vy = aff[2] * X0 + aff[3] * Y0;
            for (int k = 0; k < 4; k++) {
                double g = exp(-((x - b[k][0]) * (x - b[k][0]) + (y - b[k][1]) * (y - b[k][1])) / (2 * b[k][2] * b[k][2]));
                vx += b[k][3] * g;
                vy += b[k][4] * g;
            }
        } else {
            double g = exp(-X0 * X0 / (2 * sx * sx) - Y0 * Y0 / (2 * sy * sy));
            vx = g;
            vy = dyb * g;
        }
        d[n] = (float)vx;
        d[NODES + n] = (float)vy;
    }
    float tmp[DOF];
    memcpy(tmp, d, sizeof tmp);
    plx_lat_sim_gauge(&L, tmp);
    double s = 0;
    int c = 0;
    for (int n = 0; n < NODES; n++)
        if (core_node(n)) s += (double)tmp[n] * tmp[n] + (double)tmp[NODES + n] * tmp[NODES + n], c++;
    double scale = rms / sqrt(s / c);
    if (type == 1 && plx_rng_uniform(r) < 0.5) scale = -scale; /* turns to either side */
    for (int k = 0; k < DOF; k++) d[k] = (float)(d[k] * scale);
}

/* The photo J with J(T(p + d(p))) = I(T(p)): I resampled (bicubic) at T(φ(T⁻¹(x))) where
 * φ = (id + d)⁻¹, by fixed-point iteration. If W(p) = T(p + u(p)) aligns I, then
 * W_J(p) = T(p + u(p) + d(p + u(p))) aligns J. */
static plx_img deform_photo(const plx_img *I, plx_sim T, const float *d) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    plx_img map = plx_img_new(I->w, I->h, 2), J = plx_img_new(I->w, I->h, 1);
    for (int y = 0; y < I->h; y++)
        for (int x = 0; x < I->w; x++) {
            double qx, qy, px, py, v[2], ix, iy;
            plx_sim_unapply(T, x, y, &qx, &qy);
            px = qx, py = qy;
            for (int it = 0; it < 30; it++) {
                plx_lat_eval(&L, d, px, py, v);
                px = qx - v[0];
                py = qy - v[1];
            }
            plx_sim_apply(T, px, py, &ix, &iy);
            PLX_PX(&map, x, y, 0) = (float)ix;
            PLX_PX(&map, x, y, 1) = (float)iy;
        }
    plx_warp_map(I, &map, PLX_INTERP_BICUBIC, &J, NULL);
    plx_img_free(&map);
    return J;
}

/* ---- evaluation tasks (one per sampled image) --------------------------------------------- */
typedef struct {
    const plx_cg_source *src;
    const plx_fl_fitter *fit;
    const plx_sim       *Tc;      /* congealed T (start of every single-image fit) */
    const plx_sim       *T;       /* population result */
    const float         *u;
    const int           *idx;
    uint64_t             seed;
    int                  ncore;
    /* outputs, ncore entries per sample */
    double *path_w, *path_f, *path_e;   /* single fit vs population */
    double *mir_w, *mir_f, *mir_e;      /* flipped fit vs twin of population */
    double *equi_w, *equi_f, *equi_e;   /* flipped fit vs twin of single fit (should be ~0) */
    double *rec_w, *rec_f, *rec_e;      /* deformed fit vs D composed with the population result */
    double *recs_w, *recs_f, *recs_e;   /* deformed fit vs D composed with the single fit */
    int    *rec_type;
    double *rec_rms;
} eval_ctx;

static void twin_field(const float *u, float *out) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    plx_lat_mirror(&L, u, out);
}

/* (T, u) composed with the canonical deformation d: the node map p -> p + u(p) + d(p + u(p)),
 * gauge-projected. */
static void compose_deformation(plx_sim T, const float *u, const float *d, plx_sim *To, float *uo) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    for (int n = 0; n < NODES; n++) {
        double x = plx_lat_x(&L, n % NX) + u[n], y = plx_lat_y(&L, n / NX) + u[NODES + n], v[2];
        plx_lat_eval(&L, d, x, y, v);
        uo[n] = (float)(u[n] + v[0]);
        uo[NODES + n] = (float)(u[NODES + n] + v[1]);
    }
    *To = T;
    gauge(To, uo);
}

static void eval_one(int64_t j, int tid, void *pc) {
    eval_ctx *c = pc;
    int i = c->idx[j], nc = c->ncore;
    size_t o = (size_t)j * nc;
    plx_img g = plx_cg_source_image(c->src, i), gf = plx_img_new(g.w, g.h, 1);
    plx_flip_h(&g, &gf);
    const float *up = c->u + (size_t)i * DOF;
    plx_sim Ts, Tf, Td;
    float us[DOF], uf[DOF], ud[DOF], tw[DOF], tws[DOF];
    plx_fl_fit_image(c->fit, &g, c->Tc[i], &Ts, us);
    plx_fl_fit_image(c->fit, &gf, plx_sim_twin(c->Tc[i]), &Tf, uf);
    compare(Ts, us, c->T[i], up, c->T[i], c->path_w + o, c->path_f + o, c->path_e + o);
    twin_field(up, tw);
    twin_field(us, tws);
    plx_sim Ttw = plx_sim_twin(c->T[i]);
    compare(Tf, uf, Ttw, tw, Ttw, c->mir_w + o, c->mir_f + o, c->mir_e + o);
    compare(Tf, uf, plx_sim_twin(Ts), tws, Ttw, c->equi_w + o, c->equi_f + o, c->equi_e + o);

    /* known-warp recovery */
    plx_rng r;
    plx_rng_seed(&r, c->seed + (uint64_t)i * 7919);
    int type = (int)(j % 2);
    double rms = 2 + 2 * plx_rng_uniform(&r);
    float d[DOF], upred[DOF], uspred[DOF];
    make_deformation(&r, type, rms, d);
    plx_img J = deform_photo(&g, c->T[i], d);
    plx_fl_fit_image(c->fit, &J, c->T[i], &Td, ud);
    plx_sim Tpred, Tspred;
    compose_deformation(c->T[i], up, d, &Tpred, upred);
    /* the single fit, re-expressed relative to the population T so d (canonical in that frame)
     * applies: its node map p -> T⁻¹(T_s(p + u_s(p))) */
    {
        plx_lattice L = plx_lattice_make(PLX_FL_STEP);
        float rel[DOF];
        for (int n = 0; n < NODES; n++) {
            double x = plx_lat_x(&L, n % NX), y = plx_lat_y(&L, n / NX), ix, iy, cx, cy;
            plx_sim_apply(Ts, x + us[n], y + us[NODES + n], &ix, &iy);
            plx_sim_unapply(c->T[i], ix, iy, &cx, &cy);
            rel[n] = (float)(cx - x);
            rel[NODES + n] = (float)(cy - y);
        }
        compose_deformation(c->T[i], rel, d, &Tspred, uspred);
    }
    compare(Td, ud, Tpred, upred, c->T[i], c->rec_w + o, c->rec_f + o, c->rec_e + o);
    compare(Td, ud, Tspred, uspred, c->T[i], c->recs_w + o, c->recs_f + o, c->recs_e + o);
    c->rec_type[j] = type;
    c->rec_rms[j] = rms;
    plx_img_free(&J);
    plx_img_free(&g);
    plx_img_free(&gf);
}

/* ---- pictures ----------------------------------------------------------------------------- */
static plx_img stretch_rgb(const plx_img *gray, int up) {
    plx_img u = plx_viz_upscale(gray, up, PLX_UP_BILINEAR);
    float lo = 1, hi = 0;
    for (size_t k = 0; k < (size_t)u.w * u.h; k++) lo = fminf(lo, u.px[k]), hi = fmaxf(hi, u.px[k]);
    plx_img_scale_offset(&u, 1.0f / fmaxf(hi - lo, 1e-6f), -lo / fmaxf(hi - lo, 1e-6f));
    plx_img rgb = plx_viz_to_rgb(&u);
    plx_img_free(&u);
    return rgb;
}

/* Arrows of an 8-px lattice field drawn on an Ω image upscaled by `up` (node (i, j) sits at
 * canonical (x_j, y_i)); only nodes inside Ω. */
static void draw_field(plx_img *im, const float *u, int up, float scale, const float rgb[3]) {
    float dx[NODES], dy[NODES];
    for (int n = 0; n < NODES; n++) {
        bool in = n % NX > 0 && n % NX < NX - 1 && n / NX > 0 && n / NX < NY - 1;
        dx[n] = in ? u[n] * up : NAN;
        dy[n] = in ? u[NODES + n] * up : NAN;
    }
    /* canonical x_j = -0.5 + 8 j -> upscaled pixel (x + 0.5) up - 0.5 */
    plx_viz_flow_arrows(im, dx, dy, NX, NY, -0.5f, -0.5f, 8.0f * up, scale, rgb);
}

static plx_img field_color(const float *u, float max_mag, int up) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    int w = PLX_CANON_W, h = PLX_CANON_H;
    float *dx = PLX_NEW(float, (size_t)w * h), *dy = PLX_NEW(float, (size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double v[2];
            plx_lat_eval(&L, u, x, y, v);
            dx[(size_t)y * w + x] = (float)v[0];
            dy[(size_t)y * w + x] = (float)v[1];
        }
    plx_img c = plx_viz_flow_color(dx, dy, w, h, max_mag);
    plx_img cu = plx_viz_upscale(&c, up, PLX_UP_NEAREST);
    plx_img_free(&c);
    free(dx);
    free(dy);
    return cu;
}

static plx_img sim_crop_rgb(const plx_img *g, plx_sim T, bool guides) {
    plx_img c = plx_canon_new(1, 0);
    plx_canon_warp(g, T, 0, PLX_INTERP_BILINEAR, &c, NULL);
    plx_img rgb = plx_viz_to_rgb(&c);
    if (guides) {
        plx_draw_line(&rgb, 0, 63, PLX_CANON_W - 1, 63, 1, GUIDE, 0.35f);
        plx_draw_line(&rgb, 0, 103, PLX_CANON_W - 1, 103, 1, GUIDE, 0.35f);
        plx_draw_line(&rgb, 63.5f, 0, 63.5f, PLX_CANON_H - 1, 1, GUIDE, 0.2f);
    }
    plx_img_free(&c);
    return rgb;
}

static plx_img flow_crop_rgb(const plx_img *g, plx_sim T, const float *u) {
    plx_lattice L = plx_lattice_make(PLX_FL_STEP);
    plx_img c = plx_fl_warp_crop(g, T, &L, u), rgb = plx_viz_to_rgb(&c);
    plx_draw_line(&rgb, 0, 63, PLX_CANON_W - 1, 63, 1, GUIDE, 0.35f);
    plx_draw_line(&rgb, 0, 103, PLX_CANON_W - 1, 103, 1, GUIDE, 0.35f);
    plx_draw_line(&rgb, 63.5f, 0, 63.5f, PLX_CANON_H - 1, 1, GUIDE, 0.2f);
    plx_img_free(&c);
    return rgb;
}

/* Triplets: similarity crop with the field's arrows (x3) | flow-warped crop | colour code. */
static void diag_faces(const char *dir, const char *name, const plx_cg_source *src, const plx_sim *T, const float *u,
                       const int *idx, int m, const float *q) {
    plx_img *tiles = PLX_NEW(plx_img, 3 * m);
    char **lab = q ? PLX_NEW(char *, 3 * m) : NULL;
    for (int j = 0; j < m; j++) {
        int i = idx[j];
        plx_img g = plx_cg_source_image(src, i);
        const float *ui = u + (size_t)i * DOF;
        tiles[3 * j] = sim_crop_rgb(&g, T[i], false);
        draw_field(&tiles[3 * j], ui, 1, 3.0f, ARROW);
        tiles[3 * j + 1] = flow_crop_rgb(&g, T[i], ui);
        tiles[3 * j + 2] = field_color(ui, 6.0f, 1);
        if (lab) {
            for (int k = 0; k < 3; k++) lab[3 * j + k] = plx_malloc(48);
            const float *qi = q + (size_t)i * QCOLS;
            snprintf(lab[3 * j], 48, "#%d", i);
            snprintf(lab[3 * j + 1], 48, "c%.1f j%.2f", qi[0], qi[3]);
            snprintf(lab[3 * j + 2], 48, "|u|%.1f f%d", qi[4], (int)qi[6]);
        }
        plx_img_free(&g);
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 12;
    o.pad = 3;
    plx_img im = plx_viz_montage(tiles, 3 * m, lab ? (const char *const *)lab : NULL, &o);
    save_png(dir, name, &im);
    plx_img_free(&im);
    for (int k = 0; k < 3 * m; k++) {
        plx_img_free(&tiles[k]);
        if (lab) free(lab[k]);
    }
    free(tiles);
    free(lab);
}

static void diag_rounds(const char *dir, const plx_img *congeal_face, const plx_fl_round *rounds, int nr) {
    int nt = nr + 1;
    plx_img *tiles = PLX_NEW(plx_img, nt);
    char **lab = PLX_NEW(char *, nt);
    for (int r = 0; r < nt; r++) {
        const plx_img *f = r == 0 ? congeal_face : &rounds[r - 1].mean_gray;
        tiles[r] = stretch_rgb(f, 2);
        lab[r] = plx_malloc(48);
        if (r == 0) snprintf(lab[r], 48, "congeal %.2f", plx_cg_sharpness(f));
        else if (r == nt - 1) snprintf(lab[r], 48, "final %.2f", plx_cg_sharpness(f));
        else snprintf(lab[r], 48, "round %d %.2f", r - 1, plx_cg_sharpness(f));
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = (nt + 1) / 2;
    o.text_scale = 2;
    plx_img m = plx_viz_montage(tiles, nt, (const char *const *)lab, &o);
    save_png(dir, "rounds_mean_face.png", &m);
    plx_img_free(&m);
    /* the headline comparison, large: congealing's mean face next to the flow's */
    plx_img pair[2] = {stretch_rgb(congeal_face, 3), stretch_rgb(&rounds[nr - 1].mean_gray, 3)};
    const char *pl[2] = {"similarity only (congeal)", "similarity + flow"};
    o.cols = 2;
    plx_img pm = plx_viz_montage(pair, 2, pl, &o);
    save_png(dir, "mean_face.png", &pm);
    plx_img_free(&pm);
    plx_img_free(&pair[0]);
    plx_img_free(&pair[1]);
    for (int r = 0; r < nt; r++) {
        plx_img_free(&tiles[r]);
        free(lab[r]);
    }
    free(tiles);
    free(lab);
}

/* ---- PCA of the fields over the population ------------------------------------------------ */
typedef struct { double frac, e, o, v, q, centre_side; } pc_desc;

/* Parity energies of a core-node field a (layout [2][15][11] = component, core row, core col). */
static pc_desc describe_pc(const double *a, int cr, int cc) {
    pc_desc d = {0};
    double tot = 0;
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < cr; i++)
            for (int j = 0; j < cc; j++) {
                double v = a[(c * cr + i) * cc + j], vm = a[(c * cr + i) * cc + (cc - 1 - j)];
                double ev = 0.5 * (v + vm), od = 0.5 * (v - vm);
                tot += v * v;
                if (c == 0) d.e += ev * ev, d.o += od * od;
                else d.v += ev * ev, d.q += od * od;
            }
    d.e /= tot, d.o /= tot, d.v /= tot, d.q /= tot;
    /* E profile: centre column vs outermost columns, over the middle rows (eyes to mouth) */
    double cen = 0, side = 0;
    for (int i = cr / 4; i < 3 * cr / 4; i++) {
        double at[2] = {0};
        for (int j = 0; j < cc; j++) {
            double ev = 0.5 * (a[i * cc + j] + a[i * cc + (cc - 1 - j)]);
            if (j == cc / 2) at[0] += ev;
            if (j == 0 || j == cc - 1) at[1] += 0.5 * ev;
        }
        cen += at[0];
        side += at[1];
    }
    d.centre_side = fabs(side) > 1e-12 ? cen / side : INFINITY;
    return d;
}

static void field_from_core(const double *a, double scale, int cr, int cc, float *u) {
    memset(u, 0, sizeof(float) * DOF);
    int k = 0;
    for (int n = 0; n < NODES; n++) {
        if (!core_node(n)) continue;
        int i = k / cc, j = k % cc;
        u[n] = (float)(scale * a[(0 * cr + i) * cc + j]);
        u[NODES + n] = (float)(scale * a[(1 * cr + i) * cc + j]);
        k++;
    }
}

static void diag_pca(const char *dir, FILE *f, const float *u, const uint8_t *use, int n, const plx_img *mean_face) {
    int cc = 0, cr = 0;
    {
        plx_lattice L = plx_lattice_make(PLX_FL_STEP);
        for (int j = 0; j < NX; j++) cc += plx_lat_x(&L, j) >= PLX_LAT_CORE_X0 && plx_lat_x(&L, j) <= PLX_LAT_CORE_X1;
        for (int i = 0; i < NY; i++) cr += plx_lat_y(&L, i) >= PLX_LAT_CORE_Y0 && plx_lat_y(&L, i) <= PLX_LAT_CORE_Y1;
    }
    int d = 2 * cr * cc, m = 0;
    for (int i = 0; i < n; i++) m += use[i];
    float *X = PLX_NEW(float, (size_t)m * d);
    for (int i = 0, r = 0; i < n; i++) {
        if (!use[i]) continue;
        int k = 0;
        for (int nn = 0; nn < NODES; nn++) {
            if (!core_node(nn)) continue;
            X[(size_t)r * d + k] = u[(size_t)i * DOF + nn];
            X[(size_t)r * d + cr * cc + k] = u[(size_t)i * DOF + NODES + nn];
            k++;
        }
        r++;
    }
    double *mean = PLX_NEW(double, d), tot = 0;
    plx_col_mean_f32(X, m, d, mean);
    for (int r = 0; r < m; r++)
        for (int k = 0; k < d; k++) {
            double v = X[(size_t)r * d + k] - mean[k];
            tot += v * v;
        }
    enum { KP = 8 };
    plx_pca pca;
    plx_pca_opts po = {.method = PLX_PCA_COV};
    if (plx_pca_f32(X, m, d, KP, &po, &pca) != PLX_OK) plx_die("PCA of the fields failed");
    fprintf(f, "\nPCA OF THE FIELDS: %d trusted images, core nodes (%d x %d, both components), centred\n", m, cc, cr);
    fprintf(f, "  total variance %.3f px^2 per image (sum over %d dims); mean field |max| %.3f px\n", tot / m, d,
            fabs(mean[0]));
    fprintf(f, "  energy fractions of each axis: E = even(u_x) [yaw parallax], O = odd(u_x) [widths, foreshortening],\n"
               "  V = even(u_y) [pitch, heights], Q = odd(u_y) [roll residue, asymmetry];\n"
               "  E centre/side = mean even(u_x) at the centre column / at the outermost core columns (middle rows)\n");
    fprintf(f, "  pc  var_frac  cum     sd_px    E      O      V      Q    E_centre/side\n");
    plx_img tiles[2 * KP];
    char *lab[2 * KP];
    double cum = 0;
    float uf[DOF];
    for (int k = 0; k < KP; k++) {
        double var = pca.sv[k] * pca.sv[k] / m, frac = var / (tot / m);
        cum += frac;
        const double *a = &PLX_M(&pca.axes, k, 0);
        pc_desc ds = describe_pc(a, cr, cc);
        fprintf(f, "  %2d  %7.4f  %6.4f  %6.3f  %5.3f  %5.3f  %5.3f  %5.3f  %8.2f\n", k + 1, frac, cum, sqrt(var), ds.e,
                ds.o, ds.v, ds.q, ds.centre_side);
        /* drawn at +2 standard deviations */
        field_from_core(a, 2 * sqrt(var), cr, cc, uf);
        tiles[2 * k] = stretch_rgb(mean_face, 3);
        draw_field(&tiles[2 * k], uf, 3, 3.0f, ARROW);
        tiles[2 * k + 1] = field_color(uf, 0, 3);
        lab[2 * k] = plx_malloc(64);
        lab[2 * k + 1] = plx_malloc(64);
        snprintf(lab[2 * k], 64, "PC%d %.1f%% (+2sd, x3)", k + 1, 100 * frac);
        snprintf(lab[2 * k + 1], 64, "E%.2f O%.2f V%.2f Q%.2f", ds.e, ds.o, ds.v, ds.q);
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 4;
    o.text_scale = 2;
    plx_img im = plx_viz_montage(tiles, 2 * KP, (const char *const *)lab, &o);
    save_png(dir, "pca.png", &im);
    plx_img_free(&im);
    for (int k = 0; k < 2 * KP; k++) {
        plx_img_free(&tiles[k]);
        free(lab[k]);
    }
    plx_pca_free(&pca);
    free(mean);
    free(X);
}

/* ---- node reliability ---------------------------------------------------------------------- */
/* w_n = (data share of the node's stiffness) x min(1, median cost / node cost): nodes the
 * regulariser has to invent (border) and nodes whose population residual is high (hair, jaw
 * against background, ears) are both down-weighted. Symmetrised. */
static void node_weights(const plx_fl_model *m, const plx_fl_nodestat *ns, float *w, float *datafrac, float *costn) {
    plx_fl_ref ref = plx_fl_ref_build(&m->tmpl, &m->p, &m->stages[m->n_stages - 1]);
    plx_fl_data_fraction(&ref, datafrac);
    plx_fl_ref_free(&ref);
    double c[NODES], med[NODES];
    int nm = 0;
    for (int n = 0; n < NODES; n++) {
        int mm = (n / NX) * NX + (NX - 1 - n % NX);
        double cost = ns->cost[n] + ns->cost[mm], mass = ns->mass[n] + ns->mass[mm];
        c[n] = mass > 0 ? cost / mass : NAN;
        costn[n] = (float)c[n];
        if (mass > 0 && core_node(n)) med[nm++] = c[n];
    }
    double cm = quantile(med, nm, 0.5);
    for (int n = 0; n < NODES; n++) w[n] = isfinite(c[n]) ? datafrac[n] * (float)fmin(1.0, cm / c[n]) : 0.0f;
}

static void diag_nodes(const char *dir, const float *w, const float *df, const float *cn) {
    const float *planes[3] = {df, cn, w};
    const char *names[3] = {"data share", "population cost", "node_weight"};
    plx_img tiles[3];
    for (int k = 0; k < 3; k++) {
        plx_img t = plx_img_new(NX, NY, 1);
        memcpy(t.px, planes[k], sizeof(float) * NODES);
        plx_img c = plx_viz_scalar(&t, PLX_CMAP_SEQ);
        tiles[k] = plx_viz_upscale(&c, 12, PLX_UP_NEAREST);
        plx_img_free(&t);
        plx_img_free(&c);
    }
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 3;
    o.text_scale = 2;
    plx_img im = plx_viz_montage(tiles, 3, names, &o);
    save_png(dir, "node_weight.png", &im);
    plx_img_free(&im);
    for (int k = 0; k < 3; k++) plx_img_free(&tiles[k]);
}

typedef struct { float key; int idx; } keyed;
static int cmp_keyed_desc(const void *a, const void *b) {
    float x = ((const keyed *)a)->key, y = ((const keyed *)b)->key;
    return (x < y) - (x > y);
}

/* ---- main ---------------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    const char *pack = plx_arg_str(&a, "pack", "runs/main/pack");
    const char *cgdir = plx_arg_str(&a, "congeal", "runs/main/congeal");
    const char *out = plx_arg_str(&a, "out", "runs/main/flow");
    plx_fl_model model = {0};
    plx_fl_params *p = &model.p;
    *p = plx_fl_default();
    p->cg.eps_frac = (float)plx_arg_num(&a, "eps-frac", p->cg.eps_frac);
    p->cg.huber = (float)plx_arg_num(&a, "huber", p->cg.huber);
    p->lambda0 = (float)plx_arg_num(&a, "lambda0", p->lambda0);
    p->damp = (float)plx_arg_num(&a, "damp", p->damp);
    p->tol = (float)plx_arg_num(&a, "tol", p->tol);
    if (plx_arg_flag(&a, "plain")) p->weighting = PLX_FL_W_PLAIN;
    const char *sched_str = plx_arg_str(&a, "schedule", DEFAULT_SCHEDULE);
    model.n_stages = parse_schedule(sched_str, model.stages);
    for (int s = 0; s < model.n_stages; s++) {
        const plx_fl_stage *st = &model.stages[s];
        if (st->level < 0 || st->level >= p->cg.levels || st->iters < 1 || (st->step != 8 && st->step != 16 && st->step != 32))
            plx_die("schedule: bad stage %d", s);
    }
    if (model.stages[model.n_stages - 1].step != PLX_FL_STEP) plx_die("schedule: the last stage must use the 8-px lattice");
    int eval_n = (int)plx_arg_int(&a, "eval-n", 400);
    uint64_t seed = (uint64_t)plx_arg_int(&a, "seed", 1);
    bool skip_eval = plx_arg_flag(&a, "skip-eval");

    /* ---- inputs ---- */
    char path[8192], dir[4096];
    snprintf(path, sizeof path, "%s/gray.plxt", pack);
    plx_tensor gt, st, qt, mt;
    if (plx_tensor_map(path, &gt) != PLX_OK) plx_die("cannot map %s", path);
    if (gt.dtype != PLX_U8 || gt.ndim != 3 || gt.dim[1] != PLX_CANON_IMG_SIDE || gt.dim[2] != PLX_CANON_IMG_SIDE)
        plx_die("%s: want u8 [N, 250, 250]", path);
    snprintf(path, sizeof path, "%s/sim.plxt", cgdir);
    if (plx_tensor_load(path, &st) != PLX_OK || st.dtype != PLX_F32 || st.dim[0] != gt.dim[0]) plx_die("bad %s", path);
    snprintf(path, sizeof path, "%s/quality.plxt", cgdir);
    if (plx_tensor_load(path, &qt) != PLX_OK || qt.dtype != PLX_F32 || qt.dim[0] != gt.dim[0] || qt.dim[1] < 6)
        plx_die("bad %s", path);
    snprintf(path, sizeof path, "%s/mean_face.plxt", cgdir);
    if (plx_tensor_load(path, &mt) != PLX_OK || mt.dim[0] != PLX_CANON_H || mt.dim[1] != PLX_CANON_W)
        plx_die("bad %s", path);
    plx_img congeal_face = plx_img_new(PLX_CANON_W, PLX_CANON_H, 1);
    memcpy(congeal_face.px, mt.data, mt.nbytes);
    int n = (int)PLX_MIN(gt.dim[0], plx_arg_int(&a, "n", gt.dim[0]));
    plx_cg_source src = {gt.data, n, (int)gt.dim[2], (int)gt.dim[1]};
    plx_sim *Tc = PLX_NEW(plx_sim, n), *T = PLX_NEW(plx_sim, n);
    uint8_t *trusted = PLX_NEW(uint8_t, n);
    const float *sv = st.data, *cq = qt.data;
    int qc = (int)qt.dim[1], n_untrusted = 0;
    for (int i = 0; i < n; i++) {
        Tc[i] = T[i] = (plx_sim){sv[4 * i], sv[4 * i + 1], sv[4 * i + 2], sv[4 * i + 3]};
        /* congeal quality: column 3 disp (px a fresh solve would still move), 5 bound */
        trusted[i] = !(cq[(size_t)i * qc + 5] > 0 || cq[(size_t)i * qc + 3] > 1.0f);
        n_untrusted += !trusted[i];
    }
    snprintf(dir, sizeof dir, "%s/diag", out);
    if (plx_mkdir_p(dir) != PLX_OK) plx_die("cannot create %s", dir);
    plx_log("flow: %d images (%d untrusted by congealing), %d threads, schedule %s", n, n_untrusted, plx_num_threads(),
            sched_str);

    /* ---- the stage itself ---- */
    double t_start = plx_now();
    float *u = PLX_NEW(float, (size_t)n * DOF);
    plx_fl_round *rounds = PLX_NEW(plx_fl_round, model.n_stages + 2);
    plx_fl_quality *qual = PLX_NEW(plx_fl_quality, n);
    plx_fl_nodestat ns = {PLX_NEW(double, NODES), PLX_NEW(double, NODES)};
    plx_fl_congeal(&src, p, model.stages, model.n_stages, trusted, T, u, &model.tmpl, rounds, qual, &ns);
    double t_flow = plx_now() - t_start;

    /* ---- outputs ---- */
    float *sim = PLX_NEW(float, (size_t)n * 4), *q = PLX_NEW(float, (size_t)n * QCOLS);
    int n_fold = 0, n_flag = 0;
    for (int i = 0; i < n; i++) {
        float v[4] = {(float)T[i].a, (float)T[i].b, (float)T[i].tx, (float)T[i].ty};
        memcpy(sim + 4 * i, v, sizeof v);
        const plx_fl_quality *qi = &qual[i];
        int flags = (!trusted[i]) | (qi->min_jdet <= 0) << 1 | (!qi->converged) << 2 | (qi->valid < 0.9f) << 3 |
                    (qi->step > 1.0f) << 4;
        float r[QCOLS] = {qi->cost, qi->inlier, qi->valid, qi->min_jdet, qi->max_u, qi->step, (float)flags};
        memcpy(q + (size_t)QCOLS * i, r, sizeof r);
        n_fold += qi->min_jdet <= 0;
        n_flag += (flags & ~4) != 0;
    }
    float nw[NODES], df[NODES], cn[NODES];
    node_weights(&model, &ns, nw, df, cn);
    const plx_img *mean_face = &rounds[model.n_stages + 1].mean_gray;
    struct { const char *name; int ndim; int64_t dim[4]; const float *data; } outs[] = {
        {"u.plxt", 4, {n, 2, NY, NX}, u},
        {"sim.plxt", 2, {n, 4}, sim},
        {"quality.plxt", 2, {n, QCOLS}, q},
        {"node_weight.plxt", 2, {NY, NX}, nw},
        {"mean_face.plxt", 2, {mean_face->h, mean_face->w}, mean_face->px},
    };
    for (size_t j = 0; j < sizeof outs / sizeof outs[0]; j++) {
        snprintf(path, sizeof path, "%s/%s", out, outs[j].name);
        if (plx_tensor_save(path, PLX_F32, outs[j].ndim, outs[j].dim, outs[j].data) != PLX_OK)
            plx_die("cannot write %s", path);
        plx_log("wrote %s", path);
    }
    snprintf(path, sizeof path, "%s/model", out);
    if (plx_fl_model_save(&model, path) != PLX_OK) plx_die("cannot write the model to %s", path);
    plx_log("wrote %s", path);

    /* ---- summary: parameters, rounds, distributions ---- */
    snprintf(path, sizeof path, "%s/summary.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) plx_die("cannot write %s", path);
    fprintf(f, "plx_flow summary: %d images (%d untrusted by congealing, excluded from template and mean gauge), "
               "flow %.1f s on %d threads\n\n",
            n, n_untrusted, t_flow, plx_num_threads());
    fprintf(f, "features: K=%d %s sigma_d=%.2f sigma_n=%.2f sigma_s=%.2f eps=%.4f; margin %d, %s render\n", p->cg.feat.k,
            p->cg.feat.signed_ ? "signed" : "unsigned", p->cg.feat.sigma_d, p->cg.feat.sigma_n, p->cg.feat.sigma_s,
            p->cg.feat.eps, p->cg.margin, p->cg.interp == PLX_INTERP_BICUBIC ? "bicubic" : "bilinear");
    fprintf(f, "solver: weights %s, eps_frac=%.3f huber=%.2f*sqrt(K) ramp=%.1f lambda0=%g damp=%g tol=%.3f px\n",
            p->weighting == PLX_FL_W_INVVAR ? "omega/(var+eps)" : "omega (plain)", p->cg.eps_frac, p->cg.huber,
            p->cg.ramp, p->lambda0, p->damp, p->tol);
    fprintf(f, "schedule (step:level:blur:lambda:prox:iters): %s\n\n", sched_str);
    fprintf(f, "rounds (template of round r = population at the start of round r; sharpness of the mean WARPED gray face):\n");
    fprintf(f, "  round step level blur  lambda   core_var  var/2nd-moment  sharpness  mean_cost  median_last_step  "
               "rms|u|  unconverged   sec\n");
    fprintf(f, "  congealing's final mean face (similarity only): sharpness %.3f\n", plx_cg_sharpness(&congeal_face));
    for (int r = 0; r < model.n_stages + 2; r++) {
        const plx_fl_round *ri = &rounds[r];
        bool al = r >= 1 && r <= model.n_stages;
        const plx_fl_stage *s = al ? &model.stages[r - 1] : NULL;
        fprintf(f, "  %5d %4d %5d %4.1f %7.3f %10.2f %15.4f %10.3f %10.3f %17.4f %7.3f %12d %5.1f\n", r, s ? s->step : 0,
                s ? s->level : -1, s ? s->blur : 0.0f, s ? s->lambda : 0.0f, ri->core_var, ri->core_ratio,
                plx_cg_sharpness(&ri->mean_gray), ri->mean_cost, ri->median_step, ri->mean_rms_u, ri->n_unconverged,
                ri->seconds);
    }
    fprintf(f, "(sharpness = 1e4 * mean |grad|^2 of the mean gray face over the face core, the same definition as\n"
               " congealing's summary; core_var = Σ feature variance over the core box)\n");
    int last = model.n_stages + 1;
    fprintf(f, "feature variance drop over the flow: %.3f (core var %.2f -> %.2f; var/2nd-moment %.4f -> %.4f); "
               "sharpness %.3f -> %.3f\n\n",
            rounds[0].core_var / rounds[last].core_var, rounds[0].core_var, rounds[last].core_var, rounds[0].core_ratio,
            rounds[last].core_ratio, plx_cg_sharpness(&rounds[0].mean_gray), plx_cg_sharpness(mean_face));
    double *col = PLX_NEW(double, n);
    fprintf(f, "quality distributions (all images):\n");
    for (int c = 0; c < QCOLS - 1; c++) {
        for (int i = 0; i < n; i++) col[i] = q[(size_t)i * QCOLS + c];
        describe(f, QUALITY_COLUMNS[c], col, n);
    }
    for (int i = 0; i < n; i++) {
        const float *ui = u + (size_t)i * DOF;
        double s = 0;
        int c = 0;
        for (int k = 0; k < NODES; k++)
            if (core_node(k)) s += (double)ui[k] * ui[k] + (double)ui[NODES + k] * ui[NODES + k], c++;
        col[i] = sqrt(s / c);
    }
    describe(f, "rms|u|", col, n);
    for (int i = 0; i < n; i++) col[i] = plx_sim_scale(T[i]) / plx_sim_scale(Tc[i]) - 1;
    describe(f, "dscale", col, n); /* how much scale the similarity gauge moved from congealing's T */
    int n_unconv = 0;
    for (int i = 0; i < n; i++) n_unconv += !qual[i].converged;
    fprintf(f, "\nFOLDING: images with min det(I + grad u) <= 0 over the core: %d (%.3f%%)\n", n_fold, 100.0 * n_fold / n);
    for (int i = 0; i < n; i++) col[i] = qual[i].min_jdet;
    describe(f, "min_jdet", col, n);
    fprintf(f, "flagged (any flag except 'last round not converged'): %d; last round not converged: %d\n", n_flag, n_unconv);
    fprintf(f, "\nNODE WEIGHT (node_weight.plxt): data share x min(1, median cost / node cost), symmetric\n");
    for (int i = 0; i < NY; i++) {
        fprintf(f, "  ");
        for (int j = 0; j < NX; j++) fprintf(f, "%5.2f", nw[i * NX + j]);
        fprintf(f, "\n");
    }

    /* ---- acceptance metrics ---- */
    plx_rng rng;
    plx_rng_seed(&rng, seed);
    if (!skip_eval) {
        double t0 = plx_now();
        plx_fl_fitter fitter;
        plx_fl_fitter_init(&fitter, &model);
        int *perm = PLX_NEW(int, n), *idx = PLX_NEW(int, n), m = 0;
        for (int i = 0; i < n; i++) perm[i] = i;
        plx_rng_shuffle(&rng, perm, n);
        for (int k = 0; k < n && m < eval_n; k++) {
            int i = perm[k];
            if (trusted[i] && qual[i].converged && qual[i].valid > 0.95f && qual[i].min_jdet > 0) idx[m++] = i;
        }
        int nc = n_core_nodes();
        size_t cap = (size_t)m * nc;
        errs path_e, mir_e, equi_e, rec_e, recs_e;
        errs_init(&path_e, (int)cap);
        errs_init(&mir_e, (int)cap);
        errs_init(&equi_e, (int)cap);
        errs_init(&rec_e, (int)cap);
        errs_init(&recs_e, (int)cap);
        int *rtype = PLX_NEW(int, m);
        double *rrms = PLX_NEW(double, m);
        eval_ctx c = {.src = &src, .fit = &fitter, .Tc = Tc, .T = T, .u = u, .idx = idx, .seed = seed, .ncore = nc,
                      .path_w = path_e.warp, .path_f = path_e.field, .path_e = path_e.even,
                      .mir_w = mir_e.warp, .mir_f = mir_e.field, .mir_e = mir_e.even,
                      .equi_w = equi_e.warp, .equi_f = equi_e.field, .equi_e = equi_e.even,
                      .rec_w = rec_e.warp, .rec_f = rec_e.field, .rec_e = rec_e.even,
                      .recs_w = recs_e.warp, .recs_f = recs_e.field, .recs_e = recs_e.even,
                      .rec_type = rtype, .rec_rms = rrms};
        plx_parallel_for(m, 1, eval_one, &c);
        path_e.n = mir_e.n = equi_e.n = rec_e.n = recs_e.n = (int)cap;
        fprintf(f, "\nEVALUATION on %d random trusted, converged images (core nodes: %d per image); every single-image\n"
                   "fit runs the full schedule against the frozen final model (plx_fl_fit_image)\n\n", m, nc);
        fprintf(f, "KNOWN-WARP RECOVERY: photo resampled through a known smooth canonical deformation d (RMS of its\n"
                   "non-similarity part 2-4 px; half random bumps + affine, half 'head turn' x-bumps even in x),\n"
                   "fitted from the original's T with u = 0, compared with W_true(p) = T(p + u(p) + d(p + u(p))):\n");
        report_errs(f, "deformed fit vs d composed with the POPULATION result", &rec_e);
        report_errs(f, "deformed fit vs d composed with the SINGLE-IMAGE fit of the original (path dependence removed)",
                    &recs_e);
        for (int type = 0; type < 2; type++) {
            errs sub;
            errs_init(&sub, (int)cap);
            for (int j = 0; j < m; j++)
                if (rtype[j] == type)
                    for (int k = 0; k < nc; k++) {
                        sub.warp[sub.n] = rec_e.warp[(size_t)j * nc + k];
                        sub.field[sub.n] = rec_e.field[(size_t)j * nc + k];
                        sub.even[sub.n] = rec_e.even[(size_t)j * nc + k];
                        sub.n++;
                    }
            report_errs(f, type ? "  head-turn deformations only (vs population)" : "  random deformations only (vs population)",
                        &sub);
            errs_free(&sub);
        }
        fprintf(f, "\nPATH INDEPENDENCE: single-image fit from (congealed T, u = 0) vs the population result\n");
        report_errs(f, "single fit vs population", &path_e);
        fprintf(f, "\nMIRROR: flipped photo fitted from twin(congealed T) vs the twin of the population result\n");
        report_errs(f, "flipped fit vs twin(population)", &mir_e);
        fprintf(f, "mirror equivariance sanity check (flipped fit vs twin of the original's single fit, both from\n"
                   "the congealed T; ~0 by construction, measures the machinery's symmetry):\n");
        report_errs(f, "flipped fit vs twin(single fit)", &equi_e);
        fprintf(f, "\nevaluation time %.1f s\n", plx_now() - t0);
        errs_free(&path_e);
        errs_free(&mir_e);
        errs_free(&equi_e);
        errs_free(&rec_e);
        errs_free(&recs_e);
        free(rtype);
        free(rrms);
        free(perm);
        free(idx);
        plx_fl_fitter_free(&fitter);
    }

    /* ---- pictures and the field PCA ---- */
    diag_pca(dir, f, u, trusted, n, mean_face);
    fclose(f);
    plx_log("wrote %s", path);
    diag_rounds(dir, &congeal_face, rounds, model.n_stages + 2);
    {
        enum { SHOW = 40 };
        int m = PLX_MIN(SHOW, n), *perm = PLX_NEW(int, n);
        for (int i = 0; i < n; i++) perm[i] = i;
        plx_rng_shuffle(&rng, perm, n);
        diag_faces(dir, "before_after.png", &src, T, u, perm, m, NULL);
        keyed *order = PLX_NEW(keyed, n);
        for (int i = 0; i < n; i++) order[i] = (keyed){q[(size_t)i * QCOLS], i};
        qsort(order, n, sizeof(keyed), cmp_keyed_desc);
        for (int j = 0; j < m; j++) perm[j] = order[j].idx;
        diag_faces(dir, "worst.png", &src, T, u, perm, m, q);
        /* the most folded */
        for (int i = 0; i < n; i++) order[i] = (keyed){-q[(size_t)i * QCOLS + 3], i};
        qsort(order, n, sizeof(keyed), cmp_keyed_desc);
        for (int j = 0; j < m; j++) perm[j] = order[j].idx;
        diag_faces(dir, "most_folded.png", &src, T, u, perm, m, q);
        free(order);
        free(perm);
    }
    diag_nodes(dir, nw, df, cn);
    {
        plx_img leg = plx_viz_flow_wheel(96);
        plx_draw_text(&leg, 2, 2, "6 px", 1, WHITE);
        save_png(dir, "flow_legend.png", &leg);
        plx_img_free(&leg);
    }
    plx_log("total %.1f s", plx_now() - t_start);

    for (int r = 0; r < model.n_stages + 2; r++) plx_img_free(&rounds[r].mean_gray);
    plx_fl_model_free(&model);
    plx_img_free(&congeal_face);
    free(rounds);
    free(qual);
    free(T);
    free(Tc);
    free(trusted);
    free(u);
    free(sim);
    free(q);
    free(col);
    free(ns.cost);
    free(ns.mass);
    plx_tensor_free(&gt);
    plx_tensor_free(&st);
    plx_tensor_free(&qt);
    plx_tensor_free(&mt);
    return 0;
}
