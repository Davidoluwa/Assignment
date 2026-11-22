/* Dense surface, fused texture and the clip record. See fuse.h. */
#include "fuse/fuse.h"

#include "img/imgio.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- dense surface ---------------------------------------------------------------------------- */
/* NaN nodes filled by repeated averaging of finite 4-neighbours (a harmonic-like extension), so
 * the bicubic interpolation near the border of the estimated region has values to lean on. */
static void fill_nan(int nx, int ny, const double *f, double *g) {
    int P = nx * ny;
    double *h = PLX_NEW(double, P);
    uint8_t *fixed = PLX_NEW(uint8_t, P);
    memcpy(g, f, sizeof(double) * P);
    for (int k = 0; k < P; k++) fixed[k] = isfinite(f[k]);
    for (int it = 0; it < 4 * (nx + ny); it++) {
        memcpy(h, g, sizeof(double) * P);
        for (int k = 0; k < P; k++) {
            if (fixed[k]) continue;
            int r = k / nx, c = k % nx, nb[4][2] = {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}}, m = 0;
            double s = 0;
            for (int j = 0; j < 4; j++)
                if (nb[j][0] >= 0 && nb[j][0] < ny && nb[j][1] >= 0 && nb[j][1] < nx && isfinite(h[nb[j][0] * nx + nb[j][1]]))
                    s += h[nb[j][0] * nx + nb[j][1]], m++;
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

void plx_fu_dense_surface(const plx_lattice *L, const double *S_lat, const uint8_t *used, bool mask, double *dense) {
    int nx = L->nx, ny = L->ny, P = nx * ny, W = PLX_CANON_W, H = PLX_CANON_H;
    double *f = PLX_NEW(double, P), *g = PLX_NEW(double, P);
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < P; k++) f[k] = used[k] ? S_lat[c * P + k] : NAN;
        fill_nan(nx, ny, f, g);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double gx = (x - PLX_LAT_ORIGIN) / L->step, gy = (y - PLX_LAT_ORIGIN) / L->step;
                int ix = PLX_MIN((int)floor(gx), nx - 2), iy = PLX_MIN((int)floor(gy), ny - 2);
                double tx = gx - ix, ty = gy - iy, row[4];
                double *out = dense + (size_t)c * W * H + (size_t)y * W + x;
                if (mask && !(used[iy * nx + ix] && used[iy * nx + ix + 1] && used[(iy + 1) * nx + ix] &&
                              used[(iy + 1) * nx + ix + 1])) {
                    *out = NAN;
                    continue;
                }
                for (int j = 0; j < 4; j++) {
                    int r = PLX_CLAMP(iy - 1 + j, 0, ny - 1);
                    double p[4];
                    for (int i = 0; i < 4; i++) p[i] = g[r * nx + PLX_CLAMP(ix - 1 + i, 0, nx - 1)];
                    row[j] = catmull(p[0], p[1], p[2], p[3], tx);
                }
                *out = catmull(row[0], row[1], row[2], row[3], ty);
            }
    }
    free(f), free(g);
}

/* ---- fused texture ------------------------------------------------------------------------------ */
typedef struct {
    const plx_img      *frames;
    const plx_fu_track *tr;
    const plx_fu_shape *sh;
    const double       *dense;
    plx_img            *warped; /* [F] */
    double             *vis;    /* [F][W H] */
} tex_ctx;

static void tex_one(int64_t f, int tid, void *pc) {
    (void)tid;
    tex_ctx *c = pc;
    const plx_fu_track *t = c->tr;
    int dof = plx_lat_dof(&t->lat), WH = PLX_CANON_W * PLX_CANON_H;
    c->warped[f] = plx_fl_warp_crop(&c->frames[f], t->T[f], &t->lat, t->u + (size_t)f * dof);
    plx_fu_grid_visibility(c->dense, NULL, PLX_CANON_W, PLX_CANON_H, c->sh->R + f * 9, c->sh->scale[f], c->sh->tx[f],
                           c->sh->ty[f], c->frames[f].w, c->frames[f].h, c->vis + (size_t)f * WH);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

void plx_fu_fuse_texture(const plx_img *frames, int F, const plx_fu_track *tr, const plx_fu_shape *sh,
                         const double *dense_S, plx_img *texture, plx_img *weight) {
    int W = PLX_CANON_W, H = PLX_CANON_H, WH = W * H, C = frames[0].c;
    tex_ctx c = {.frames = frames, .tr = tr, .sh = sh, .dense = dense_S, .warped = PLX_NEW(plx_img, F),
                 .vis = PLX_NEW(double, (size_t)F * WH)};
    plx_parallel_for(F, 1, tex_one, &c);
    /* gain: each frame's visibility-weighted mean brightness, matched to the median frame's */
    double *mean = PLX_NEW(double, F), *srt = PLX_NEW(double, F), *fw = PLX_NEW(double, F);
    for (int f = 0; f < F; f++) {
        double sw = 0, sv = 0;
        for (int k = 0; k < WH; k++) {
            double v = c.vis[(size_t)f * WH + k];
            if (v <= 0.5) continue;
            double lum = 0;
            for (int ch = 0; ch < C; ch++) lum += c.warped[f].px[(size_t)ch * WH + k] / C;
            sw += v, sv += v * lum;
        }
        mean[f] = sw > 0 ? sv / sw : NAN;
        srt[f] = isfinite(mean[f]) ? mean[f] : 0;
        double r = sh->frame_rms[f] / (sh->rms > 0 ? sh->rms : 1);
        fw[f] = isfinite(r) ? 1 / (1 + r * r) : 0; /* a frame the rigid model explains badly counts less */
    }
    qsort(srt, F, sizeof(double), cmp_double);
    double ref = srt[F / 2];
    *texture = plx_img_new(W, H, C);
    *weight = plx_img_new(W, H, 1);
    for (int k = 0; k < WH; k++) {
        double sw = 0, acc[3] = {0}, plain[3] = {0};
        for (int f = 0; f < F; f++) {
            double g = isfinite(mean[f]) && mean[f] > 0 ? ref / mean[f] : 1;
            double wv = fw[f] * c.vis[(size_t)f * WH + k];
            for (int ch = 0; ch < C; ch++) {
                double v = g * c.warped[f].px[(size_t)ch * WH + k];
                acc[ch] += wv * v;
                plain[ch] += v / F;
            }
            sw += wv;
        }
        /* outside the estimated surface (hair, background) there is no visibility: plain mean */
        for (int ch = 0; ch < C; ch++) texture->px[(size_t)ch * WH + k] = (float)(sw > 1e-3 ? acc[ch] / sw : plain[ch]);
        weight->px[k] = (float)sw;
    }
    for (int f = 0; f < F; f++) plx_img_free(&c.warped[f]);
    free(c.warped), free(c.vis), free(mean), free(srt), free(fw);
}

/* ---- record -------------------------------------------------------------------------------------- */
static int save(const char *dir, const char *name, int ndim, const int64_t *dim, const float *v) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    return plx_tensor_save(path, PLX_F32, ndim, dim, v);
}

static int save_img(const char *dir, const char *name, const plx_img *im) {
    if (!im->px) return PLX_OK;
    int64_t d3[3] = {im->c, im->h, im->w};
    return im->c == 1 ? save(dir, name, 2, d3 + 1, im->px) : save(dir, name, 3, d3, im->px);
}

int plx_fu_record_save(const char *dir, const plx_fu_record *r) {
    if (plx_mkdir_p(dir) != PLX_OK) return PLX_EIO;
    plx_lattice L8 = plx_lattice_make(PLX_FL_STEP);
    int st = PLX_OK;
    char path[4096];
    snprintf(path, sizeof path, "%s/record.txt", dir);
    if (plx_write_file(path, r->info ? r->info : "", r->info ? strlen(r->info) : 0) != PLX_OK) st = PLX_EIO;
    int64_t ds[3] = {7, L8.ny, L8.nx}, dl[3] = {7, r->lat.ny, r->lat.nx}, dp[3] = {2, L8.ny, L8.nx};
    int64_t dpose[2] = {r->F, 9}, dt[4] = {r->F, 3, r->lat.ny, r->lat.nx};
    if (st == PLX_OK) st = save(dir, "shape.plxt", 3, ds, r->shape);
    if (st == PLX_OK) st = save(dir, "shape_lat.plxt", 3, dl, r->shape_lat);
    if (st == PLX_OK) st = save(dir, "posterior.plxt", 3, dp, r->post);
    if (st == PLX_OK) st = save(dir, "pose.plxt", 2, dpose, r->pose);
    if (st == PLX_OK) st = save(dir, "tracks.plxt", 4, dt, r->tracks);
    if (st == PLX_OK) st = save_img(dir, "depth.plxt", &r->depth);
    if (st == PLX_OK) st = save_img(dir, "texture.plxt", &r->texture);
    if (st == PLX_OK) st = save_img(dir, "texture_w.plxt", &r->texture_w);
    if (st == PLX_OK) st = save_img(dir, "template.plxt", &r->tmpl);
    return st;
}

static float *load_f32(const char *dir, const char *name, int64_t count, int64_t *dim, int ndim, int *st) {
    if (*st != PLX_OK) return NULL;
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    plx_tensor t;
    if ((*st = plx_tensor_load(path, &t)) != PLX_OK) return NULL;
    if (t.dtype != PLX_F32 || t.ndim != ndim || (count >= 0 && plx_tensor_count(&t) != count)) {
        plx_tensor_free(&t);
        *st = PLX_EFORMAT;
        return NULL;
    }
    if (dim) memcpy(dim, t.dim, sizeof(int64_t) * ndim);
    float *v = PLX_NEW(float, plx_tensor_count(&t));
    memcpy(v, t.data, t.nbytes);
    plx_tensor_free(&t);
    return v;
}

/* [H, W] (one channel) or [C, H, W] */
static plx_img load_img(const char *dir, const char *name, int *st) {
    plx_img im = {0};
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    plx_tensor t;
    if ((*st = plx_tensor_load(path, &t)) != PLX_OK) return im;
    if (t.dtype == PLX_F32 && (t.ndim == 2 || t.ndim == 3)) {
        int c = t.ndim == 3 ? (int)t.dim[0] : 1;
        im = plx_img_new((int)t.dim[t.ndim - 1], (int)t.dim[t.ndim - 2], c);
        memcpy(im.px, t.data, t.nbytes);
    } else {
        *st = PLX_EFORMAT;
    }
    plx_tensor_free(&t);
    return im;
}

int plx_fu_record_load(const char *dir, plx_fu_record *r) {
    memset(r, 0, sizeof *r);
    plx_lattice L8 = plx_lattice_make(PLX_FL_STEP);
    int st = PLX_OK;
    char path[4096];
    snprintf(path, sizeof path, "%s/record.txt", dir);
    uint8_t *txt;
    size_t len;
    if ((st = plx_read_file(path, &txt, &len)) != PLX_OK) return st;
    r->info = plx_malloc(len + 1);
    memcpy(r->info, txt, len);
    r->info[len] = 0;
    free(txt);
    int64_t dim[4];
    r->shape = load_f32(dir, "shape.plxt", 7 * plx_lat_nodes(&L8), NULL, 3, &st);
    r->shape_lat = load_f32(dir, "shape_lat.plxt", -1, dim, 3, &st);
    if (st == PLX_OK) {
        int step = (int)lrint(PLX_FL_STEP * (L8.nx - 1) / (double)(dim[2] - 1));
        r->lat = plx_lattice_make(step);
        if (dim[0] != 7 || dim[1] != r->lat.ny || dim[2] != r->lat.nx) st = PLX_EFORMAT;
    }
    r->post = load_f32(dir, "posterior.plxt", 2 * plx_lat_nodes(&L8), NULL, 3, &st);
    r->pose = load_f32(dir, "pose.plxt", -1, dim, 2, &st);
    if (st == PLX_OK) r->F = (int)dim[0];
    r->tracks = load_f32(dir, "tracks.plxt", (int64_t)r->F * 3 * plx_lat_nodes(&r->lat), NULL, 4, &st);
    if (st == PLX_OK) r->depth = load_img(dir, "depth.plxt", &st);
    if (st == PLX_OK) r->texture = load_img(dir, "texture.plxt", &st);
    if (st == PLX_OK) r->texture_w = load_img(dir, "texture_w.plxt", &st);
    if (st == PLX_OK) r->tmpl = load_img(dir, "template.plxt", &st);
    r->C = r->texture.c;
    if (st != PLX_OK) plx_fu_record_free(r);
    return st;
}

void plx_fu_record_free(plx_fu_record *r) {
    free(r->shape), free(r->shape_lat), free(r->post), free(r->pose), free(r->tracks), free(r->info);
    plx_img_free(&r->depth), plx_img_free(&r->texture), plx_img_free(&r->texture_w), plx_img_free(&r->tmpl);
    memset(r, 0, sizeof *r);
}
