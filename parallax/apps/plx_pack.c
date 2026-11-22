// plx-deps: data img core
/* plx_pack: decode the whole LFW set once into flat tensors, so no later stage touches JPEG.
 *
 *   plx_pack --lfw data/lfw --out runs/main/pack [--rgb]
 *
 * Writes  <out>/gray.plxt   u8 [N, 250, 250]      Rec.601 luma, rounded
 *         <out>/rgb.plxt    u8 [N, 250, 250, 3]   only with --rgb (for textured renders)
 *         <out>/index.csv   global image index -> identity, number, path
 * Row i is global image index i (identities by name, images by number). */
#include "core/core.h"
#include "data/lfw.h"
#include "img/jpeg.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 250, PIXELS = SIDE * SIDE };

typedef struct {
    const plx_lfw *db;
    uint8_t *gray, *rgb; /* rgb may be NULL */
    atomic_int failures;
} pack_ctx;

static void pack_one(int64_t i, int tid, void *p) {
    pack_ctx *c = p;
    char path[PLX_LFW_PATH_MAX];
    plx_lfw_path(c->db, (int)i, path, sizeof path);

    uint8_t *file = NULL, *px = NULL;
    size_t len = 0;
    int w = 0, h = 0, ch = 0;
    int rc = plx_read_file(path, &file, &len);
    if (rc == PLX_OK) rc = plx_jpeg_decode_u8(file, len, &w, &h, &ch, &px);
    if (rc == PLX_OK && (w != SIDE || h != SIDE || (ch != 1 && ch != 3))) rc = PLX_EUNSUPPORTED;
    if (rc != PLX_OK) {
        plx_log("FAILED %s: %s", path, plx_status_str(rc));
        atomic_fetch_add(&c->failures, 1);
    } else {
        uint8_t *g = c->gray + (size_t)i * PIXELS;
        uint8_t *rgb = c->rgb ? c->rgb + (size_t)i * PIXELS * 3 : NULL;
        for (int k = 0; k < PIXELS; k++) {
            int r = px[k * ch], gr = px[k * ch + (ch == 3)], b = px[k * ch + 2 * (ch == 3)];
            g[k] = (uint8_t)((299 * r + 587 * gr + 114 * b + 500) / 1000);
            if (rgb) { rgb[3 * k] = (uint8_t)r; rgb[3 * k + 1] = (uint8_t)gr; rgb[3 * k + 2] = (uint8_t)b; }
        }
    }
    free(file);
    free(px);
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    const char *root = plx_arg_str(&a, "lfw", "data/lfw");
    const char *out = plx_arg_str(&a, "out", "runs/main/pack");
    bool want_rgb = plx_arg_flag(&a, "rgb");

    plx_lfw db;
    if (plx_lfw_open(&db, root) != PLX_OK) plx_die("cannot open LFW at %s (run scripts/fetch_lfw.sh)", root);
    if (plx_mkdir_p(out) != PLX_OK) plx_die("cannot create %s", out);

    size_t n = (size_t)db.n_images;
    pack_ctx c = {.db = &db, .gray = plx_malloc(n * PIXELS), .rgb = want_rgb ? plx_malloc(n * PIXELS * 3) : NULL};
    atomic_init(&c.failures, 0);
    double t0 = plx_now();
    plx_parallel_for((int64_t)n, 16, pack_one, &c);
    int failures = atomic_load(&c.failures);
    plx_log("decoded %zu images in %.1fs (%d failed)", n, plx_now() - t0, failures);
    if (failures) plx_die("refusing to write a pack with missing images");

    char path[4096];
    snprintf(path, sizeof path, "%s/gray.plxt", out);
    if (plx_tensor_save(path, PLX_U8, 3, (int64_t[]){(int64_t)n, SIDE, SIDE}, c.gray) != PLX_OK) plx_die("write failed: %s", path);
    plx_log("wrote %s", path);
    if (want_rgb) {
        snprintf(path, sizeof path, "%s/rgb.plxt", out);
        if (plx_tensor_save(path, PLX_U8, 4, (int64_t[]){(int64_t)n, SIDE, SIDE, 3}, c.rgb) != PLX_OK) plx_die("write failed: %s", path);
        plx_log("wrote %s", path);
    }
    snprintf(path, sizeof path, "%s/index.csv", out);
    if (plx_lfw_write_csv(&db, path) != PLX_OK) plx_die("write failed: %s", path);

    free(c.gray);
    free(c.rgb);
    plx_lfw_free(&db);
    return 0;
}
