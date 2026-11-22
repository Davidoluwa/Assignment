// plx-deps: img core
/* Decodes a list of JPEGs with the Parallax decoder into one u8 tensor [N, H, W, 3] so that
 * tools/oracle/jpeg_vs_pil.py can compare it against PIL. Grayscale images are replicated to
 * three channels (what PIL's convert("RGB") does). All images must share one size.
 *
 *   plx_decode_dump --list paths.txt --out dump.plxt [--png DIR]   (PLX_THREADS=1 for 1-thread timing)
 *
 * --png also writes image i as DIR/<i>.png (decoded channels), DIR/<i>_a.png (plus an alpha
 * channel = first channel ^ 0x5A) and DIR/<i>_16.png (16-bit gray = first << 8 | last channel),
 * so the oracle can check that PIL reads every PNG variant back exactly. */
#include "core/core.h"
#include "img/jpeg.h"
#include "img/png.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    char **paths;
    const char *png_dir;
    int w, h;
    uint8_t *out;
    int *status;
    double *decode_s; /* per-thread decode-only seconds */
} dump_ctx;

static int write_pngs(const char *dir, int64_t i, const uint8_t *px, int w, int h, int c) {
    char path[4096];
    size_t n = (size_t)w * h;
    snprintf(path, sizeof path, "%s/%06lld.png", dir, (long long)i);
    int st = plx_png_write_u8(path, px, w, h, c);
    uint8_t *alpha = plx_malloc(n * (c + 1));
    uint16_t *g16 = plx_malloc(sizeof *g16 * n);
    for (size_t k = 0; k < n; k++) {
        memcpy(alpha + k * (c + 1), px + k * c, (size_t)c);
        alpha[k * (c + 1) + c] = px[k * c] ^ 0x5A;
        g16[k] = (uint16_t)(px[k * c] << 8 | px[k * c + c - 1]);
    }
    snprintf(path, sizeof path, "%s/%06lld_a.png", dir, (long long)i);
    if (st == PLX_OK) st = plx_png_write_u8(path, alpha, w, h, c + 1);
    snprintf(path, sizeof path, "%s/%06lld_16.png", dir, (long long)i);
    if (st == PLX_OK) st = plx_png_write_gray16(path, g16, w, h);
    free(alpha);
    free(g16);
    return st;
}

static void dump_one(int64_t i, int tid, void *p) {
    dump_ctx *ctx = p;
    uint8_t *data, *px;
    size_t len;
    int st = plx_read_file(ctx->paths[i], &data, &len);
    if (st < 0) {
        ctx->status[i] = st;
        return;
    }
    int w, h, c;
    double t0 = plx_now();
    st = plx_jpeg_decode_u8(data, len, &w, &h, &c, &px);
    ctx->decode_s[tid] += plx_now() - t0;
    free(data);
    if (st == PLX_OK && (w != ctx->w || h != ctx->h)) st = PLX_EINVAL;
    if (st == PLX_OK) {
        uint8_t *dst = ctx->out + (size_t)i * ctx->w * ctx->h * 3;
        size_t n = (size_t)w * h;
        for (size_t k = 0; k < n; k++)
            for (int ch = 0; ch < 3; ch++) dst[k * 3 + ch] = px[k * c + (c == 3 ? ch : 0)];
    }
    if (st == PLX_OK && ctx->png_dir) st = write_pngs(ctx->png_dir, i, px, w, h, c);
    free(px);
    ctx->status[i] = st;
}

static char **read_lines(const char *path, int *n) {
    uint8_t *data;
    size_t len;
    if (plx_read_file(path, &data, &len) < 0) plx_die("cannot read %s", path);
    char **lines = NULL;
    int count = 0;
    size_t start = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i < len && data[i] != '\n') continue;
        size_t end = i;
        if (end > start && data[end - 1] == '\r') end--;
        if (end > start) {
            lines = plx_realloc(lines, sizeof *lines * (size_t)(count + 1));
            lines[count] = plx_malloc(end - start + 1);
            memcpy(lines[count], data + start, end - start);
            lines[count][end - start] = 0;
            count++;
        }
        start = i + 1;
    }
    free(data);
    *n = count;
    return lines;
}

int main(int argc, char **argv) {
    plx_args a = {argc, argv};
    const char *list = plx_arg_str(&a, "list", NULL), *outp = plx_arg_str(&a, "out", NULL);
    if (!list || !outp) {
        fprintf(stderr, "usage: %s --list paths.txt --out dump.plxt [--png DIR]\n", argv[0]);
        return 2;
    }
    int n;
    char **paths = read_lines(list, &n);
    if (n == 0) plx_die("%s lists no images", list);

    dump_ctx ctx = {.paths = paths, .png_dir = plx_arg_str(&a, "png", NULL)};
    if (ctx.png_dir && plx_mkdir_p(ctx.png_dir) < 0) plx_die("cannot create %s", ctx.png_dir);
    uint8_t *data;
    size_t len;
    int c, st = plx_read_file(paths[0], &data, &len);
    if (st == PLX_OK) {
        st = plx_jpeg_info(data, len, &ctx.w, &ctx.h, &c);
        free(data);
    }
    if (st < 0) plx_die("%s: %s", paths[0], plx_status_str(st));

    ctx.out = plx_malloc((size_t)n * ctx.w * ctx.h * 3);
    ctx.status = PLX_NEW(int, n);
    ctx.decode_s = PLX_NEW(double, plx_num_threads());
    double t0 = plx_now();
    plx_parallel_for(n, 4, dump_one, &ctx);
    double wall = plx_now() - t0, dec = 0;
    for (int t = 0; t < plx_num_threads(); t++) dec += ctx.decode_s[t];

    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (ctx.status[i] == PLX_OK) continue;
        if (bad++ < 20) plx_log("%s: %s", paths[i], plx_status_str(ctx.status[i]));
    }
    plx_log("%d images %dx%d, %d threads: wall %.2f s (read+decode), decode-only %.2f s = %.3f ms/image",
            n, ctx.w, ctx.h, plx_num_threads(), wall, dec, 1e3 * dec / n);
    if (bad) plx_die("%d images failed to decode", bad);

    int64_t dim[4] = {n, ctx.h, ctx.w, 3};
    st = plx_tensor_save(outp, PLX_U8, 4, dim, ctx.out);
    if (st < 0) plx_die("%s: %s", outp, plx_status_str(st));
    for (int i = 0; i < n; i++) free(paths[i]);
    free(paths);
    free(ctx.out);
    free(ctx.status);
    free(ctx.decode_s);
    return 0;
}
