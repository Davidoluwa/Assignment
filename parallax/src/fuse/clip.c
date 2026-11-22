/* Clip container (.plxv) and frame directories. See fuse.h. */
#include "fuse/fuse.h"

#include "img/imgio.h"

#include <ctype.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

enum { PLXV_VERSION = 1, PLXV_MAX_FRAMES = 1 << 20 };

void plx_fu_clip_append(plx_fu_clip *c, double t_ms, const uint8_t *data, size_t len) {
    c->t_ms = plx_realloc(c->t_ms, sizeof(double) * (c->n + 1));
    c->data = plx_realloc(c->data, sizeof(uint8_t *) * (c->n + 1));
    c->len = plx_realloc(c->len, sizeof(size_t) * (c->n + 1));
    c->t_ms[c->n] = t_ms;
    c->data[c->n] = plx_malloc(len ? len : 1);
    memcpy(c->data[c->n], data, len);
    c->len[c->n] = len;
    c->n++;
}

void plx_fu_clip_free(plx_fu_clip *c) {
    for (int i = 0; i < c->n; i++) free(c->data[i]);
    free(c->data);
    free(c->len);
    free(c->t_ms);
    memset(c, 0, sizeof *c);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static void put_u32(uint8_t *p, uint32_t v) {
    for (int k = 0; k < 4; k++) p[k] = (uint8_t)(v >> (8 * k));
}
/* IEEE-754 doubles are stored as their little-endian bit pattern. */
static double get_f64(const uint8_t *p) {
    uint64_t b = 0;
    for (int k = 0; k < 8; k++) b |= (uint64_t)p[k] << (8 * k);
    double v;
    memcpy(&v, &b, 8);
    return v;
}
static void put_f64(uint8_t *p, double v) {
    uint64_t b;
    memcpy(&b, &v, 8);
    for (int k = 0; k < 8; k++) p[k] = (uint8_t)(b >> (8 * k));
}

static int read_plxv(const char *path, plx_fu_clip *c) {
    uint8_t *buf;
    size_t len;
    int st = plx_read_file(path, &buf, &len);
    if (st != PLX_OK) return st;
    st = PLX_EFORMAT;
    if (len < 12 || memcmp(buf, "PLXV", 4) != 0 || get_u32(buf + 4) != PLXV_VERSION) goto done;
    uint32_t n = get_u32(buf + 8);
    if (n > PLXV_MAX_FRAMES) goto done;
    size_t off = 12;
    for (uint32_t i = 0; i < n; i++) {
        if (len - off < 12) goto done;
        double t = get_f64(buf + off);
        uint32_t fl = get_u32(buf + off + 8);
        off += 12;
        if (len - off < fl) goto done;
        plx_fu_clip_append(c, t, buf + off, fl);
        off += fl;
    }
    st = off == len ? PLX_OK : PLX_EFORMAT; /* trailing bytes mean a writer bug, not a frame */
done:
    if (st != PLX_OK) {
        plx_log("%s: not a valid .plxv v%d container", path, PLXV_VERSION);
        plx_fu_clip_free(c);
    }
    free(buf);
    return st;
}

int plx_fu_clip_write(const char *path, const plx_fu_clip *c) {
    size_t total = 12;
    for (int i = 0; i < c->n; i++) total += 12 + c->len[i];
    uint8_t *buf = plx_malloc(total), *p = buf;
    memcpy(p, "PLXV", 4);
    put_u32(p + 4, PLXV_VERSION);
    put_u32(p + 8, (uint32_t)c->n);
    p += 12;
    for (int i = 0; i < c->n; i++) {
        PLX_ASSERT(c->len[i] <= UINT32_MAX);
        put_f64(p, c->t_ms[i]);
        put_u32(p + 8, (uint32_t)c->len[i]);
        memcpy(p + 12, c->data[i], c->len[i]);
        p += 12 + c->len[i];
    }
    int st = plx_write_file(path, buf, total);
    free(buf);
    return st;
}

/* ---- directories -------------------------------------------------------------------------- */
typedef struct { long num; char *name; } dir_entry;

static int cmp_entry(const void *a, const void *b) {
    const dir_entry *x = a, *y = b;
    if (x->num != y->num) return x->num < y->num ? -1 : 1;
    return strcmp(x->name, y->name);
}

static bool frame_name(const char *name, long *num) {
    const char *dot = strrchr(name, '.');
    if (!dot || (strcasecmp(dot, ".jpg") && strcasecmp(dot, ".jpeg") && strcasecmp(dot, ".ppm") &&
                 strcasecmp(dot, ".pgm")))
        return false;
    /* the last run of digits before the extension */
    const char *e = dot;
    while (e > name && !isdigit((unsigned char)e[-1])) e--;
    if (e == name) return false;
    const char *s = e;
    while (s > name && isdigit((unsigned char)s[-1])) s--;
    *num = strtol(s, NULL, 10);
    return true;
}

static int read_dir(const char *path, plx_fu_clip *c) {
    DIR *d = opendir(path);
    if (!d) return PLX_EIO;
    dir_entry *ent = NULL;
    int n = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        long num;
        if (!frame_name(de->d_name, &num)) continue;
        ent = plx_realloc(ent, sizeof(dir_entry) * (n + 1));
        ent[n].num = num;
        ent[n].name = plx_strdup(de->d_name);
        n++;
    }
    closedir(d);
    qsort(ent, n, sizeof(dir_entry), cmp_entry);
    int st = n ? PLX_OK : PLX_EIO;
    for (int i = 0; i < n; i++) {
        char fp[4096];
        snprintf(fp, sizeof fp, "%s/%s", path, ent[i].name);
        uint8_t *buf;
        size_t len;
        if (st == PLX_OK && (st = plx_read_file(fp, &buf, &len)) == PLX_OK) {
            plx_fu_clip_append(c, i * 1000.0 / 30.0, buf, len);
            free(buf);
        }
        free(ent[i].name);
    }
    free(ent);
    if (st != PLX_OK) plx_fu_clip_free(c);
    return st;
}

int plx_fu_clip_read(const char *path, plx_fu_clip *c) {
    memset(c, 0, sizeof *c);
    struct stat sb;
    if (stat(path, &sb) != 0) return PLX_EIO;
    return S_ISDIR(sb.st_mode) ? read_dir(path, c) : read_plxv(path, c);
}

int plx_fu_clip_decode(const plx_fu_clip *c, int i, plx_img *out) {
    PLX_ASSERT(i >= 0 && i < c->n);
    return plx_img_decode(c->data[i], c->len[i], out);
}
