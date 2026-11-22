#include "data/lfw.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ---- index -------------------------------------------------------------------------- */

typedef struct { char **v; int n, cap; } str_list;

static void str_push(str_list *l, const char *s) {
    if (l->n == l->cap) l->v = plx_realloc(l->v, sizeof(char *) * (size_t)(l->cap = l->cap ? 2 * l->cap : 64));
    l->v[l->n++] = plx_strdup(s);
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static int cmp_int(const void *a, const void *b) { int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }

static bool is_dir(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* Returns the image number if `file` is exactly "<name>_<NNNN>.jpg" in canonical %04d form,
 * else -1. Canonical form is required so that a path regenerated from (name, num) is
 * byte-identical to the file on disk. */
static int parse_image_name(const char *file, const char *name) {
    size_t ln = strlen(name);
    if (strncmp(file, name, ln) != 0 || file[ln] != '_') return -1;
    const char *p = file + ln + 1;
    if (!isdigit((unsigned char)*p)) return -1;
    char *end;
    errno = 0;
    long num = strtol(p, &end, 10);
    if (errno || num < 0 || num > INT_MAX || strcmp(end, ".jpg") != 0) return -1;
    char canon[32];
    snprintf(canon, sizeof canon, "%04ld.jpg", num);
    return strcmp(p, canon) == 0 ? (int)num : -1;
}

int plx_lfw_open(plx_lfw *db, const char *root) {
    memset(db, 0, sizeof *db);
    char *abs = realpath(root, NULL);
    DIR *d = abs ? opendir(abs) : NULL;
    if (!d) {
        plx_log("lfw: cannot open dataset root '%s'", root);
        free(abs);
        return PLX_EIO;
    }
    str_list names = {0};
    char path[PLX_LFW_PATH_MAX];
    for (struct dirent *e; (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        snprintf(path, sizeof path, "%s/%s", abs, e->d_name);
        if (is_dir(path)) str_push(&names, e->d_name);
    }
    closedir(d);
    if (names.n > 1) qsort(names.v, (size_t)names.n, sizeof(char *), cmp_str);

    /* Second pass: images per identity. Identities without any image are dropped. */
    int img_cap = 1024, *nums = NULL, nums_cap = 0, skipped = 0;
    db->root = abs;
    db->id_name = PLX_NEW(char *, names.n > 0 ? names.n : 1);
    db->id_first = PLX_NEW(int, names.n + 1);
    db->img_id = PLX_NEW(int, img_cap);
    db->img_num = PLX_NEW(int, img_cap);
    for (int i = 0; i < names.n; i++) {
        snprintf(path, sizeof path, "%s/%s", abs, names.v[i]);
        DIR *sd = opendir(path);
        int cnt = 0;
        for (struct dirent *e; sd && (e = readdir(sd));) {
            if (e->d_name[0] == '.') continue;
            int num = parse_image_name(e->d_name, names.v[i]);
            if (num < 0) { skipped++; continue; }
            if (cnt == nums_cap) nums = plx_realloc(nums, sizeof(int) * (size_t)(nums_cap = nums_cap ? 2 * nums_cap : 256));
            nums[cnt++] = num;
        }
        if (sd) closedir(sd);
        if (cnt == 0) { free(names.v[i]); continue; }
        qsort(nums, (size_t)cnt, sizeof(int), cmp_int);
        int id = db->n_ids++;
        db->id_name[id] = names.v[i];
        db->id_first[id] = db->n_images;
        if (db->n_images + cnt > img_cap) {
            while (db->n_images + cnt > img_cap) img_cap *= 2;
            db->img_id = plx_realloc(db->img_id, sizeof(int) * (size_t)img_cap);
            db->img_num = plx_realloc(db->img_num, sizeof(int) * (size_t)img_cap);
        }
        for (int k = 0; k < cnt; k++) {
            db->img_id[db->n_images] = id;
            db->img_num[db->n_images++] = nums[k];
        }
    }
    db->id_first[db->n_ids] = db->n_images;
    free(nums);
    free(names.v);
    if (skipped) plx_log("lfw: ignored %d non-canonical file name(s) under %s", skipped, abs);
    if (db->n_images == 0) {
        plx_log("lfw: no images found under '%s'", abs);
        plx_lfw_free(db);
        return PLX_EIO;
    }
    return PLX_OK;
}

void plx_lfw_free(plx_lfw *db) {
    for (int i = 0; i < db->n_ids; i++) free(db->id_name[i]);
    free(db->id_name);
    free(db->id_first);
    free(db->img_id);
    free(db->img_num);
    free(db->root);
    memset(db, 0, sizeof *db);
}

int plx_lfw_find_id(const plx_lfw *db, const char *name) {
    int lo = 0, hi = db->n_ids - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2, c = strcmp(db->id_name[mid], name);
        if (c == 0) return mid;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int plx_lfw_find(const plx_lfw *db, const char *name, int num) {
    int id = plx_lfw_find_id(db, name);
    if (id < 0) return -1;
    int lo = db->id_first[id], hi = db->id_first[id + 1] - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (db->img_num[mid] == num) return mid;
        if (db->img_num[mid] < num) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int plx_lfw_relpath(const plx_lfw *db, int img, char *buf, size_t cap) {
    PLX_ASSERT(img >= 0 && img < db->n_images);
    const char *name = db->id_name[db->img_id[img]];
    int n = snprintf(buf, cap, "%s/%s_%04d.jpg", name, name, db->img_num[img]);
    PLX_ASSERT(n >= 0 && (size_t)n < cap);
    return n;
}

int plx_lfw_path(const plx_lfw *db, int img, char *buf, size_t cap) {
    int n = snprintf(buf, cap, "%s/", db->root);
    PLX_ASSERT(n >= 0 && (size_t)n < cap);
    return n + plx_lfw_relpath(db, img, buf + n, cap - (size_t)n);
}

int plx_lfw_write_csv(const plx_lfw *db, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { plx_log("lfw: cannot write '%s'", path); return PLX_EIO; }
    char rel[PLX_LFW_PATH_MAX];
    fprintf(f, "index,id,name,num,relpath\n");
    for (int i = 0; i < db->n_images; i++) {
        plx_lfw_relpath(db, i, rel, sizeof rel);
        fprintf(f, "%d,%d,%s,%d,%s\n", i, db->img_id[i], db->id_name[db->img_id[i]], db->img_num[i], rel);
    }
    return fclose(f) == 0 ? PLX_OK : PLX_EIO;
}

/* ---- pair lists --------------------------------------------------------------------- */

#define MAX_TOK 5

/* Splits a line in place on spaces/tabs/CR. Returns the token count (capped at MAX_TOK+1 so
 * that "too many tokens" is still detectable). */
static int tokenize(char *line, char **tok) {
    int n = 0;
    for (char *p = line; *p;) {
        while (*p == ' ' || *p == '\t' || *p == '\r') *p++ = '\0';
        if (!*p) break;
        if (n <= MAX_TOK) tok[n] = p;
        n++;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r') p++;
    }
    return PLX_MIN(n, MAX_TOK + 1);
}

static bool parse_pos_int(const char *s, int *out) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end || v <= 0 || v > INT_MAX) return false;
    *out = (int)v;
    return true;
}

/* Next non-blank line; returns NULL at end of buffer. *lineno is 1-based. */
static char *next_line(char **cursor, int *lineno) {
    while (**cursor) {
        char *line = *cursor, *nl = strchr(line, '\n');
        if (nl) { *nl = '\0'; *cursor = nl + 1; }
        else *cursor = line + strlen(line);
        (*lineno)++;
        if (line[strspn(line, " \t\r")] != '\0') return line;
    }
    return NULL;
}

static int resolve(const plx_lfw *db, const char *path, int lineno, const char *name, const char *num_s, int *img) {
    int num;
    if (!parse_pos_int(num_s, &num)) {
        plx_log("%s:%d: bad image number '%s'", path, lineno, num_s);
        return PLX_EFORMAT;
    }
    int id = plx_lfw_find_id(db, name);
    if (id < 0) {
        plx_log("%s:%d: unknown identity '%s' (not in %s)", path, lineno, name, db->root);
        return PLX_EFORMAT;
    }
    *img = plx_lfw_find(db, name, num);
    if (*img < 0) {
        plx_log("%s:%d: identity '%s' has no image %d (it has %d images)", path, lineno, name, num,
                plx_lfw_id_count(db, id));
        return PLX_EFORMAT;
    }
    return PLX_OK;
}

int plx_pairs_load(plx_pairs *out, const char *path, const plx_lfw *db) {
    memset(out, 0, sizeof *out);
    uint8_t *raw;
    size_t len;
    if (plx_read_file(path, &raw, &len) != PLX_OK) {
        plx_log("pairs: cannot read '%s'", path);
        return PLX_EIO;
    }
    char *text = plx_realloc(raw, len + 1); /* NUL-terminate for the line scanner */
    text[len] = '\0';
    char *cur = text, *tok[MAX_TOK + 1], *line;
    int lineno = 0, rc = PLX_OK, n_folds = 0, per = 0;

    if (!(line = next_line(&cur, &lineno))) {
        plx_log("%s: empty pair list", path);
        rc = PLX_EFORMAT;
    } else {
        int nt = tokenize(line, tok);
        bool ok = false;
        if (nt == 1) { n_folds = 1; ok = parse_pos_int(tok[0], &per); }
        else if (nt == 2) ok = parse_pos_int(tok[0], &n_folds) && parse_pos_int(tok[1], &per);
        if (!ok || (int64_t)n_folds * per * 2 > INT_MAX / 2) {
            plx_log("%s:%d: header must be \"N\" or \"folds\\tN\"", path, lineno);
            rc = PLX_EFORMAT;
        }
    }
    if (rc == PLX_OK) {
        out->n_folds = n_folds;
        out->p = PLX_NEW(plx_pair, (size_t)n_folds * per * 2);
        for (int f = 0; f < n_folds && rc == PLX_OK; f++) {
            for (int k = 0; k < 2 * per && rc == PLX_OK; k++) {
                int same = k < per;
                if (!(line = next_line(&cur, &lineno))) {
                    plx_log("%s: ended after %d of %d pairs", path, out->n, n_folds * per * 2);
                    rc = PLX_EFORMAT;
                    break;
                }
                int nt = tokenize(line, tok);
                plx_pair p = {.same = same, .fold = f};
                if (nt != (same ? 3 : 4)) {
                    plx_log("%s:%d: expected a %s pair with %d fields, got %d", path, lineno,
                            same ? "matched" : "mismatched", same ? 3 : 4, nt);
                    rc = PLX_EFORMAT;
                } else if (same) {
                    rc = resolve(db, path, lineno, tok[0], tok[1], &p.a);
                    if (rc == PLX_OK) rc = resolve(db, path, lineno, tok[0], tok[2], &p.b);
                } else {
                    rc = resolve(db, path, lineno, tok[0], tok[1], &p.a);
                    if (rc == PLX_OK) rc = resolve(db, path, lineno, tok[2], tok[3], &p.b);
                    if (rc == PLX_OK && db->img_id[p.a] == db->img_id[p.b]) {
                        plx_log("%s:%d: mismatched pair names the same identity twice", path, lineno);
                        rc = PLX_EFORMAT;
                    }
                }
                if (rc == PLX_OK) out->p[out->n++] = p;
            }
        }
        if (rc == PLX_OK && (line = next_line(&cur, &lineno))) {
            plx_log("%s:%d: unexpected content after the last pair", path, lineno);
            rc = PLX_EFORMAT;
        }
    }
    free(text);
    if (rc != PLX_OK) plx_pairs_free(out);
    return rc;
}

void plx_pairs_free(plx_pairs *pl) {
    free(pl->p);
    memset(pl, 0, sizeof *pl);
}

void plx_pairs_fold_identity_mask(const plx_pairs *pl, const plx_lfw *db, int fold, uint8_t *mask) {
    PLX_ASSERT(fold >= 0 && fold < pl->n_folds);
    uint8_t *id_hit = PLX_NEW(uint8_t, db->n_ids);
    for (int i = 0; i < pl->n; i++) {
        if (pl->p[i].fold != fold) continue;
        id_hit[db->img_id[pl->p[i].a]] = 1;
        id_hit[db->img_id[pl->p[i].b]] = 1;
    }
    for (int i = 0; i < db->n_images; i++) mask[i] = id_hit[db->img_id[i]];
    free(id_hit);
}

void plx_pairs_image_mask(const plx_pairs *pl, const plx_lfw *db, uint8_t *mask) {
    memset(mask, 0, (size_t)db->n_images);
    for (int i = 0; i < pl->n; i++) {
        PLX_ASSERT(pl->p[i].a >= 0 && pl->p[i].a < db->n_images && pl->p[i].b >= 0 && pl->p[i].b < db->n_images);
        mask[pl->p[i].a] = mask[pl->p[i].b] = 1;
    }
}
