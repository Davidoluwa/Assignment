#include "core/core.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ---- status / logging --------------------------------------------------------------- */

const char *plx_status_str(int status) {
    switch (status) {
    case PLX_OK: return "ok";
    case PLX_EIO: return "i/o error";
    case PLX_EFORMAT: return "malformed data";
    case PLX_EUNSUPPORTED: return "unsupported variant";
    case PLX_EINVAL: return "invalid argument";
    default: return "unknown status";
    }
}

double plx_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;
static double g_log_t0 = -1.0;

static void vlog(const char *fmt, va_list ap) {
    pthread_mutex_lock(&g_log_mu);
    if (g_log_t0 < 0) g_log_t0 = plx_now();
    fprintf(stderr, "[%8.2fs] ", plx_now() - g_log_t0);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    pthread_mutex_unlock(&g_log_mu);
}

void plx_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlog(fmt, ap);
    va_end(ap);
}

_Noreturn void plx_die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlog(fmt, ap);
    va_end(ap);
    abort();
}

/* ---- memory ------------------------------------------------------------------------- */

void *plx_malloc(size_t nbytes) {
    void *p = malloc(nbytes ? nbytes : 1);
    if (!p) plx_die("out of memory (%zu bytes)", nbytes);
    return p;
}

void *plx_calloc(size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) plx_die("allocation overflow (%zu x %zu)", count, size);
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) plx_die("out of memory (%zu x %zu bytes)", count, size);
    return p;
}

void *plx_realloc(void *p, size_t nbytes) {
    void *q = realloc(p, nbytes ? nbytes : 1);
    if (!q) plx_die("out of memory (%zu bytes)", nbytes);
    return q;
}

char *plx_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    return memcpy(plx_malloc(n), s, n);
}

/* ---- RNG ---------------------------------------------------------------------------- */

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

void plx_rng_seed(plx_rng *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) { /* splitmix64 expands the seed into a full state */
        uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        r->s[i] = z ^ (z >> 31);
    }
}

uint64_t plx_rng_u64(plx_rng *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl64(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

uint32_t plx_rng_below(plx_rng *r, uint32_t n) {
    PLX_ASSERT(n > 0);
    uint32_t limit = UINT32_MAX - UINT32_MAX % n; /* rejection keeps it unbiased */
    uint32_t x;
    do x = (uint32_t)(plx_rng_u64(r) >> 32); while (x >= limit);
    return x % n;
}

double plx_rng_uniform(plx_rng *r) { return (double)(plx_rng_u64(r) >> 11) * 0x1.0p-53; }

double plx_rng_normal(plx_rng *r) {
    double u, v, s;
    do { /* Marsaglia polar; the second deviate is discarded to keep the state trivially copyable */
        u = 2.0 * plx_rng_uniform(r) - 1.0;
        v = 2.0 * plx_rng_uniform(r) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    return u * sqrt(-2.0 * log(s) / s);
}

void plx_rng_shuffle(plx_rng *r, int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)plx_rng_below(r, (uint32_t)i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

/* ---- threads ------------------------------------------------------------------------ */

static int g_num_threads = 0;
static _Thread_local int tl_in_parallel = 0;

int plx_num_threads(void) {
    if (g_num_threads > 0) return g_num_threads;
    int n = 0;
    const char *env = getenv("PLX_THREADS");
    if (env) n = atoi(env);
    if (n <= 0) n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) n = 1;
    g_num_threads = n;
    return n;
}

void plx_set_num_threads(int n) { g_num_threads = n > 0 ? n : 0; }

typedef struct {
    atomic_int_fast64_t next;
    int64_t n, chunk;
    plx_par_fn fn;
    void *ctx;
} par_job;

typedef struct { par_job *job; int tid; } par_arg;

static void *par_worker(void *p) {
    par_arg *a = p;
    par_job *j = a->job;
    int was = tl_in_parallel;
    tl_in_parallel = 1;
    for (;;) {
        int64_t s = atomic_fetch_add(&j->next, j->chunk);
        if (s >= j->n) break;
        int64_t e = PLX_MIN(s + j->chunk, j->n);
        for (int64_t i = s; i < e; i++) j->fn(i, a->tid, j->ctx);
    }
    tl_in_parallel = was;
    return NULL;
}

void plx_parallel_for(int64_t n, int64_t chunk, plx_par_fn fn, void *ctx) {
    if (n <= 0) return;
    int nt = plx_num_threads();
    if (nt > n) nt = (int)n;
    if (nt <= 1 || tl_in_parallel) {
        for (int64_t i = 0; i < n; i++) fn(i, 0, ctx);
        return;
    }
    if (chunk <= 0) chunk = PLX_MAX(1, n / ((int64_t)nt * 8));
    par_job job = {.n = n, .chunk = chunk, .fn = fn, .ctx = ctx};
    atomic_init(&job.next, 0);
    pthread_t *th = PLX_NEW(pthread_t, nt);
    par_arg *args = PLX_NEW(par_arg, nt);
    for (int t = 0; t < nt; t++) args[t] = (par_arg){&job, t};
    for (int t = 1; t < nt; t++)
        if (pthread_create(&th[t], NULL, par_worker, &args[t]) != 0) plx_die("pthread_create failed");
    par_worker(&args[0]); /* the caller is worker 0 */
    for (int t = 1; t < nt; t++) pthread_join(th[t], NULL);
    free(th);
    free(args);
}

/* ---- files -------------------------------------------------------------------------- */

int plx_read_file(const char *path, uint8_t **data, size_t *len) {
    *data = NULL;
    *len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return PLX_EIO;
    size_t cap = 1 << 16, n = 0;
    uint8_t *buf = plx_malloc(cap);
    for (;;) {
        size_t got = fread(buf + n, 1, cap - n, f);
        n += got;
        if (got == 0) break;
        if (n == cap) buf = plx_realloc(buf, cap *= 2);
    }
    int bad = ferror(f);
    fclose(f);
    if (bad) { free(buf); return PLX_EIO; }
    *data = buf;
    *len = n;
    return PLX_OK;
}

int plx_write_file(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return PLX_EIO;
    size_t put = len ? fwrite(data, 1, len, f) : 0;
    int bad = (put != len) | (fclose(f) != 0);
    return bad ? PLX_EIO : PLX_OK;
}

int plx_mkdir_p(const char *path) {
    char *tmp = plx_strdup(path);
    int rc = PLX_OK;
    for (char *p = tmp + 1;; p++) {
        if (*p != '/' && *p != '\0') continue;
        char saved = *p;
        *p = '\0';
        if (mkdir(tmp, 0777) != 0 && errno != EEXIST) { rc = PLX_EIO; break; }
        if (saved == '\0') break;
        *p = saved;
    }
    free(tmp);
    return rc;
}

/* ---- tensor container --------------------------------------------------------------- */

typedef struct {
    char     magic[4];
    uint32_t version, dtype, ndim;
    int64_t  dim[PLX_MAX_DIMS];
} plxt_header;
_Static_assert(sizeof(plxt_header) == 64, "plxt header must be exactly 64 bytes");

size_t plx_dtype_size(plx_dtype t) {
    switch (t) {
    case PLX_U8: return 1;
    case PLX_I32: return 4;
    case PLX_F32: return 4;
    case PLX_F64: return 8;
    }
    plx_die("bad dtype %d", (int)t);
}

int64_t plx_tensor_count(const plx_tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->dim[i];
    return n;
}

static void require_little_endian(void) {
    const uint16_t probe = 1;
    if (*(const uint8_t *)&probe != 1) plx_die(".plxt I/O assumes a little-endian host");
}

static plxt_header make_header(plx_dtype dtype, int ndim, const int64_t *dim) {
    PLX_ASSERT(ndim >= 1 && ndim <= PLX_MAX_DIMS);
    plxt_header h = {.magic = {'P', 'L', 'X', 'T'}, .version = 1, .dtype = (uint32_t)dtype, .ndim = (uint32_t)ndim};
    for (int i = 0; i < ndim; i++) h.dim[i] = dim[i];
    return h;
}

/* Validates a header and fills everything in *t except the data pointer. */
static int parse_header(const plxt_header *h, uint64_t file_len, plx_tensor *t) {
    if (memcmp(h->magic, "PLXT", 4) != 0 || h->version != 1) return PLX_EFORMAT;
    if (h->dtype > PLX_F64 || h->ndim < 1 || h->ndim > PLX_MAX_DIMS) return PLX_EFORMAT;
    memset(t, 0, sizeof *t);
    t->dtype = (plx_dtype)h->dtype;
    t->ndim = (int)h->ndim;
    uint64_t count = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (h->dim[i] < 0) return PLX_EFORMAT;
        if (h->dim[i] && count > (UINT64_MAX / 16) / (uint64_t)h->dim[i]) return PLX_EFORMAT;
        count *= (uint64_t)h->dim[i];
        t->dim[i] = h->dim[i];
    }
    uint64_t nbytes = count * plx_dtype_size(t->dtype);
    if (file_len < sizeof *h || nbytes != file_len - sizeof *h) return PLX_EFORMAT;
    t->nbytes = (size_t)nbytes;
    return PLX_OK;
}

int plx_tensor_save(const char *path, plx_dtype dtype, int ndim, const int64_t *dim, const void *data) {
    plx_tensor_writer *w = plx_tensor_writer_open(path, dtype, ndim, dim);
    if (!w) return PLX_EIO;
    plx_tensor shape = {.dtype = dtype, .ndim = ndim};
    for (int i = 0; i < ndim; i++) shape.dim[i] = dim[i];
    int rc = plx_tensor_writer_append(w, data, (size_t)plx_tensor_count(&shape) * plx_dtype_size(dtype));
    int rc2 = plx_tensor_writer_close(w);
    return rc != PLX_OK ? rc : rc2;
}

int plx_tensor_load(const char *path, plx_tensor *t) {
    require_little_endian();
    memset(t, 0, sizeof *t);
    FILE *f = fopen(path, "rb");
    if (!f) return PLX_EIO;
    struct stat st;
    plxt_header h;
    int rc = PLX_OK;
    if (fstat(fileno(f), &st) != 0 || fread(&h, sizeof h, 1, f) != 1) rc = PLX_EIO;
    if (rc == PLX_OK) rc = parse_header(&h, (uint64_t)st.st_size, t);
    if (rc == PLX_OK) {
        t->data = plx_malloc(t->nbytes);
        if (t->nbytes && fread(t->data, 1, t->nbytes, f) != t->nbytes) {
            free(t->data);
            memset(t, 0, sizeof *t);
            rc = PLX_EIO;
        }
    }
    fclose(f);
    return rc;
}

int plx_tensor_map(const char *path, plx_tensor *t) {
    require_little_endian();
    memset(t, 0, sizeof *t);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return PLX_EIO;
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(plxt_header)) { close(fd); return PLX_EIO; }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return PLX_EIO;
    int rc = parse_header(base, (uint64_t)st.st_size, t);
    if (rc != PLX_OK) { munmap(base, (size_t)st.st_size); memset(t, 0, sizeof *t); return rc; }
    t->data = (uint8_t *)base + sizeof(plxt_header);
    t->map_base = base;
    t->map_len = (size_t)st.st_size;
    return PLX_OK;
}

void plx_tensor_free(plx_tensor *t) {
    if (t->map_base) munmap(t->map_base, t->map_len);
    else free(t->data);
    memset(t, 0, sizeof *t);
}

struct plx_tensor_writer {
    FILE *f;
    plxt_header h;
    uint64_t written;
};

plx_tensor_writer *plx_tensor_writer_open(const char *path, plx_dtype dtype, int ndim, const int64_t *dim) {
    require_little_endian();
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;
    plx_tensor_writer *w = PLX_NEW(plx_tensor_writer, 1);
    w->f = f;
    w->h = make_header(dtype, ndim, dim);
    if (fwrite(&w->h, sizeof w->h, 1, f) != 1) { fclose(f); free(w); return NULL; }
    return w;
}

int plx_tensor_writer_append(plx_tensor_writer *w, const void *data, size_t nbytes) {
    if (nbytes && fwrite(data, 1, nbytes, w->f) != nbytes) return PLX_EIO;
    w->written += nbytes;
    return PLX_OK;
}

int plx_tensor_writer_close(plx_tensor_writer *w) {
    int rc = PLX_OK;
    uint64_t row = plx_dtype_size((plx_dtype)w->h.dtype);
    for (uint32_t i = 1; i < w->h.ndim; i++) row *= (uint64_t)w->h.dim[i];
    if (w->h.dim[0] < 0) { /* leading dimension discovered from what was appended */
        if (row == 0 || w->written % row != 0) rc = PLX_EINVAL;
        else w->h.dim[0] = (int64_t)(w->written / row);
        if (rc == PLX_OK && (fseek(w->f, 0, SEEK_SET) != 0 || fwrite(&w->h, sizeof w->h, 1, w->f) != 1)) rc = PLX_EIO;
    } else if (w->written != row * (uint64_t)w->h.dim[0]) {
        rc = PLX_EINVAL;
    }
    if (fclose(w->f) != 0 && rc == PLX_OK) rc = PLX_EIO;
    free(w);
    return rc;
}

/* ---- CLI args ----------------------------------------------------------------------- */

/* Returns the value for --name, "" for a bare flag, NULL if absent. Last occurrence wins. */
static const char *arg_find(const plx_args *a, const char *name) {
    size_t n = strlen(name);
    const char *found = NULL;
    for (int i = 1; i < a->argc; i++) {
        const char *s = a->argv[i];
        if (s[0] != '-' || s[1] != '-' || strncmp(s + 2, name, n) != 0) continue;
        if (s[2 + n] == '=') found = s + 3 + n;
        else if (s[2 + n] == '\0')
            found = (i + 1 < a->argc && strncmp(a->argv[i + 1], "--", 2) != 0) ? a->argv[i + 1] : "";
    }
    return found;
}

const char *plx_arg_str(const plx_args *a, const char *name, const char *def) {
    const char *v = arg_find(a, name);
    return (v && *v) ? v : def;
}

int64_t plx_arg_int(const plx_args *a, const char *name, int64_t def) {
    const char *v = arg_find(a, name);
    if (!v || !*v) return def;
    char *end;
    long long x = strtoll(v, &end, 10);
    if (*end) plx_die("--%s expects an integer, got '%s'", name, v);
    return (int64_t)x;
}

double plx_arg_num(const plx_args *a, const char *name, double def) {
    const char *v = arg_find(a, name);
    if (!v || !*v) return def;
    char *end;
    double x = strtod(v, &end);
    if (*end) plx_die("--%s expects a number, got '%s'", name, v);
    return x;
}

bool plx_arg_flag(const plx_args *a, const char *name) { return arg_find(a, name) != NULL; }
