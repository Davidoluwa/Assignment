/* Parallax core runtime: memory, logging, RNG, threads, files, tensor container, CLI args.
 * Everything else in the system sits on this and on libc/libm/pthreads. Nothing else. */
#ifndef PLX_CORE_H
#define PLX_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ---- status ------------------------------------------------------------------------
 * Recoverable failures (bad input file, unsupported format) return a negative status.
 * Programmer errors and out-of-memory abort via plx_die: no NULL checks at call sites. */
typedef enum {
    PLX_OK = 0,
    PLX_EIO = -1,          /* file missing / short read / write failed */
    PLX_EFORMAT = -2,      /* malformed data */
    PLX_EUNSUPPORTED = -3, /* well-formed, but a variant we do not handle */
    PLX_EINVAL = -4        /* bad argument */
} plx_status;
const char *plx_status_str(int status);

/* ---- logging ----------------------------------------------------------------------- */
#if defined(__GNUC__)
#define PLX_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define PLX_PRINTF(a, b)
#endif
void plx_log(const char *fmt, ...) PLX_PRINTF(1, 2);            /* stderr, elapsed-time prefix, thread-safe */
_Noreturn void plx_die(const char *fmt, ...) PLX_PRINTF(1, 2);  /* log + abort() */
#define PLX_ASSERT(cond) \
    do { if (!(cond)) plx_die("assert failed: %s (%s:%d)", #cond, __FILE__, __LINE__); } while (0)

/* ---- memory (never returns NULL) --------------------------------------------------- */
void *plx_malloc(size_t nbytes);
void *plx_calloc(size_t count, size_t size);
void *plx_realloc(void *p, size_t nbytes);
char *plx_strdup(const char *s);
#define PLX_NEW(type, count) ((type *)plx_calloc((size_t)(count), sizeof(type)))

/* ---- time -------------------------------------------------------------------------- */
double plx_now(void); /* monotonic seconds */

/* ---- RNG: xoshiro256**, explicit state so every experiment is reproducible --------- */
typedef struct { uint64_t s[4]; } plx_rng;
void     plx_rng_seed(plx_rng *r, uint64_t seed);
uint64_t plx_rng_u64(plx_rng *r);
uint32_t plx_rng_below(plx_rng *r, uint32_t n); /* uniform in [0, n), unbiased */
double   plx_rng_uniform(plx_rng *r);           /* [0, 1) */
double   plx_rng_normal(plx_rng *r);            /* N(0, 1) */
void     plx_rng_shuffle(plx_rng *r, int *idx, int n);

/* ---- threads -----------------------------------------------------------------------
 * plx_parallel_for runs fn(i, tid, ctx) for i in [0, n) over a dynamic work queue.
 * tid is in [0, plx_num_threads()) and is stable within one call: index per-thread scratch by it.
 * chunk <= 0 picks a default. Nested calls run serially on the calling thread (tid 0).
 * Thread count: PLX_THREADS env var, else online CPUs. */
typedef void (*plx_par_fn)(int64_t i, int tid, void *ctx);
int  plx_num_threads(void);
void plx_set_num_threads(int n);
void plx_parallel_for(int64_t n, int64_t chunk, plx_par_fn fn, void *ctx);

/* ---- files ------------------------------------------------------------------------- */
int plx_read_file(const char *path, uint8_t **data, size_t *len); /* caller frees *data */
int plx_write_file(const char *path, const void *data, size_t len);
int plx_mkdir_p(const char *path);

/* ---- tensor container (.plxt) ------------------------------------------------------
 * The one on-disk format every pipeline stage speaks: a 64-byte header followed by raw
 * little-endian row-major data. Layout: magic "PLXT", u32 version=1, u32 dtype, u32 ndim,
 * i64 dim[6]. Data starts at byte 64, so it is mmap-friendly and trivially readable elsewhere. */
typedef enum { PLX_U8 = 0, PLX_I32 = 1, PLX_F32 = 2, PLX_F64 = 3 } plx_dtype;
#define PLX_MAX_DIMS 6
typedef struct {
    plx_dtype dtype;
    int       ndim;
    int64_t   dim[PLX_MAX_DIMS];
    void     *data;
    size_t    nbytes;
    void     *map_base; /* private: non-NULL when memory-mapped */
    size_t    map_len;
} plx_tensor;
size_t  plx_dtype_size(plx_dtype t);
int64_t plx_tensor_count(const plx_tensor *t); /* product of dims */
int  plx_tensor_save(const char *path, plx_dtype dtype, int ndim, const int64_t *dim, const void *data);
int  plx_tensor_load(const char *path, plx_tensor *t); /* heap copy, writable */
int  plx_tensor_map(const char *path, plx_tensor *t);  /* read-only mmap, for multi-GB packs */
void plx_tensor_free(plx_tensor *t);

/* Streaming writer for tensors too big to hold in RAM. Pass dim[0] = -1 if the leading
 * dimension is not known up front; it is derived from the bytes appended at close. */
typedef struct plx_tensor_writer plx_tensor_writer;
plx_tensor_writer *plx_tensor_writer_open(const char *path, plx_dtype dtype, int ndim, const int64_t *dim);
int plx_tensor_writer_append(plx_tensor_writer *w, const void *data, size_t nbytes);
int plx_tensor_writer_close(plx_tensor_writer *w); /* validates size, patches header, frees w */

/* ---- CLI args: "--name value", "--name=value", bare "--flag" ------------------------ */
typedef struct { int argc; char **argv; } plx_args;
const char *plx_arg_str(const plx_args *a, const char *name, const char *def);
int64_t     plx_arg_int(const plx_args *a, const char *name, int64_t def);
double      plx_arg_num(const plx_args *a, const char *name, double def);
bool        plx_arg_flag(const plx_args *a, const char *name);

#define PLX_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PLX_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PLX_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

#endif
