// plx-deps: img core
#include "test.h"

#include "core/core.h"
#include "img/image.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char g_tmp[256];
static const char *tmp_path(const char *name) {
    snprintf(g_tmp, sizeof g_tmp, "/tmp/plx_test_core_%d_%s", (int)getpid(), name);
    return g_tmp;
}

TEST(rng_is_deterministic_and_sane) {
    plx_rng a, b;
    plx_rng_seed(&a, 42);
    plx_rng_seed(&b, 42);
    for (int i = 0; i < 100; i++) CHECK(plx_rng_u64(&a) == plx_rng_u64(&b));

    double sum = 0, sum2 = 0, usum = 0;
    int n = 200000, hist[7] = {0};
    for (int i = 0; i < n; i++) {
        double z = plx_rng_normal(&a), u = plx_rng_uniform(&a);
        sum += z; sum2 += z * z; usum += u;
        CHECK(u >= 0.0 && u < 1.0);
        hist[plx_rng_below(&a, 7)]++;
    }
    CHECK_NEAR(sum / n, 0.0, 0.01);
    CHECK_NEAR(sum2 / n, 1.0, 0.02);
    CHECK_NEAR(usum / n, 0.5, 0.005);
    for (int k = 0; k < 7; k++) CHECK_NEAR(hist[k] / (double)n, 1.0 / 7, 0.005);

    int idx[50], seen[50] = {0};
    for (int i = 0; i < 50; i++) idx[i] = i;
    plx_rng_shuffle(&a, idx, 50);
    for (int i = 0; i < 50; i++) seen[idx[i]]++;
    for (int i = 0; i < 50; i++) CHECK(seen[i] == 1);
}

typedef struct { atomic_long sum; atomic_int hits[1000]; int max_tid; } par_ctx;
static void par_body(int64_t i, int tid, void *p) {
    par_ctx *c = p;
    atomic_fetch_add(&c->sum, (long)i);
    atomic_fetch_add(&c->hits[i], 1);
    if (tid < 0 || tid >= c->max_tid) atomic_fetch_add(&c->sum, 1000000000L); /* poison on a bad tid */
}
static void par_nested(int64_t i, int tid, void *p) { plx_parallel_for(10, 0, par_body, p); }

TEST(parallel_for_visits_each_index_once) {
    par_ctx c = {.max_tid = plx_num_threads()};
    plx_parallel_for(1000, 0, par_body, &c);
    CHECK(atomic_load(&c.sum) == 999L * 1000 / 2);
    for (int i = 0; i < 1000; i++) CHECK(atomic_load(&c.hits[i]) == 1);

    par_ctx d = {.max_tid = plx_num_threads()};
    plx_parallel_for(8, 1, par_nested, &d); /* nested loops must degrade to serial, not explode */
    CHECK(atomic_load(&d.sum) == 8L * 45);
    plx_parallel_for(0, 0, par_body, &d);
}

TEST(tensor_roundtrip_load_map_and_stream) {
    int64_t dim[3] = {4, 3, 5};
    float src[60];
    for (int i = 0; i < 60; i++) src[i] = (float)i * 0.5f - 7.0f;
    char path[256];
    strcpy(path, tmp_path("t.plxt"));
    CHECK(plx_tensor_save(path, PLX_F32, 3, dim, src) == PLX_OK);

    plx_tensor t;
    CHECK(plx_tensor_load(path, &t) == PLX_OK);
    CHECK(t.dtype == PLX_F32 && t.ndim == 3 && t.dim[0] == 4 && t.dim[2] == 5 && t.nbytes == sizeof src);
    CHECK(plx_tensor_count(&t) == 60 && memcmp(t.data, src, sizeof src) == 0);
    plx_tensor_free(&t);

    CHECK(plx_tensor_map(path, &t) == PLX_OK);
    CHECK(t.map_base != NULL && memcmp(t.data, src, sizeof src) == 0);
    plx_tensor_free(&t);

    int64_t open_dim[3] = {-1, 3, 5}; /* leading dim discovered at close */
    plx_tensor_writer *w = plx_tensor_writer_open(path, PLX_F32, 3, open_dim);
    CHECK(w != NULL);
    for (int r = 0; r < 4; r++) CHECK(plx_tensor_writer_append(w, src + 15 * r, 15 * sizeof(float)) == PLX_OK);
    CHECK(plx_tensor_writer_close(w) == PLX_OK);
    CHECK(plx_tensor_load(path, &t) == PLX_OK);
    CHECK(t.dim[0] == 4 && memcmp(t.data, src, sizeof src) == 0);
    plx_tensor_free(&t);

    w = plx_tensor_writer_open(path, PLX_F32, 3, dim); /* short write must be rejected */
    CHECK(plx_tensor_writer_append(w, src, 15 * sizeof(float)) == PLX_OK);
    CHECK(plx_tensor_writer_close(w) == PLX_EINVAL);

    CHECK(plx_write_file(path, "PLXTgarbage", 11) == PLX_OK);
    CHECK(plx_tensor_load(path, &t) != PLX_OK);
    CHECK(plx_tensor_map(path, &t) != PLX_OK);
    CHECK(plx_tensor_load("/nonexistent/x.plxt", &t) == PLX_EIO);
    unlink(path);
}

TEST(files_and_mkdir) {
    char dir[256], file[300];
    strcpy(dir, tmp_path("d"));
    snprintf(file, sizeof file, "%s/a/b/c.bin", dir);
    char sub[300];
    snprintf(sub, sizeof sub, "%s/a/b", dir);
    CHECK(plx_mkdir_p(sub) == PLX_OK);
    CHECK(plx_mkdir_p(sub) == PLX_OK); /* idempotent */
    size_t n = 200000;
    uint8_t *src = plx_malloc(n), *back;
    for (size_t i = 0; i < n; i++) src[i] = (uint8_t)(i * 31 + 7);
    CHECK(plx_write_file(file, src, n) == PLX_OK);
    size_t len;
    CHECK(plx_read_file(file, &back, &len) == PLX_OK);
    CHECK(len == n && memcmp(src, back, n) == 0);
    free(src); free(back);
    CHECK(plx_read_file("/nonexistent/file", &back, &len) == PLX_EIO);
    unlink(file);
}

TEST(args) {
    char *argv[] = {"prog", "--n", "12", "--rate=0.5", "--verbose", "--name", "x", "--n", "13", "pos"};
    plx_args a = {10, argv};
    CHECK(plx_arg_int(&a, "n", 0) == 13); /* last wins */
    CHECK_NEAR(plx_arg_num(&a, "rate", 0), 0.5, 0);
    CHECK(plx_arg_flag(&a, "verbose") && !plx_arg_flag(&a, "quiet"));
    CHECK(strcmp(plx_arg_str(&a, "name", "d"), "x") == 0);
    CHECK(strcmp(plx_arg_str(&a, "missing", "d"), "d") == 0);
    CHECK(plx_arg_int(&a, "rat", 7) == 7); /* prefix of another option is not a match */
}

TEST(image_container) {
    uint8_t rgb[2 * 2 * 3] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    plx_img im = plx_img_from_u8(rgb, 2, 2, 3, 1.0f / 255.0f);
    CHECK_NEAR(PLX_PX(&im, 0, 0, 0), 1.0, 1e-6);
    CHECK_NEAR(PLX_PX(&im, 1, 0, 1), 1.0, 1e-6);
    CHECK_NEAR(PLX_PX(&im, 0, 1, 2), 1.0, 1e-6);
    plx_img y = plx_img_luma(&im);
    CHECK(y.c == 1);
    CHECK_NEAR(PLX_PX(&y, 0, 0, 0), 0.299, 1e-6);
    CHECK_NEAR(PLX_PX(&y, 1, 1, 0), 1.0, 1e-6);
    uint8_t back[12];
    plx_img_to_u8(&im, back, 1.0f / 255.0f);
    CHECK(memcmp(back, rgb, 12) == 0);
    plx_img cl = plx_img_clone(&im);
    PLX_PX(&cl, 0, 0, 0) = 9.0f;
    CHECK_NEAR(PLX_PX(&im, 0, 0, 0), 1.0, 1e-6);
    plx_img_to_u8(&cl, back, 1.0f / 255.0f);
    CHECK(back[0] == 255); /* clamped */
    plx_img_free(&im); plx_img_free(&y); plx_img_free(&cl);
    CHECK(im.px == NULL);
}

int main(void) {
    RUN(rng_is_deterministic_and_sane);
    RUN(parallel_for_visits_each_index_once);
    RUN(tensor_roundtrip_load_map_and_stream);
    RUN(files_and_mkdir);
    RUN(args);
    RUN(image_container);
    return TEST_REPORT();
}
