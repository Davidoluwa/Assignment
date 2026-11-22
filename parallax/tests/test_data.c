// plx-deps: eval data core
#include "test.h"

#include "core/core.h"
#include "data/lfw.h"
#include "eval/verif.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Scratch space: $PLX_TEST_TMP (default /tmp)/plx_test_data_<pid>. Real-data checks use
 * $PLX_LFW_ROOT (default data/lfw) and data/protocol, and are skipped when absent. */
static char g_dir[512];
static const char *scratch(const char *name, char *buf, size_t cap) {
    snprintf(buf, cap, "%s/%s", g_dir, name);
    return buf;
}

/* Everything the tests create, removed in reverse order at exit. */
static char *g_made[256];
static int g_nmade;
static void track(const char *p) { PLX_ASSERT(g_nmade < 256); g_made[g_nmade++] = plx_strdup(p); }
static void make_dir(const char *p) { PLX_ASSERT(mkdir(p, 0777) == 0); track(p); }
static void make_file(const char *p, const char *text) {
    PLX_ASSERT(plx_write_file(p, text, strlen(text)) == PLX_OK);
    track(p);
}
static void cleanup(void) {
    for (int i = g_nmade - 1; i >= 0; i--) {
        struct stat st;
        if (stat(g_made[i], &st) == 0) (S_ISDIR(st.st_mode) ? rmdir : unlink)(g_made[i]);
        free(g_made[i]);
    }
}

static bool have_python_oracle(const char *script) {
    static int ok = -1;
    if (ok < 0) ok = system("python3 -I -c 'import numpy' >/dev/null 2>&1") == 0;
    return ok && access(script, R_OK) == 0;
}

static bool files_equal(const char *a, const char *b) {
    uint8_t *x, *y;
    size_t nx, ny;
    if (plx_read_file(a, &x, &nx) != PLX_OK) return false;
    if (plx_read_file(b, &y, &ny) != PLX_OK) { free(x); return false; }
    bool eq = nx == ny && memcmp(x, y, nx) == 0;
    free(x);
    free(y);
    return eq;
}

/* ---- synthetic tree ------------------------------------------------------------------- */

static char g_root[600];

static void build_tree(void) {
    char p[700];
    snprintf(g_root, sizeof g_root, "%s/lfw", g_dir);
    make_dir(g_root);
    const char *dirs[] = {"bob", "Zed", "Alice", "Bob", "Empty", ".hidden"};
    for (int i = 0; i < 6; i++) { snprintf(p, sizeof p, "%s/%s", g_root, dirs[i]); make_dir(p); }
    const char *files[] = {"Alice/Alice_0003.jpg", "Alice/Alice_0001.jpg", "Alice/Alice_0012.jpg",
                           "Alice/Alice_2.jpg" /* non-canonical */, "Alice/notes.txt", "Alice/Bob_0001.jpg",
                           "Bob/Bob_0001.jpg", "Bob/Bob_0002.jpg", "Zed/Zed_0001.jpg", "Zed/Zed_0002.jpg",
                           "bob/bob_0007.jpg", ".hidden/.hidden_0001.jpg", "README"};
    for (int i = 0; i < 13; i++) { snprintf(p, sizeof p, "%s/%s", g_root, files[i]); make_file(p, "x"); }
}

TEST(index_synthetic_tree) {
    plx_lfw db;
    CHECK(plx_lfw_open(&db, g_root) == PLX_OK);
    /* Byte-wise order puts upper case first: Alice < Bob < Zed < bob. Empty and hidden dropped. */
    CHECK(db.n_ids == 4 && db.n_images == 8);
    const char *want[] = {"Alice", "Bob", "Zed", "bob"};
    for (int i = 0; i < 4 && i < db.n_ids; i++) CHECK(strcmp(db.id_name[i], want[i]) == 0);
    int nums[] = {1, 3, 12, 1, 2, 1, 2, 7}, ids[] = {0, 0, 0, 1, 1, 2, 2, 3};
    for (int i = 0; i < 8 && i < db.n_images; i++) CHECK(db.img_num[i] == nums[i] && db.img_id[i] == ids[i]);
    CHECK(db.id_first[0] == 0 && db.id_first[1] == 3 && db.id_first[4] == 8 && plx_lfw_id_count(&db, 0) == 3);
    CHECK(plx_lfw_find(&db, "Alice", 12) == 2 && plx_lfw_find(&db, "bob", 7) == 7);
    CHECK(plx_lfw_find(&db, "Alice", 2) == -1 && plx_lfw_find(&db, "Carol", 1) == -1);
    CHECK(plx_lfw_find_id(&db, "Zed") == 2 && plx_lfw_find_id(&db, "Empty") == -1);
    char buf[PLX_LFW_PATH_MAX], want_abs[PLX_LFW_PATH_MAX];
    plx_lfw_relpath(&db, 2, buf, sizeof buf);
    CHECK(strcmp(buf, "Alice/Alice_0012.jpg") == 0);
    plx_lfw_path(&db, 2, buf, sizeof buf);
    snprintf(want_abs, sizeof want_abs, "%s/Alice/Alice_0012.jpg", db.root);
    CHECK(strcmp(buf, want_abs) == 0 && buf[0] == '/' && access(buf, R_OK) == 0);

    char csv[700], ocsv[700], cmd[2000];
    CHECK(plx_lfw_write_csv(&db, scratch("syn.csv", csv, sizeof csv)) == PLX_OK);
    track(csv);
    if (have_python_oracle("tools/oracle/lfw_index.py")) {
        snprintf(cmd, sizeof cmd, "python3 -I tools/oracle/lfw_index.py '%s' '%s'", g_root,
                 scratch("syn_oracle.csv", ocsv, sizeof ocsv));
        CHECK(system(cmd) == 0);
        track(ocsv);
        CHECK(files_equal(csv, ocsv));
    }
    plx_lfw_free(&db);
    CHECK(plx_lfw_open(&db, "/nonexistent/lfw") == PLX_EIO);
    char empty[700];
    snprintf(empty, sizeof empty, "%s/Empty", g_root);
    CHECK(plx_lfw_open(&db, empty) == PLX_EIO);
}

static int load_pairs_text(const char *name, const char *text, const plx_lfw *db, plx_pairs *pl) {
    char p[700];
    scratch(name, p, sizeof p);
    make_file(p, text);
    return plx_pairs_load(pl, p, db);
}

TEST(pairs_synthetic_formats_and_errors) {
    plx_lfw db;
    PLX_ASSERT(plx_lfw_open(&db, g_root) == PLX_OK);
    plx_pairs pl;
    CHECK(load_pairs_text("p10.txt",
                          "2\t1\nAlice\t1\t12\nAlice\t3\tbob\t7\r\nBob\t1\t2\n\nZed\t1\tAlice\t1\n\n",
                          &db, &pl) == PLX_OK);
    CHECK(pl.n == 4 && pl.n_folds == 2);
    if (pl.n == 4) {
        CHECK(pl.p[0].a == 0 && pl.p[0].b == 2 && pl.p[0].same == 1 && pl.p[0].fold == 0);
        CHECK(pl.p[1].a == 1 && pl.p[1].b == 7 && pl.p[1].same == 0 && pl.p[1].fold == 0);
        CHECK(pl.p[2].a == 3 && pl.p[2].b == 4 && pl.p[2].same == 1 && pl.p[2].fold == 1);
        CHECK(pl.p[3].a == 5 && pl.p[3].b == 0 && pl.p[3].same == 0 && pl.p[3].fold == 1);
        uint8_t mask[8];
        plx_pairs_fold_identity_mask(&pl, &db, 0, mask); /* fold 0 names Alice and bob */
        const uint8_t want0[8] = {1, 1, 1, 0, 0, 0, 0, 1};
        CHECK(memcmp(mask, want0, 8) == 0);
        plx_pairs_fold_identity_mask(&pl, &db, 1, mask); /* fold 1: Bob, Zed, Alice */
        const uint8_t want1[8] = {1, 1, 1, 1, 1, 1, 1, 0};
        CHECK(memcmp(mask, want1, 8) == 0);
        plx_pairs_image_mask(&pl, &db, mask);
        const uint8_t wantimg[8] = {1, 1, 1, 1, 1, 1, 0, 1};
        CHECK(memcmp(mask, wantimg, 8) == 0);
    }
    plx_pairs_free(&pl);

    CHECK(load_pairs_text("dev.txt", "1\nZed\t1\t2\nBob\t2\tbob\t7\n", &db, &pl) == PLX_OK);
    CHECK(pl.n == 2 && pl.n_folds == 1 && pl.p[0].same && !pl.p[1].same && pl.p[1].fold == 0);
    plx_pairs_free(&pl);

    /* Malformed lists fail cleanly with PLX_EFORMAT and leave the output empty. */
    const char *bad[] = {
        "1\nCarol\t1\t2\nBob\t1\tZed\t1\n", /* unknown identity */
        "1\nAlice\t1\t2\nBob\t1\tZed\t1\n", /* Alice has no image 2 */
        "1\nAlice\t1\t3\nBob\t1\tZed\n",    /* mismatched line with 3 fields */
        "1\nAlice\t1\t3\n",                 /* truncated */
        "1\nAlice\t1\t3\nBob\t1\tZed\t1\nZed\t1\t2\n", /* trailing extra pair */
        "1\nAlice\t1\tx3\nBob\t1\tZed\t1\n",           /* bad number */
        "1\nAlice\t1\t3\nBob\t1\tBob\t2\n",            /* "mismatched" pair of one identity */
        "two\n", "",
    };
    for (int i = 0; i < (int)(sizeof bad / sizeof bad[0]); i++) {
        char name[32];
        snprintf(name, sizeof name, "bad%d.txt", i);
        int rc = load_pairs_text(name, bad[i], &db, &pl);
        CHECK(rc == PLX_EFORMAT && pl.n == 0 && pl.p == NULL);
    }
    CHECK(plx_pairs_load(&pl, "/nonexistent/pairs.txt", &db) == PLX_EIO);
    plx_lfw_free(&db);
}

/* ---- the real dataset ----------------------------------------------------------------- */

static const char *lfw_root(void) {
    const char *r = getenv("PLX_LFW_ROOT");
    return r ? r : "data/lfw";
}

TEST(index_real_lfw) {
    struct stat st;
    if (stat(lfw_root(), &st) != 0) { printf("  (skipped: no dataset at %s)\n", lfw_root()); return; }
    plx_lfw db;
    CHECK(plx_lfw_open(&db, lfw_root()) == PLX_OK);
    CHECK(db.n_images == 13233 && db.n_ids == 5749);
    CHECK(strcmp(db.id_name[0], "AJ_Cook") == 0);
    int gwb = plx_lfw_find_id(&db, "George_W_Bush");
    CHECK(gwb >= 0 && plx_lfw_id_count(&db, gwb) == 530);
    char buf[PLX_LFW_PATH_MAX];
    plx_lfw_relpath(&db, 0, buf, sizeof buf);
    CHECK(strcmp(buf, "AJ_Cook/AJ_Cook_0001.jpg") == 0);
    int img = plx_lfw_find(&db, "George_W_Bush", 530);
    CHECK(img == db.id_first[gwb + 1] - 1);
    plx_lfw_path(&db, img, buf, sizeof buf);
    CHECK(access(buf, R_OK) == 0);
    for (int i = 0; i + 1 < db.n_images; i++) { /* index order is (id, num) lexicographic */
        bool ordered = db.img_id[i] < db.img_id[i + 1] ||
                       (db.img_id[i] == db.img_id[i + 1] && db.img_num[i] < db.img_num[i + 1]);
        if (!ordered) { CHECK(ordered); break; }
    }
    for (int i = 0; i + 1 < db.n_ids; i++)
        if (strcmp(db.id_name[i], db.id_name[i + 1]) >= 0) { CHECK(!"identities not sorted"); break; }
    if (have_python_oracle("tools/oracle/lfw_index.py")) {
        char csv[700], ocsv[700], cmd[2000];
        CHECK(plx_lfw_write_csv(&db, scratch("lfw.csv", csv, sizeof csv)) == PLX_OK);
        track(csv);
        snprintf(cmd, sizeof cmd, "python3 -I tools/oracle/lfw_index.py '%s' '%s'", lfw_root(),
                 scratch("lfw_oracle.csv", ocsv, sizeof ocsv));
        CHECK(system(cmd) == 0);
        track(ocsv);
        CHECK(files_equal(csv, ocsv));
    }
    plx_lfw_free(&db);
}

TEST(pairs_real_protocol) {
    struct stat st;
    if (stat(lfw_root(), &st) != 0 || stat("data/protocol/pairs.txt", &st) != 0) {
        printf("  (skipped: no dataset / protocol files)\n");
        return;
    }
    plx_lfw db;
    PLX_ASSERT(plx_lfw_open(&db, lfw_root()) == PLX_OK);
    plx_pairs pl;
    CHECK(plx_pairs_load(&pl, "data/protocol/pairs.txt", &db) == PLX_OK);
    CHECK(pl.n == 6000 && pl.n_folds == 10);
    int same = 0, per_fold[10] = {0}, same_fold[10] = {0};
    for (int i = 0; i < pl.n; i++) {
        same += pl.p[i].same;
        per_fold[pl.p[i].fold]++;
        same_fold[pl.p[i].fold] += pl.p[i].same;
        bool sid = db.img_id[pl.p[i].a] == db.img_id[pl.p[i].b];
        if (sid != (pl.p[i].same != 0)) { CHECK(sid == (pl.p[i].same != 0)); break; }
    }
    CHECK(same == 3000);
    for (int k = 0; k < 10; k++) CHECK(per_fold[k] == 600 && same_fold[k] == 300);
    /* First and last lines of the file, resolved. */
    CHECK(pl.p[0].a == plx_lfw_find(&db, "Abel_Pacheco", 1) && pl.p[0].b == plx_lfw_find(&db, "Abel_Pacheco", 4));
    CHECK(pl.p[5999].b == plx_lfw_find(&db, "Sok_An", 1));

    /* Identity-disjoint folds: no identity occurs in two folds. Checked directly, and through
     * the hygiene mask (no image of another fold's pair may be masked for fold k). */
    int *id_fold = PLX_NEW(int, db.n_ids), clash = 0;
    for (int i = 0; i < db.n_ids; i++) id_fold[i] = -1;
    for (int i = 0; i < pl.n; i++)
        for (int e = 0; e < 2; e++) {
            int id = db.img_id[e ? pl.p[i].b : pl.p[i].a];
            if (id_fold[id] >= 0 && id_fold[id] != pl.p[i].fold) clash++;
            id_fold[id] = pl.p[i].fold;
        }
    CHECK(clash == 0);
    uint8_t *mask = PLX_NEW(uint8_t, db.n_images);
    for (int k = 0; k < 10; k++) {
        plx_pairs_fold_identity_mask(&pl, &db, k, mask);
        int leak = 0, own = 0, masked = 0;
        for (int i = 0; i < pl.n; i++) {
            bool m = mask[pl.p[i].a] || mask[pl.p[i].b];
            if (pl.p[i].fold == k) own += mask[pl.p[i].a] && mask[pl.p[i].b];
            else leak += m;
        }
        for (int i = 0; i < db.n_images; i++) masked += mask[i];
        CHECK(leak == 0 && own == 600 && masked > 600);
    }
    plx_pairs_image_mask(&pl, &db, mask);
    int used = 0;
    for (int i = 0; i < db.n_images; i++) used += mask[i];
    CHECK(used > 1000 && used < 12000);
    printf("  pairs.txt: %d distinct images in pairs\n", used);
    free(mask);
    free(id_fold);
    plx_pairs_free(&pl);

    plx_pairs tr, te;
    CHECK(plx_pairs_load(&tr, "data/protocol/pairsDevTrain.txt", &db) == PLX_OK);
    CHECK(plx_pairs_load(&te, "data/protocol/pairsDevTest.txt", &db) == PLX_OK);
    CHECK(tr.n == 2200 && te.n == 1000 && tr.n_folds == 1 && te.n_folds == 1);
    int trs = 0, tes = 0;
    for (int i = 0; i < tr.n; i++) trs += tr.p[i].same;
    for (int i = 0; i < te.n; i++) tes += te.p[i].same;
    CHECK(trs == 1100 && tes == 500 && tr.p[1099].same && !tr.p[1100].same);
    plx_pairs_free(&tr);
    plx_pairs_free(&te);
    plx_lfw_free(&db);
}

/* ---- metrics ---------------------------------------------------------------------------- */

static void pairs_from_labels(const int *same, const int *fold, int n, plx_pair *p) {
    for (int i = 0; i < n; i++) p[i] = (plx_pair){.a = 0, .b = 0, .same = same[i], .fold = fold ? fold[i] : 0};
}

TEST(metrics_hand_computed) {
    /* same {0.9, 0.8, 0.4}, diff {0.6, 0.3, 0.1}: two optimal runs at 5/6 accuracy,
     * [0.3, 0.4) and [0.6, 0.8); the wider one wins -> t = 0.7. AUC = 8/9. */
    double s[] = {0.9, 0.8, 0.4, 0.6, 0.3, 0.1};
    int y[] = {1, 1, 1, 0, 0, 0};
    double acc, t = plx_verif_best_threshold(s, y, 6, &acc);
    CHECK_NEAR(t, 0.7, 1e-15);
    CHECK_NEAR(acc, 5.0 / 6, 1e-15);
    CHECK_NEAR(plx_verif_accuracy(s, y, 6, t), 5.0 / 6, 1e-15);
    CHECK_NEAR(plx_verif_auc(s, y, 6), 8.0 / 9, 1e-15);

    /* Ties: same {1, 1} vs diff {1, 0} -> AUC = (0.5 + 1 + 0.5 + 1) / 4. */
    double s2[] = {1, 1, 1, 0};
    int y2[] = {1, 1, 0, 0};
    CHECK_NEAR(plx_verif_auc(s2, y2, 4), 0.75, 1e-15);
    /* Optimal set is [0, 1) -> midpoint 0.5 (acc 3/4). */
    CHECK_NEAR(plx_verif_best_threshold(s2, y2, 4, &acc), 0.5, 0);
    CHECK_NEAR(acc, 0.75, 0);

    /* ROC: same {3, 2}, diff {2, 1}: vertices (0,0) (0,.5) (.5,1) (1,1). */
    double s3[] = {3, 2, 2, 1};
    int y3[] = {1, 1, 0, 0};
    plx_roc roc;
    plx_roc_compute(s3, y3, 4, &roc);
    CHECK(roc.n == 4);
    CHECK(roc.far[1] == 0 && roc.tar[1] == 0.5 && roc.far[2] == 0.5 && roc.tar[2] == 1 && roc.thr[2] == 2);
    CHECK_NEAR(plx_roc_tar_at_far(&roc, 0.1), 0.6, 1e-15);
    CHECK_NEAR(plx_roc_tar_at_far(&roc, 0.0), 0.5, 0); /* vertical segment at FAR 0: highest TAR */
    CHECK_NEAR(plx_roc_eer(&roc), 0.25, 1e-15);
    plx_roc_free(&roc);

    /* Perfect separation: AUC 1, EER 0, TAR 1 everywhere, every fold accuracy 1, S_E 0. */
    int n = 40, ys[40], fs[40];
    double sc[40];
    plx_pair p[40];
    for (int i = 0; i < n; i++) { ys[i] = i % 2; fs[i] = (i / 2) % 10; sc[i] = ys[i] ? 1 + i : -1 - i; }
    pairs_from_labels(ys, fs, n, p);
    plx_verif_report r;
    CHECK(plx_verif_evaluate(p, n, sc, &r) == PLX_OK);
    CHECK(r.n_folds == 10 && r.mean_acc == 1 && r.sem == 0 && r.auc == 1 && r.eer == 0);
    CHECK(r.tar_at_far[0] == 1 && r.tar_at_far[2] == 1 && r.n_nan == 0);

    /* Reversed scores: everything wrong in the ranking sense. */
    for (int i = 0; i < n; i++) sc[i] = -sc[i];
    CHECK(plx_verif_evaluate(p, n, sc, &r) == PLX_OK);
    CHECK(r.auc == 0 && r.eer == 1 && r.tar_at_far[1] == 0);

    /* S_E is the sample std (n - 1) of the fold accuracies over sqrt(folds). */
    plx_rng g;
    plx_rng_seed(&g, 3);
    for (int i = 0; i < n; i++) sc[i] = plx_rng_normal(&g) + ys[i];
    CHECK(plx_verif_evaluate(p, n, sc, &r) == PLX_OK);
    double m = 0, ss = 0;
    for (int k = 0; k < 10; k++) m += r.fold_acc[k] / 10;
    for (int k = 0; k < 10; k++) ss += (r.fold_acc[k] - m) * (r.fold_acc[k] - m);
    CHECK_NEAR(r.mean_acc, m, 1e-15);
    CHECK_NEAR(r.sem, sqrt(ss / 9) / sqrt(10), 1e-15);

    /* NaN = lowest similarity: same {NaN, 0.9} vs diff {0.1, 0.2} -> AUC 2/4. */
    double s4[] = {NAN, 0.9, 0.1, 0.2};
    int y4[] = {1, 1, 0, 0};
    plx_pair p4[4];
    pairs_from_labels(y4, NULL, 4, p4);
    CHECK(plx_verif_evaluate(p4, 4, s4, &r) == PLX_OK);
    CHECK(r.n_nan == 1 && r.n_folds == 0 && isnan(r.mean_acc));
    CHECK_NEAR(r.auc, 0.5, 0);
    /* Best rule for these is [0.2, 0.9) (acc 3/4) -> t = 0.55; NaN predicted different. */
    CHECK_NEAR(plx_verif_best_threshold(s4, y4, 4, &acc), 0.55, 1e-15);
    CHECK_NEAR(acc, 0.75, 0);

    /* One class only is a reportable error, not a crash. */
    int y5[] = {1, 1, 1, 1};
    pairs_from_labels(y5, NULL, 4, p4);
    CHECK(plx_verif_evaluate(p4, 4, s, &r) == PLX_EINVAL);
}

TEST(metrics_split_and_output) {
    double trs[] = {0.9, 0.8, 0.4, 0.6, 0.3, 0.1}, tes[] = {0.75, 0.5, 0.72, 0.2};
    int try_[] = {1, 1, 1, 0, 0, 0}, tey[] = {1, 1, 0, 0};
    plx_pair tr[6], te[4];
    pairs_from_labels(try_, NULL, 6, tr);
    pairs_from_labels(tey, NULL, 4, te);
    plx_verif_report r;
    CHECK(plx_verif_split(tr, 6, trs, te, 4, tes, &r) == PLX_OK);
    CHECK(r.split && r.n_train == 6 && r.n_pairs == 4);
    CHECK_NEAR(r.fold_thr[0], 0.7, 1e-15);   /* fitted on train */
    CHECK_NEAR(r.mean_acc, 0.5, 0);          /* test: 0.75 yes, 0.5 no(miss), 0.72 yes(false), 0.2 no */
    CHECK_NEAR(r.fold_train_acc[0], 5.0 / 6, 1e-15);
    CHECK_NEAR(r.auc, 0.75, 0);

    char path[700], text[4096];
    FILE *f = fmemopen(text, sizeof text, "w");
    plx_verif_print(f, "split", &r);
    fclose(f);
    CHECK(strstr(text, "[split] verification, train/test split: 4 test pairs (2 same, 2 diff, 0 NaN)") != NULL);
    CHECK(strstr(text, "accuracy 0.5000 (test)") != NULL && strstr(text, "AUC 0.7500") != NULL);

    te[1].a = 11; te[1].b = 12; te[1].fold = 3;
    tes[1] = NAN;
    CHECK(plx_verif_write_scores(scratch("scores.plxt", path, sizeof path), te, 4, tes) == PLX_OK);
    track(path);
    plx_tensor t;
    CHECK(plx_tensor_load(path, &t) == PLX_OK);
    CHECK(t.dtype == PLX_F64 && t.ndim == 2 && t.dim[0] == 4 && t.dim[1] == 5);
    const double *d = t.data;
    CHECK(d[5] == 11 && d[6] == 12 && d[7] == 1 && d[8] == 3 && isnan(d[9]) && d[4] == 0.75);
    plx_tensor_free(&t);
    CHECK(plx_verif_write_scores(scratch("scores.csv", path, sizeof path), te, 4, tes) == PLX_OK);
    track(path);
    uint8_t *raw;
    size_t len;
    CHECK(plx_read_file(path, &raw, &len) == PLX_OK);
    CHECK(len > 20 && memcmp(raw, "a,b,same,fold,score\n0,0,1,0,0.75\n11,12,1,3,nan\n", 47) == 0);
    free(raw);
    CHECK(plx_verif_write_roc(scratch("roc.csv", path, sizeof path), te, 4, tes) == PLX_OK);
    track(path);
    CHECK(plx_read_file(path, &raw, &len) == PLX_OK);
    CHECK(len > 20 && memcmp(raw, "threshold,far,tar\ninf,0,0\n0.75,0,0.5\n", 37) == 0);
    free(raw);
}

TEST(metrics_against_numpy_oracle) {
    if (!have_python_oracle("tools/oracle/verif_oracle.py")) {
        printf("  (skipped: python3 + numpy oracle unavailable)\n");
        return;
    }
    plx_rng g;
    plx_rng_seed(&g, 2024);
    for (int trial = 0; trial < 4; trial++) {
        int n = trial == 0 ? 60 : 1200; /* quantised scores: many ties, some NaN */
        double *in = PLX_NEW(double, (size_t)n * 3), *sc = PLX_NEW(double, n);
        plx_pair *p = PLX_NEW(plx_pair, n);
        double step = trial == 3 ? 1.0 : 0.25; /* coarse grid on the last trial: ties everywhere */
        for (int i = 0; i < n; i++) {
            int same = i % 2, fold = (i / 2) % 10;
            double s = round((plx_rng_normal(&g) + 1.2 * same) / step) * step;
            if (plx_rng_below(&g, 50) == 0) s = NAN;
            sc[i] = s;
            p[i] = (plx_pair){0, 0, same, fold};
            in[3 * i] = s; in[3 * i + 1] = same; in[3 * i + 2] = fold;
        }
        char ip[700], op[700], cmd[2000];
        int64_t dim[2] = {n, 3};
        CHECK(plx_tensor_save(scratch("oracle_in.plxt", ip, sizeof ip), PLX_F64, 2, dim, in) == PLX_OK);
        scratch("oracle_out.plxt", op, sizeof op);
        snprintf(cmd, sizeof cmd, "python3 -I tools/oracle/verif_oracle.py '%s' '%s'", ip, op);
        CHECK(system(cmd) == 0);
        plx_verif_report r;
        CHECK(plx_verif_evaluate(p, n, sc, &r) == PLX_OK);
        plx_tensor t;
        if (plx_tensor_load(op, &t) == PLX_OK && t.dim[0] == 8 + 30) {
            const double *o = t.data;
            CHECK_NEAR(r.auc, o[0], 1e-12);
            CHECK_NEAR(r.eer, o[1], 1e-12);
            for (int k = 0; k < 3; k++) CHECK_NEAR(r.tar_at_far[k], o[2 + k], 1e-12);
            CHECK_NEAR(r.mean_acc, o[5], 1e-12);
            CHECK_NEAR(r.sem, o[6], 1e-12);
            CHECK(r.n_folds == (int)o[7]);
            for (int k = 0; k < 10; k++) {
                CHECK_NEAR(r.fold_acc[k], o[8 + 3 * k], 1e-12);
                CHECK(r.fold_thr[k] == o[9 + 3 * k] || (isinf(r.fold_thr[k]) && r.fold_thr[k] == o[9 + 3 * k]));
                CHECK_NEAR(r.fold_train_acc[k], o[10 + 3 * k], 1e-12);
            }
            if (trial == 1) plx_verif_print(stdout, "oracle-trial", &r);
            plx_tensor_free(&t);
        } else {
            CHECK(!"oracle output missing or malformed");
        }
        unlink(ip);
        unlink(op);
        free(in); free(sc); free(p);
    }
}

int main(void) {
    const char *base = getenv("PLX_TEST_TMP");
    snprintf(g_dir, sizeof g_dir, "%s/plx_test_data_%d", base ? base : "/tmp", (int)getpid());
    PLX_ASSERT(plx_mkdir_p(g_dir) == PLX_OK);
    track(g_dir);
    build_tree();
    RUN(index_synthetic_tree);
    RUN(pairs_synthetic_formats_and_errors);
    RUN(index_real_lfw);
    RUN(pairs_real_protocol);
    RUN(metrics_hand_computed);
    RUN(metrics_split_and_output);
    RUN(metrics_against_numpy_oracle);
    cleanup();
    return TEST_REPORT();
}
