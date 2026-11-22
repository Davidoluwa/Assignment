#include "eval/verif.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

const double plx_verif_far_points[PLX_VERIF_NFAR] = {1e-1, 1e-2, 1e-3};

typedef struct { double s; int y; } item;

static double clean(double s) { return isnan(s) ? -INFINITY : s; }

static int cmp_item_asc(const void *a, const void *b) {
    double x = ((const item *)a)->s, y = ((const item *)b)->s;
    return (x > y) - (x < y);
}

static item *make_items(const double *score, const int *same, int n) {
    item *it = PLX_NEW(item, n);
    for (int i = 0; i < n; i++) it[i] = (item){clean(score[i]), same[i] != 0};
    qsort(it, (size_t)n, sizeof(item), cmp_item_asc);
    return it;
}

/* ---- threshold fitting ---------------------------------------------------------------- */

double plx_verif_accuracy(const double *score, const int *same, int n, double thr) {
    if (n <= 0) return NAN;
    int ok = 0;
    for (int i = 0; i < n; i++) ok += (clean(score[i]) > thr) == (same[i] != 0);
    return (double)ok / n;
}

/* One maximal run of optimal threshold intervals. lo/hi are the score values bounding it;
 * open_lo: the run starts with the "t below every score" interval; open_hi: it ends with
 * the "t at or above the largest score" interval. */
typedef struct { double lo, hi; bool open_lo, open_hi; } run;

static double run_threshold(const run *r, double s_min, double s_max) {
    if (r->open_lo) return nextafter(s_min, -INFINITY);
    if (r->open_hi) return s_max;
    if (r->lo == -INFINITY) return -INFINITY; /* "> -inf" separates the NaN group from the rest */
    if (r->hi == INFINITY) return DBL_MAX;    /* "> DBL_MAX" keeps only +inf scores as same */
    double t = 0.5 * r->lo + 0.5 * r->hi;     /* halves first: no overflow for huge |scores| */
    return t < r->hi ? t : r->lo;             /* adjacent doubles can round the midpoint up */
}

double plx_verif_best_threshold(const double *score, const int *same, int n, double *acc) {
    PLX_ASSERT(n > 0);
    item *it = make_items(score, same, n);
    /* Sweep t upwards through the distinct scores. Interval j (j = 0..G) is
     *   j = 0: t < v_0;   j >= 1: v_{j-1} <= t < v_j  (v_G = +inf).
     * Crossing v_j turns its group from predicted-same into predicted-different. */
    int G = 0;
    for (int i = 0; i < n; i++) G += i == 0 || it[i].s != it[i - 1].s;
    double *v = PLX_NEW(double, G);
    int *correct = PLX_NEW(int, G + 1), c = 0;
    for (int i = 0; i < n; i++) c += it[i].y;
    correct[0] = c;
    for (int i = 0, g = 0; i < n; g++) {
        v[g] = it[i].s;
        for (; i < n && it[i].s == v[g]; i++) c += it[i].y ? -1 : 1;
        correct[g + 1] = c;
    }
    int best = -1;
    for (int j = 0; j <= G; j++)
        if (correct[j] > best && !(j == 0 && v[0] == -INFINITY)) best = correct[j];

    run chosen = {0};
    bool have = false;
    double chosen_w = -1;
    for (int j = 0; j <= G;) {
        if (correct[j] != best || (j == 0 && v[0] == -INFINITY)) { j++; continue; }
        int e = j;
        while (e + 1 <= G && correct[e + 1] == best) e++;
        run r = {.open_lo = j == 0, .open_hi = e == G};
        r.lo = j == 0 ? -INFINITY : v[j - 1];
        r.hi = e == G ? INFINITY : v[e];
        bool bounded = !r.open_lo && !r.open_hi;
        double w = bounded ? r.hi - r.lo : -1;
        bool chosen_bounded = have && !chosen.open_lo && !chosen.open_hi;
        if (!have || (bounded && (!chosen_bounded || w > chosen_w))) { chosen = r; chosen_w = w; have = true; }
        j = e + 1;
    }
    double t = run_threshold(&chosen, v[0], v[G - 1]);
    if (acc) *acc = (double)best / n;
    free(v);
    free(correct);
    free(it);
    return t;
}

/* ---- threshold-free metrics ----------------------------------------------------------- */

void plx_roc_compute(const double *score, const int *same, int n, plx_roc *roc) {
    item *it = make_items(score, same, n);
    int ns = 0;
    for (int i = 0; i < n; i++) ns += it[i].y;
    int nd = n - ns;
    PLX_ASSERT(ns > 0 && nd > 0);
    roc->thr = PLX_NEW(double, n + 1);
    roc->far = PLX_NEW(double, n + 1);
    roc->tar = PLX_NEW(double, n + 1);
    roc->thr[0] = INFINITY;
    roc->n = 1;
    int tp = 0, fp = 0;
    for (int i = n - 1; i >= 0;) { /* descending, one vertex per tie group */
        double s = it[i].s;
        for (; i >= 0 && it[i].s == s; i--) { tp += it[i].y; fp += !it[i].y; }
        roc->thr[roc->n] = s;
        roc->far[roc->n] = (double)fp / nd;
        roc->tar[roc->n] = (double)tp / ns;
        roc->n++;
    }
    free(it);
}

void plx_roc_free(plx_roc *roc) {
    free(roc->thr);
    free(roc->far);
    free(roc->tar);
    memset(roc, 0, sizeof *roc);
}

double plx_roc_tar_at_far(const plx_roc *roc, double far) {
    if (far <= 0) far = 0;
    int i = 0;
    while (i + 1 < roc->n && roc->far[i + 1] <= far) i++; /* last vertex with FAR <= far */
    if (roc->far[i] == far || i + 1 == roc->n) return roc->tar[i];
    double f0 = roc->far[i], f1 = roc->far[i + 1];
    return roc->tar[i] + (far - f0) / (f1 - f0) * (roc->tar[i + 1] - roc->tar[i]);
}

double plx_roc_eer(const plx_roc *roc) {
    /* d = FAR - FRR = FAR + TAR - 1 is non-decreasing along the polyline, from -1 to +1. */
    for (int i = 1; i < roc->n; i++) {
        double d1 = roc->far[i] + roc->tar[i] - 1;
        if (d1 < 0) continue;
        double d0 = roc->far[i - 1] + roc->tar[i - 1] - 1;
        double lam = -d0 / (d1 - d0);
        return roc->far[i - 1] + lam * (roc->far[i] - roc->far[i - 1]);
    }
    return 1.0;
}

double plx_verif_auc(const double *score, const int *same, int n) {
    item *it = make_items(score, same, n);
    double rank_sum = 0; /* sum of 1-based mid-ranks of the "same" pairs */
    int ns = 0;
    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && it[j].s == it[i].s) j++;
        double mid = 0.5 * ((i + 1) + j);
        for (int k = i; k < j; k++)
            if (it[k].y) { rank_sum += mid; ns++; }
        i = j;
    }
    free(it);
    int nd = n - ns;
    PLX_ASSERT(ns > 0 && nd > 0);
    return (rank_sum - 0.5 * ns * (ns + 1.0)) / ((double)ns * nd);
}

/* ---- protocol-level evaluation -------------------------------------------------------- */

static void labels_of(const plx_pair *p, int n, int *same) {
    for (int i = 0; i < n; i++) same[i] = p[i].same != 0;
}

static int threshold_free(const plx_pair *p, int n, const double *score, plx_verif_report *r) {
    r->n_pairs = n;
    r->n_same = r->n_diff = r->n_nan = 0;
    for (int i = 0; i < n; i++) {
        if (p[i].same) r->n_same++;
        else r->n_diff++;
        r->n_nan += isnan(score[i]) != 0;
    }
    if (r->n_same == 0 || r->n_diff == 0) {
        plx_log("verif: need both same and different pairs (got %d same, %d different)", r->n_same, r->n_diff);
        return PLX_EINVAL;
    }
    int *same = PLX_NEW(int, n);
    labels_of(p, n, same);
    plx_roc roc;
    plx_roc_compute(score, same, n, &roc);
    r->auc = plx_verif_auc(score, same, n);
    r->eer = plx_roc_eer(&roc);
    for (int k = 0; k < PLX_VERIF_NFAR; k++) r->tar_at_far[k] = plx_roc_tar_at_far(&roc, plx_verif_far_points[k]);
    plx_roc_free(&roc);
    free(same);
    return PLX_OK;
}

static void report_reset(plx_verif_report *r) {
    memset(r, 0, sizeof *r);
    r->mean_acc = r->sem = r->auc = r->eer = NAN;
    for (int k = 0; k < PLX_VERIF_NFAR; k++) r->tar_at_far[k] = NAN;
}

int plx_verif_evaluate(const plx_pair *p, int n, const double *score, plx_verif_report *r) {
    report_reset(r);
    int rc = threshold_free(p, n, score, r);
    if (rc != PLX_OK) return rc;
    int nf = 0;
    for (int i = 0; i < n; i++) {
        PLX_ASSERT(p[i].fold >= 0 && p[i].fold < PLX_VERIF_MAX_FOLDS);
        nf = PLX_MAX(nf, p[i].fold + 1);
    }
    if (nf < 2) return PLX_OK; /* nothing to cross-validate over: see plx_verif_split */

    double *ts = PLX_NEW(double, n), *vs = PLX_NEW(double, n);
    int *ty = PLX_NEW(int, n), *vy = PLX_NEW(int, n);
    for (int k = 0; k < nf && rc == PLX_OK; k++) {
        int nt = 0, nv = 0;
        for (int i = 0; i < n; i++) {
            if (p[i].fold == k) { vs[nv] = score[i]; vy[nv++] = p[i].same != 0; }
            else { ts[nt] = score[i]; ty[nt++] = p[i].same != 0; }
        }
        if (nv == 0) {
            plx_log("verif: fold %d of %d has no pairs", k, nf);
            rc = PLX_EINVAL;
            break;
        }
        r->fold_thr[k] = plx_verif_best_threshold(ts, ty, nt, &r->fold_train_acc[k]);
        r->fold_acc[k] = plx_verif_accuracy(vs, vy, nv, r->fold_thr[k]);
    }
    free(ts); free(vs); free(ty); free(vy);
    if (rc != PLX_OK) return rc;
    r->n_folds = nf;
    double m = 0, ss = 0;
    for (int k = 0; k < nf; k++) m += r->fold_acc[k];
    m /= nf;
    for (int k = 0; k < nf; k++) ss += (r->fold_acc[k] - m) * (r->fold_acc[k] - m);
    r->mean_acc = m;
    r->sem = sqrt(ss / (nf - 1)) / sqrt((double)nf);
    return PLX_OK;
}

int plx_verif_split(const plx_pair *train, int ntrain, const double *train_score,
                    const plx_pair *test, int ntest, const double *test_score, plx_verif_report *r) {
    report_reset(r);
    if (ntrain <= 0) {
        plx_log("verif: empty training set");
        return PLX_EINVAL;
    }
    int rc = threshold_free(test, ntest, test_score, r);
    if (rc != PLX_OK) return rc;
    int *ty = PLX_NEW(int, PLX_MAX(ntrain, ntest));
    labels_of(train, ntrain, ty);
    r->fold_thr[0] = plx_verif_best_threshold(train_score, ty, ntrain, &r->fold_train_acc[0]);
    labels_of(test, ntest, ty);
    r->fold_acc[0] = r->mean_acc = plx_verif_accuracy(test_score, ty, ntest, r->fold_thr[0]);
    free(ty);
    r->split = true;
    r->n_train = ntrain;
    return PLX_OK;
}

/* ---- reporting ------------------------------------------------------------------------ */

void plx_verif_print(FILE *f, const char *label, const plx_verif_report *r) {
    const char *lab = label ? label : "verif";
    if (r->split) {
        fprintf(f, "[%s] verification, train/test split: %d test pairs (%d same, %d diff, %d NaN), %d train pairs\n",
                lab, r->n_pairs, r->n_same, r->n_diff, r->n_nan, r->n_train);
        fprintf(f, "  accuracy %.4f (test)   train %.4f   threshold %.6g\n", r->mean_acc, r->fold_train_acc[0],
                r->fold_thr[0]);
    } else {
        fprintf(f, "[%s] verification, %d-fold: %d pairs (%d same, %d diff, %d NaN)\n", lab, r->n_folds, r->n_pairs,
                r->n_same, r->n_diff, r->n_nan);
        if (r->n_folds >= 2) {
            fprintf(f, "  accuracy %.4f +/- %.4f (mean +/- S_E over %d folds)\n", r->mean_acc, r->sem, r->n_folds);
            fprintf(f, "  folds   ");
            for (int k = 0; k < r->n_folds; k++) fprintf(f, " %.4f", r->fold_acc[k]);
            fputc('\n', f);
        } else {
            fprintf(f, "  accuracy n/a (single fold; use a train/test split)\n");
        }
    }
    fprintf(f, "  AUC %.4f  EER %.4f  TAR@FAR=10%% %.4f  1%% %.4f  0.1%% %.4f\n", r->auc, r->eer, r->tar_at_far[0],
            r->tar_at_far[1], r->tar_at_far[2]);
}

static bool ends_with(const char *s, const char *suf) {
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && strcmp(s + a - b, suf) == 0;
}

int plx_verif_write_scores(const char *path, const plx_pair *p, int n, const double *score) {
    if (ends_with(path, ".csv")) {
        FILE *f = fopen(path, "w");
        if (!f) { plx_log("verif: cannot write '%s'", path); return PLX_EIO; }
        fprintf(f, "a,b,same,fold,score\n");
        for (int i = 0; i < n; i++)
            fprintf(f, "%d,%d,%d,%d,%.17g\n", p[i].a, p[i].b, p[i].same, p[i].fold, score[i]);
        return fclose(f) == 0 ? PLX_OK : PLX_EIO;
    }
    double *t = PLX_NEW(double, (size_t)n * 5);
    for (int i = 0; i < n; i++) {
        double *row = t + (size_t)i * 5;
        row[0] = p[i].a; row[1] = p[i].b; row[2] = p[i].same; row[3] = p[i].fold; row[4] = score[i];
    }
    int64_t dim[2] = {n, 5};
    int rc = plx_tensor_save(path, PLX_F64, 2, dim, t);
    free(t);
    if (rc != PLX_OK) plx_log("verif: cannot write '%s'", path);
    return rc;
}

int plx_verif_write_roc(const char *path, const plx_pair *p, int n, const double *score) {
    int *same = PLX_NEW(int, n), ns = 0;
    labels_of(p, n, same);
    for (int i = 0; i < n; i++) ns += same[i];
    if (ns == 0 || ns == n) {
        free(same);
        plx_log("verif: ROC needs both same and different pairs");
        return PLX_EINVAL;
    }
    plx_roc roc;
    plx_roc_compute(score, same, n, &roc);
    free(same);
    FILE *f = fopen(path, "w");
    int rc = PLX_OK;
    if (!f) {
        plx_log("verif: cannot write '%s'", path);
        rc = PLX_EIO;
    } else {
        fprintf(f, "threshold,far,tar\n");
        for (int i = 0; i < roc.n; i++) fprintf(f, "%.17g,%.17g,%.17g\n", roc.thr[i], roc.far[i], roc.tar[i]);
        if (fclose(f) != 0) rc = PLX_EIO;
    }
    plx_roc_free(&roc);
    return rc;
}
