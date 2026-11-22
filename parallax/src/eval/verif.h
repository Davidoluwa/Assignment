/* Face verification metrics for the LFW protocol.
 *
 * Input everywhere: pairs (labels + folds) and one similarity score per pair, higher meaning
 * "more likely the same person". A NaN score is treated as the lowest possible similarity
 * (-infinity) and counted in n_nan, so a pipeline that fails on some images still gets a
 * well-defined, pessimistic score instead of silently skewing thresholds.
 *
 * Decision rules (stated once, used consistently):
 *   - accuracy uses a threshold t and predicts "same" iff score > t.
 *     The fitted t maximises training accuracy. Accuracy as a function of t is piecewise
 *     constant on intervals [s_i, s_{i+1}) between consecutive distinct scores; adjacent
 *     optimal intervals are merged, and t is the midpoint of the widest merged optimal
 *     interval (bounded intervals preferred over the two unbounded ends; ties -> lowest).
 *     Unbounded ends: "all same" uses t = nextafter(s_min, -inf), "all different" t = s_max.
 *   - ROC vertices: for every distinct score v (descending), "same" iff score >= v. The
 *     polyline through (FAR, TAR) vertices starts at (0, 0) and ends at (1, 1); a group of
 *     tied scores therefore contributes one diagonal segment (what a randomised threshold
 *     inside the tie achieves).
 *   - AUC: exact Mann-Whitney statistic P(s_same > s_diff) + 0.5 P(s_same == s_diff); equals
 *     the trapezoidal area under that polyline.
 *   - EER: the point on the polyline where FAR = 1 - TAR, by linear interpolation along the
 *     crossing segment.
 *   - TAR@FAR=f: the polyline evaluated at FAR = f by linear interpolation between the two
 *     vertices bracketing f; if a vertex lies exactly at f, the highest TAR at that FAR.
 *
 * LFW 10-fold protocol (View 2): for each fold k, t_k is fitted on the other 9 folds and
 * accuracy measured on fold k. Reported: mean of the 10 accuracies and the standard error
 * S_E = sigma / sqrt(10), sigma the sample std (n - 1 denominator), as in the LFW report. */
#ifndef PLX_VERIF_H
#define PLX_VERIF_H

#include "core/core.h"
#include "data/lfw.h"

#define PLX_VERIF_MAX_FOLDS 16
#define PLX_VERIF_NFAR 3
extern const double plx_verif_far_points[PLX_VERIF_NFAR]; /* 0.1, 0.01, 0.001 */

typedef struct {
    int    n_pairs, n_same, n_diff, n_nan; /* threshold-free metrics are over these pairs */
    bool   split;    /* true: one threshold fitted on a separate train set (plx_verif_split) */
    int    n_train;  /* split only: pairs the threshold was fitted on */
    int    n_folds;  /* cross-validation folds (0 when the pairs have a single fold) */
    double fold_acc[PLX_VERIF_MAX_FOLDS];       /* held-out accuracy per fold */
    double fold_thr[PLX_VERIF_MAX_FOLDS];       /* threshold fitted for that fold */
    double fold_train_acc[PLX_VERIF_MAX_FOLDS]; /* accuracy of that threshold on its training set */
    double mean_acc, sem;                       /* NaN when undefined */
    double auc, eer;
    double tar_at_far[PLX_VERIF_NFAR];
} plx_verif_report;

/* Threshold fitting on raw arrays (same[i] is 0/1). Returns t; *acc gets the training accuracy. */
double plx_verif_best_threshold(const double *score, const int *same, int n, double *acc);
double plx_verif_accuracy(const double *score, const int *same, int n, double thr);

/* ROC polyline (see rules above). thr[0] = +inf at (0, 0); thr[i] = the score v of vertex i. */
typedef struct { int n; double *thr, *far, *tar; } plx_roc;
void   plx_roc_compute(const double *score, const int *same, int n, plx_roc *roc); /* needs both classes */
double plx_roc_tar_at_far(const plx_roc *roc, double far);
double plx_roc_eer(const plx_roc *roc);
void   plx_roc_free(plx_roc *roc);
double plx_verif_auc(const double *score, const int *same, int n);

/* Full evaluation over pairs. 10-fold accuracy when the pairs carry >= 2 folds, plus
 * threshold-free metrics over all pairs. PLX_EINVAL (logged) if a class or a fold is empty. */
int plx_verif_evaluate(const plx_pair *p, int n, const double *score, plx_verif_report *r);
/* View 1 / dev protocol: threshold fitted on (train), applied to (test). The report holds
 * fold_acc[0] = mean_acc = test accuracy, fold_thr[0], fold_train_acc[0]; threshold-free
 * metrics are over the test pairs. */
int plx_verif_split(const plx_pair *train, int ntrain, const double *train_score,
                    const plx_pair *test, int ntest, const double *test_score, plx_verif_report *r);

/* Stable, compact, paste-able text block. */
void plx_verif_print(FILE *f, const char *label, const plx_verif_report *r);

/* Scores + labels for later analysis: CSV "a,b,same,fold,score" if path ends in ".csv",
 * otherwise a .plxt F64 tensor [n, 5] with the same columns. Raw scores (NaN kept). */
int plx_verif_write_scores(const char *path, const plx_pair *p, int n, const double *score);
/* ROC polyline as CSV "threshold,far,tar" (NaN scores mapped to -inf as everywhere). */
int plx_verif_write_roc(const char *path, const plx_pair *p, int n, const double *score);

#endif
