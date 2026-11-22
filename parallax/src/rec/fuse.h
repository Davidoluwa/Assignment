/* Score-level fusion of several similarity sources (rung L4: texture + geometry codes).
 *
 * A source scores pairs. It is either
 *   - a matcher (match.h) fitted per split on the split's fit images, contributing its
 *     unsupervised score and/or its supervised score as columns, or
 *   - a fixed pair score that fits nothing (e.g. a depth-code likelihood ratio from the geometry
 *     model, an aspect-ratio difference, a pose term).
 * For one split, plx_fuse_columns scores the training pairs and the test pairs with every column.
 * Training-pair scores are what a fusion is fitted on, so they must not be optimistic:
 * unsupervised and fixed columns are scored directly (they saw no labels), supervised columns
 * out-of-fold (proto.h), and their test scores come from the same out-of-fold models.
 *
 * Fusions over those columns:
 *   learned (image-restricted)  L2-regularised logistic regression on the standardised columns,
 *                               fitted on the split's training pairs only;
 *   fixed weights               s = sum_c w_c (x_c - m_c) / sd_c with m, sd the mean and standard
 *                               deviation of column c over the training pairs (label-free) and w
 *                               chosen by the caller (on View 1, then frozen).
 * A NaN score (a source with nothing to say about that pair) is replaced by the training mean of
 * its column, i.e. contributes nothing. */
#ifndef PLX_FUSE_H
#define PLX_FUSE_H

#include "rec/proto.h"

/* ---- logistic regression (shared with match.c) ----------------------------------------------
 * Phi: n x F design matrix whose LAST column is the constant 1 (bias). Minimises
 * mean log-loss + 0.5 l2 |w[0..F-2]|^2 by Newton's method (IRLS); w (F) gets the weights.
 * Returns PLX_OK or PLX_ENOTSPD. */
int plx_logreg_fit(const plx_mat *Phi, const int *y, double l2, double *w);

/* ---- sources ------------------------------------------------------------------------------- */
typedef double (*plx_pair_score_fn)(int a, int b, void *ctx);

typedef struct {
    const char *name;
    /* matcher source (fn == NULL) */
    const plx_match_cfg   *cfg;  /* cfg->sup selects the supervised method (or NONE) */
    const plx_desc_layout *lay;
    const plx_feats       *feats;
    bool unsup, sup;             /* which of its scores become columns */
    /* fixed source */
    plx_pair_score_fn fn;
    void *ctx;
} plx_score_source;

#define PLX_FUSE_MAX_COLS 16
typedef struct {
    int  n_cols;
    char name[PLX_FUSE_MAX_COLS][48];
    double *train[PLX_FUSE_MAX_COLS]; /* [n_train] */
    double *test[PLX_FUSE_MAX_COLS];  /* [n_test] */
} plx_fuse_cols;

int  plx_fuse_columns(const plx_score_source *src, int n_src, const plx_rec_split *s, plx_fuse_cols *out);
void plx_fuse_cols_free(plx_fuse_cols *c);

/* Column standardisation over the training pairs: m, sd (n_cols each). */
void plx_fuse_standardise(const plx_fuse_cols *c, int n_train, double *m, double *sd);

/* Learned fusion on the columns in `use` (indices, n_use of them): fits on the training pairs and
 * writes fused scores for the training and test pairs. w_out (nullable, n_use + 1) receives the
 * weights on the STANDARDISED columns, bias last, so they compare across columns. */
int plx_fuse_learned(const plx_fuse_cols *c, const plx_pair *train, int n_train, int n_test, const int *use,
                     int n_use, double l2, double *train_score, double *test_score, double *w_out);
/* Fixed-weight fusion: weights w[n_use] on the standardised columns `use`. */
void plx_fuse_fixed(const plx_fuse_cols *c, int n_train, int n_test, const int *use, int n_use, const double *w,
                    double *train_score, double *test_score);

#endif
