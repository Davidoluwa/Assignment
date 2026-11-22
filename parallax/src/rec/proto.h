/* LFW protocol runner: who may be fitted on what, and the evaluation loops.
 *
 * A split says, for one evaluation unit (a View 2 fold, or View 1), which images a model may be
 * fitted on, which labelled pairs it may learn from, and which pairs it is scored on.
 *   View 2, fold k:  fit images = every image whose identity occurs in NO pair of fold k
 *                    (complement of plx_pairs_fold_identity_mask); training pairs = the pairs
 *                    of the other nine folds; test = fold k's pairs.
 *   View 1:          fit images = every image whose identity occurs in no pairsDevTest pair;
 *                    training pairs = pairsDevTrain; test = pairsDevTest.
 * The split constructors verify every split with plx_rec_split_check and abort on a violation.
 *
 * plx_rec_run_split copies ONLY the fit images into a compact matrix and fits the unsupervised
 * stage on it (no labels exist at that point); the restricted paradigm then embeds the
 * training-pair images (each asserted to be a fit image) and fits the supervised stage on those
 * embeddings and the training-pair labels. Test pairs are scored as unlabelled index pairs. The
 * model never holds a reference to a test descriptor or a test label, so leakage is
 * structurally impossible rather than avoided by care. */
#ifndef PLX_PROTO_H
#define PLX_PROTO_H

#include "data/lfw.h"
#include "eval/verif.h"
#include "rec/match.h"

typedef struct {
    int        n_images;
    const int *img_id;          /* [n_images] identity of each image (borrowed from the index) */
    uint8_t   *fit_img;         /* [n_images]: 1 = may be used for fitting */
    plx_pair  *train;           /* labelled training pairs (global image indices) */
    int        n_train;
    plx_pair  *test;
    int        n_test;
} plx_rec_split;

void plx_rec_split_view2(const plx_pairs *pl, const plx_lfw *db, int fold, plx_rec_split *s);
void plx_rec_split_view1(const plx_pairs *dev_train, const plx_pairs *dev_test, const plx_lfw *db,
                         plx_rec_split *s);
void plx_rec_split_free(plx_rec_split *s);
/* Number of violations (0 = clean): a fit image sharing an identity with a test pair, a
 * training-pair image that is not a fit image, or a pair occurring in both lists. */
int  plx_rec_split_check(const plx_rec_split *s, const plx_lfw *db);

/* Outputs of one split. test_unsup is required; every other array is optional (NULL skips it).
 * test_sup needs cfg->sup != NONE. train_* are scores of the split's training pairs, for
 * fitting a decision threshold: unsupervised ones directly (the model saw no labels), restricted
 * ones out-of-fold (5-fold over identity-disjoint groups of the training pairs: a supervised
 * model scores the pairs it was fitted on far too confidently). When train_sup is requested,
 * test_sup is the mean of those five out-of-fold models' scores, so threshold and test scores
 * come from models fitted on the same number of pairs. */
typedef struct {
    double *test_unsup, *test_sup;   /* [n_test] */
    double *train_unsup, *train_sup; /* [n_train] */
} plx_rec_scores;

#define PLX_REC_OOF_FOLDS 5

int plx_rec_run_split(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *all,
                      const plx_rec_split *s, const plx_rec_scores *out);

/* View 2: ten splits; score_*[pl->n] get every pair's held-out score (score_sup / r_sup NULL:
 * unsupervised only), each paradigm then evaluated with plx_verif_evaluate. */
int plx_rec_eval_view2(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *all,
                       const plx_pairs *pl, const plx_lfw *db, double *score_unsup, double *score_sup,
                       plx_verif_report *r_unsup, plx_verif_report *r_sup);
/* View 1: fit on the dev-train side, threshold on dev-train pair scores, test on dev-test
 * (plx_verif_split). score_*[dev_test->n]. */
int plx_rec_eval_view1(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *all,
                       const plx_pairs *dev_train, const plx_pairs *dev_test, const plx_lfw *db,
                       double *score_unsup, double *score_sup, plx_verif_report *r_unsup,
                       plx_verif_report *r_sup);

#endif
