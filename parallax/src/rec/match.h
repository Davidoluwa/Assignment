/* Recognition back half: descriptors -> similarity score.
 *
 * Two paradigms, kept apart by the API itself:
 *   unsupervised   plx_match_fit sees unlabelled training images and nothing else:
 *                  Hellinger-mapped histograms, PCA + whitening fitted on them, cosine (or a
 *                  fixed raw distance that fits nothing);
 *   restricted     plx_match_fit_pairs adds a supervised stage learnt from same/different labels
 *                  of training PAIRS (never identity names), on top of the unsupervised embedding.
 * Both stages receive copies of training data only (see proto.h), and scoring takes unlabelled
 * index pairs, so a model has no way to reach a test descriptor or a test label. */
#ifndef PLX_MATCH_H
#define PLX_MATCH_H

#include "la/la.h"
#include "rec/rec.h"

/* n descriptors, row-major n x d, with n x n_cells per-cell weights (NULL: all 1). */
typedef struct {
    int n, d, n_cells;
    const float *x;
    const float *cw;
} plx_feats;

typedef enum {
    PLX_SCORE_EMBED = 0, /* learned embedding, cosine (or the supervised score below) */
    PLX_SCORE_CHI2,      /* -sum (a - b)^2 / (a + b) on raw descriptors, nothing fitted */
    PLX_SCORE_COS,       /* cosine on raw descriptors, nothing fitted */
    PLX_SCORE_CELL       /* confidence-weighted mean of per-cell dot products (with Hellinger
                            cells: Bhattacharyya coefficients), weight sqrt(w_a w_b) per cell */
} plx_score_kind;

typedef enum {
    PLX_SUP_NONE = 0,
    PLX_SUP_LDA,   /* pair-LDA: directions maximising total / within-pair variance, then cosine */
    PLX_SUP_JB,    /* joint Bayesian likelihood ratio, covariances from pairs (diagonal in the
                      pair-LDA basis) */
    PLX_SUP_LOGIT  /* L2-regularised logistic regression on per-dimension pair features
                      (u_i v_i, (u_i - v_i)^2) in the pair-LDA basis */
} plx_sup_method;

typedef struct {
    plx_score_kind score;
    /* unsupervised stage */
    int    pca_dim;       /* components per PCA group; 0 = no PCA (cosine on imputed raw) */
    bool   per_block;     /* one PCA per descriptor block, each block then gets one vote */
    bool   whiten;
    int    pca_max_train; /* PCA fitted on at most this many training images (deterministic subsample) */
    int    pca_iters;     /* power iterations of the randomised PCA */
    /* supervised stage */
    plx_sup_method sup;
    int    sup_pca;       /* label-free PCA of the embedding before the pair statistics */
    int    sup_dim;       /* pair-LDA dimensions kept */
    double sup_ridge;     /* within-pair covariance += ridge * mean eigenvalue * I */
    double logit_l2;      /* logistic regression penalty on the mean log-loss */
    uint64_t seed;
} plx_match_cfg;
plx_match_cfg plx_match_default(void); /* the best View-1 configuration, unsupervised (sup = NONE) */
/* unsupervised part only, or both when sup != NONE */
void plx_match_describe(const plx_match_cfg *c, bool supervised, char *buf, size_t cap);

typedef struct {
    plx_match_cfg   cfg;
    plx_desc_layout lay;
    /* unsupervised */
    double *mean;               /* d: training mean descriptor, target of confidence imputation */
    int     n_groups;
    int     g_off[PLX_DESC_MAX_BLOCKS], g_len[PLX_DESC_MAX_BLOCKS]; /* descriptor dims of each group */
    plx_pca pca[PLX_DESC_MAX_BLOCKS];
    int     k;                  /* unsupervised embedding dimension */
    /* label-free reduction feeding the supervised stage (only when cfg.sup != NONE) */
    double *mu;                 /* k: mean embedding of the PCA subsample */
    plx_mat T0;                 /* r x k: whitened top-r principal directions of the embedding */
    /* supervised (has_sup) */
    bool    has_sup;
    int     m;                  /* pair-LDA dims */
    plx_mat P;                  /* m x k: z = P (y - mu) */
    double *lam;                /* m: between / within variance ratio of each direction */
    double *w;                  /* JB: 2m quadratic coefficients; LOGIT: 2m + 1 weights */
    double *f_mean, *f_scale;   /* LOGIT: feature standardisation, 2m each */
} plx_match_model;

/* Unsupervised stage from training images only. PLX_OK or a numerical status from la (logged). */
int  plx_match_fit(const plx_match_cfg *cfg, const plx_desc_layout *lay, const plx_feats *train,
                   plx_match_model *m);
/* Supervised stage (replacing any earlier one) from labelled pairs (pa[i], pb[i], same[i]) over
 * the rows of y: n x m->k unsupervised embeddings (plx_match_embed with supervised = false) of
 * training images. Requires cfg.sup != NONE and cfg.score == EMBED. */
int  plx_match_fit_pairs(plx_match_model *m, const float *y, int n, int n_pairs, const int *pa,
                         const int *pb, const int *same);
void plx_match_free(plx_match_model *m);

/* Embedding of n descriptor rows (x: n x d, cw: n x n_cells or NULL) for EMBED models; y gets
 * n x plx_match_embed_dim floats: the unsupervised embedding, or the pair-LDA coordinates. */
int  plx_match_embed_dim(const plx_match_model *m, bool supervised);
void plx_match_embed(const plx_match_model *m, const float *x, const float *cw, int n, bool supervised,
                     float *y);

/* Similarity of n unlabelled pairs (rows a[i], b[i] of f); higher = more likely the same person.
 * supervised selects the supervised stage (which must have been fitted). */
void plx_match_score(const plx_match_model *m, const plx_feats *f, const int *a, const int *b, int n,
                     bool supervised, double *score);
/* The same on already-embedded rows (y: rows x plx_match_embed_dim(m, supervised)). */
double plx_match_score_embedded(const plx_match_model *m, const float *u, const float *v, bool supervised);

#endif
