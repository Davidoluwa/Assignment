/* Labeled Faces in the Wild: dataset index and the verification protocol pair lists.
 *
 * Index: the dataset root holds one directory per identity, <Name>/<Name>_<NNNN>.jpg.
 * Every image gets a deterministic global index 0..n_images-1: identities sorted by
 * byte-wise name (strcmp order, so "AJ_Cook" < "Aaron_Eckhart"), images within an identity
 * sorted by number. Images of identity `id` are exactly [id_first[id], id_first[id+1]).
 * Files that do not match the canonical name pattern are ignored (and counted in the log),
 * so the index never contains a path that plx_lfw_relpath would spell differently.
 *
 * Pairs: both official formats are parsed into records with resolved image indices.
 *   pairs.txt       first line "F\tP": F folds, each P matched then P mismatched lines.
 *   pairsDev*.txt   first line "N": N matched then N mismatched lines (all fold 0).
 *   matched line    "name\tn1\tn2"          mismatched line "name1\tn1\tname2\tn2" */
#ifndef PLX_LFW_H
#define PLX_LFW_H

#include "core/core.h"

#define PLX_LFW_PATH_MAX 4096

typedef struct {
    char  *root;      /* absolute, canonical dataset root (no trailing slash) */
    int    n_ids, n_images;
    char **id_name;   /* [n_ids] */
    int   *id_first;  /* [n_ids + 1] first image index of each identity; id_first[n_ids] = n_images */
    int   *img_id;    /* [n_images] identity of each image */
    int   *img_num;   /* [n_images] the NNNN in the file name (1-based in LFW) */
} plx_lfw;

int  plx_lfw_open(plx_lfw *db, const char *root); /* scans the tree; PLX_EIO if unreadable/empty */
void plx_lfw_free(plx_lfw *db);

static inline int plx_lfw_id_count(const plx_lfw *db, int id) { return db->id_first[id + 1] - db->id_first[id]; }
int plx_lfw_find_id(const plx_lfw *db, const char *name);        /* -1 if absent */
int plx_lfw_find(const plx_lfw *db, const char *name, int num);  /* image index, -1 if absent */
/* "<Name>/<Name>_<NNNN>.jpg" / "<root>/<Name>/<Name>_<NNNN>.jpg". Return the string length;
 * the buffer must be large enough (PLX_LFW_PATH_MAX always is). */
int plx_lfw_relpath(const plx_lfw *db, int img, char *buf, size_t cap);
int plx_lfw_path(const plx_lfw *db, int img, char *buf, size_t cap);
/* CSV "index,id,name,num,relpath" with one row per image, for analysis and oracle checks. */
int plx_lfw_write_csv(const plx_lfw *db, const char *path);

/* ---- protocol pairs ------------------------------------------------------------------ */
typedef struct { int a, b; int same; int fold; } plx_pair;
typedef struct {
    plx_pair *p;
    int n;
    int n_folds; /* 10 for pairs.txt, 1 for the dev lists */
} plx_pairs;

/* Recoverable: PLX_EIO (unreadable), PLX_EFORMAT (malformed, unknown name or image number;
 * the log names file, line and offending token). On failure *out is empty. */
int  plx_pairs_load(plx_pairs *out, const char *path, const plx_lfw *db);
void plx_pairs_free(plx_pairs *pl);

/* Fold hygiene. mask has db->n_images entries; set to 1/0.
 *   fold_identity_mask: images whose IDENTITY occurs in any pair of fold k. Anything learnt
 *     for scoring fold k (PCA bases, metric, 3D model) must be trained with these excluded.
 *   image_mask: images that occur themselves in any pair of the list. */
void plx_pairs_fold_identity_mask(const plx_pairs *pl, const plx_lfw *db, int fold, uint8_t *mask);
void plx_pairs_image_mask(const plx_pairs *pl, const plx_lfw *db, uint8_t *mask);

#endif
