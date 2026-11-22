// plx-deps: render viz img core
#include "test.h"

#include "core/core.h"
#include "img/image.h"
#include "render/render.h"
#include "viz/viz.h"

#include <stdlib.h>
#include <string.h>

static const float WHITE[3] = {1, 1, 1};
static const double DEG = M_PI / 180.0;

static double srgb_lin(double c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
static double luminance(const float rgb[3]) {
    return 0.2126 * srgb_lin(rgb[0]) + 0.7152 * srgb_lin(rgb[1]) + 0.0722 * srgb_lin(rgb[2]);
}

static double ink(const plx_img *im) { /* sum of channel 0: coverage when drawing white on black */
    double s = 0;
    for (int i = 0; i < im->w * im->h; i++) s += im->px[i];
    return s;
}

/* ---- colour maps ------------------------------------------------------------------------- */

TEST(colormaps_ordered_and_centred) {
    float rgb[3], prev = -1;
    bool mono = true, in_range = true;
    for (int i = 0; i <= 255; i++) {
        plx_cmap_eval(PLX_CMAP_SEQ, i / 255.0f, rgb);
        float y = (float)luminance(rgb);
        mono &= y > prev;
        prev = y;
        for (int k = 0; k < 3; k++) in_range &= rgb[k] >= 0 && rgb[k] <= 1;
    }
    CHECK(mono && in_range);
    plx_cmap_eval(PLX_CMAP_SEQ, 0, rgb);
    CHECK(luminance(rgb) < 0.03 && rgb[2] > rgb[1]); /* dark, violet-blue */
    plx_cmap_eval(PLX_CMAP_SEQ, 1, rgb);
    CHECK(luminance(rgb) > 0.8 && rgb[0] > rgb[2] && rgb[1] > rgb[2]); /* pale yellow */
    plx_cmap_eval(PLX_CMAP_SEQ, -3, rgb); /* clamped */
    float lo0[3];
    plx_cmap_eval(PLX_CMAP_SEQ, 0, lo0);
    CHECK(rgb[0] == lo0[0] && rgb[1] == lo0[1] && rgb[2] == lo0[2]);

    /* Diverging: luminance rises to the centre and falls after, ends dark blue / dark red. */
    bool up = true, down = true;
    prev = -1;
    for (int i = 0; i <= 128; i++) { plx_cmap_eval(PLX_CMAP_DIV, i / 256.0f, rgb); up &= luminance(rgb) > prev; prev = (float)luminance(rgb); }
    for (int i = 129; i <= 256; i++) { plx_cmap_eval(PLX_CMAP_DIV, i / 256.0f, rgb); down &= luminance(rgb) < prev; prev = (float)luminance(rgb); }
    CHECK(up && down);
    plx_cmap_eval(PLX_CMAP_DIV, 0.5f, rgb);
    CHECK(luminance(rgb) > 0.85 && fabsf(rgb[0] - rgb[1]) < 0.02f && fabsf(rgb[1] - rgb[2]) < 0.02f);
    plx_cmap_eval(PLX_CMAP_DIV, 0, rgb);
    CHECK(rgb[2] > rgb[0] + 0.2f);
    plx_cmap_eval(PLX_CMAP_DIV, 1, rgb);
    CHECK(rgb[0] > rgb[2] + 0.2f);

    /* Image path: zero maps to the neutral centre even for an asymmetric range; NaN -> black;
     * out-of-range values clamp to the end colours. */
    float v[5] = {0, NAN, -1, 3, 100};
    plx_img im = plx_viz_colormap(v, 5, 1, -1, 3, PLX_CMAP_DIV);
    float mid[3], end[3];
    plx_cmap_eval(PLX_CMAP_DIV, 0.5f, mid);
    plx_cmap_eval(PLX_CMAP_DIV, 1.0f, end);
    for (int k = 0; k < 3; k++) {
        CHECK_NEAR(PLX_PX(&im, 0, 0, k), mid[k], 2e-3);
        CHECK(PLX_PX(&im, 1, 0, k) == 0);
        CHECK_NEAR(PLX_PX(&im, 3, 0, k), end[k], 1e-6);
        CHECK_NEAR(PLX_PX(&im, 4, 0, k), end[k], 1e-6);
    }
    CHECK(PLX_PX(&im, 2, 0, 2) > PLX_PX(&im, 2, 0, 0)); /* -1 is on the blue side */
    plx_img_free(&im);

    float data[104], lo, hi;
    for (int i = 0; i < 100; i++) data[i] = (float)(99 - i);
    data[100] = NAN; data[101] = INFINITY; data[102] = -INFINITY; data[103] = 1e30f;
    plx_viz_auto_range(data, 103, 0.01f, 0.99f, &lo, &hi); /* 1e30 excluded by n */
    CHECK(lo == 1 && hi == 98);
    plx_viz_auto_range(data + 100, 3, 0.01f, 0.99f, &lo, &hi);
    CHECK(lo == 0 && hi == 1);
    float flat[4] = {2, 2, 2, 2};
    plx_viz_auto_range(flat, 4, 0, 1, &lo, &hi);
    CHECK(lo < 2 && hi > 2);
}

/* ---- text ---------------------------------------------------------------------------------- */

TEST(text_glyph_pixels) {
    /* 'A' drawn independently here; it must match the rendered pixels exactly. */
    static const char *A[8] = {".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#", "....."};
    for (int scale = 1; scale <= 3; scale++) {
        plx_img im = plx_img_new(2 + 6 * scale, 2 + 9 * scale, 3);
        plx_draw_text(&im, 1, 1, "A", scale, WHITE);
        int wrong = 0, set = 0;
        for (int y = 0; y < im.h; y++)
            for (int x = 0; x < im.w; x++) {
                int gx = (x - 1) / scale, gy = (y - 1) / scale;
                bool want = x >= 1 && y >= 1 && gx < 5 && gy < 8 && A[gy][gx] == '#';
                bool got = PLX_PX(&im, x, y, 0) == 1 && PLX_PX(&im, x, y, 2) == 1;
                wrong += want != got;
                set += got;
            }
        CHECK(wrong == 0 && set == 18 * scale * scale);
        plx_img_free(&im);
    }
    int w, h;
    plx_text_size("AB\nC", 1, &w, &h);
    CHECK(w == 11 && h == 17);
    plx_text_size("", 2, &w, &h);
    CHECK(w == 0 && h == 16);
    /* Every printable glyph has ink except space; unknown bytes draw '?'. */
    int empty = 0;
    for (int c = 33; c < 127; c++) {
        int n = 0;
        for (int r = 0; r < PLX_FONT_H; r++)
            for (int q = 0; q < PLX_FONT_W; q++) n += plx_font_pixel((char)c, q, r);
        empty += n == 0;
    }
    CHECK(empty == 0);
    for (int r = 0; r < PLX_FONT_H; r++)
        for (int q = 0; q < PLX_FONT_W; q++) CHECK(plx_font_pixel('\x01', q, r) == plx_font_pixel('?', q, r));
    /* Clipping: text partly outside the image must not write out of bounds (ASan checks). */
    plx_img im = plx_img_new(7, 5, 3);
    plx_draw_text(&im, -3, -4, "Wg|\nxyz", 2, WHITE);
    plx_img_free(&im);
}

/* ---- drawing ------------------------------------------------------------------------------- */

TEST(drawing_coverage) {
    plx_img im = plx_img_new(40, 30, 3);
    plx_draw_rect(&im, 1.5f, 1.5f, 5.5f, 4.5f, WHITE, 1); /* exactly pixels 2..5 x 2..4 */
    CHECK_NEAR(ink(&im), 12, 1e-5);
    CHECK(PLX_PX(&im, 2, 2, 0) == 1 && PLX_PX(&im, 5, 4, 1) == 1 && PLX_PX(&im, 6, 4, 0) == 0);
    plx_img_free(&im);

    im = plx_img_new(40, 30, 3);
    plx_draw_rect(&im, 10.2f, 3.7f, 17.9f, 9.1f, WHITE, 1); /* area-exact for any rectangle */
    CHECK_NEAR(ink(&im), 7.7 * 5.4, 1e-4);
    plx_img_free(&im);

    im = plx_img_new(40, 30, 3);
    plx_draw_line(&im, 2, 5, 12, 5, 1, WHITE, 1); /* one full row of 11 pixels, nothing else */
    CHECK_NEAR(ink(&im), 11, 1e-5);
    CHECK(PLX_PX(&im, 2, 5, 0) == 1 && PLX_PX(&im, 12, 5, 0) == 1 && PLX_PX(&im, 7, 4, 0) == 0);
    plx_img_free(&im);

    im = plx_img_new(40, 40, 3);
    plx_draw_line(&im, 3.3f, 4.1f, 33.7f, 29.2f, 2, WHITE, 1); /* ink ~ length * width */
    double len = hypot(30.4, 25.1);
    CHECK(fabs(ink(&im) - 2 * len) < 0.1 * 2 * len);
    plx_img_free(&im);

    im = plx_img_new(60, 60, 3);
    plx_draw_disc(&im, 30.3f, 29.6f, 12, WHITE, 1);
    CHECK(fabs(ink(&im) - M_PI * 144) < 0.01 * M_PI * 144);
    plx_img_free(&im);

    im = plx_img_new(60, 60, 3);
    plx_draw_circle(&im, 30, 30, 15, 2, WHITE, 1);
    CHECK(fabs(ink(&im) - 2 * M_PI * 15 * 2) < 0.03 * 2 * M_PI * 15 * 2);
    plx_img_free(&im);

    im = plx_img_new(20, 20, 3);
    plx_draw_cross(&im, 10, 10, 4, 1, WHITE, 1); /* 9 + 4 + 4 pixels, centre counted once */
    CHECK_NEAR(ink(&im), 17, 1e-5);
    plx_draw_rect_outline(&im, 0, 0, 19, 19, 1, WHITE, 0.5f); /* edge pixels half-covered */
    plx_draw_line(&im, -50, -50, 100, 70, 3, WHITE, 1);        /* clipping */
    plx_draw_disc(&im, NAN, 3, 4, WHITE, 1);                    /* rejected, no crash */
    plx_img_free(&im);

    im = plx_img_new(12, 12, 3);
    plx_draw_rect_outline(&im, 2, 2, 9, 9, 1, WHITE, 1); /* ring of pixel width: 4 * 7 (half-covered rims) */
    CHECK_NEAR(ink(&im), 4 * 8 * 1.0, 1e-4);
    plx_img_free(&im);
}

/* ---- montage, upscale, flow ---------------------------------------------------------------- */

TEST(montage_layout) {
    plx_img t[2] = {plx_img_new(4, 3, 1), plx_img_new(5, 2, 3)};
    for (int i = 0; i < 12; i++) t[0].px[i] = 0.5f;
    for (int i = 0; i < 10; i++) t[1].px[i] = 1.0f; /* red */
    plx_montage_opts o = plx_montage_defaults();
    o.cols = 2;
    o.pad = 1;
    plx_img m = plx_viz_montage(t, 2, NULL, &o);
    CHECK(m.w == 2 * 5 + 3 && m.h == 3 + 2 && m.c == 3);
    CHECK(PLX_PX(&m, 0, 0, 0) == o.bg[0] && PLX_PX(&m, 5, 1, 1) == o.bg[1]); /* pad, then cell slack */
    CHECK(PLX_PX(&m, 1, 1, 0) == 0.5f && PLX_PX(&m, 1, 1, 2) == 0.5f && PLX_PX(&m, 4, 3, 1) == 0.5f);
    CHECK(PLX_PX(&m, 7, 1, 0) == 1 && PLX_PX(&m, 7, 1, 1) == 0 && PLX_PX(&m, 11, 2, 0) == 1);
    CHECK(PLX_PX(&m, 7, 3, 0) == o.bg[0]); /* below the shorter tile */
    plx_img_free(&m);

    const char *labels[2] = {"a", NULL};
    o.cols = 1;
    m = plx_viz_montage(t, 2, labels, &o);
    CHECK(m.w == 5 + 2 && m.h == 2 * (3 + 10) + 3);
    CHECK(PLX_PX(&m, 1 + 1, 1 + 2, 0) == 1); /* 'a' row 2, column 1 is ink */
    plx_img_free(&m);
    plx_img_free(&t[0]);
    plx_img_free(&t[1]);
}

TEST(upscale_modes) {
    plx_img s = plx_img_new(2, 2, 1);
    s.px[0] = 0; s.px[1] = 1; s.px[2] = 2; s.px[3] = 3;
    plx_img n = plx_viz_upscale(&s, 3, PLX_UP_NEAREST), b = plx_viz_upscale(&s, 2, PLX_UP_BILINEAR);
    CHECK(n.w == 6 && n.h == 6 && PLX_PX(&n, 2, 2, 0) == 0 && PLX_PX(&n, 3, 0, 0) == 1 && PLX_PX(&n, 5, 5, 0) == 3);
    CHECK(PLX_PX(&b, 0, 0, 0) == 0);                     /* clamped at the border */
    CHECK_NEAR(PLX_PX(&b, 1, 0, 0), 0.25, 1e-6);         /* source x = 0.25 */
    CHECK_NEAR(PLX_PX(&b, 1, 1, 0), 0.25 + 0.5, 1e-6);   /* (0.25, 0.25) */
    CHECK(PLX_PX(&b, 3, 3, 0) == 3);
    s.px[1] = NAN; /* a NaN neighbour must not leak into a pixel sitting on a valid sample */
    plx_img_free(&b);
    b = plx_viz_upscale(&s, 2, PLX_UP_BILINEAR);
    CHECK(PLX_PX(&b, 0, 0, 0) == 0 && PLX_PX(&b, 0, 3, 0) == 2 && isnan(PLX_PX(&b, 1, 0, 0)));
    plx_img_free(&s); plx_img_free(&n); plx_img_free(&b);
}

TEST(flow_visualisation) {
    float dx[5] = {2, 0, 0, NAN, -1}, dy[5] = {0, 2, 0, 0, 0};
    plx_img c = plx_viz_flow_color(dx, dy, 5, 1, 2);
    CHECK(PLX_PX(&c, 0, 0, 0) == 1 && PLX_PX(&c, 0, 0, 1) == 0 && PLX_PX(&c, 0, 0, 2) == 0); /* right: red */
    CHECK_NEAR(PLX_PX(&c, 1, 0, 0), 0.5, 1e-5);                                          /* down: 90 deg */
    CHECK(PLX_PX(&c, 1, 0, 1) == 1 && PLX_PX(&c, 1, 0, 2) == 0);
    CHECK(PLX_PX(&c, 2, 0, 0) == 0 && PLX_PX(&c, 2, 0, 1) == 0 && PLX_PX(&c, 2, 0, 2) == 0); /* zero: black */
    CHECK(PLX_PX(&c, 3, 0, 1) == 0.5f);                                                  /* NaN: grey */
    CHECK(PLX_PX(&c, 4, 0, 0) == 0 && PLX_PX(&c, 4, 0, 1) == 0.5f && PLX_PX(&c, 4, 0, 2) == 0.5f); /* left, half */
    plx_img_free(&c);

    plx_img im = plx_img_new(30, 30, 3);
    float ax[1] = {5}, ay[1] = {0};
    plx_viz_flow_arrows(&im, ax, ay, 1, 1, 10, 10, 8, 2, WHITE); /* (10,10) -> (20,10) */
    CHECK(PLX_PX(&im, 15, 10, 0) == 1 && PLX_PX(&im, 20, 10, 0) == 1 && PLX_PX(&im, 22, 10, 0) == 0);
    CHECK(PLX_PX(&im, 18, 9, 0) > 0.2f && PLX_PX(&im, 18, 11, 0) > 0.2f); /* barbs behind the tip */
    CHECK(PLX_PX(&im, 10, 15, 0) == 0);
    plx_img_free(&im);
}

/* ---- rasteriser ---------------------------------------------------------------------------- */

static plx_mesh quad(float x0, float y0, float x1, float y1, float z, float a) {
    plx_mesh m = plx_mesh_new(4, 2, 1);
    const float v[12] = {x0, y0, z, x1, y0, z, x0, y1, z, x1, y1, z};
    memcpy(m.v, v, sizeof v);
    const int t[6] = {0, 1, 2, 1, 3, 2};
    memcpy(m.tri, t, sizeof t);
    for (int i = 0; i < 4; i++) m.attr[i] = a;
    return m;
}

static int count_mask(const plx_raster *r) {
    int n = 0;
    for (int i = 0; i < r->w * r->h; i++) n += r->mask[i];
    return n;
}

TEST(raster_flat_square_exact_count) {
    plx_camera id = plx_camera_make(0, 0, 0, 1, NULL, 0, 0);
    plx_raster_opts o = {.w = 20, .h = 12, .cull = PLX_CULL_BACK};
    plx_raster r;
    plx_mesh m = quad(2.5f, 2.5f, 12.5f, 7.5f, 1, 1); /* centres x 3..12, y 3..7 */
    plx_render(&m, &id, &o, &r);
    CHECK(count_mask(&r) == 50 && r.mask[3 * 20 + 3] && r.mask[7 * 20 + 12] && !r.mask[8 * 20 + 12]);
    plx_raster_free(&r);
    plx_mesh_free(&m);
    /* Corners exactly on pixel centres: top-left rule keeps x = 2..11 and y = 2..6. */
    m = quad(2, 2, 12, 7, 1, 1);
    plx_render(&m, &id, &o, &r);
    CHECK(count_mask(&r) == 50 && r.mask[2 * 20 + 2] && !r.mask[2 * 20 + 12] && !r.mask[7 * 20 + 2]);
    CHECK(PLX_PX(&r.depth, 5, 5, 0) == 1 && isnan(PLX_PX(&r.depth, 0, 0, 0)));
    plx_raster_free(&r);
    /* Reversed winding = back face: culled, or drawn with CULL_NONE. */
    int t = m.tri[1]; m.tri[1] = m.tri[2]; m.tri[2] = t;
    t = m.tri[4]; m.tri[4] = m.tri[5]; m.tri[5] = t;
    plx_render(&m, &id, &o, &r);
    CHECK(count_mask(&r) == 0);
    plx_raster_free(&r);
    o.cull = PLX_CULL_NONE;
    plx_render(&m, &id, &o, &r);
    CHECK(count_mask(&r) == 50);
    plx_raster_free(&r);
    plx_mesh_free(&m);
}

/* Each triangle rendered alone; per-pixel hit counts summed. */
static int *hit_counts(const plx_mesh *m, const plx_camera *cam, int w, int h) {
    int *hits = PLX_NEW(int, w * h);
    plx_mesh one = plx_mesh_new(m->nv, 1, 0);
    memcpy(one.v, m->v, sizeof(float) * 3 * m->nv);
    plx_raster_opts o = {.w = w, .h = h, .cull = PLX_CULL_NONE};
    for (int t = 0; t < m->nt; t++) {
        memcpy(one.tri, m->tri + 3 * t, sizeof(int) * 3);
        plx_raster r;
        plx_render(&one, cam, &o, &r);
        for (int i = 0; i < w * h; i++) hits[i] += r.mask[i];
        plx_raster_free(&r);
    }
    plx_mesh_free(&one);
    return hits;
}

TEST(raster_shared_edges_no_cracks_or_double_hits) {
    /* (1) Integer grid, spacing 4: many pixel centres lie exactly on shared edges and on
     * the diagonals. Coverage must be exactly the half-open box [4, 36) x [4, 28). */
    plx_img z = plx_img_new(9, 7, 1);
    plx_mesh m = plx_mesh_heightfield(&z, NULL, NULL);
    for (int i = 0; i < m.nv; i++) { m.v[3 * i] = 4 + 4 * m.v[3 * i]; m.v[3 * i + 1] = 4 + 4 * m.v[3 * i + 1]; }
    plx_camera id = plx_camera_make(0, 0, 0, 1, NULL, 0, 0);
    int *hits = hit_counts(&m, &id, 40, 32), bad = 0;
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 40; x++) bad += hits[y * 40 + x] != (x >= 4 && x < 36 && y >= 4 && y < 28);
    CHECK(bad == 0);
    free(hits);

    /* (2) Jittered interior vertices, then rotated by an arbitrary roll: every pixel centre
     * clearly inside the outer square is hit exactly once, every one outside never. */
    plx_rng g;
    plx_rng_seed(&g, 7);
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 9; x++) {
            float *v = m.v + 3 * (y * 9 + x);
            v[0] = 4 + 4.3f * x;
            v[1] = 4 + 4.3f * y;
            if (x > 0 && x < 8 && y > 0 && y < 6) {
                v[0] += (float)(1.6 * (plx_rng_uniform(&g) - 0.5));
                v[1] += (float)(1.6 * (plx_rng_uniform(&g) - 0.5));
            }
        }
    double c[3] = {21.2, 16.9, 0};
    plx_camera cam = plx_camera_make(0, 0, 0.37, 1, c, 21.2, 16.9);
    hits = hit_counts(&m, &cam, 48, 40);
    int inside_bad = 0, outside_bad = 0, total = 0;
    plx_raster_opts o = {.w = 48, .h = 40};
    plx_raster r;
    plx_render(&m, &cam, &o, &r);
    for (int y = 0; y < 40; y++)
        for (int x = 0; x < 48; x++) {
            /* back to object coordinates: inverse roll about c */
            double dx = x - c[0], dy = y - c[1], cr = cos(0.37), sr = sin(0.37);
            double ox = cr * dx + sr * dy + c[0], oy = -sr * dx + cr * dy + c[1];
            double margin = PLX_MIN(PLX_MIN(ox - 4, 4 + 4.3 * 8 - ox), PLX_MIN(oy - 4, 4 + 4.3 * 6 - oy));
            int hcount = hits[y * 48 + x];
            if (margin > 1e-3) inside_bad += hcount != 1;
            if (margin < -1e-3) outside_bad += hcount != 0;
            total += hcount;
            if (hcount > 1) inside_bad++;
        }
    CHECK(inside_bad == 0 && outside_bad == 0 && total == count_mask(&r));
    plx_raster_free(&r);
    free(hits);
    plx_mesh_free(&m);
    plx_img_free(&z);
}

TEST(raster_z_order) {
    plx_mesh a = quad(2, 2, 14, 14, 1, 10), b = quad(8, 8, 20, 20, 2, 20);
    plx_mesh both = plx_mesh_new(8, 4, 1);
    for (int order = 0; order < 2; order++) {
        const plx_mesh *first = order ? &b : &a, *second = order ? &a : &b;
        memcpy(both.v, first->v, sizeof(float) * 12);
        memcpy(both.v + 12, second->v, sizeof(float) * 12);
        both.attr[0] = both.attr[1] = both.attr[2] = both.attr[3] = first->attr[0];
        both.attr[4] = both.attr[5] = both.attr[6] = both.attr[7] = second->attr[0];
        for (int i = 0; i < 6; i++) { both.tri[i] = first->tri[i]; both.tri[6 + i] = second->tri[i] + 4; }
        plx_camera id = plx_camera_make(0, 0, 0, 1, NULL, 0, 0);
        plx_raster_opts o = {.w = 24, .h = 24, .want_ids = true, .bg = -1};
        plx_raster r;
        plx_render(&both, &id, &o, &r);
        CHECK(PLX_PX(&r.attr, 10, 10, 0) == 20 && PLX_PX(&r.depth, 10, 10, 0) == 2); /* overlap: z=2 wins */
        CHECK(PLX_PX(&r.attr, 4, 4, 0) == 10 && PLX_PX(&r.attr, 18, 18, 0) == 20 && PLX_PX(&r.attr, 22, 3, 0) == -1);
        CHECK(r.tri[10 * 24 + 10] == (order ? 0 : 2) || r.tri[10 * 24 + 10] == (order ? 1 : 3));
        CHECK(r.tri[0] == -1);
        plx_raster_free(&r);
    }
    /* Rotating the pair by yaw = 180 deg puts the z=1 square in front. */
    double c[3] = {12, 12, 1.5};
    plx_camera back = plx_camera_make(M_PI, 0, 0, 1, c, 12, 12);
    plx_raster_opts o = {.w = 24, .h = 24, .cull = PLX_CULL_NONE};
    plx_raster r;
    plx_render(&both, &back, &o, &r);
    CHECK(PLX_PX(&r.attr, 12, 12, 0) == 10 && PLX_PX(&r.depth, 12, 12, 0) == 0.5f);
    plx_raster_free(&r);
    plx_mesh_free(&a); plx_mesh_free(&b); plx_mesh_free(&both);
}

#define HEMI_N 101
#define HEMI_R 40.0

static plx_img hemisphere(void) {
    plx_img z = plx_img_new(HEMI_N, HEMI_N, 1);
    for (int y = 0; y < HEMI_N; y++)
        for (int x = 0; x < HEMI_N; x++) {
            double dx = x - 50, dy = y - 50, q = HEMI_R * HEMI_R - dx * dx - dy * dy;
            PLX_PX(&z, x, y, 0) = q >= 0 ? (float)sqrt(q) : NAN;
        }
    return z;
}

TEST(raster_hemisphere_yaw_shift_and_self_occlusion) {
    plx_img z = hemisphere();
    plx_mesh m = plx_mesh_heightfield(&z, NULL, NULL);
    double c[3] = {50, 50, 0};
    int lo[2], hi[2], hidden_wrong[2] = {0, 0}, visible_wrong[2] = {0, 0};
    plx_raster rr[2];
    for (int s = 0; s < 2; s++) {
        double yaw = (s ? -30 : 30) * DEG;
        plx_camera cam = plx_camera_make(yaw, 0, 0, 1, c, 50, 50);
        plx_raster_opts o = {.w = HEMI_N, .h = HEMI_N, .cull = PLX_CULL_BACK};
        plx_render(&m, &cam, &o, &rr[s]);
        lo[s] = HEMI_N; hi[s] = -1;
        for (int x = 0; x < HEMI_N; x++)
            if (rr[s].mask[50 * HEMI_N + x]) { lo[s] = PLX_MIN(lo[s], x); hi[s] = PLX_MAX(hi[s], x); }
        /* Equator vertices: visible iff their projected depth reaches the z-buffer. */
        for (int x = 10; x <= 90; x++) {
            double p[3];
            plx_camera_project(&cam, m.v + 3 * (50 * HEMI_N + x), p);
            int px = (int)lround(p[0]);
            float d = PLX_PX(&rr[s].depth, PLX_CLAMP(px, 0, HEMI_N - 1), 50, 0);
            bool vis = !isnan(d) && p[2] >= d - 0.75;
            double ox = (x - 50) * (s ? -1 : 1); /* + = the side that turns away */
            if (ox >= 37) hidden_wrong[s] += vis;  /* beyond r sin 60 = 34.6: self-occluded */
            if (ox <= 30) visible_wrong[s] += !vis;
        }
        /* Keeping back faces must not change what is visible: the z-buffer hides them. */
        plx_raster keep;
        o.cull = PLX_CULL_NONE;
        plx_render(&m, &cam, &o, &keep);
        int diff = 0;
        for (int i = 0; i < HEMI_N * HEMI_N; i++) {
            float a = rr[s].depth.px[i], b = keep.depth.px[i];
            diff += isnan(a) != isnan(b) || (!isnan(a) && a != b);
        }
        CHECK(diff == 0);
        plx_raster_free(&keep);
    }
    /* yaw +30: silhouette spans [50 - r cos 30, 50 + r] = [15.4, 90]; shifted towards image-right. */
    CHECK(lo[0] >= 15 && lo[0] <= 17 && hi[0] >= 88 && hi[0] <= 90);
    CHECK(lo[1] >= 10 && lo[1] <= 12 && hi[1] >= 83 && hi[1] <= 85); /* mirror image */
    CHECK(hidden_wrong[0] == 0 && hidden_wrong[1] == 0 && visible_wrong[0] == 0 && visible_wrong[1] == 0);
    /* The two masks are mirror images (up to the triangulation's diagonal direction). */
    int mism = 0, cov = 0;
    for (int y = 0; y < HEMI_N; y++)
        for (int x = 0; x < HEMI_N; x++) {
            mism += rr[0].mask[y * HEMI_N + x] != rr[1].mask[y * HEMI_N + (100 - x)];
            cov += rr[0].mask[y * HEMI_N + x];
        }
    CHECK(mism < cov / 100);
    /* Nose (apex) moves to image-right for positive yaw: the apex x' = r sin 30 = +20. */
    double p[3];
    plx_camera cam = plx_camera_make(30 * DEG, 0, 0, 1, c, 50, 50);
    plx_camera_project(&cam, m.v + 3 * (50 * HEMI_N + 50), p);
    CHECK_NEAR(p[0], 70, 1e-4);
    cam = plx_camera_make(0, 30 * DEG, 0, 1, c, 50, 50); /* positive pitch: apex moves up */
    plx_camera_project(&cam, m.v + 3 * (50 * HEMI_N + 50), p);
    CHECK_NEAR(p[1], 30, 1e-4);
    cam = plx_camera_make(0, 0, 90 * DEG, 1, c, 50, 50); /* positive roll: +x goes to +y (clockwise) */
    plx_camera_project(&cam, m.v + 3 * (50 * HEMI_N + 60), p);
    CHECK_NEAR(p[0], 50, 1e-4);
    CHECK_NEAR(p[1], 60, 1e-4);
    plx_raster_free(&rr[0]);
    plx_raster_free(&rr[1]);
    plx_mesh_free(&m);
    plx_img_free(&z);
}

TEST(raster_linear_attributes_exact) {
    /* Frontal: a tilted plane with a texture ramp; integer vertices, exact interpolation. */
    plx_img z = plx_img_new(30, 20, 1), tex = plx_img_new(30, 20, 1);
    for (int y = 0; y < 20; y++)
        for (int x = 0; x < 30; x++) {
            PLX_PX(&z, x, y, 0) = 0.3f * x - 0.2f * y + 5;
            PLX_PX(&tex, x, y, 0) = 2.0f * x + 3.0f * y + 1;
        }
    plx_mesh m = plx_mesh_heightfield(&z, NULL, &tex);
    plx_camera id = plx_camera_make(0, 0, 0, 1, NULL, 0, 0);
    plx_raster_opts o = {.w = 30, .h = 20, .want_ids = true};
    plx_raster r;
    plx_render(&m, &id, &o, &r);
    double err = 0, berr = 0;
    for (int y = 0; y < 19; y++)
        for (int x = 0; x < 29; x++) {
            err = fmax(err, fabs(PLX_PX(&r.attr, x, y, 0) - (2.0 * x + 3.0 * y + 1)));
            err = fmax(err, fabs(PLX_PX(&r.depth, x, y, 0) - (0.3 * x - 0.2 * y + 5)));
        }
    CHECK(err < 1e-4 && count_mask(&r) == 29 * 19);
    plx_raster_free(&r);
    plx_mesh_free(&m);

    /* Arbitrary pose: attributes = each vertex's own projected (x, y, z). The rendered
     * attributes must reproduce the pixel coordinates and the depth buffer everywhere. */
    plx_img tex3 = plx_img_new(30, 20, 3);
    double c[3] = {14.5, 9.5, 5};
    plx_camera cam = plx_camera_make(0.4, -0.3, 0.2, 1.7, c, 32, 24);
    for (int y = 0; y < 20; y++)
        for (int x = 0; x < 30; x++) {
            float v[3] = {(float)x, (float)y, PLX_PX(&z, x, y, 0)};
            double p[3];
            plx_camera_project(&cam, v, p);
            for (int k = 0; k < 3; k++) PLX_PX(&tex3, x, y, k) = (float)p[k];
        }
    m = plx_mesh_heightfield(&z, NULL, &tex3);
    o = (plx_raster_opts){.w = 64, .h = 48, .want_ids = true, .cull = PLX_CULL_NONE};
    plx_render(&m, &cam, &o, &r);
    err = 0;
    int covered = 0;
    for (int y = 0; y < 48; y++)
        for (int x = 0; x < 64; x++) {
            int i = y * 64 + x;
            if (!r.mask[i]) continue;
            covered++;
            err = fmax(err, fabs(PLX_PX(&r.attr, x, y, 0) - x));
            err = fmax(err, fabs(PLX_PX(&r.attr, x, y, 1) - y));
            err = fmax(err, fabs(PLX_PX(&r.attr, x, y, 2) - PLX_PX(&r.depth, x, y, 0)));
            const int *t = m.tri + 3 * r.tri[i];
            double bs = 0, bx = 0;
            for (int k = 0; k < 3; k++) {
                bs += PLX_PX(&r.bary, x, y, k);
                bx += PLX_PX(&r.bary, x, y, k) * tex3.px[t[k]];
            }
            berr = fmax(berr, fmax(fabs(bs - 1), fabs(bx - x)));
        }
    CHECK(covered > 500 && err < 1e-4 && berr < 1e-4);
    plx_raster_free(&r);
    plx_mesh_free(&m);
    plx_img_free(&z); plx_img_free(&tex); plx_img_free(&tex3);
}

TEST(shading_lambert) {
    plx_img z = plx_img_new(20, 20, 1);
    for (int i = 0; i < 400; i++) z.px[i] = 0.5f * (float)(i % 20); /* z = 0.5 x, normal (-0.5, 0, 1) */
    plx_mesh m = plx_mesh_heightfield(&z, NULL, NULL);
    plx_camera id = plx_camera_make(0, 0, 0, 1, NULL, 0, 0);
    plx_raster_opts o = {.w = 20, .h = 20, .bg = -1};
    plx_shade sh = {.light = {0, 0, 1}, .ambient = 0, .smooth = false};
    plx_img s = plx_render_shaded(&m, &id, &o, &sh);
    CHECK_NEAR(PLX_PX(&s, 8, 8, 0), 1 / sqrt(1.25), 1e-5);
    CHECK(PLX_PX(&s, 19, 19, 0) == -1);
    plx_img_free(&s);
    sh.light[0] = -1; sh.light[2] = 0; sh.smooth = true; /* from image-left: n.l = 0.5 / sqrt(1.25) */
    s = plx_render_shaded(&m, &id, &o, &sh);
    CHECK_NEAR(PLX_PX(&s, 8, 8, 0), 0.5 / sqrt(1.25), 1e-5);
    plx_img_free(&s);
    sh.light[0] = 1; sh.ambient = 0.2f; /* facing away from the light: ambient only */
    s = plx_render_shaded(&m, &id, &o, &sh);
    CHECK_NEAR(PLX_PX(&s, 8, 8, 0), 0.2, 1e-6);
    plx_img_free(&s);
    plx_mesh_free(&m);
    plx_img_free(&z);
}

/* ---- demo pictures (written only when PLX_DEMO_DIR is set) ----------------------------------- */

static float gauss2(double x, double y, double cx, double cy, double sx, double sy) {
    double u = (x - cx) / sx, v = (y - cy) / sy;
    return (float)exp(-0.5 * (u * u + v * v));
}

/* A face-like relief: ellipsoidal head, nose ridge and tip, eye sockets, brow ridge, lips and
 * chin, on a 120 x 150 grid with an elliptical validity mask. */
static void face_surface(plx_img *z, uint8_t *mask, plx_img *tex) {
    *z = plx_img_new(120, 150, 1);
    *tex = plx_img_new(120, 150, 3);
    for (int y = 0; y < 150; y++)
        for (int x = 0; x < 120; x++) {
            double u = (x - 60) / 48.0, v = (y - 75) / 66.0, q = 1 - u * u - v * v;
            size_t i = (size_t)y * 120 + x;
            mask[i] = q > 0;
            double h = 38 * sqrt(PLX_MAX(q, 0));
            h += 14 * gauss2(x, y, 60, 86, 4.5, 6) + 6 * gauss2(x, y, 60, 72, 3.5, 13);  /* nose tip + ridge */
            h -= 7 * (gauss2(x, y, 41, 64, 8, 5) + gauss2(x, y, 79, 64, 8, 5));       /* eye sockets */
            h += 3 * (gauss2(x, y, 41, 54, 10, 3) + gauss2(x, y, 79, 54, 10, 3));     /* brows */
            h += 3 * gauss2(x, y, 60, 107, 11, 3) - 2 * gauss2(x, y, 60, 111, 12, 1.5); /* lips, mouth line */
            h += 4 * gauss2(x, y, 60, 128, 12, 6);                                    /* chin */
            z->px[i] = (float)h;
            float skin[3] = {0.85f, 0.68f, 0.56f};
            float eye = gauss2(x, y, 41, 64, 5, 2.5) + gauss2(x, y, 79, 64, 5, 2.5);
            float lip = gauss2(x, y, 60, 108, 10, 2.5);
            for (int k = 0; k < 3; k++) PLX_PX(tex, x, y, k) = skin[k] * (1 - 0.8f * eye);
            PLX_PX(tex, x, y, 1) *= 1 - 0.5f * lip;
            PLX_PX(tex, x, y, 2) *= 1 - 0.4f * lip;
        }
}

static void save_demo(const char *dir, const char *name, const plx_img *im) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    CHECK(plx_viz_save(p, im) == PLX_OK);
}

TEST(demo_pictures) {
    const char *dir = getenv("PLX_DEMO_DIR");
    if (!dir) { printf("  (skipped: set PLX_DEMO_DIR to write demo images)\n"); return; }
    CHECK(plx_mkdir_p(dir) == PLX_OK);
    plx_img z, tex;
    uint8_t *mask = PLX_NEW(uint8_t, 120 * 150);
    face_surface(&z, mask, &tex);

    plx_pose yaws[5] = {{-45 * DEG, 0, 0}, {-20 * DEG, 0, 0}, {0, 0, 0}, {20 * DEG, 0, 0}, {45 * DEG, 0, 0}};
    plx_img v = plx_render_views(&z, mask, &tex, yaws, 5);
    save_demo(dir, "face_yaw.plxt", &v);
    plx_img_free(&v);
    plx_pose mix[4] = {{0, 25 * DEG, 0}, {0, -25 * DEG, 0}, {30 * DEG, 15 * DEG, 0}, {0, 0, 20 * DEG}};
    v = plx_render_views(&z, mask, NULL, mix, 4);
    save_demo(dir, "face_pitch_roll.plxt", &v);
    plx_img_free(&v);

    /* Depth maps through both colour maps, a signed difference, flow field and legend. */
    for (int i = 0; i < 120 * 150; i++) if (!mask[i]) z.px[i] = NAN;
    plx_img tiles[4];
    tiles[0] = plx_viz_scalar(&z, PLX_CMAP_SEQ);
    plx_img grad = plx_img_new(120, 150, 1), fx = plx_img_new(120, 150, 1), fy = plx_img_new(120, 150, 1);
    for (int y = 1; y < 149; y++)
        for (int x = 1; x < 119; x++) {
            PLX_PX(&grad, x, y, 0) = PLX_PX(&z, x + 1, y, 0) - PLX_PX(&z, x - 1, y, 0);
            PLX_PX(&fx, x, y, 0) = -0.5f * (PLX_PX(&z, x + 1, y, 0) - PLX_PX(&z, x - 1, y, 0));
            PLX_PX(&fy, x, y, 0) = -0.5f * (PLX_PX(&z, x, y + 1, 0) - PLX_PX(&z, x, y - 1, 0));
        }
    tiles[1] = plx_viz_scalar(&grad, PLX_CMAP_DIV);
    tiles[2] = plx_viz_flow_color(fx.px, fy.px, 120, 150, 0);
    float gx[15 * 19], gy[15 * 19];
    for (int j = 0; j < 19; j++)
        for (int i = 0; i < 15; i++) {
            gx[j * 15 + i] = PLX_PX(&fx, 4 + 8 * i, 4 + 8 * j, 0);
            gy[j * 15 + i] = PLX_PX(&fy, 4 + 8 * i, 4 + 8 * j, 0);
        }
    plx_img lum = plx_img_luma(&tex);
    tiles[3] = plx_viz_to_rgb(&lum);
    plx_viz_flow_arrows(&tiles[3], gx, gy, 15, 19, 4, 4, 8, 3, (const float[3]){1, 0.2f, 0.1f});
    float green[3] = {0.2f, 1, 0.3f};
    plx_draw_cross(&tiles[3], 60, 86, 4, 1, green, 1);
    plx_draw_circle(&tiles[3], 41, 64, 7, 1.5f, green, 1);
    plx_draw_circle(&tiles[3], 79, 64, 7, 1.5f, green, 1);
    const char *labels[4] = {"depth (seq)", "dz/dx (div)", "-grad z (hue)", "arrows+marks"};
    plx_montage_opts mo = plx_montage_defaults();
    mo.cols = 4;
    plx_img sheet = plx_viz_montage(tiles, 4, labels, &mo);
    plx_img big = plx_viz_upscale(&sheet, 2, PLX_UP_NEAREST);
    save_demo(dir, "maps.plxt", &big);
    for (int i = 0; i < 4; i++) plx_img_free(&tiles[i]);
    plx_img_free(&sheet); plx_img_free(&big); plx_img_free(&grad); plx_img_free(&fx); plx_img_free(&fy);
    plx_img_free(&lum);

    /* Colour bars, the flow wheel and the full font. */
    plx_img bars = plx_img_new(256, 3 * 24 + 4 * 6 + 120, 3);
    for (int m = 0; m < 3; m++)
        for (int x = 0; x < 256; x++) {
            float rgb[3];
            plx_cmap_eval((plx_cmap)m, x / 255.0f, rgb);
            for (int y = 0; y < 24; y++)
                for (int k = 0; k < 3; k++) PLX_PX(&bars, x, 6 + m * 30 + y, k) = rgb[k];
        }
    plx_img wheel = plx_viz_flow_wheel(64);
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
            for (int k = 0; k < 3; k++) PLX_PX(&bars, 4 + x, 100 + y, k) = PLX_PX(&wheel, x, y, k);
    char font_text[200];
    int n = 0;
    for (int c = 32; c < 127; c++) {
        font_text[n++] = (char)c;
        if ((c - 31) % 30 == 0) font_text[n++] = '\n';
    }
    font_text[n] = '\0';
    plx_draw_text(&bars, 74, 100, font_text, 1, WHITE);
    plx_draw_text(&bars, 74, 140, "Yaw +30", 2, WHITE);
    plx_img bars2 = plx_viz_upscale(&bars, 3, PLX_UP_NEAREST);
    save_demo(dir, "legend.plxt", &bars2);
    plx_img_free(&bars); plx_img_free(&bars2); plx_img_free(&wheel);

    /* Hemisphere at yaw -30 / 0 / +30 for checking occlusion by eye. */
    plx_img hz = hemisphere();
    plx_pose hp[3] = {{-30 * DEG, 0, 0}, {0, 0, 0}, {30 * DEG, 0, 0}};
    v = plx_render_views(&hz, NULL, NULL, hp, 3);
    save_demo(dir, "hemisphere_yaw.plxt", &v);
    plx_img_free(&v);
    plx_img_free(&hz);
    plx_img_free(&z);
    plx_img_free(&tex);
    free(mask);
}

int main(void) {
    RUN(colormaps_ordered_and_centred);
    RUN(text_glyph_pixels);
    RUN(drawing_coverage);
    RUN(montage_layout);
    RUN(upscale_modes);
    RUN(flow_visualisation);
    RUN(raster_flat_square_exact_count);
    RUN(raster_shared_edges_no_cracks_or_double_hits);
    RUN(raster_z_order);
    RUN(raster_hemisphere_yaw_shift_and_self_occlusion);
    RUN(raster_linear_attributes_exact);
    RUN(shading_lambert);
    RUN(demo_pictures);
    return TEST_REPORT();
}
