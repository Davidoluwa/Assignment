/* Synthetic clips with known geometry: a textured height-field head rendered by the z-buffer
 * renderer through a known pose track. See fuse.h. */
#include "fuse/fuse.h"

#include "ops/ops.h"
#include "render/render.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DEG (M_PI / 180.0)

static double gauss2(double x, double y, double sx, double sy) {
    return exp(-0.5 * (x * x / (sx * sx) + y * y / (sy * sy)));
}

plx_fu_head plx_fu_head_preset(int which) {
    /* 0 reproduces geom's synthetic mean face (plx_synth_depth with nose_px = 20): the nose tip
     * stands ~25 px above the cheeks on a face ~100 px wide */
    plx_fu_head h = {.dome_depth = 38, .dome_ax = 62, .dome_ay = 82, .nose = 20, .nose_w = 7, .nose_len = 26,
                     .eye_depth = 7, .eye_x = 19, .eye_y = -15, .brow = 3, .lips = 3, .chin = 4};
    if (which == 1) { /* long narrow nose, flat face, deep-set eyes, strong chin */
        h.dome_depth = 28, h.nose = 27, h.nose_w = 5.5, h.nose_len = 34, h.eye_depth = 11, h.chin = 7, h.brow = 5;
    } else if (which == 2) { /* short wide nose, round full face, shallow eyes, full lips */
        h.dome_depth = 46, h.dome_ax = 66, h.dome_ay = 78, h.nose = 13, h.nose_w = 10, h.nose_len = 18;
        h.eye_depth = 4, h.brow = 1.5, h.lips = 6, h.chin = 2;
    }
    return h;
}

double plx_fu_head_depth(const plx_fu_head *h, double X, double Y) {
    double ex = X / h->dome_ax, ey = (Y - 4) / h->dome_ay, e = 1 - ex * ex - ey * ey;
    if (e <= 0) return NAN;
    double z = h->dome_depth * sqrt(e);
    /* nose: a ridge from the brows down to the tip at Y0 = +10, then a quick drop */
    double r = Y <= 10 ? (10 - Y) / h->nose_len : (Y - 10) / 6;
    z += h->nose * exp(-0.5 * X * X / (h->nose_w * h->nose_w)) * exp(-r * r);
    z -= h->eye_depth * (gauss2(X - h->eye_x, Y - h->eye_y, 8, 8) + gauss2(X + h->eye_x, Y - h->eye_y, 8, 8));
    z += h->brow * gauss2(X, Y + 28, 25, 5);
    z += h->lips * gauss2(X, Y - 30, 12, 5);
    z += h->chin * gauss2(X, Y - 48, 10, 7);
    return z;
}

plx_fu_synth_opts plx_fu_synth_defaults(void) {
    return (plx_fu_synth_opts){.F = 40, .yaw_deg = 30, .pitch_deg = 12, .roll_deg = 3, .scale_drift = 0.03,
                               .trans_drift = 3, .yaw_cycles = 1, .pitch_cycles = 2, .noise = 0.01, .gain = 0.05,
                               .shade = 0.3, .frame_w = 250, .frame_h = 250, .face_scale = 1,
                               .face_x = PLX_CANON_IMG_CX, .face_y = PLX_CANON_IMG_CY, .clutter = false, .seed = 1};
}

plx_fu_pose plx_fu_synth_pose(const plx_fu_synth_opts *o, int f) {
    double u = o->F > 1 ? (double)f / o->F : 0, tau = 2 * M_PI;
    plx_fu_pose p;
    p.yaw = o->yaw_deg * DEG * sin(tau * o->yaw_cycles * u);
    p.pitch = o->pitch_deg * DEG * sin(tau * o->pitch_cycles * u + 0.6);
    p.roll = o->roll_deg * DEG * sin(tau * u + 1.1);
    p.s = o->face_scale * (1 + o->scale_drift * sin(tau * 0.5 * u + 0.3));
    p.x = o->face_x + o->trans_drift * sin(tau * u + 2.0);
    p.y = o->face_y + o->trans_drift * cos(tau * u + 0.5);
    return p;
}

void plx_fu_synth_project(const plx_fu_head *h, const plx_fu_synth_opts *o, plx_fu_pose P, double X0, double Y0,
                          double *x, double *y) {
    (void)o;
    double R[9], v[3] = {X0, Y0, plx_fu_head_depth(h, X0, Y0)};
    plx_rot_ypr(P.yaw, P.pitch, P.roll, R);
    *x = P.s * (R[0] * v[0] + R[1] * v[1] + R[2] * v[2]) + P.x;
    *y = P.s * (R[3] * v[0] + R[4] * v[1] + R[5] * v[2]) + P.y;
}

plx_img plx_fu_synth_background(const plx_fu_synth_opts *o, int channels) {
    int W = o->frame_w, H = o->frame_h;
    plx_img bg = plx_img_new(W, H, channels);
    plx_rng r;
    plx_rng_seed(&r, o->seed ^ 0xb6c0);
    float base[3];
    for (int c = 0; c < channels; c++) base[c] = (float)(0.35 + 0.3 * plx_rng_uniform(&r));
    for (int c = 0; c < channels; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) PLX_PX(&bg, x, y, c) = base[c] * (float)(0.8 + 0.4 * (x + 0.5 * y) / (W + 0.5 * H));
    if (!o->clutter) return bg;
    /* clutter: rectangles, discs, line bundles and gratings in random colours (an office wall,
     * shelves, a window), the kind of structure a sliding face template must reject */
    int n_items = 12 + W * H / 4000;
    for (int k = 0; k < n_items; k++) {
        float col[3];
        for (int c = 0; c < 3; c++) col[c] = (float)plx_rng_uniform(&r);
        int kind = (int)plx_rng_below(&r, 4);
        double cx = plx_rng_uniform(&r) * W, cy = plx_rng_uniform(&r) * H, sz = 8 + plx_rng_uniform(&r) * 0.25 * W;
        double ang = plx_rng_uniform(&r) * M_PI, per = 3 + 10 * plx_rng_uniform(&r);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double dx = x - cx, dy = y - cy, a = 0;
                if (kind == 0) a = fabs(dx) < sz && fabs(dy) < 0.6 * sz;
                else if (kind == 1) a = dx * dx + dy * dy < sz * sz * 0.4;
                else if (kind == 2) a = fabs(dx * sin(ang) - dy * cos(ang)) < 1.5 && hypot(dx, dy) < 2 * sz;
                else if (fabs(dx) < sz && fabs(dy) < sz) a = 0.5 + 0.5 * sin(2 * M_PI * (dx * cos(ang) + dy * sin(ang)) / per);
                if (a <= 0) continue;
                for (int c = 0; c < channels; c++) {
                    float v = channels == 1 ? (col[0] + col[1] + col[2]) / 3 : col[c];
                    float *p = &PLX_PX(&bg, x, y, c);
                    *p = (float)((1 - a) * *p + a * v);
                }
            }
    }
    plx_blur(&bg, &bg, 0.7f);
    return bg;
}

plx_img plx_fu_synth_frame(const plx_fu_head *h, const plx_img *tex, const plx_img *background,
                           const plx_fu_synth_opts *o, int f) {
    PLX_ASSERT(tex->c == 1 || tex->c == 3);
    int margin = (tex->w - PLX_CANON_W) / 2;
    PLX_ASSERT(margin >= 0 && tex->w == PLX_CANON_W + 2 * margin && tex->h == PLX_CANON_H + 2 * margin);
    int W = o->frame_w, H = o->frame_h, SS = 2, C = tex->c;
    plx_img own_bg = {0};
    if (!background) own_bg = plx_fu_synth_background(o, C), background = &own_bg;
    PLX_ASSERT(background->w == W && background->h == H && background->c == C);

    /* height field over the texture's grid (canonical point = grid point - margin) */
    plx_img z = plx_img_new(tex->w, tex->h, 1);
    for (int y = 0; y < tex->h; y++)
        for (int x = 0; x < tex->w; x++)
            PLX_PX(&z, x, y, 0) = (float)plx_fu_head_depth(h, x - margin - PLX_CANON_CX, y - margin - PLX_CANON_CY);
    plx_mesh m = plx_mesh_heightfield(&z, NULL, tex);
    plx_img_free(&z);

    /* render at SS x resolution: supersampled pixel X' = SS x + (SS - 1) / 2 (pixel areas aligned) */
    plx_fu_pose P = plx_fu_synth_pose(o, f);
    double centre[3] = {margin + PLX_CANON_CX, margin + PLX_CANON_CY, 0};
    plx_camera cam = plx_camera_make(P.yaw, P.pitch, P.roll, SS * P.s, centre, SS * P.x + 0.5 * (SS - 1),
                                     SS * P.y + 0.5 * (SS - 1));
    plx_raster_opts ro = {.w = SS * W, .h = SS * H, .cull = PLX_CULL_NONE, .bg = NAN};
    plx_shade sh = plx_shade_defaults();
    sh.ambient = (float)(1 - o->shade);
    sh.use_texture = true;
    plx_img head = plx_render_shaded(&m, &cam, &ro, &sh);
    plx_mesh_free(&m);

    double u = o->F > 1 ? (double)f / o->F : 0;
    float gain = (float)(1 + o->gain * sin(2 * M_PI * 1.3 * u + 0.4));
    plx_img big = plx_img_new(SS * W, SS * H, C);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < SS * H; y++)
            for (int x = 0; x < SS * W; x++) {
                float v = PLX_PX(&head, x, y, c);
                PLX_PX(&big, x, y, c) = isfinite(v) ? gain * v : PLX_PX(background, x / SS, y / SS, c);
            }
    plx_img_free(&head);
    plx_img out = plx_resize(&big, W, H);
    plx_img_free(&big);
    plx_rng r;
    plx_rng_seed(&r, o->seed * 7919 + (uint64_t)f);
    for (size_t k = 0; k < (size_t)W * H * C; k++)
        out.px[k] = PLX_CLAMP(out.px[k] + (float)(o->noise * plx_rng_normal(&r)), 0.0f, 1.0f);
    plx_img_free(&own_bg);
    return out;
}
