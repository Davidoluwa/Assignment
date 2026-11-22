/* Z-buffer triangle rasteriser with the system's weak-perspective camera.
 *
 * COORDINATES. Object and camera space share the image axes: x right, y DOWN, z TOWARDS the
 * viewer (larger z = closer; for a face height field the nose tip has the largest z).
 * Note this frame is left-handed; every rotation below is defined by its matrix, so no
 * handedness rule is needed to read the conventions.
 *
 * CAMERA. p = s * R * (v - centre) + t. (p.x, p.y) is the image position (pixel centres at
 * integers), p.z the depth (t[2] is an optional depth offset; orthographic, so it never
 * affects image positions). R = Rz(roll) * Rx(pitch) * Ry(yaw), angles in radians, so yaw
 * is applied first, in the object's own frame:
 *
 *   Ry(yaw)   = [ c 0 s ; 0 1 0 ; -s 0 c ]   maps +z (facing the viewer) to (sin yaw, 0, cos yaw):
 *               POSITIVE YAW TURNS THE FACE TOWARDS IMAGE-RIGHT (the nose moves to +x; the
 *               image-right side of the face turns away and is the side that self-occludes).
 *   Rx(pitch) = [ 1 0 0 ; 0 c -s ; 0 s c ]   maps +z to (0, -sin pitch, cos pitch):
 *               POSITIVE PITCH TILTS THE FACE UP (nose moves to -y, up in the image; the chin
 *               comes forward).
 *   Rz(roll)  = [ c -s 0 ; s c 0 ; 0 0 1 ]   maps +x to (cos roll, sin roll, 0):
 *               POSITIVE ROLL ROTATES THE IMAGE CLOCKWISE on screen (image-right side moves down).
 *
 * TRIANGLES. Front-facing means the projected triangle has positive signed area
 *   E = (x1 - x0)(y2 - y0) - (y1 - y0)(x2 - x0) > 0,
 * which with y down is CLOCKWISE as seen on screen, and equals the z-component of
 * (v1 - v0) x (v2 - v0): that cross product is the outward normal pointing at the viewer.
 * The height-field mesh is built with this orientation, so a surface z(x, y) seen frontally
 * is front-facing.
 *
 * RASTERISATION. A pixel is covered when its centre lies inside the triangle. Vertex
 * positions are snapped to 1/256 pixel and edge functions are evaluated exactly in 64-bit
 * integers with the top-left rule (centres exactly on an edge belong to top / left edges
 * only), so two triangles sharing an edge never both cover, and never both miss, a pixel.
 * Nearest surface wins (largest z; strictly greater, so on exact depth ties the triangle
 * drawn first keeps the pixel). Attributes and depth are interpolated linearly in screen
 * space with barycentrics computed from the unsnapped positions; under an orthographic
 * camera that is exact (no perspective correction exists to be made). */
#ifndef PLX_RENDER_H
#define PLX_RENDER_H

#include "img/image.h"
#include "viz/viz.h"

typedef struct {
    int    nv, nt, na; /* vertices, triangles, attribute channels per vertex */
    float *v;          /* [nv][3] x, y, z */
    float *attr;       /* [nv][na]; NULL when na == 0 */
    int   *tri;        /* [nt][3] vertex indices */
} plx_mesh;

plx_mesh plx_mesh_new(int nv, int nt, int na); /* zero-filled */
void     plx_mesh_free(plx_mesh *m);
/* Height-field mesh from channel 0 of z (W x H). Vertex index = y * W + x, at (x, y, z(x, y)).
 * A grid point is valid when z is finite and (mask == NULL or mask[y * W + x] != 0); every
 * cell whose 4 corners are valid becomes two triangles, (x,y)(x+1,y)(x,y+1) and
 * (x+1,y)(x+1,y+1)(x,y+1). Invalid vertices get z = 0 and are referenced by no triangle.
 * tex (optional, same W x H, any channel count) becomes the per-vertex attributes. */
plx_mesh plx_mesh_heightfield(const plx_img *z, const uint8_t *mask, const plx_img *tex);
/* Mean of the vertices referenced by at least one triangle (the natural rotation centre). */
void     plx_mesh_centroid(const plx_mesh *m, double c[3]);

/* ---- camera ---------------------------------------------------------------------------- */
typedef struct {
    double R[9];      /* row-major */
    double s;
    double centre[3];
    double t[3];
} plx_camera;

void       plx_rot_ypr(double yaw, double pitch, double roll, double R[9]); /* R = Rz * Rx * Ry */
plx_camera plx_camera_make(double yaw, double pitch, double roll, double s, const double centre[3], double tx,
                           double ty);
void       plx_camera_project(const plx_camera *cam, const float v[3], double p[3]);

/* ---- rasteriser ------------------------------------------------------------------------ */
typedef enum { PLX_CULL_NONE = 0, PLX_CULL_BACK = 1 } plx_cull;

typedef struct {
    int      w, h;
    plx_cull cull;
    bool     want_ids; /* also fill raster.tri and raster.bary */
    float    bg;       /* attribute value where nothing is drawn */
} plx_raster_opts;

typedef struct {
    int      w, h;
    plx_img  attr;  /* na channels (absent when na == 0) */
    plx_img  depth; /* 1 channel: p.z of the visible surface, NaN where uncovered */
    uint8_t *mask;  /* [h * w] 1 where covered */
    int32_t *tri;   /* [h * w] visible triangle id, -1 where uncovered (want_ids only) */
    plx_img  bary;  /* 3 channels: weights of the triangle's vertices 0, 1, 2 (want_ids only) */
} plx_raster;

void plx_render(const plx_mesh *m, const plx_camera *cam, const plx_raster_opts *o, plx_raster *out);
void plx_raster_free(plx_raster *r);

/* ---- shading -----------------------------------------------------------------------------
 * Lambertian: intensity = ambient + (1 - ambient) * max(0, n . l), n the unit normal in
 * camera space (two-sided: a visible back face uses its flipped normal), l the unit
 * direction TOWARDS the light in camera coordinates, e.g. {-0.4, -0.5, 1} = from the
 * upper-left, in front. Normals per face (flat) or per vertex (area-weighted, smooth).
 * With use_texture, intensity multiplies the mesh attributes (1 or 3 channels). */
typedef struct {
    float light[3];
    float ambient;
    bool  smooth;
    bool  use_texture;
} plx_shade;
plx_shade plx_shade_defaults(void); /* upper-left frontal light, ambient 0.15, smooth, untextured */
/* Returns a 1-channel image (3 when textured with an RGB mesh); bg where uncovered. */
plx_img plx_render_shaded(const plx_mesh *m, const plx_camera *cam, const plx_raster_opts *o, const plx_shade *sh);

/* ---- the diagnostic picture -------------------------------------------------------------
 * Render a height field from k poses and lay the results out in one labelled montage:
 * row 1 = shaded geometry, row 2 (only when tex is given) = the texture re-rendered at that
 * pose. All views share one scale, chosen so every pose fits its tile (never enlarged
 * beyond s = 1); rotation is about the centroid of the valid surface. Tiles are W x H;
 * tex must have 1 or 3 channels. */
typedef struct { double yaw, pitch, roll; } plx_pose; /* radians */
plx_img plx_render_views(const plx_img *z, const uint8_t *mask, const plx_img *tex, const plx_pose *poses, int k);

#endif
