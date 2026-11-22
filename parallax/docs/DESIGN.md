# Parallax — design

A face recognition system that builds its own 3D representation of faces from 2D pixels and
matches people through it. Hand-written C11 (libc, libm, pthreads). No neural networks, no
third-party code, no human-labelled landmarks, no 3D scans, no shading model.

This file is the contract between modules. If code and this file disagree, one of them is a bug.

## 1. The idea

Depth is not in a single pixel, but it leaks into **where things land**. Two geometric facts
carry everything:

1. **Pose parallax.** When a head turns by yaw θ, a point at depth z slides sideways by
   `sin θ · z`. Across thousands of photos of heads at different yaws, the dominant way that
   faces fail to line up *is* the depth map.
2. **Mirror stereo.** A face is bilaterally symmetric, so one photo of a turned face is a
   stereo pair: the image and its mirror are views at +θ and −θ. The midpoint between a point
   and its mirror partner sits at `sin θ · z`. Seen this way, the facial midline of a 3/4
   photo *is the side profile*, compressed by sin θ.

Both facts are expressed in one canonical frame through a **parity decomposition** (§5), which
cleanly separates pose (even) from identity shape (odd) with no supervision. Video is the same
machinery with more views: every frame adds `sin²θ` of precision to the same shape (§8).

"Self-representing" means the canonical frame, the correspondence, the depth, the shape bases
and the uncertainty model are all estimated by the system from its own alignment of the data.
The only prior knowledge supplied is: faces are roughly symmetric, roughly rigid, and convex
(nose towards the camera — this picks the sign that orthographic projection cannot).

## 2. Conventions

| thing | convention |
|---|---|
| image | 250×250, x right, y **down**, pixel centres at integers, intensity [0,1] |
| canonical frame Ω | 128 wide × 160 tall at unit scale, centre `c0 = (63.5, 79.5)`; centred coords `X0 = x − 63.5`, `Y0 = y − 79.5`. In the unaligned LFW mean, eyes land near canonical y ≈ 63, mouth ≈ 103, chin ≈ 136 |
| mirror | `σ(x, y) = (127 − x, y)`; on vectors `σ*(vx, vy) = (−vx, vy)` |
| 3D axes | X right, Y down, **Z towards the viewer** (depth in canonical pixels) |
| yaw θ | `x' = cosθ·X + sinθ·Z`, `z' = −sinθ·X + cosθ·Z`. Positive = face turns to image-right |
| pitch φ | `y' = cosφ·Y − sinφ·Z`, `z' = sinφ·Y + cosφ·Z`. Positive = face looks up |
| roll ρ | `x' = cosρ·x − sinρ·y`, `y' = sinρ·x + cosρ·y`. Positive = clockwise on screen |
| camera | weak perspective: `p = s · Rz(ρ) · [Rx(φ) · Ry(θ) · v]xy + t` (yaw first, then pitch, then roll) |
| similarity `T` | `T(p) = [a −b; b a](p − c0) + (124.5 + tx, 128.5 + ty)`, `a = s cosρ`, `b = s sinρ`; maps Ω → image (the face sits slightly below the crop centre) |
| lattice | nodes every 8 px on Ω: 17 columns at x ∈ {−0.5, 7.5, …, 127.5} × 21 rows at y ∈ {−0.5, …, 159.5}; σ-symmetric, centre column on the axis. Coarser (16, 32 px) levels for coarse-to-fine; a 4 px level (33×41) is optional |
| full warp | `W_i(p) = T_i(p + u_i(p))`, `u_i` = lattice displacement field in canonical pixels |
| global image index | identities sorted by name, images by number: 0 … 13,232 (see `data/lfw.h`) |

Stage outputs are `.plxt` tensors under `runs/<run>/`, so every stage can be run, inspected
and replaced on its own.

## 3. Pipeline

```
 JPEG ──pack──▶ gray pack ──congeal──▶ T_i, template μ ──flow──▶ u_i ──emerge──▶ z̄, κ, bases
                                                              │                       │
                 video frames ───────────── same flow ────────┤                       ▼
                                                              └──────fit──▶ FaceCode ──match──▶ score
```

| stage | module / app | in | out |
|---|---|---|---|
| pack | `apps/plx_pack` | LFW JPEGs | `pack/gray.plxt` u8 [N,250,250] |
| congeal | `align` / `plx_congeal` | pack | `congeal/sim.plxt` f32 [N,4] (a,b,tx,ty), template mean + variance |
| flow | `align` / `plx_flow` | pack, sim, template | `flow/u.plxt` f32 [N,2,21,17] (component, row, col), refined template, per-image fit quality |
| emerge | `geom` / `plx_emerge` | u | `geom/model.plxt…`: z̄, κ, pose per image, depth basis, shape basis |
| fit | `geom` / `plx_fit` | image(s) + model | FaceCode |
| match | `rec` / `plx_eval` | FaceCodes + pairs | LFW accuracy ladder |
| fuse | `geom` / `plx_fuse` | video frames | FaceCode with multi-frame shape |

## 4. Correspondence (congeal + flow)

Everything downstream is only as good as the dense map from the canonical frame into each
photo, and it has to be learned without labels.

**Features.** Alignment never compares intensities. It compares orientation channels
(`feat`): K soft-binned gradient-orientation maps, locally contrast-normalised, so the residual
says "an edge of this direction should be here" regardless of lighting and skin tone.

**Re-render, then refine.** Each round warps the *gray image* into Ω with the current warp,
computes features there, and solves for a small incremental warp against the template. Because
features are always computed in the canonical frame, scale and in-plane rotation are handled
exactly and the incremental problem is always small.

**Congealing (similarity, 4 dof).** Alternate: (1) template μ = mean of canonically warped
features, variance σ² alongside; (2) symmetrise: `μ ← ½(μ + flipperm(μ))` — equivalent to
adding every image's mirror twin to the population, and it pins the symmetry axis to x = 63.5;
(3) align every image to μ by inverse-compositional Gauss-Newton with weights `1/(σ² + ε)`
(the population itself decides which pixels are trustworthy), coarse to fine; (4) gauge-fix:
mean log-scale and mean ty over the population are reset to 0 so the template cannot drift or
shrink.

**Flow (deformable, 714 dof).** Same loop with the lattice field `u`. Inverse-compositional:
the Jacobian comes from template gradients, so `H = JᵀWJ + λR` is shared by all images and is
factorised once per round; each image costs one warp, one feature pass and one back-substitution.
Requirements that the geometry depends on:

- the regulariser `R` is **bending energy** (second differences): its null space is affine, so
  foreshortening (which is how metric scale is recovered, §5.4) is not penalised;
- gauge: each `u_i` has its similarity component projected into `T_i`, and the population-mean
  of `u` (which by symmetry is x-odd / y-even) is folded back into the template each outer
  round, so the template stays the mean shape;
- robustness: Huber-clipped residuals; images whose fit does not converge are flagged, not trusted;
- the 32 and 16 px rounds re-render through the field; the 8 px rounds keep the last render and
  refine by sampling. At 8 px the orientation features respond to a sub-cell displacement too
  differently from the displacement model for re-rendering to converge on real faces. Each
  round's increment is damped (terms that vanish at convergence, so the affine part stays unbiased).

Flow has no ground truth, so it is judged by self-consistency: (a) variance of warped features
drops and the mean warped face sharpens; (b) **mirror consistency** — the flow of a mirrored
image must be the mirror of the flow, and the violation is an honest error bar; (c) recovery of
known synthetic warps; (d) the downstream rank-1 structure of §5.

## 5. Mirror-parallax geometry (emerge)

### 5.1 Model
Person i has canonical 3D shape `S_i(p) = (X0 + a_i(p), Y0 + b_i(p), z_i(p))` with `a_i` odd,
`b_i` even, `z_i` even under σ (bilateral symmetry). Viewed at yaw θ, pitch φ, and with a small
residual similarity `(δs, δρ, δt)` not absorbed by `T_i`, the flow is

```
u_x = (cosθ − 1 + δs)·X0 + cosθ·a + sinθ·z − δρ·Y0 + δtx
u_y = (cosφ − 1 + δs)·Y0 + cosφ·b − sinφ·cosθ·z + (sinφ·sinθ + δρ)·X0 + sinφ·sinθ·a + δty
```

### 5.2 Parity decomposition
Split each component into σ-even and σ-odd parts, `even f = ½(f(p) + f(σp))`:

| part | contains | meaning |
|---|---|---|
| **E** = even(u_x) | `sinθ·z(p) − δρ·Y0 + δtx` | **yaw parallax = depth** |
| O = odd(u_x) | `(cosθ − 1 + δs)·X0 + cosθ·a(p)` | foreshortening + identity widths |
| V = even(u_y) | `(cosφ − 1 + δs)·Y0 + cosφ·b(p) − sinφ·cosθ·z(p) + δty` | pitch parallax + identity heights |
| Q = odd(u_y) | `(sinφ·sinθ + δρ)·X0 + …` | roll residue, asymmetry |

Identity cannot leak into E (a symmetric face's width variation is odd) and X0-linear
foreshortening cannot either. E is exactly half the disparity between the image and its own
mirror. Gauge: project `span{1, Y0}` out of E (and of V) by weighted least squares → E⊥, V⊥.
Depth is therefore defined up to a plane `α + γ·Y`; we fix z̄ to have zero mean and zero
Y-slope over the face support.

### 5.3 Depth as a singular vector
Stack E⊥ over the population: `E⊥_i(p) = s_i · v(p) + noise`, rank 1. `v` (unit weighted RMS)
is the top right singular vector, sign fixed by convexity (centre of the face positive).
`s_i = ⟨E⊥_i, v⟩ / ⟨v, v⟩` is yaw in units of depth-RMS·sin θ. Pitch from the same pattern:
`t_i = −⟨V⊥_i, q⟩` with `q ∝ (C + μI)⁻¹ v`, `qᵀv = 1`, where C is the scatter of V⊥ and μ its
mean eigenvalue. (The plain projection onto v is dominated by identity heights `b` that happen
to resemble v; the shrunk generalised-least-squares direction suppresses them without knowing
the pitch variance.) That the *one* pattern v explains both the horizontal channel (yaw) and
the vertical channel (pitch) is the rigid-body check, and it is reported. Pitch, like the
Y-tilt of z̄, is defined relative to the population-mean viewpoint: the two are the same gauge.

The robust weights of the rank-1 fit are judged against photos of similar s²; a global
residual threshold would discard exactly the turned faces that carry the signal.

### 5.4 Metric scale without a ruler
`z̄ = κ·v` for one unknown κ. The similarity gauge divides every field by that photo's own
foreshortened scale `ρ_i = f_x·cosθ_i + (1 − f_x)·cosφ_i` (`f_x = Σg·X0² / Σg·(X0² + Y0²)` over
the gauge nodes), so the pose relations are `sinθ_i = ρ_i·s_i/κ` and
`sinφ_i·cosθ_i = ρ_i·t_i/κ`, solved by fixed point. A turned face is also *narrower*:
the anisotropy `A_i = (1 + slope of O on X0) / (1 + slope of V on Y0) = (cosθ_i / cosφ_i)·ar_i`
where `ar_i` is the person's own aspect ratio (independent of pose). κ is the value that makes
`log A_i ≈ c + ½·log(1 − s_i²/κ²) − ½·log(1 − t_i²/κ²)` hold across the population (robust 1-D
fit). This is Tomasi-Kanade's orthonormality constraint, applied statistically across different
people instead of across frames.

### 5.5 Personal depth, with honest uncertainty
`R_i = E⊥_i − s_i·v = s_i·B·c_i + n_i`, `c_i ~ N(0, I)`, `B ⟂ v` (the bas-relief gauge: depth
scale along v is not observable in one view, so it is defined away), and `n_i` is
pose-independent nuisance (flow error, asymmetry, lighting bias) with covariance Σ0. Then
`Cov(R | s) = s²·BBᵀ + Σ0`: **personal depth variation is exactly the part of the residual
covariance that grows with sin²(yaw)**. Regressing `R_i R_iᵀ` on `s_i²` gives the slope matrix
Σ_B (its leading eigenvectors are B) and the intercept Σ0. For one image, with
`N_i = Σ0 + s_i²·Σ_rest` (Σ_rest = the part of Σ_B beyond the kept modes):

```
precision  Λ_i = I + s_i²·Bᵀ N_i⁻¹ B          ĉ_i = Λ_i⁻¹ · s_i · Bᵀ N_i⁻¹ R_i
```

A frontal photo (s ≈ 0) returns the prior with full uncertainty: no parallax, no depth claim.
A 30° photo returns a confident personal depth. The representation knows what it does not know.

### 5.6 Relief: fine depth from the population, not from the photo
What real data showed: a per-photo field has to be stiff (one photo is weak evidence, and a
soft field chases beards and glasses), and that stiffness erases high-curvature parallax, so
§5.3 recovers pose and a smooth convex dome but no nose. The depth pattern, however, is shared
by everyone. Once each photo's yaw `s_i` is known, add one dense population-shared layer,
`W_i(p) = T_i(p + u_i(p) + s_i·w(p)·x̂)`, and estimate `w` at pixel resolution as the regression
coefficient of residual horizontal misalignment on yaw:

```
b_ik(p) = s_i·∂x F_ik(p)      Δw(p) = − G_σ*[Σ_i,k ω·b·ψ(r)] / (G_σ*[Σ_i,k ω·b²] + λ)
```

with each photo's own feature gradient, mirror twins folded in (`N(p) + N(σp)`), the plane
gauge of §5.2, and the scale gauge `⟨v + w, v⟩ = ⟨v, v⟩` so κ and the yaw angles stay valid.
13,233 photos per pixel replace the per-photo regulariser. Judged by split-half agreement of
the high-pass part of `w`, a permutation control (shuffle `s_i`: the map must vanish), and
recovery of a known synthetic relief.

### 5.7 Direct mirror stereo (single-image refinement)
The template flow gives E through cross-identity matching. The same quantity can be measured
*within* the image: in the roll-aligned frame, mirror partners share a row, so for each row y
and half-width w there is a midpoint `m(w, y)` with `F(m + w, y) ≈ flipperm F(m − w, y)`, and
`m(w, y) − axis = sinθ·z`. Solved as regularised Gauss-Newton on the (w, y) lattice,
initialised from E. Same person on both sides of the match, so it is sharper than any
cross-identity flow; occluded partners are handled by the robust loss.

## 6. FaceCode — what a face is, internally

```
pose      θ, φ, ρ, s, t                nuisance; drives visibility, never compared
depth     ĉ ∈ R^Kz , Λ                 personal depth coefficients + precision
shape     â ∈ R^Ks                     in-plane identity shape: PCA of O and V after removing pose terms
texture   d ∈ R^D , per-cell weights   descriptors on the frontalised texture G(p) = I(W(p))
frames    F                            1 for a photo; video accumulates
```

**Visibility.** From pose and depth, each canonical pixel gets a weight
`vis(p) = clamp(n(p)·view)` (equivalently the foreshortening Jacobian of W). Grazing or
self-occluded regions are down-weighted and optionally filled from their mirror partner.

## 7. Matching

`score = geometry term + texture term`.

- geometry: log-likelihood ratio of "same latent c" vs "independent c" given `(ĉ₁, Λ₁)`,
  `(ĉ₂, Λ₂)` — a heteroscedastic two-sample test, so two frontal photos contribute ~0 instead of noise;
- texture: similarity of descriptors over cells, weighted by the joint visibility of both faces;
- protocol: unsupervised parts (template, flow, depth model, PCA) never see identity labels.
  Supervised fusion / metric learning uses only the 9 training folds of LFW View 2
  (image-restricted, no outside data). Development on View 1.

Reported as an ablation ladder, because the point is to learn what the 3D representation buys:

| level | representation |
|---|---|
| L0 | raw centre crop |
| L1 | + similarity congealing |
| L2 | + deformable frontalisation |
| L3 | + depth-derived visibility and symmetric fill |
| L4 | + geometry code |
| L5 | + supervised metric (image-restricted) |

## 8. Video: many frames, one shape

Frames f = 1…F of one person. Track the lattice in every frame (warm-started, and re-tracked
against the *person's own* mean texture, which is far sharper than the population template).
The lattice positions form `W ∈ R^{2F×P}`; after centring, `W = M·S` with rank 3. Weighted
alternating least squares handles occluded nodes; the metric upgrade finds the 3×3 `Q` making
each frame's two rows of `M·Q` orthogonal with equal norm. Result: the person's full
`(x, y, z)` shape with no symmetry assumption and no population prior, plus per-frame pose.

It then drops into the same posterior as §5.5: every frame contributes precision `s_f²/σ²`,
the population prior fills what the motion never revealed, and textures are fused by
visibility-weighted averaging in Ω. A single photo is the F = 1 case of the same estimator.

## 9. Honest expectations

Classical, label-free LFW verification sits around 70–88 % in the literature; hand-engineered
supervised methods reach the low 90s. Neural networks own 99 %+. This project is not trying to
beat them. It is trying to find out how much real 3D structure can be pulled out of 2D by
geometry alone, and how much that structure helps recognition. Geometry alone will be a weak
verifier on single in-the-wild photos (little pose ⇒ little depth evidence); the interesting
numbers are L2 → L3 → L4, and single photo vs video.
