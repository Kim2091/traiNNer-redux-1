# BGCC — Bilateral Guided Color Correction Architecture

**Status:** Design spec (pre-implementation)
**Date:** 2026-04-20
**Target project:** traiNNer-redux (`traiNNer/archs/`)

> Name is tentative — `BGCC` is used throughout for concreteness. Final name TBD during implementation.

---

## 1. Problem Statement

When building SISR/VSR datasets for 90s/2000s anime, training pairs are typically assembled from:

- **HR source**: Blu-ray (BD) releases — clean, sharp, high resolution. Colors reflect the BD's mastering, which often differs from the original broadcast look (different scans, remaster color grading, etc.).
- **LR source**: DVD or LaserDisc releases — degraded (block compression, chroma subsampling, color bleed, mosquito noise, interlacing remnants) but carry the *intended* color look closer to the original broadcast master.

To produce usable training pairs, the HR must be color-corrected to match the LR's palette while preserving HR's structural detail. Existing techniques all fail in specific ways:

| Method | Failure mode |
|---|---|
| Mean+std color transfer | Global only; washes blacks, inaccurate |
| Linear histogram matching | Overgeneralized; frequent total failures |
| Wavelet color fix | Transfers LR low-frequency into HR — blooms luma/chroma around sharp edges. Catastrophic on anime (flat colors + sharp lines + high contrast) |
| Manual (wavelet-fix → Photoshop color-blend → PixTransform) | Trains a fresh NN per image for ~2.5k iters → ~20s/image; inconsistent; can lean on HR colors and wash blacks |

**Goal:** A learned, one-shot, lightweight neural architecture that takes `(HR, LR)` and produces a color-corrected HR. It must:

1. Preserve HR's structural detail (no softening, no bloom).
2. Transfer LR's target color palette (chroma + luma).
3. Be robust to LR degradation (compression, chroma subsampling, color bleed, mosquito noise).
4. De-bleed — when LR has color bleeding outside its intended spatial region (thin straps, small features), the model must use HR's structure to localize the *intended* color placement rather than copying the bleed.
5. Not lean on HR colors for the final answer (HR colors are structurally useful as a segmentation prior, but are the *wrong* palette and must not be copied into the output).
6. Run in sub-second per HD image, inference-only at deployment.
7. Support flexible scale factors at inference (train at 2x; work at 1x/2x/3x/4x without retraining).
8. Export to ONNX (opset 16+) with broad runtime compatibility (TensorRT, ONNX Runtime CPU/CUDA).

**Param budget:** DIS tier (~100-400K parameters). Color correction is a transform-estimation problem, not a pixel-synthesis problem — heavy capacity is unnecessary.

---

## 2. Inputs, Outputs, Supervision

### Inference (2 inputs)

- `hr`: `(B, 3, H_hr, W_hr)` — high-resolution input with wrong colors, correct structure.
- `lr`: `(B, 3, H_lr, W_lr)` — low-resolution input with target colors, degraded.
- **Returns** `output`: `(B, 3, H_hr, W_hr)` — HR's structure with corrected colors.

### Training (3 inputs from dataset)

- `lr` — same as inference.
- `hr` — same as inference.
- `cc_hr`: `(B, 3, H_hr, W_hr)` — ground-truth color-corrected HR. The supervision target.

Loss is computed as `L(output, cc_hr)`.

### Scale relationship

- Training uses `H_hr / H_lr = 2` (2x pairs, matching DVD→BD typical ratio after cropping).
- No fixed scale is baked into the architecture. Inference supports arbitrary `H_hr, W_hr` pairs.

---

## 3. Core Design Insight

Color correction is a **per-pixel color transform prediction** problem, not a pixel-synthesis problem. HR already provides all spatial detail; the model only needs to predict, for each HR pixel, a local color operator (a 3×4 affine matrix in RGB) that pushes HR's color toward LR's target palette.

The structural move is to **estimate the transform at low resolution** from the LR (where the color signal lives and LR degradations average out under a large receptive field) and **apply the transform at high resolution** to the HR (preserving structure exactly).

Consequences of this framing:

- **Scale-flexible by construction** — a per-pixel operator naturally "upsamples" to any HR size.
- **Robust to LR degradation** — the encoder reasons at low res with aggregated context; local bleed/compression noise averages out before being baked into the predicted transform.
- **HR structure preserved perfectly** — output is `matrix @ hr_pixel`; edges, textures, fine detail are untouched unless the predicted matrix actively modifies them.
- **Cannot lean on HR colors by default** — the transform is produced by the LR-side encoder. HR structure enters only as a segmentation/guidance signal (see §4.2), never as a direct color source for the output.

---

## 4. Architecture: BGCC (Bilateral Guided Color Correction)

Inspired by HDRNet (Gharbi et al., SIGGRAPH 2017) — "Deep Bilateral Learning for Real-Time Image Enhancement." Adapted for dual-input color correction and ONNX-friendly deployment.

### 4.1 High-level data flow

```
LR ─┐                               ┌─────────────────┐
    ├──▶ [LR encoder] ──▶ bilateral │ predicts 3×4    │
HR ─┘      6ch input         grid   │ affine matrix   │
                             (B,12, │ per voxel       │
                             D,H',W)└─────────────────┘
                                              │
HR ──▶ [guidance head]                        │
         tiny, per-pixel ──▶ guidance (B,1,H_hr,W_hr)
                                              │
                    ┌─────────────────────────┘
                    ▼
       [manual trilinear slice]
        spatially bilinear sample of grid → per (x,y)
        then linearly interpolate adjacent luma bins
        by guidance value
                    │
                    ▼
       per-pixel 3×4 affine matrix M(x,y) at HR res
                    │
                    ▼
       output[p] = M(p) @ [HR[p].R, HR[p].G, HR[p].B, 1]
                    │
                    ▼
       + global residual (+ HR) ──▶ final output
```

### 4.2 LR encoder — 6-channel input (LR + downsampled HR)

**Motivation for 6-channel input:** The "de-bleed" behavior requires the encoder to see HR structure alongside LR colors. If LR has bled cyan into skin territory, the encoder must use HR structure to recognize "this spot is skin, not strap" and pull the canonical cyan from the real strap region nearby. A pure-LR encoder cannot do this.

At the first conv, input is the channel-wise concatenation of:
- `lr` (3 channels) — degraded colors at LR res.
- `F.interpolate(hr, size=lr.shape[-2:], mode='bilinear')` (3 channels) — HR structure at LR res.

This costs ~450 extra parameters at the first conv (3→6 input channels into 16 output channels, 3×3 kernel).

**Encoder body:** A small conv stack with progressive spatial downsampling from LR res down to the bilateral grid's spatial resolution (`LR_res / 8` or `LR_res / 16`, configurable). DIS-style building blocks:
- `Conv 6→F 3×3` (stem)
- N × `FastResBlock(F)` at LR res (PReLU, no BN)
- `Conv F→F 3×3 stride=2` (downsample)
- N × `FastResBlock(F)` at LR/2
- `Conv F→F 3×3 stride=2` (downsample)
- N × `FastResBlock(F)` at LR/4
- (optionally further downsamples to LR/8 or LR/16)
- `Conv F → (12 × D) 1×1` — produces the bilateral grid coefficients.

**Output shape:** `(B, 12, D, H', W')` where:
- `D` = luma bin count (default 8)
- `H' = H_lr / stride_factor`, `W' = W_lr / stride_factor`
- `12 = 3 × 4` coefficients per voxel (3×4 affine: `[R', G', B']ᵀ = M · [R, G, B, 1]ᵀ`)

**Param target:** feature width `F = 16` or `F = 24`, N = 1 or 2 per stage. Exact tuning during implementation.

### 4.3 HR guidance head

A tiny per-pixel network producing a scalar guidance value per HR pixel. Used to index the luma axis of the bilateral grid.

Two candidate implementations (decide during implementation):
- **Fixed**: `guidance = 0.299*R + 0.587*G + 0.114*B` (Rec.601 luma). Zero parameters.
- **Learned**: `Conv 3→8 1×1 → PReLU → Conv 8→1 1×1 → sigmoid`. ~80 params.

HDRNet original uses a learned 1×1 MLP with a nonlinear tone curve. For anime (mostly flat colors, well-defined luma bands), a fixed luma projection may suffice. Implementation plan will A/B both.

**Output:** `(B, 1, H_hr, W_hr)` in [0, 1].

### 4.4 Bilateral slicing (manual, ONNX-friendly)

Given:
- Grid `G`: `(B, 12, D, H', W')`
- Guidance `g`: `(B, 1, H_hr, W_hr)` in [0, 1]

Compute per-pixel affine coefficients `M`: `(B, 12, H_hr, W_hr)`.

**Manual implementation (preferred over 5D `grid_sample`):**

```python
# Map guidance to luma bin coordinate in [0, D-1]
bin_f = g.squeeze(1) * (D - 1)          # (B, H_hr, W_hr)
bin_lo = bin_f.floor().clamp(0, D-1).long()   # (B, H_hr, W_hr)
bin_hi = (bin_lo + 1).clamp(max=D-1)
w_hi = (bin_f - bin_lo.float()).unsqueeze(1)  # (B, 1, H_hr, W_hr)
w_lo = 1 - w_hi

# For each luma bin, spatially upsample its 12-channel slice to HR res
# Reshape G from (B, 12, D, H', W') into D slices, each (B, 12, H', W')
# Use F.grid_sample (opset 16+) or F.interpolate with a normalized sampling grid
per_bin_hr = []  # list of D tensors, each (B, 12, H_hr, W_hr)
for d in range(D):
    slice_d = G[:, :, d]  # (B, 12, H', W')
    # Spatial bilinear to HR size
    per_bin_hr.append(F.interpolate(
        slice_d, size=(H_hr, W_hr), mode='bilinear', align_corners=False
    ))
per_bin_hr = torch.stack(per_bin_hr, dim=2)  # (B, 12, D, H_hr, W_hr)

# Gather adjacent bins per pixel and interpolate by guidance.
# Expand bin_lo / bin_hi to (B, 12, 1, H_hr, W_hr) and use torch.gather on dim=2.
idx_lo = bin_lo.unsqueeze(1).unsqueeze(1).expand(-1, 12, 1, -1, -1)   # (B,12,1,H,W)
idx_hi = bin_hi.unsqueeze(1).unsqueeze(1).expand(-1, 12, 1, -1, -1)
M_lo = torch.gather(per_bin_hr, 2, idx_lo).squeeze(2)   # (B,12,H_hr,W_hr)
M_hi = torch.gather(per_bin_hr, 2, idx_hi).squeeze(2)
M = w_lo * M_lo + w_hi * M_hi   # (B, 12, H_hr, W_hr)
```

All ops (`interpolate`, `gather`, `stack`, `floor`, `clamp`, arithmetic) are ONNX opset 16+ and TensorRT-compatible. No 5D `grid_sample` required.

**Optimization note for implementation:** the per-bin `interpolate` loop can be collapsed into a single 2D `interpolate` call by reshaping `(B, 12, D, H', W')` → `(B, 12*D, H', W')`, interpolating, and reshaping back. This is what the real implementation will do.

### 4.5 Apply the transform

Given per-pixel `M`: `(B, 12, H_hr, W_hr)` reshaped as `(B, 3, 4, H_hr, W_hr)`, and HR input:

```python
# Append 1 to the color channel dim to form [R, G, B, 1]
hr_aug = torch.cat([hr, torch.ones_like(hr[:, :1])], dim=1)  # (B, 4, H_hr, W_hr)

# Per-pixel matrix-vector: output[b, c, h, w] = sum_k M[b, c, k, h, w] * hr_aug[b, k, h, w]
# Implemented as einsum or explicit expand+sum
M_reshaped = M.view(B, 3, 4, H_hr, W_hr)
out = (M_reshaped * hr_aug.unsqueeze(1)).sum(dim=2)  # (B, 3, H_hr, W_hr)
```

### 4.6 Global residual (default)

```python
final = out + hr
```

**Default behavior**: zero-initialize the final conv in the bilateral-grid head. At init, `M ≈ 0` everywhere → `out ≈ 0` → `final ≈ hr`. The model starts as a no-op and learns deviations. This is the canonical stability device for restoration / correction networks and is the default for v1.

**Alternative to evaluate during tuning**: bias the final conv so each voxel predicts an identity 3×4 affine matrix at init, and skip the residual (`final = out` directly). Arithmetically equivalent at init (output = HR) but has different gradient dynamics. Tracked in §10 as an open question; v1 ships with the residual form above.

### 4.7 Why this handles the specified requirements

| Requirement | Mechanism |
|---|---|
| Preserve HR detail | Output is a per-pixel linear transform of HR. High-frequency structure is passed through unless the matrix actively blurs it (and nothing in training pressures it to). |
| Transfer LR colors | The transform is predicted from LR features; gradient flows from CC_HR loss into the LR encoder. |
| Robust to LR degradation | LR encoder runs at LR res with a large receptive field (pooled/downsampled body). Local bleed/compression averages out into coherent per-voxel transforms. |
| De-bleed using HR structure | LR encoder takes HR_downsampled as part of its input; can learn to associate LR colors with HR's structural regions. |
| Do not lean on HR colors | HR enters the final output ONLY through the per-pixel linear transform. The matrix coefficients are predicted from the LR branch. HR colors bias the output only if the model explicitly chooses to predict identity-like transforms in certain regions, and the CC_HR loss directly penalizes doing that in regions where colors should change. |
| Edge-aware (no bloom) | Bilateral slicing indexes the transform by HR luma. Colors do not leak across luma-level boundaries during slicing. This is the structural anti-bloom mechanism. |
| Scale-flexible | Architecture contains no scale-specific layers. The slicing step upsamples to whatever `H_hr, W_hr` the user provides. |
| ONNX exportable | All ops are opset 16+ primitives. No custom CUDA. No 5D `grid_sample`. |

---

## 5. Parameter Budget

Rough accounting, `F = 16`, `D = 8`, 2 residual blocks per stage, 3 downsampling stages:

- Stem (Conv 6→16, 3×3): ~880
- 2 × FastResBlock(16) at LR: ~9.3K
- Downsample Conv 16→16: ~2.3K
- 2 × FastResBlock(16): ~9.3K
- Downsample Conv 16→16: ~2.3K
- 2 × FastResBlock(16): ~9.3K
- Head Conv 16 → (12 × 8) = 96 channels, 1×1: ~1.5K
- Guidance (if learned): ~80

**Estimated total: ~35K parameters** at `F=16`. At `F=24`: ~75K. At `F=32`: ~130K.

Well within the DIS tier. Room to scale up if robustness to heavy degradations demands it.

---

## 6. Framework Integration

### 6.1 File layout

- **Arch:** `traiNNer/archs/bgcc_arch.py`
  - Registered under both `ARCH_REGISTRY` and `SPANDREL_REGISTRY` (following the pattern used by `dis_arch.py`, `span_arch.py`, etc.).
  - One or more variants registered as functions: `bgcc()` (default), possibly `bgcc_tiny()` / `bgcc_small()`.
- **Model wrapper:** likely a new `traiNNer/models/cc_model.py` exposing a `ColorCorrectionModel`, registered under `MODEL_REGISTRY`. It overrides `SRModel`'s data loading to feed two tensors into the network and supervises against a separate `gt` (the CC_HR). If the `SRModel` interface can be extended minimally with a flag, that's preferred over duplicating.
- **Dataset:** new `traiNNer/data/paired_cc_dataset.py` — loads triplets `(lr, hr, gt=cc_hr)`. Derived from `paired_image_dataset.py`. The `scale` config field determines the HR/LR ratio during cropping.
- **Config option struct:** extend `ReduxOptions` / dataset option structs in `traiNNer/utils/redux_options.py` to accept a third data root (`dataroot_hr` for the HR input), in addition to existing `dataroot_gt` and `dataroot_lq`.
- **Arch info:** add `BGCC` entry to `traiNNer/archs/arch_info.py` `ALL_ARCHS`. Add to `ARCHS_WITHOUT_FP16` if needed (unlikely — all DIS-style ops are FP16-safe).
- **Example config:** `options/train/BGCC/bgcc_2x.yml`.

### 6.2 Arch forward signature

```python
class BGCC(nn.Module):
    def forward(self, hr: Tensor, lr: Tensor) -> Tensor:
        ...
```

Note HR is first argument — matches "apply transform to HR" mental model. Model wrapper calls `net_g(hr=batch['hr'], lr=batch['lq'])`.

### 6.3 Dataset semantics

Three folders in the config:

```yaml
datasets:
  train:
    name: my_anime_cc_train
    type: PairedCCDataset
    dataroot_gt: /path/to/cc_hr   # color-corrected HR (supervision target)
    dataroot_hr: /path/to/hr      # original BD HR (input 1)
    dataroot_lq: /path/to/lr      # DVD/LD LR (input 2)
    scale: 2
    lq_size: 64                   # random crop patch size at LR res
    ...
```

At sampling time:
- Pick a random LR crop of `lq_size × lq_size`.
- Take the spatially-aligned HR crop at `(lq_size × scale) × (lq_size × scale)` from both `dataroot_hr` and `dataroot_gt`.
- All three crops share augmentation (hflip/rot must be applied identically).

### 6.4 Model wrapper

A thin extension over `SRModel`:

- Pulls `hr`, `lq`, `gt` from the batch.
- Calls `output = self.net_g(hr, lq)`.
- Loss: `L(output, gt)` using configured pixel/perceptual losses.
- Validation: same flow; metric comparison is `output` vs `gt`.

### 6.5 Registry naming

- Registered names (lowercase): `bgcc` (default), `bgcc_tiny` (smaller variant if needed).
- `ALL_ARCHS` entry: `{"names": ["BGCC"], "scales": [1, 2, 3, 4]}`.

---

## 7. Training

### 7.1 Loss

Color correction is a transform-accuracy problem. The primary losses:

- **L1 in RGB** — baseline pixel accuracy.
- **L1 in YCbCr with a chroma weighting** — weights Cb/Cr channels higher than Y to emphasize color fidelity over luma (since HR already has correct luma structure locally). Configurable ratio, starting point `0.25 * L1(Y) + 1.0 * L1(Cb) + 1.0 * L1(Cr)`.

Secondary / optional:

- **Gradient L1** — penalizes predicting transforms that alter HR's detail structure. `L1(∇output, ∇hr)` or similar. Keeps the model honest about "don't destroy HR structure."

Not recommended:
- LPIPS / perceptual losses — this problem isn't about perceptual realism; it's about accuracy against a ground-truth CC_HR. Perceptual losses could *bias* the model toward plausible-looking-but-wrong colors.
- GAN / discriminator losses — same reason.

Loss config in the YAML follows the existing traiNNer-redux loss registration pattern.

### 7.2 Schedule (starting point — to refine in implementation plan)

Borrowing from DIS / SPAN-style lightweight training:

```yaml
total_iter: 500000
milestones: [250000, 400000, 450000, 475000]
warmup_iter: -1
lr: !!float 5e-4
lq_size: 64      # 64x64 LR crop → 128x128 HR crop at 2x
batch_size_per_gpu: 16
accum_iter: 1
```

Dataset size / quality will dominate final training time. For a small curated dataset, fewer iters is fine. Add to `OFFICIAL_SETTINGS_FROMSCRATCH` once a solid recipe is found.

### 7.3 Dataset construction

The user's existing manual pipeline (wavelet-fix → Photoshop color-blend → PixTransform) produces the CC_HR supervision. Out-of-scope for this spec; the arch consumes whatever (LR, HR, CC_HR) triplets the user provides.

---

## 8. ONNX Export

- Target opset: **16+** (primary), verified up to opset 20.
- Runtime targets: TensorRT, ONNX Runtime (CPU + CUDA + DirectML).
- Dynamic shapes: `H_hr, W_hr, H_lr, W_lr` must all be dynamic in the exported graph. Input names: `hr`, `lr`. Output name: `output`.
- Dynamic scale: the exported model must run with arbitrary HR/LR size ratios (any `H_hr/H_lr`), not just 2x. Verify explicitly at 1x, 2x, 3x, 4x during export tests.
- `convert_to_onnx.py` needs a small patch to pass two input tensors when exporting BGCC. Implementation plan owns this.

---

## 9. Testing

### 9.1 Unit tests (`tests/test_archs/test_archs.py`)

- Shape test: random `hr` (B=2, 3, 128, 128) + `lr` (B=2, 3, 64, 64) → output `(2, 3, 128, 128)`.
- Scale-flexibility test: same model, test with (`hr`=256, `lr`=64) → output `(B, 3, 256, 256)`.
- FP16 forward test: no NaN/Inf.
- Parameter count test: confirm within budget.
- Gradient test: loss.backward() succeeds, grads non-zero.

### 9.2 ONNX export test (`tests/test_archs/` new test)

- Export with opset 16.
- Load into `onnxruntime` CPU.
- Compare output with PyTorch forward on random input — MSE should be < 1e-4.
- Repeat with dynamic shapes at 1x/2x/3x/4x.

### 9.3 Integration test (small synthetic data)

A tiny end-to-end test: 4 fake triplets, 50 iters, confirm loss decreases monotonically. Not required for merge but useful for the implementation plan.

### 9.4 Visual regression (manual, outside CI)

- Run the trained model on a reference DVD/BD clip provided by the user.
- Compare against wavelet-fix and against manual Photoshop+PixTransform output.
- Inspect for bloom, color accuracy, detail preservation.

---

## 10. Open Questions / Deferred to Implementation

1. **Grid resolution tuning.** `LR/8` vs `LR/16` vs fixed small (HDRNet original uses 16×16). Try two or three during tuning.
2. **Luma bin count `D`.** 4, 8, 16. HDRNet uses 8.
3. **Feature width `F`.** 16 / 24 / 32. Start at 16, scale up only if robustness insufficient.
4. **Learned vs fixed guidance head.** Both cheap; A/B during tuning.
5. **Identity-init vs residual-to-HR.** Two stability mechanisms; pick one.
6. **Whether to include gradient-preservation auxiliary loss.** Default to no; enable if HR-detail loss is observed during initial runs.
7. **Model wrapper: extend `SRModel` with a flag, or new `ColorCorrectionModel`.** Implementation plan decides after reading SRModel's full interface.
8. **Dataset augmentation.** Standard hflip/rot applied identically to LR/HR/CC_HR. No degradation augmentation on LR at train time (LR already has real degradation).

---

## 11. Out of Scope

- Video / temporal consistency (image-only for v1; a temporal variant could be layered on later).
- Automated CC_HR generation (dataset pipeline is user-owned).
- Adversarial training or perceptual losses (explicitly avoided — see §7.1).
- Handling unaligned HR/LR pairs (the design assumes spatial alignment up to scale).

---

## 12. Summary for Review

**What:** A ~35-150K parameter bilateral-guided color-correction architecture that takes HR (correct structure, wrong colors) and LR (target colors, degraded) and produces a color-corrected HR.

**How:** Predict a per-pixel 3×4 affine color transform at low resolution from an LR encoder that also sees HR structure (for de-bleeding). Apply the transform at HR resolution via bilateral slicing indexed by an HR-derived guidance channel. All ops are ONNX opset 16+ compatible.

**Why it beats existing methods:**
- Unlike mean+std / histogram / wavelet fix: learned, context-aware, non-blooming by construction.
- Unlike manual Photoshop+PixTransform: one-shot inference in <1s, fully consistent, no per-image training.
- Unlike generic dual-input CNNs: LR-side transform prediction is structurally robust to LR degradation and architecturally cannot copy HR colors.

**Performance targets:**
- Parameters: ~35-150K (DIS tier).
- Inference: sub-second at 1080p HR on consumer GPU.
- Scale: trained at 2x, supports any scale at inference.
- ONNX: opset 16+, TensorRT / ONNX Runtime compatible.
