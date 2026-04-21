# BGCC Color Correction Arch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement BGCC, a dual-input (HR+LR) bilateral guided color correction architecture in traiNNer-redux, including full framework integration (arch, dataset, model wrapper, ONNX export, example training config, and tests).

**Architecture:** The arch predicts a per-pixel 3×4 affine color transform from a low-res encoder that sees both LR and downsampled HR, stores the transforms in a compact bilateral grid (`D × LR/8 × LR/8`), and applies them at HR resolution via learned-guidance bilateral slicing. All ONNX opset-16 ops only.

**Tech Stack:** PyTorch 2.10+, msgspec-based config, traiNNer-redux registry system, pytest for testing, torch.onnx.export for ONNX.

**Spec:** [`docs/superpowers/specs/2026-04-20-bgcc-color-correction-arch-design.md`](../specs/2026-04-20-bgcc-color-correction-arch-design.md)

---

## File Structure

**Create:**
- `traiNNer/archs/bgcc_arch.py` — BGCC architecture + building blocks
- `traiNNer/data/paired_cc_dataset.py` — triplet (LR, HR, CC_HR) dataset class
- `traiNNer/models/bgcc_model.py` — BGCCModel wrapping SRModel with dual-input forward
- `tests/test_archs/test_bgcc_arch.py` — BGCC-specific tests (shapes, scale flex, ONNX, gradient)
- `options/train/BGCC/bgcc_2x.yml` — example training config

**Modify:**
- `traiNNer/archs/arch_info.py` — add BGCC entry to `ALL_ARCHS`
- `traiNNer/data/transforms.py` — add `paired_random_crop_triplet_vips` and `augment_vips_triplet`
- `traiNNer/models/sr_model.py` — add minimal `_run_net` hook method; replace direct `net(lq)` call sites with hook
- `traiNNer/models/__init__.py` — register `BGCCModel` in `build_model`
- `traiNNer/utils/redux_options.py` — add `dataroot_hr` to `DatasetOptions`; add `bgcc_mode` to `ReduxOptions`
- `convert_to_onnx.py` — handle dual-input export for BGCC
- `tests/test_archs/test_archs.py` — exclude BGCC from generic single-input parametrized test

---

## Task 1: BGCC building blocks — `FastResBlock`, `GuidanceHead`, `BilateralSlicer`

**Files:**
- Create: `traiNNer/archs/bgcc_arch.py`
- Create: `tests/test_archs/test_bgcc_arch.py`

- [ ] **Step 1.1: Create the new test file with failing tests for the building blocks**

Write `tests/test_archs/test_bgcc_arch.py`:

```python
import pytest
import torch
from torch import Tensor


def test_fast_res_block_preserves_shape():
    from traiNNer.archs.bgcc_arch import FastResBlock

    block = FastResBlock(channels=16)
    x = torch.randn(2, 16, 32, 32)
    y = block(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_guidance_head_outputs_single_channel_in_range():
    from traiNNer.archs.bgcc_arch import GuidanceHead

    head = GuidanceHead(hidden=8)
    x = torch.randn(2, 3, 64, 64).clamp(0, 1)
    g = head(x)
    assert g.shape == (2, 1, 64, 64)
    assert g.min() >= 0.0 and g.max() <= 1.0


def test_bilateral_slicer_output_shape():
    from traiNNer.archs.bgcc_arch import BilateralSlicer

    slicer = BilateralSlicer()
    # grid: (B=2, 12 coeffs, D=8 bins, H'=4, W'=4), guidance: (B=2, 1, H_hr=32, W_hr=32)
    grid = torch.randn(2, 12, 8, 4, 4)
    guidance = torch.rand(2, 1, 32, 32)
    m = slicer(grid, guidance)
    assert m.shape == (2, 12, 32, 32)


def test_bilateral_slicer_identity_grid_gives_identity_matrices():
    """If every voxel of the grid holds the same identity 3x4 matrix,
    every output pixel should be the identity 3x4 matrix too."""
    from traiNNer.archs.bgcc_arch import BilateralSlicer

    slicer = BilateralSlicer()
    identity = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
    grid = identity.view(1, 12, 1, 1, 1).expand(2, 12, 8, 4, 4).contiguous()
    guidance = torch.rand(2, 1, 32, 32)
    m = slicer(grid, guidance)  # (2, 12, 32, 32)
    expected = identity.view(1, 12, 1, 1).expand(2, 12, 32, 32)
    assert torch.allclose(m, expected, atol=1e-5)
```

- [ ] **Step 1.2: Run the tests and verify they fail**

Run: `pytest tests/test_archs/test_bgcc_arch.py -v`

Expected: All four tests FAIL with `ImportError` / `ModuleNotFoundError` for `traiNNer.archs.bgcc_arch`.

- [ ] **Step 1.3: Create `bgcc_arch.py` with the three building blocks**

Write `traiNNer/archs/bgcc_arch.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY


class FastResBlock(nn.Module):
    """Residual block: Conv 3x3 -> PReLU -> Conv 3x3 -> add."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y


class GuidanceHead(nn.Module):
    """Tiny per-pixel MLP on HR RGB producing a learned 1-channel guidance in [0, 1].

    Structure: Conv 3->H 1x1 -> PReLU -> Conv H->1 1x1 -> sigmoid.
    Removes the fixed-luma inductive bias; model learns what projection of RGB
    best indexes the bilateral grid.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden, 1, bias=True)
        self.act = nn.PReLU(hidden)
        self.conv2 = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, hr: Tensor) -> Tensor:
        y = self.act(self.conv1(hr))
        y = self.conv2(y)
        return torch.sigmoid(y)


class BilateralSlicer(nn.Module):
    """Manual trilinear slicing of a bilateral grid, ONNX-opset-16 friendly.

    Inputs:
        grid: (B, C=12, D, H', W') of transform coefficients.
        guidance: (B, 1, H_hr, W_hr) in [0, 1].

    Output:
        M: (B, 12, H_hr, W_hr) — per-HR-pixel transform coefficients.

    Algorithm:
        1. Spatially upsample every D-bin slice of the grid to HR resolution
           using bilinear interpolate (done as one interpolate call via reshape).
        2. Compute bin_lo, bin_hi from guidance * (D - 1) with floor + clamp.
        3. Gather the two adjacent bins per HR pixel and linearly interpolate
           by the fractional guidance weight.
    """

    def forward(self, grid: Tensor, guidance: Tensor) -> Tensor:
        b, c, d, h_grid, w_grid = grid.shape
        _, _, h_hr, w_hr = guidance.shape

        # Spatial upsample all D bins at once.
        # Reshape (B, C, D, H', W') -> (B, C*D, H', W') for a single interpolate call.
        grid_flat = grid.reshape(b, c * d, h_grid, w_grid)
        grid_up = F.interpolate(
            grid_flat, size=(h_hr, w_hr), mode="bilinear", align_corners=False
        )
        # (B, C, D, H_hr, W_hr)
        grid_up = grid_up.reshape(b, c, d, h_hr, w_hr)

        # Map guidance to continuous bin coordinate in [0, D-1].
        bin_f = guidance.squeeze(1) * (d - 1)   # (B, H_hr, W_hr)
        bin_lo = bin_f.floor().clamp(0, d - 1).long()
        bin_hi = (bin_lo + 1).clamp(max=d - 1)
        w_hi = (bin_f - bin_lo.float()).unsqueeze(1)   # (B, 1, H_hr, W_hr)
        w_lo = 1.0 - w_hi

        # Gather along bin axis.
        # idx shape must match grid_up except on the gather axis:
        # grid_up is (B, C, D, H_hr, W_hr); idx must be (B, C, 1, H_hr, W_hr).
        idx_lo = bin_lo.unsqueeze(1).unsqueeze(1).expand(b, c, 1, h_hr, w_hr)
        idx_hi = bin_hi.unsqueeze(1).unsqueeze(1).expand(b, c, 1, h_hr, w_hr)
        m_lo = torch.gather(grid_up, 2, idx_lo).squeeze(2)   # (B, C, H_hr, W_hr)
        m_hi = torch.gather(grid_up, 2, idx_hi).squeeze(2)

        return w_lo * m_lo + w_hi * m_hi


# BGCC main module is added in a later task.
```

- [ ] **Step 1.4: Run the tests and verify they pass**

Run: `pytest tests/test_archs/test_bgcc_arch.py -v`

Expected: All four tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add traiNNer/archs/bgcc_arch.py tests/test_archs/test_bgcc_arch.py
git commit -m "BGCC: add FastResBlock, GuidanceHead, BilateralSlicer building blocks"
```

---

## Task 2: BGCC encoder and main module

**Files:**
- Modify: `traiNNer/archs/bgcc_arch.py`
- Modify: `tests/test_archs/test_bgcc_arch.py`

- [ ] **Step 2.1: Add failing tests for `BGCC.forward` shape and scale flexibility**

Append to `tests/test_archs/test_bgcc_arch.py`:

```python
def test_bgcc_forward_shape_2x():
    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=32, d=8)
    hr = torch.randn(2, 3, 128, 128)
    lr = torch.randn(2, 3, 64, 64)
    out = model(hr, lr)
    assert out.shape == hr.shape


def test_bgcc_forward_shape_flexible_scale():
    """Model trained at 2x should still run at 3x at inference without code changes."""
    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=32, d=8)
    hr = torch.randn(1, 3, 192, 192)
    lr = torch.randn(1, 3, 64, 64)
    out = model(hr, lr)
    assert out.shape == hr.shape


def test_bgcc_initial_output_matches_hr_within_tolerance():
    """Zero-init + residual => output should equal hr at init (zero training)."""
    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=32, d=8).eval()
    hr = torch.randn(1, 3, 64, 64)
    lr = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(hr, lr)
    assert torch.allclose(out, hr, atol=1e-5)


def test_bgcc_gradients_flow_end_to_end():
    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=16, d=4)
    hr = torch.randn(1, 3, 64, 64, requires_grad=False)
    lr = torch.randn(1, 3, 32, 32, requires_grad=False)
    target = torch.randn(1, 3, 64, 64)
    out = model(hr, lr)
    loss = (out - target).abs().mean()
    loss.backward()
    # Every trainable parameter should have received a gradient.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
```

- [ ] **Step 2.2: Run the tests and verify they fail**

Run: `pytest tests/test_archs/test_bgcc_arch.py -v -k bgcc_forward or bgcc_initial or bgcc_gradients`

Expected: All four new tests FAIL with `ImportError` (no `bgcc` function exported yet).

- [ ] **Step 2.3: Implement the `BGCC` class and `bgcc()` factory**

Append to `traiNNer/archs/bgcc_arch.py`:

```python
class BGCC(nn.Module):
    """Bilateral Guided Color Correction.

    Takes HR (correct structure, wrong colors) and LR (target colors, degraded)
    and produces an HR image with HR's structure and LR's colors.

    Args:
        feat: Feature width of the encoder (default 32).
        d: Bilateral bin count (default 8).
        n_blocks_per_stage: Number of FastResBlocks at each encoder stage (default 2).
        guidance_hidden: Hidden width of the guidance MLP (default 8).
    """

    def __init__(
        self,
        feat: int = 32,
        d: int = 8,
        n_blocks_per_stage: int = 2,
        guidance_hidden: int = 8,
    ) -> None:
        super().__init__()
        self.feat = feat
        self.d = d
        self.coeffs_per_voxel = 12  # 3 output channels * 4 input components (RGB+1)

        # Encoder: takes (LR RGB || HR_downsampled RGB) = 6 channels
        self.stem = nn.Conv2d(6, feat, 3, padding=1, bias=True)
        self.stem_act = nn.PReLU(feat)

        self.stage0 = nn.Sequential(
            *[FastResBlock(feat) for _ in range(n_blocks_per_stage)]
        )
        self.down1 = nn.Conv2d(feat, feat, 3, stride=2, padding=1, bias=False)
        self.stage1 = nn.Sequential(
            *[FastResBlock(feat) for _ in range(n_blocks_per_stage)]
        )
        self.down2 = nn.Conv2d(feat, feat, 3, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(
            *[FastResBlock(feat) for _ in range(n_blocks_per_stage)]
        )
        self.down3 = nn.Conv2d(feat, feat, 3, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(
            *[FastResBlock(feat) for _ in range(n_blocks_per_stage)]
        )

        # Grid head: produces (12 * D) channels at LR/8 spatial resolution.
        self.grid_head = nn.Conv2d(feat, self.coeffs_per_voxel * d, 1, bias=True)

        # Zero-init the grid head so output = HR + 0 at init.
        nn.init.zeros_(self.grid_head.weight)
        if self.grid_head.bias is not None:
            nn.init.zeros_(self.grid_head.bias)

        self.guidance = GuidanceHead(hidden=guidance_hidden)
        self.slicer = BilateralSlicer()

    def forward(self, hr: Tensor, lr: Tensor) -> Tensor:
        b, _, h_hr, w_hr = hr.shape
        _, _, h_lr, w_lr = lr.shape

        # Downsample HR to LR size and concat with LR along channel dim.
        hr_ds = F.interpolate(
            hr, size=(h_lr, w_lr), mode="bilinear", align_corners=False
        )
        x = torch.cat([lr, hr_ds], dim=1)   # (B, 6, H_lr, W_lr)

        # Encoder
        x = self.stem_act(self.stem(x))
        x = self.stage0(x)
        x = self.stage1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.stage3(self.down3(x))

        # Predict the bilateral grid: (B, 12*D, H_lr/8, W_lr/8)
        grid_flat = self.grid_head(x)
        # Reshape to (B, 12, D, H', W')
        grid = grid_flat.reshape(b, self.coeffs_per_voxel, self.d, *grid_flat.shape[-2:])

        # Learned 1-ch guidance from HR
        guidance = self.guidance(hr)   # (B, 1, H_hr, W_hr)

        # Slice: per-HR-pixel 3x4 affine matrix coefficients.
        m = self.slicer(grid, guidance)   # (B, 12, H_hr, W_hr)

        # Apply per-pixel affine to HR.
        # M reshaped to (B, 3, 4, H_hr, W_hr); [R,G,B,1] to (B, 4, H_hr, W_hr).
        m = m.view(b, 3, 4, h_hr, w_hr)
        hr_aug = torch.cat([hr, torch.ones_like(hr[:, :1])], dim=1)   # (B, 4, H, W)
        out = (m * hr_aug.unsqueeze(1)).sum(dim=2)   # (B, 3, H_hr, W_hr)

        # Residual against HR for init stability (grid_head is zero-init -> out starts 0).
        return out + hr


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def bgcc(
    feat: int = 32,
    d: int = 8,
    n_blocks_per_stage: int = 2,
    guidance_hidden: int = 8,
    scale: int = 2,   # accepted for framework compatibility; not used architecturally
) -> BGCC:
    return BGCC(
        feat=feat,
        d=d,
        n_blocks_per_stage=n_blocks_per_stage,
        guidance_hidden=guidance_hidden,
    )


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def bgcc_tiny(
    feat: int = 16,
    d: int = 8,
    n_blocks_per_stage: int = 1,
    guidance_hidden: int = 8,
    scale: int = 2,
) -> BGCC:
    return BGCC(
        feat=feat,
        d=d,
        n_blocks_per_stage=n_blocks_per_stage,
        guidance_hidden=guidance_hidden,
    )
```

- [ ] **Step 2.4: Run the tests and verify they pass**

Run: `pytest tests/test_archs/test_bgcc_arch.py -v`

Expected: All eight tests PASS.

- [ ] **Step 2.5: Verify parameter count is within budget**

Run:

```bash
python -c "from traiNNer.archs.bgcc_arch import bgcc; m = bgcc(feat=32, d=8); print(sum(p.numel() for p in m.parameters()))"
```

Expected: Output between 100_000 and 400_000 (target ~180K per spec §5).

- [ ] **Step 2.6: Commit**

```bash
git add traiNNer/archs/bgcc_arch.py tests/test_archs/test_bgcc_arch.py
git commit -m "BGCC: add main arch module with zero-init residual and two factory variants"
```

---

## Task 3: Arch info registration

**Files:**
- Modify: `traiNNer/archs/arch_info.py`

- [ ] **Step 3.1: Add BGCC to the `ALL_ARCHS` list**

Open `traiNNer/archs/arch_info.py`. Find the closing `]` of `ALL_ARCHS` near the bottom of the list. Add before it:

```python
    {
        "names": ["BGCC", "BGCC_tiny"],
        "scales": ALL_SCALES,
    },
```

- [ ] **Step 3.2: Confirm the arch is auto-discovered**

Run:

```bash
python -c "from traiNNer.archs import ARCH_REGISTRY; print('bgcc' in ARCH_REGISTRY, 'bgcc_tiny' in ARCH_REGISTRY)"
```

Expected: `True True`.

- [ ] **Step 3.3: Commit**

```bash
git add traiNNer/archs/arch_info.py
git commit -m "BGCC: register in arch_info ALL_ARCHS"
```

---

## Task 4: Exclude BGCC from the generic single-input parametrized test

**Files:**
- Modify: `tests/test_archs/test_archs.py`

The existing `test_archs.py` calls every registered arch with a single `lq` tensor and asserts `output = model(lq)`. BGCC takes two inputs. Adding BGCC to the `EXCLUDE_ARCHS` set prevents spurious failures and defers BGCC testing to `test_bgcc_arch.py`.

- [ ] **Step 4.1: Update `EXCLUDE_ARCHS`**

Open `tests/test_archs/test_archs.py`. In the `EXCLUDE_ARCHS` set (around lines 16-31), add two entries alphabetically:

```python
EXCLUDE_ARCHS = {
    "autoencoder",
    "bgcc",
    "bgcc_tiny",
    "dunet",
    # ... rest unchanged ...
}
```

- [ ] **Step 4.2: Run the full arch test to confirm BGCC is excluded and nothing else breaks**

Run: `pytest tests/test_archs/test_archs.py -v -x --tb=short -q 2>&1 | head -50`

Expected: Tests run without any `bgcc` / `bgcc_tiny` tests appearing. Other arch tests pass or have their pre-existing status (no new failures attributable to BGCC).

- [ ] **Step 4.3: Commit**

```bash
git add tests/test_archs/test_archs.py
git commit -m "BGCC: exclude dual-input arch from generic single-input parametrized test"
```

---

## Task 5: Triplet crop and augment helpers in `transforms.py`

**Files:**
- Modify: `traiNNer/data/transforms.py`
- Create: `tests/test_data/test_triplet_transforms.py`

- [ ] **Step 5.1: Inspect existing `paired_random_crop_vips` and `augment_vips_pair`**

Open `traiNNer/data/transforms.py`. Locate `paired_random_crop_vips` (around line 263) and `augment_vips_pair` (around line 388). Confirm their signatures match what the explore agent reported (see spec §6.3 note on alignment).

- [ ] **Step 5.2: Write failing tests for triplet helpers**

Create `tests/test_data/test_triplet_transforms.py`:

```python
import numpy as np
import pyvips
import pytest


def _make_vips_rgb(h: int, w: int, seed: int) -> pyvips.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return pyvips.Image.new_from_array(arr)


def test_triplet_random_crop_returns_aligned_regions():
    from traiNNer.data.transforms import paired_random_crop_triplet_vips

    # LR 32x32, HR 64x64 (2x), CC_HR 64x64 (2x). All same scene => crops must align.
    img_lr = _make_vips_rgb(32, 32, seed=0)
    img_hr = _make_vips_rgb(64, 64, seed=1)
    img_cc = _make_vips_rgb(64, 64, seed=2)

    arr_lr, arr_hr, arr_cc = paired_random_crop_triplet_vips(
        img_hr_ref=img_hr,
        img_cc_hr=img_cc,
        img_lq=img_lr,
        gt_patch_size=32,   # HR/CC_HR patch size
        scale=2,
    )
    assert arr_lr.shape == (16, 16, 3)   # lq patch = gt/scale
    assert arr_hr.shape == (32, 32, 3)
    assert arr_cc.shape == (32, 32, 3)


def test_triplet_augment_applies_identical_flip_rot_to_all_three():
    from traiNNer.data.transforms import augment_vips_triplet

    img_lr = _make_vips_rgb(32, 32, seed=0)
    img_hr = _make_vips_rgb(64, 64, seed=1)
    img_cc = _make_vips_rgb(64, 64, seed=2)

    # Force hflip only.
    out_hr, out_cc, out_lr = augment_vips_triplet(
        (img_hr, img_cc, img_lr),
        hflip=True,
        vflip=False,
        rot90=False,
        force_hflip=True,
        force_vflip=False,
        force_rot90=False,
    )
    # Compare numpy: flipped HR equals manually-flipped source.
    expected_hr = np.flip(img_hr.numpy(), axis=1)
    expected_cc = np.flip(img_cc.numpy(), axis=1)
    expected_lr = np.flip(img_lr.numpy(), axis=1)
    assert np.array_equal(out_hr.numpy(), expected_hr)
    assert np.array_equal(out_cc.numpy(), expected_cc)
    assert np.array_equal(out_lr.numpy(), expected_lr)
```

- [ ] **Step 5.3: Run the tests and verify they fail**

Run: `pytest tests/test_data/test_triplet_transforms.py -v`

Expected: Both tests FAIL with `ImportError` for `paired_random_crop_triplet_vips` / `augment_vips_triplet`.

- [ ] **Step 5.4: Implement `paired_random_crop_triplet_vips`**

Open `traiNNer/data/transforms.py`. After the existing `paired_random_crop_vips` function, add:

```python
def paired_random_crop_triplet_vips(
    img_hr_ref: "pyvips.Image",
    img_cc_hr: "pyvips.Image",
    img_lq: "pyvips.Image",
    gt_patch_size: int,
    scale: int,
    x: int | None = None,
    y: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triplet-aligned random crop for BGCC training.

    HR reference and CC_HR target are both at HR resolution; LQ is at LR resolution.
    All three crops come from the same spatial location (scaled appropriately).

    Args:
        img_hr_ref: HR reference image (color-incorrect, HR resolution).
        img_cc_hr: CC_HR target image (color-correct, HR resolution).
        img_lq: LQ image (degraded, LR resolution).
        gt_patch_size: Patch size for HR crops (cc_hr and hr_ref).
        scale: HR/LR ratio.
        x, y: Optional deterministic top-left in LR coordinates.

    Returns:
        (hr_ref_np, cc_hr_np, lq_np) as HWC uint8 numpy arrays.
    """
    h_lq, w_lq = img_lq.height, img_lq.width
    lq_patch = gt_patch_size // scale

    if x is None:
        x = int(np.random.randint(0, max(w_lq - lq_patch + 1, 1)))
    if y is None:
        y = int(np.random.randint(0, max(h_lq - lq_patch + 1, 1)))

    lq_crop = img_lq.crop(x, y, lq_patch, lq_patch)
    hr_x, hr_y = x * scale, y * scale
    hr_ref_crop = img_hr_ref.crop(hr_x, hr_y, gt_patch_size, gt_patch_size)
    cc_hr_crop = img_cc_hr.crop(hr_x, hr_y, gt_patch_size, gt_patch_size)

    return hr_ref_crop.numpy(), cc_hr_crop.numpy(), lq_crop.numpy()
```

- [ ] **Step 5.5: Implement `augment_vips_triplet`**

After `augment_vips_pair` in `traiNNer/data/transforms.py`, add:

```python
def augment_vips_triplet(
    imgs: tuple["pyvips.Image", "pyvips.Image", "pyvips.Image"],
    hflip: bool = True,
    vflip: bool = True,
    rot90: bool = True,
    force_hflip: bool | None = None,
    force_vflip: bool | None = None,
    force_rot90: bool | None = None,
) -> tuple["pyvips.Image", "pyvips.Image", "pyvips.Image"]:
    """Apply the same random flip/rotation to all three images.

    Thin wrapper that calls `augment_vips_pair` twice with identical force_*
    flags derived from a single random draw, to keep augmentation consistent
    across (hr_ref, cc_hr, lq).
    """
    import random

    do_hflip = force_hflip if force_hflip is not None else (hflip and random.random() < 0.5)
    do_vflip = force_vflip if force_vflip is not None else (vflip and random.random() < 0.5)
    do_rot90 = force_rot90 if force_rot90 is not None else (rot90 and random.random() < 0.5)

    a, b = augment_vips_pair(
        (imgs[0], imgs[1]),
        hflip=hflip, vflip=vflip, rot90=rot90,
        force_hflip=do_hflip, force_vflip=do_vflip, force_rot90=do_rot90,
    )
    _, c = augment_vips_pair(
        (imgs[0], imgs[2]),
        hflip=hflip, vflip=vflip, rot90=rot90,
        force_hflip=do_hflip, force_vflip=do_vflip, force_rot90=do_rot90,
    )
    return a, b, c
```

- [ ] **Step 5.6: Run the tests and verify they pass**

Run: `pytest tests/test_data/test_triplet_transforms.py -v`

Expected: Both tests PASS.

- [ ] **Step 5.7: Commit**

```bash
git add traiNNer/data/transforms.py tests/test_data/test_triplet_transforms.py
git commit -m "BGCC: add paired_random_crop_triplet_vips and augment_vips_triplet"
```

---

## Task 6: Extend `DatasetOptions` with `dataroot_hr`

**Files:**
- Modify: `traiNNer/utils/redux_options.py`

- [ ] **Step 6.1: Add `dataroot_hr` field to `DatasetOptions`**

Open `traiNNer/utils/redux_options.py`. Find the `DatasetOptions` struct (around lines 17-143). Near the existing `dataroot_lq` / `dataroot_gt` fields, add:

```python
    dataroot_hr: str | list[str] | None = None
    """Optional data root for HR reference input (used by BGCC color correction).

    When set, the dataset is expected to produce triplets (lq, hr_ref, gt) where
    gt is the CC_HR supervision target. Only the PairedCCDataset uses this field.
    """
```

- [ ] **Step 6.2: Confirm the struct still imports and parses**

Run:

```bash
python -c "from traiNNer.utils.redux_options import DatasetOptions; print(DatasetOptions.__struct_fields__)"
```

Expected: output includes `'dataroot_hr'` among the listed fields.

- [ ] **Step 6.3: Commit**

```bash
git add traiNNer/utils/redux_options.py
git commit -m "BGCC: add dataroot_hr field to DatasetOptions"
```

---

## Task 7: `PairedCCDataset` — the triplet dataset class

**Files:**
- Create: `traiNNer/data/paired_cc_dataset.py`
- Create: `tests/test_data/test_paired_cc_dataset.py`

- [ ] **Step 7.1: Write a failing dataset test**

Create `tests/test_data/test_paired_cc_dataset.py`:

```python
import os
import numpy as np
import pytest
import pyvips
from PIL import Image


@pytest.fixture
def triplet_dataset_root(tmp_path):
    """Create 3 folders each with 2 small PNG images sharing basenames."""
    roots = {}
    for key, (h, w) in [
        ("lr", (32, 32)),
        ("hr", (64, 64)),
        ("cc_hr", (64, 64)),
    ]:
        d = tmp_path / key
        d.mkdir()
        for i in range(2):
            arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"img{i}.png")
        roots[key] = str(d)
    return roots


def test_paired_cc_dataset_produces_triplets(triplet_dataset_root):
    from traiNNer.data.paired_cc_dataset import PairedCCDataset
    from traiNNer.utils.redux_options import DatasetOptions

    opt = DatasetOptions(
        name="test_cc",
        type="PairedCCDataset",
        dataroot_gt=[triplet_dataset_root["cc_hr"]],
        dataroot_hr=triplet_dataset_root["hr"],
        dataroot_lq=[triplet_dataset_root["lr"]],
        scale=2,
        phase="train",
        gt_size=32,
        use_hflip=True,
        use_rot=True,
        io_backend={"type": "disk"},
    )
    ds = PairedCCDataset(opt)
    sample = ds[0]
    assert "lq" in sample
    assert "hr" in sample
    assert "gt" in sample
    # Shapes: lq = (3, lq_size, lq_size); hr and gt = (3, gt_size, gt_size).
    assert sample["lq"].shape == (3, 16, 16)
    assert sample["hr"].shape == (3, 32, 32)
    assert sample["gt"].shape == (3, 32, 32)
```

- [ ] **Step 7.2: Run the test and verify it fails**

Run: `pytest tests/test_data/test_paired_cc_dataset.py -v`

Expected: FAIL with `ImportError` for `PairedCCDataset`.

- [ ] **Step 7.3: Implement `PairedCCDataset`**

Create `traiNNer/data/paired_cc_dataset.py`:

```python
import os
from typing import Any

import numpy as np

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_util import paired_paths_from_folder
from traiNNer.data.transforms import (
    augment_vips_triplet,
    paired_random_crop_triplet_vips,
)
from traiNNer.utils import FileClient, img2tensor
from traiNNer.utils.img_util import img2rgb, vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class PairedCCDataset(BaseDataset):
    """Triplet dataset for BGCC color-correction training.

    Reads three folders:
        - dataroot_lq: LR images (DVD/LD, degraded, target colors).
        - dataroot_hr: HR reference images (BD, sharp structure, wrong colors).
        - dataroot_gt: CC_HR target images (manually color-corrected HR; supervision target).

    All three folders must share image basenames.

    Returns dict:
        {"lq": LR tensor, "hr": HR_ref tensor, "gt": CC_HR tensor,
         "lq_path": ..., "hr_path": ..., "gt_path": ...}
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.mean = opt.mean
        self.std = opt.std

        assert isinstance(opt.dataroot_lq, list), "dataroot_lq must be a list"
        assert isinstance(opt.dataroot_gt, list), "dataroot_gt must be a list"
        assert opt.dataroot_hr is not None, (
            "dataroot_hr is required for PairedCCDataset"
        )

        self.lq_folder = opt.dataroot_lq
        self.gt_folder = opt.dataroot_gt
        self.hr_folder = (
            opt.dataroot_hr if isinstance(opt.dataroot_hr, list) else [opt.dataroot_hr]
        )
        self.filename_tmpl = opt.filename_tmpl

        # Use the existing LQ/GT pairing logic, then pair HR by basename against LQ.
        lq_gt_paths = paired_paths_from_folder(
            (self.lq_folder, self.gt_folder), ("lq", "gt"), self.filename_tmpl
        )
        self.paths: list[dict[str, str]] = []
        for entry in lq_gt_paths:
            lq_path = entry["lq_path"]
            gt_path = entry["gt_path"]
            basename = os.path.basename(lq_path)
            hr_path = None
            for root in self.hr_folder:
                candidate = os.path.join(root, basename)
                if os.path.exists(candidate):
                    hr_path = candidate
                    break
            if hr_path is None:
                raise FileNotFoundError(
                    f"No HR reference found for {basename} in {self.hr_folder}"
                )
            self.paths.append(
                {"lq_path": lq_path, "gt_path": gt_path, "hr_path": hr_path}
            )

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt.scale
        assert scale is not None

        entry = self.paths[index]
        lq_path, gt_path, hr_path = entry["lq_path"], entry["gt_path"], entry["hr_path"]

        vips_lq = vipsimfrompath(lq_path)
        vips_hr = vipsimfrompath(hr_path)
        vips_gt = vipsimfrompath(gt_path)

        if self.opt.phase == "train":
            assert self.opt.gt_size is not None
            assert self.opt.use_hflip is not None
            assert self.opt.use_rot is not None

            vips_hr, vips_gt, vips_lq = augment_vips_triplet(
                (vips_hr, vips_gt, vips_lq),
                self.opt.use_hflip,
                self.opt.use_rot,
                self.opt.use_rot,
            )

            img_hr, img_gt, img_lq = paired_random_crop_triplet_vips(
                img_hr_ref=vips_hr,
                img_cc_hr=vips_gt,
                img_lq=vips_lq,
                gt_patch_size=self.opt.gt_size,
                scale=scale,
            )
        else:
            img_lq = img2rgb(vips_lq.numpy())
            img_hr = img2rgb(vips_hr.numpy())
            img_gt = img2rgb(vips_gt.numpy())
            # Ensure HR/GT are trimmed to LR*scale at eval time.
            h_trim = img_lq.shape[0] * scale
            w_trim = img_lq.shape[1] * scale
            img_hr = img_hr[:h_trim, :w_trim, :]
            img_gt = img_gt[:h_trim, :w_trim, :]

        lq_t = img2tensor(img_lq, float32=True, from_bgr=False)
        hr_t = img2tensor(img_hr, float32=True, from_bgr=False)
        gt_t = img2tensor(img_gt, float32=True, from_bgr=False)

        if self.mean is not None and self.std is not None:
            from torchvision.transforms.functional import normalize
            normalize(lq_t, self.mean, self.std, inplace=True)
            normalize(hr_t, self.mean, self.std, inplace=True)
            normalize(gt_t, self.mean, self.std, inplace=True)

        return {
            "lq": lq_t,
            "hr": hr_t,
            "gt": gt_t,
            "lq_path": lq_path,
            "hr_path": hr_path,
            "gt_path": gt_path,
        }

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "triplet (lq, hr_ref, cc_hr) images"
```

- [ ] **Step 7.4: Run the dataset test and verify it passes**

Run: `pytest tests/test_data/test_paired_cc_dataset.py -v`

Expected: PASS.

- [ ] **Step 7.5: Commit**

```bash
git add traiNNer/data/paired_cc_dataset.py tests/test_data/test_paired_cc_dataset.py
git commit -m "BGCC: add PairedCCDataset loading (lq, hr_ref, cc_hr) triplets"
```

---

## Task 8: Add `_run_net` hook to `SRModel`

Rationale: BGCC needs to call `net_g(hr, lq)` instead of `net_g(lq)`. The minimal-risk way is to factor the forward-call into a single overridable helper that non-BGCC code behavior-preserves.

**Files:**
- Modify: `traiNNer/models/sr_model.py`

- [ ] **Step 8.1: Locate every `net_g(lq)` and `net_g_ema(lq)` call site**

Run: `grep -n "self\.net_g(" traiNNer/models/sr_model.py` and `grep -n "net_g_ema(" traiNNer/models/sr_model.py`.

Expected call sites (line numbers will vary slightly):
- `output = self.net_g(lq)` in `optimize_parameters`
- `tmp_out = net(lq)` in `test` (where `net` is chosen between `net_g` and `net_g_ema`)

Note the line numbers for the next step.

- [ ] **Step 8.2: Add `_run_net` helper method to `SRModel`**

In `traiNNer/models/sr_model.py`, near the top of the `SRModel` class body (before `feed_data`), add:

```python
    def _run_net(self, net: nn.Module, lq: Tensor) -> Tensor:
        """Single forward-call point. Overridable by subclasses that need
        to pass extra inputs (e.g., BGCCModel passes self.hr as first arg)."""
        return net(lq)
```

- [ ] **Step 8.3: Replace `net(lq)` call sites with `self._run_net(net, lq)`**

For each call site identified in step 8.1:

- `output = self.net_g(lq)` → `output = self._run_net(self.net_g, lq)`
- `tmp_out = net(lq)` → `tmp_out = self._run_net(net, lq)`

Leave `self.net_g_teacher(self.lq)` alone (teacher is not dual-input for BGCC scope).

- [ ] **Step 8.4: Run the existing SR model tests to confirm no regression**

Run: `pytest tests/test_models/ -v -x`

Expected: All previously-passing tests continue to pass.

- [ ] **Step 8.5: Commit**

```bash
git add traiNNer/models/sr_model.py
git commit -m "SRModel: factor net_g forward into overridable _run_net hook (no behavior change)"
```

---

## Task 9: `BGCCModel` — SRModel subclass with dual-input forward

**Files:**
- Create: `traiNNer/models/bgcc_model.py`
- Create: `tests/test_models/test_bgcc_model.py`

- [ ] **Step 9.1: Write a failing test exercising BGCCModel's feed + forward**

Create `tests/test_models/test_bgcc_model.py`:

```python
import pytest
import torch


def _make_dummy_opt():
    """Minimal ReduxOptions-like object with the fields SRModel/BGCCModel touch.

    NOTE: this fixture is for Task 9 (BGCCModel unit tests) — it does NOT set
    bgcc_mode because that field is added in Task 10 and is only used by
    build_model's routing logic. We instantiate BGCCModel directly here.
    If msgspec raises on unknown or missing fields, adjust the constructor
    call to satisfy the actual ReduxOptions struct shape; the test intent
    (BGCCModel feed_data + _run_net exercise) stays the same.
    """
    from traiNNer.utils.redux_options import ReduxOptions

    opt = ReduxOptions(
        name="bgcc_unit",
        model_type="bgcc",
        scale=2,
        num_gpu=0,
        manual_seed=0,
        use_amp=False,
        use_channels_last=False,
        amp_bf16=False,
        use_compile=False,
        high_order_degradation=False,
        network_g={"type": "bgcc", "feat": 16, "d": 4, "n_blocks_per_stage": 1},
        path={},
        datasets={},
        train={},
        logger={},
        dist_params={},
    )
    return opt


def test_bgcc_model_feed_data_pulls_hr():
    from traiNNer.models.bgcc_model import BGCCModel

    opt = _make_dummy_opt()
    model = BGCCModel(opt)
    batch = {
        "lq": torch.randn(1, 3, 32, 32),
        "hr": torch.randn(1, 3, 64, 64),
        "gt": torch.randn(1, 3, 64, 64),
    }
    model.feed_data(batch)
    assert model.hr is not None
    assert model.hr.shape == (1, 3, 64, 64)
    assert model.lq.shape == (1, 3, 32, 32)


def test_bgcc_model_forward_passes_both_inputs():
    from traiNNer.models.bgcc_model import BGCCModel

    opt = _make_dummy_opt()
    model = BGCCModel(opt)
    batch = {
        "lq": torch.randn(1, 3, 32, 32),
        "hr": torch.randn(1, 3, 64, 64),
        "gt": torch.randn(1, 3, 64, 64),
    }
    model.feed_data(batch)
    out = model._run_net(model.net_g, model.lq)
    assert out.shape == (1, 3, 64, 64)
```

If the `ReduxOptions` signature in step 9.1 does not match the actual fields (msgspec will raise on unknown fields), adjust the constructor call to the actual required fields — but DO NOT change the test intent (it exercises `feed_data` and `_run_net` with `self.hr` set).

- [ ] **Step 9.2: Run the test and verify it fails**

Run: `pytest tests/test_models/test_bgcc_model.py -v`

Expected: FAIL with `ImportError` for `BGCCModel`.

- [ ] **Step 9.3: Implement `BGCCModel`**

Create `traiNNer/models/bgcc_model.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

from traiNNer.models.sr_model import SRModel
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


class BGCCModel(SRModel):
    """Dual-input color correction model.

    Extends SRModel to:
        - pull an additional `hr` tensor from each batch (the color-incorrect
          HR reference that the network should color-correct).
        - call `net_g(hr, lq)` instead of `net_g(lq)` via the `_run_net` hook.

    The supervision target stays in the `gt` key and is the CC_HR image.
    Existing SRModel loss / optimization / validation paths are unchanged.
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)
        self.hr: Tensor | None = None

    def feed_data(self, data: DataFeed) -> None:
        super().feed_data(data)
        assert "hr" in data, (
            "BGCCModel expects batches with an 'hr' key "
            "(the color-incorrect HR reference). Use PairedCCDataset."
        )
        self.hr = data["hr"].to(
            self.device, memory_format=self.memory_format, non_blocking=True
        )

    def _run_net(self, net: nn.Module, lq: Tensor) -> Tensor:
        assert self.hr is not None, (
            "BGCCModel._run_net called before feed_data — self.hr is None"
        )
        return net(self.hr, lq)
```

- [ ] **Step 9.4: Run the model test and verify it passes**

Run: `pytest tests/test_models/test_bgcc_model.py -v`

Expected: PASS. (If the dummy `ReduxOptions` construction fails due to mismatched fields, fix the test's opt-creation to satisfy the real struct and re-run — behavioral assertions remain the same.)

- [ ] **Step 9.5: Commit**

```bash
git add traiNNer/models/bgcc_model.py tests/test_models/test_bgcc_model.py
git commit -m "BGCC: add BGCCModel subclass with dual-input feed_data and _run_net override"
```

---

## Task 10: Wire `BGCCModel` into `build_model`

**Files:**
- Modify: `traiNNer/utils/redux_options.py`
- Modify: `traiNNer/models/__init__.py`

- [ ] **Step 10.1: Add `bgcc_mode` flag to `ReduxOptions`**

Open `traiNNer/utils/redux_options.py`. In `ReduxOptions`, near existing model-selection flags (`high_order_degradation`, around lines 508-513), add:

```python
    bgcc_mode: bool = False
    """Enable BGCC dual-input color-correction model. When true, build_model
    uses BGCCModel regardless of other flags."""
```

- [ ] **Step 10.2: Route to `BGCCModel` in `build_model`**

Open `traiNNer/models/__init__.py`. Add the import and route:

```python
from traiNNer.models.bgcc_model import BGCCModel
```

Then in the `build_model` function, replace:

```python
    if opt.high_order_degradation:
```

with:

```python
    if opt.bgcc_mode:
        model = BGCCModel(opt)
    elif opt.high_order_degradation:
```

(Keep the rest of the conditional chain unchanged.)

- [ ] **Step 10.3: Sanity-check the routing**

Run:

```bash
python -c "
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.models import build_model
# Construct a minimal valid ReduxOptions with bgcc_mode=True and confirm BGCCModel is returned.
# If ReduxOptions requires many fields, use a fixture YAML load instead:
# opt = Config.load_config_from_file('options/train/BGCC/bgcc_2x.yml').opt
print('routing stub ok')
"
```

Defer a fuller sanity check to the integration smoke test in Task 13.

- [ ] **Step 10.4: Commit**

```bash
git add traiNNer/utils/redux_options.py traiNNer/models/__init__.py
git commit -m "BGCC: add bgcc_mode flag and route build_model to BGCCModel"
```

---

## Task 11: ONNX dual-input export support

**Files:**
- Modify: `convert_to_onnx.py`
- Modify: `tests/test_archs/test_bgcc_arch.py`

- [ ] **Step 11.1: Inspect the existing ONNX export path**

Open `convert_to_onnx.py`. Identify:
- Where the dummy input tensor is constructed (single-input).
- The `torch.onnx.export` call that passes `(export_input,)` as args.
- The `input_names` and `dynamic_axes` configuration.

Note the line numbers of the call site and the dummy-input construction for the next step.

- [ ] **Step 11.2: Add a BGCC branch to the export path**

In `convert_to_onnx.py`, add logic that — when the loaded arch is `bgcc` / `bgcc_tiny` — constructs two dummy inputs and passes them to `torch.onnx.export` with two input names.

Exact change depends on the file layout; the minimal version is:

```python
# After the model is built and the single dummy lq tensor is constructed:
is_bgcc = network_type in {"bgcc", "bgcc_tiny"}

if is_bgcc:
    # HR has the output resolution; LR has input resolution.
    h_hr = example_h * scale
    w_hr = example_w * scale
    dummy_hr = torch.randn(1, 3, h_hr, w_hr, device=device, dtype=model_dtype)
    dummy_lq = export_input   # existing single input is the LR
    export_args = (dummy_hr, dummy_lq)
    input_names = ["hr", "lq"]
    # Both inputs and the output need the batch + HW as dynamic axes.
    output_name = "output"   # whatever string the existing script uses for output_names
    dynamic_axes = {
        "hr": {0: "batch", 2: "height_hr", 3: "width_hr"},
        "lq": {0: "batch", 2: "height_lq", 3: "width_lq"},
        output_name: {0: "batch", 2: "height_hr", 3: "width_hr"},
    }
else:
    export_args = (export_input,)
    # keep existing input_names and dynamic_axes
```

Pass `export_args`, `input_names`, and `dynamic_axes` through to `torch.onnx.export`. If the existing code already parameterizes these values by variable, reuse them; otherwise add a minimal inline conditional before the `torch.onnx.export` call. Before writing the patch, read the full `convert_to_onnx.py` file and identify the actual variable names used for (a) the dummy input, (b) the output-name string, (c) the dynamic_axes dict. Adapt the conditional to match those.

- [ ] **Step 11.3: Add an ONNX export round-trip test**

Append to `tests/test_archs/test_bgcc_arch.py`:

```python
def test_bgcc_onnx_export_round_trip(tmp_path):
    """Export BGCC to ONNX opset 17, run with onnxruntime, and compare outputs."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not installed")

    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=16, d=4).eval()
    hr = torch.randn(1, 3, 64, 64)
    lr = torch.randn(1, 3, 32, 32)

    onnx_path = tmp_path / "bgcc.onnx"
    torch.onnx.export(
        model,
        (hr, lr),
        str(onnx_path),
        input_names=["hr", "lq"],
        output_names=["output"],
        dynamic_axes={
            "hr": {0: "b", 2: "h", 3: "w"},
            "lq": {0: "b", 2: "h", 3: "w"},
            "output": {0: "b", 2: "h", 3: "w"},
        },
        opset_version=17,
    )

    with torch.no_grad():
        torch_out = model(hr, lr).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        None,
        {"hr": hr.numpy(), "lq": lr.numpy()},
    )[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-3)


def test_bgcc_onnx_export_dynamic_scale_inference(tmp_path):
    """After exporting at 2x shapes, ONNX runtime should handle 3x shapes too."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not installed")

    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=16, d=4).eval()
    hr_export = torch.randn(1, 3, 64, 64)
    lr_export = torch.randn(1, 3, 32, 32)

    onnx_path = tmp_path / "bgcc_dyn.onnx"
    torch.onnx.export(
        model, (hr_export, lr_export), str(onnx_path),
        input_names=["hr", "lq"], output_names=["output"],
        dynamic_axes={
            "hr": {0: "b", 2: "h", 3: "w"},
            "lq": {0: "b", 2: "h", 3: "w"},
            "output": {0: "b", 2: "h", 3: "w"},
        },
        opset_version=17,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Try a 3x ratio: LR 24x24, HR 72x72
    ort_out = sess.run(
        None,
        {
            "hr": np.random.randn(1, 3, 72, 72).astype(np.float32),
            "lq": np.random.randn(1, 3, 24, 24).astype(np.float32),
        },
    )[0]
    assert ort_out.shape == (1, 3, 72, 72)
```

- [ ] **Step 11.4: Run the ONNX tests**

Run: `pytest tests/test_archs/test_bgcc_arch.py -v -k onnx`

Expected: Both tests PASS (or skip cleanly if `onnxruntime` isn't installed).

- [ ] **Step 11.5: Commit**

```bash
git add convert_to_onnx.py tests/test_archs/test_bgcc_arch.py
git commit -m "BGCC: add ONNX dual-input export support and round-trip tests"
```

---

## Task 12: Example training YAML

**Files:**
- Create: `options/train/BGCC/bgcc_2x.yml`

- [ ] **Step 12.1: Locate a simple existing training YAML as a template**

Run: `ls options/train/` and pick any small arch (e.g., SPAN or DIS) as a template. Note the top-level structure (name, model_type, scale, num_gpu, path, datasets, network_g, train, logger sections).

- [ ] **Step 12.2: Write `options/train/BGCC/bgcc_2x.yml`**

Create `options/train/BGCC/bgcc_2x.yml`:

```yaml
# BGCC: Bilateral Guided Color Correction — 2x training config
# Inputs: HR (BD, color-incorrect) + LR (DVD/LD, color-correct, degraded)
# Target: CC_HR (manually color-corrected HR)

name: bgcc_2x
model_type: sr   # BGCCModel is selected via bgcc_mode, not model_type
scale: 2
num_gpu: 1
manual_seed: 42

use_amp: true
amp_bf16: false
use_channels_last: false
use_compile: false

bgcc_mode: true   # routes to BGCCModel in build_model
high_order_degradation: false

# ----------------------------------------------------------------------
# Dataset: three roots required (lq, hr, gt=cc_hr). All folders must share
# image basenames.
# ----------------------------------------------------------------------
datasets:
  train:
    name: anime_bgcc_train
    type: PairedCCDataset
    dataroot_lq:
      - /absolute/path/to/lr_dvd
    dataroot_hr:
      - /absolute/path/to/hr_bd
    dataroot_gt:
      - /absolute/path/to/cc_hr
    filename_tmpl: "{}"
    io_backend:
      type: disk

    gt_size: 128   # HR/CC_HR patch size
    use_hflip: true
    use_rot: true

    num_worker_per_gpu: 4
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: anime_bgcc_val
    type: PairedCCDataset
    dataroot_lq:
      - /absolute/path/to/val/lr
    dataroot_hr:
      - /absolute/path/to/val/hr
    dataroot_gt:
      - /absolute/path/to/val/cc_hr
    io_backend:
      type: disk

# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------
network_g:
  type: bgcc
  feat: 32
  d: 8
  n_blocks_per_stage: 2
  guidance_hidden: 8

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
train:
  ema_decay: 0.999
  total_iter: 500000
  warmup_iter: -1

  optim_g:
    type: Adam
    lr: !!float 5e-4
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [250000, 400000, 450000, 475000]
    gamma: 0.5

  losses:
    - type: l1loss
      loss_weight: 1.0

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger:
  print_freq: 100
  save_checkpoint_freq: 5000
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

dist_params:
  backend: nccl
  port: 29500

val:
  val_freq: 5000
  save_img: false
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 2
      test_y_channel: false
```

- [ ] **Step 12.3: Validate the YAML parses**

Run:

```bash
python -c "
from traiNNer.utils.config import Config
cfg = Config.load_config_from_file('options/train/BGCC/bgcc_2x.yml')
print('parsed:', cfg.opt.name, 'bgcc_mode=', cfg.opt.bgcc_mode)
"
```

If the project exposes config parsing differently, use whatever helper the existing training flow uses. Expected: the config parses without exception and prints `parsed: bgcc_2x bgcc_mode= True`.

- [ ] **Step 12.4: Commit**

```bash
git add options/train/BGCC/bgcc_2x.yml
git commit -m "BGCC: add example 2x training config YAML"
```

---

## Task 13: End-to-end integration smoke test

**Files:**
- Create: `tests/test_models/test_bgcc_e2e.py`

- [ ] **Step 13.1: Write the integration smoke test**

Create `tests/test_models/test_bgcc_e2e.py`:

```python
import os
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def tiny_triplet_root(tmp_path):
    """4 fake 32x16/32/32 triplets with consistent basenames."""
    roots = {}
    for key, (h, w) in [("lr", (16, 16)), ("hr", (32, 32)), ("cc_hr", (32, 32))]:
        d = tmp_path / key
        d.mkdir()
        for i in range(4):
            rng = np.random.default_rng(seed=i * 7 + hash(key) % 1000)
            arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img{i}.png")
        roots[key] = str(d)
    return roots


def test_bgcc_trains_loss_decreases(tiny_triplet_root):
    """50-iter smoke: verify loss decreases monotonically-ish on a tiny batch."""
    from traiNNer.archs.bgcc_arch import bgcc
    from traiNNer.data.paired_cc_dataset import PairedCCDataset
    from traiNNer.utils.redux_options import DatasetOptions

    opt = DatasetOptions(
        name="smoke",
        type="PairedCCDataset",
        dataroot_lq=[tiny_triplet_root["lr"]],
        dataroot_hr=tiny_triplet_root["hr"],
        dataroot_gt=[tiny_triplet_root["cc_hr"]],
        scale=2,
        phase="train",
        gt_size=32,
        use_hflip=False,
        use_rot=False,
        io_backend={"type": "disk"},
    )
    ds = PairedCCDataset(opt)

    model = bgcc(feat=16, d=4)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)

    initial_losses = []
    final_losses = []
    for step in range(50):
        idx = step % len(ds)
        sample = ds[idx]
        hr = sample["hr"].unsqueeze(0)
        lr = sample["lq"].unsqueeze(0)
        gt = sample["gt"].unsqueeze(0)
        out = model(hr, lr)
        loss = (out - gt).abs().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step < 5:
            initial_losses.append(loss.item())
        if step >= 45:
            final_losses.append(loss.item())

    assert np.mean(final_losses) < np.mean(initial_losses), (
        f"Loss did not decrease: initial={np.mean(initial_losses):.4f} "
        f"final={np.mean(final_losses):.4f}"
    )
```

- [ ] **Step 13.2: Run the integration test**

Run: `pytest tests/test_models/test_bgcc_e2e.py -v`

Expected: PASS. Loss should measurably decrease over 50 iters on a tiny dataset with 4 synthetic triplets.

- [ ] **Step 13.3: Run the full BGCC-related test suite one last time**

Run:

```bash
pytest tests/test_archs/test_bgcc_arch.py tests/test_data/test_triplet_transforms.py tests/test_data/test_paired_cc_dataset.py tests/test_models/test_bgcc_model.py tests/test_models/test_bgcc_e2e.py -v
```

Expected: All tests PASS (or skip with a clean message for ONNX if runtime is missing).

- [ ] **Step 13.4: Run lint/typecheck**

Run:

```bash
ruff format --check traiNNer/archs/bgcc_arch.py traiNNer/data/paired_cc_dataset.py traiNNer/models/bgcc_model.py
ruff check traiNNer/archs/bgcc_arch.py traiNNer/data/paired_cc_dataset.py traiNNer/models/bgcc_model.py
pyright traiNNer/archs/bgcc_arch.py traiNNer/data/paired_cc_dataset.py traiNNer/models/bgcc_model.py
```

Fix any issues that surface. Re-run until clean.

- [ ] **Step 13.5: Commit**

```bash
git add tests/test_models/test_bgcc_e2e.py
git commit -m "BGCC: add end-to-end smoke test (50-iter loss-decrease check)"
```

---

## Out of Scope for v1

Tracked for future work; explicitly NOT in this implementation plan:

- Identity-init alternative to residual-to-HR (spec §4.6 alternative)
- YCbCr-weighted L1 loss variant (spec §7.1 secondary loss)
- Gradient-preservation auxiliary loss (spec §10 question 3)
- Temporal/video BGCC variant (spec §11)
- Automated CC_HR dataset generation (spec §7.3 — user owns this pipeline)
- Wider D=16 or F=48 variants (tunable via existing params; no code change needed)

---

## Verification Checklist (run at the end of implementation)

Before declaring this feature done:

- [ ] All 5 BGCC-related test files pass: `pytest tests/test_archs/test_bgcc_arch.py tests/test_data/test_triplet_transforms.py tests/test_data/test_paired_cc_dataset.py tests/test_models/test_bgcc_model.py tests/test_models/test_bgcc_e2e.py -v`
- [ ] Existing test suite shows no regressions: `pytest tests/ -q --tb=no` (same pass count as pre-BGCC baseline)
- [ ] Ruff format + check clean on all modified/created files
- [ ] Pyright clean on all modified/created files
- [ ] `python -c "from traiNNer.archs.bgcc_arch import bgcc; m = bgcc(); print(sum(p.numel() for p in m.parameters()))"` outputs a value in [100_000, 400_000]
- [ ] Example YAML parses via the config loader
- [ ] Git log shows 13 focused commits, one per task
