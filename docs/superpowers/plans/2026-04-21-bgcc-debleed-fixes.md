# BGCC De-bleeding Fixes — Implementation Plan

**Goal:** Fix BGCC's shortcut-learning failure mode where the model propagates LR's chroma bleed to the output instead of de-bleeding it. Two independent fixes, both needed:

1. Replace Gaussian-blur chroma bleed augmentation with a realistic 4:2:0-simulation that matches actual DVD/LD chroma subsampling artifact patterns.
2. Add a focal-weighted L1 loss that concentrates gradient on pixels where `LR` and `CC_HR` differ (= bled pixels).

**Why both:** Aug alone creates training examples where de-bleeding is needed but shortcut loss still dominates. Focal loss alone targets gradient correctly but only works if training distribution matches real-world bleed. Together: matched distribution + targeted gradient → model is forced to learn de-bleeding.

---

## File structure

**Modify:**
- `traiNNer/data/bgcc_augment.py` — replace `_chroma_bleed` with `_chroma_subsample_simulation`; rename option
- `traiNNer/utils/redux_options.py` — rename `lr_chroma_bleed_sigma` → `lr_chroma_subsample_factor`
- `tests/test_data/test_bgcc_augment.py` — update the chroma-bleed test for the new parameter
- `options/train/BGCC/bgcc_2x.yml` — update the `bgcc_aug` block

**Create:**
- `traiNNer/losses/bgcc_bleed_focal_loss.py` — `BleedFocalL1Loss`
- `tests/test_losses/test_bgcc_bleed_focal_loss.py` — tests for focal loss

**No backwards compatibility.** This is a follow-up to an unreleased feature branch — a clean rename is cleaner than dual-option support.

---

## Task 1: Realistic chroma-subsample augmentation

**Files:** `traiNNer/data/bgcc_augment.py`, `traiNNer/utils/redux_options.py`, `tests/test_data/test_bgcc_augment.py`

- [ ] **Step 1.1: Rename option in `BGCCAugOptions` struct**

In `traiNNer/utils/redux_options.py`, find the `BGCCAugOptions.lr_chroma_bleed_sigma` field and replace it with:

```python
    lr_chroma_subsample_factor: Annotated[
        int,
        Meta(description="Max factor for simulated 4:2:0 chroma subsampling on LR only (0 disables). A value of N means chroma (Cb, Cr) is downscaled by a uniformly-sampled integer factor in [1, N] via avg_pool2d, then optionally shifted by up to 1 pixel, then upsampled back with bilinear interpolation. This simulates real DVD/LD chroma bleed (edge-localized fringing) better than Gaussian blur. Typical values: 2 (mild), 3 (moderate), 4 (aggressive)."),
    ] = 0
```

- [ ] **Step 1.2: Update `_chroma_bleed` to `_chroma_subsample_simulation`**

In `traiNNer/data/bgcc_augment.py`, replace the existing `_chroma_bleed` helper with:

```python
def _chroma_subsample_simulation(rgb: Tensor, factor: int, shift_dy: int, shift_dx: int) -> Tensor:
    """Simulate 4:2:0-style chroma subsampling artifacts.

    Converts to YCbCr, downsamples Cb/Cr by `factor` via avg_pool2d, optionally
    rolls them by (shift_dy, shift_dx) pixels, upsamples back with bilinear,
    and converts back to RGB. This produces edge-localized directional
    chroma smear that matches real DVD/LD bleed patterns, unlike Gaussian blur.
    """
    if factor <= 1:
        return rgb
    ycbcr = _rgb_to_ycbcr(rgb)
    # Cb and Cr at (2, H, W), process as a 2-channel batch of 1 item
    chroma = ycbcr[1:3].unsqueeze(0)   # (1, 2, H, W)
    h, w = chroma.shape[-2:]
    # Need dimensions divisible by factor; if not, crop
    h_c = (h // factor) * factor
    w_c = (w // factor) * factor
    if h_c == 0 or w_c == 0:
        return rgb
    chroma_c = chroma[..., :h_c, :w_c]
    # Downsample then upsample
    downs = F.avg_pool2d(chroma_c, kernel_size=factor)
    if shift_dy != 0 or shift_dx != 0:
        downs = torch.roll(downs, shifts=(shift_dy, shift_dx), dims=(2, 3))
    ups = F.interpolate(downs, size=(h_c, w_c), mode="bilinear", align_corners=False)
    # Paste back (restore full size if we cropped)
    chroma_bled = chroma.clone()
    chroma_bled[..., :h_c, :w_c] = ups
    ycbcr_out = torch.stack([ycbcr[0], chroma_bled[0, 0], chroma_bled[0, 1]], dim=0)
    return _ycbcr_to_rgb(ycbcr_out).clamp(0.0, 1.0)
```

`torch.nn.functional as F` is already imported in the file; keep it.

- [ ] **Step 1.3: Update `apply_bgcc_augmentations` to call the new helper**

Replace the existing chroma-bleed block at the end of `apply_bgcc_augmentations`:

```python
    # 5. Chroma subsampling simulation on LR only (realistic DVD/LD bleed)
    if opts.lr_chroma_subsample_factor > 1:
        factor = random.randint(2, opts.lr_chroma_subsample_factor)
        # Optional random 1-pixel shift (50% chance) to simulate chroma misalignment
        shift_dy = random.choice([-1, 0, 1]) if random.random() < 0.5 else 0
        shift_dx = random.choice([-1, 0, 1]) if random.random() < 0.5 else 0
        lr = _chroma_subsample_simulation(lr, factor, shift_dy, shift_dx)
```

- [ ] **Step 1.4: Remove the old `_chroma_bleed` helper entirely** (don't keep it around as dead code).

- [ ] **Step 1.5: Update tests**

In `tests/test_data/test_bgcc_augment.py`:
- Rename `test_lr_chroma_bleed_affects_lr_only` → `test_lr_chroma_subsample_affects_lr_only`.
- Change `opts = BGCCAugOptions(lr_chroma_bleed_sigma=2.0)` → `opts = BGCCAugOptions(lr_chroma_subsample_factor=4)`.
- In `test_output_stays_in_unit_range`, change `lr_chroma_bleed_sigma=2.0` → `lr_chroma_subsample_factor=4`.

Test behavioral assertion stays the same (`out_lr != lr`, `out_hr == hr`, `out_cc == cc_hr`).

- [ ] **Step 1.6: Run tests, verify pass**

```bash
source venv/Scripts/activate
pytest tests/test_data/test_bgcc_augment.py -v
ruff check traiNNer/data/bgcc_augment.py traiNNer/utils/redux_options.py tests/test_data/test_bgcc_augment.py
ruff format --check traiNNer/data/bgcc_augment.py traiNNer/utils/redux_options.py tests/test_data/test_bgcc_augment.py
```

- [ ] **Step 1.7: Commit**

```
git commit -m "BGCC: replace Gaussian chroma bleed aug with 4:2:0 subsampling simulation"
```

---

## Task 2: Bleed-focal L1 loss

**Files:** `traiNNer/losses/bgcc_bleed_focal_loss.py` (new), `tests/test_losses/test_bgcc_bleed_focal_loss.py` (new)

- [ ] **Step 2.1: Write failing test**

Create `tests/test_losses/test_bgcc_bleed_focal_loss.py`:

```python
import pytest
import torch


def test_bleed_focal_l1_equals_plain_l1_when_gain_zero() -> None:
    """With focal_gain=0, the loss should reduce to plain L1."""
    from traiNNer.losses.bgcc_bleed_focal_loss import BleedFocalL1Loss

    torch.manual_seed(0)
    output = torch.rand(2, 3, 64, 64)
    cc_hr = torch.rand(2, 3, 64, 64)
    lr = torch.rand(2, 3, 32, 32)

    loss_fn = BleedFocalL1Loss(loss_weight=1.0, focal_gain=0.0, focal_threshold=0.05)
    loss = loss_fn(output, cc_hr, lq=lr)
    plain_l1 = (output - cc_hr).abs().mean()

    assert torch.allclose(loss, plain_l1, atol=1e-6)


def test_bleed_focal_l1_weights_bled_pixels_higher() -> None:
    """When a region of LR differs from CC_HR, loss at that region should be amplified."""
    from traiNNer.losses.bgcc_bleed_focal_loss import BleedFocalL1Loss

    # Construct a case where upsampled LR matches CC_HR everywhere except one patch.
    cc_hr = torch.full((1, 3, 32, 32), 0.5)
    lr = torch.full((1, 3, 16, 16), 0.5)
    lr[:, :, :4, :4] = 0.9   # "bled" region — upsamples to top-left 8x8 of HR

    # Output differs from CC_HR by a constant everywhere.
    output = cc_hr + 0.1

    loss_fn = BleedFocalL1Loss(loss_weight=1.0, focal_gain=4.0, focal_threshold=0.05)
    loss = loss_fn(output, cc_hr, lq=lr)

    # Plain L1 would be exactly 0.1. With 4x gain on ~6.25% of pixels (8x8 out of 32x32),
    # loss should be greater than 0.1.
    assert loss.item() > 0.1


def test_bleed_focal_l1_respects_loss_weight() -> None:
    from traiNNer.losses.bgcc_bleed_focal_loss import BleedFocalL1Loss

    torch.manual_seed(0)
    output = torch.rand(1, 3, 32, 32)
    cc_hr = torch.rand(1, 3, 32, 32)
    lr = torch.rand(1, 3, 16, 16)

    loss_1 = BleedFocalL1Loss(loss_weight=1.0, focal_gain=0.0)(output, cc_hr, lq=lr)
    loss_3 = BleedFocalL1Loss(loss_weight=3.0, focal_gain=0.0)(output, cc_hr, lq=lr)
    assert torch.allclose(loss_3, loss_1 * 3.0, atol=1e-6)


def test_bleed_focal_l1_registered_in_loss_registry() -> None:
    from traiNNer.utils.registry import LOSS_REGISTRY
    # Trigger imports
    import traiNNer.losses.bgcc_bleed_focal_loss   # noqa: F401
    assert "bleedfocall1loss" in LOSS_REGISTRY
```

- [ ] **Step 2.2: Run test, verify it fails (ImportError).**

- [ ] **Step 2.3: Implement the loss**

Create `traiNNer/losses/bgcc_bleed_focal_loss.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class BleedFocalL1Loss(nn.Module):
    """L1 loss with pixel-wise weighting that concentrates gradient on bled pixels.

    The weight map is 1.0 everywhere, plus `focal_gain` on pixels where
    `|bilinear_upsample(lq) - cc_hr| > focal_threshold` (aggregated over channels).
    This forces the BGCC model to actually learn de-bleeding instead of taking
    the "output = upsampled LR" shortcut that gets low L1 on the majority of
    clean pixels.

    Args:
        loss_weight: Scalar multiplier on the final loss.
        focal_gain: Additional weight applied to bled pixels. 0.0 reduces to plain L1.
        focal_threshold: Channel-mean difference threshold above which a pixel is
            considered bled. Typical: 0.03-0.10 for float images in [0, 1].
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        focal_gain: float = 4.0,
        focal_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.focal_gain = focal_gain
        self.focal_threshold = focal_threshold

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,   # noqa: ARG002
        lq: Tensor | None = None,
        **kwargs,   # noqa: ARG002
    ) -> Tensor:
        """Compute weighted L1 loss.

        Args:
            pred: Model output at HR resolution, (B, 3, H, W).
            target: CC_HR supervision, (B, 3, H, W).
            lq: LR input, (B, 3, H_lq, W_lq). If None, falls back to plain L1.
        """
        abs_diff = (pred - target).abs()
        if self.focal_gain <= 0.0 or lq is None:
            return self.loss_weight * abs_diff.mean()

        # Upsample LR to HR resolution so we can compare per-pixel.
        lq_up = F.interpolate(
            lq, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        # Per-pixel channel-mean absolute difference between upsampled LR and CC_HR.
        lr_cc_diff = (lq_up - target).abs().mean(dim=1, keepdim=True)
        # Binary mask of bled pixels, then a weight of 1 + gain * mask.
        bleed_mask = (lr_cc_diff > self.focal_threshold).to(abs_diff.dtype)
        weight_map = 1.0 + self.focal_gain * bleed_mask   # (B, 1, H, W)
        return self.loss_weight * (abs_diff * weight_map).mean()
```

Important: the model wrapper needs to pass `lq` into the loss. SRModel's loss call probably only passes `(pred, target)` today, so you'll need to verify — see Step 2.4.

- [ ] **Step 2.4: Verify loss is called with `lq`**

Read `traiNNer/models/sr_model.py` around where losses are computed (search for `loss(self.output` or similar). Determine whether `self.lq` is currently passed as a kwarg to loss forward. Options:

- If losses already accept `lq` kwarg: just pass it in the loss call for BGCCModel.
- If they don't: override the loss-call step in `BGCCModel` to pass `lq=self.lq` (and `hr=self.hr` while you're at it).

The cleanest minimal change: in `traiNNer/models/bgcc_model.py`, override whatever method in `SRModel` computes the generator loss to inject `lq=self.lq` into loss forward kwargs. If that's structurally hard (e.g., the loss loop is deep inside `optimize_parameters`), the pragmatic fallback is to modify `SRModel.optimize_parameters` to always pass `lq=self.lq` as a kwarg (losses that don't use it ignore via `**kwargs`) — we already ensured all losses accept `**kwargs` via the `BleedFocalL1Loss.forward` signature.

Pick the route that's a smaller change. Report which one you took.

- [ ] **Step 2.5: Run tests**

```bash
source venv/Scripts/activate
pytest tests/test_losses/test_bgcc_bleed_focal_loss.py -v
ruff check traiNNer/losses/bgcc_bleed_focal_loss.py tests/test_losses/test_bgcc_bleed_focal_loss.py
ruff format --check traiNNer/losses/bgcc_bleed_focal_loss.py tests/test_losses/test_bgcc_bleed_focal_loss.py
```

All 4 tests pass. Ruff clean.

- [ ] **Step 2.6: Commit**

```
git commit -m "BGCC: add BleedFocalL1Loss to concentrate gradient on bled pixels"
```

---

## Task 3: Update example YAML

**File:** `options/train/BGCC/bgcc_2x.yml`

- [ ] **Step 3.1: Update bgcc_aug and losses blocks**

Replace `lr_chroma_bleed_sigma: 0.8` with `lr_chroma_subsample_factor: 4` in the `datasets.train.bgcc_aug` block.

In the `train.losses` block, replace the `l1loss` with `bleedfocall1loss`:

```yaml
  losses:
    # Bleed-focal L1 loss: concentrates gradient on pixels where LR differs
    # from CC_HR, forcing the model to learn de-bleeding instead of taking
    # the "output = upsampled LR" shortcut.
    - type: bleedfocall1loss
      loss_weight: 1.0
      focal_gain: 4.0        # extra weight on bled pixels
      focal_threshold: 0.05  # channel-mean diff threshold for bleed detection
```

- [ ] **Step 3.2: Verify YAML parses**

```bash
python -c "
import yaml
with open('options/train/BGCC/bgcc_2x.yml') as f:
    d = yaml.safe_load(f)
print('aug subsample:', d['datasets']['train']['bgcc_aug']['lr_chroma_subsample_factor'])
print('loss:', d['train']['losses'][0])
"
```

- [ ] **Step 3.3: Commit**

```
git commit -m "BGCC: update example YAML to use realistic bleed aug + focal loss"
```

---

## Verification

After all 3 tasks:

```bash
source venv/Scripts/activate
pytest tests/test_archs/test_bgcc_arch.py tests/test_data/test_bgcc_augment.py tests/test_data/test_paired_cc_dataset.py tests/test_models/test_bgcc_model.py tests/test_models/test_bgcc_e2e.py tests/test_losses/test_bgcc_bleed_focal_loss.py -v
```

Expected: all BGCC-related tests pass (previous 22 + 4 new loss tests = 26), ONNX tests skip cleanly.
