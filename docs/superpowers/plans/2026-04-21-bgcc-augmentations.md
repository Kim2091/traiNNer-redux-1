# BGCC Training-Time Augmentations — Implementation Plan

**Goal:** Add 5 training-time augmentations to the BGCC pipeline to help the model generalize from a small (4-show) dataset.

**Architecture:** Augmentations are applied in tensor space inside `PairedCCDataset.__getitem__` (train phase only), after `img2tensor` and before optional `mean`/`std` normalization. Invariant discipline: (LR, CC_HR) share the target palette so palette perturbations apply identically to both; HR has the wrong colors by design so HR-side perturbations are independent; LR-only perturbations simulate degradation severity variation.

**Tech stack:** torchvision.transforms.functional (hue / saturation / gamma / gaussian_blur), torch tensor ops, Python `random`.

---

## File structure

**Create:**
- `traiNNer/data/bgcc_augment.py` — 5 augmentation functions + `apply_bgcc_augmentations` coordinator
- `tests/test_data/test_bgcc_augment.py` — per-augmentation behavior tests

**Modify:**
- `traiNNer/utils/redux_options.py` — add `BGCCAugOptions` struct; add `bgcc_aug: BGCCAugOptions | None = None` to `DatasetOptions`
- `traiNNer/data/paired_cc_dataset.py` — call `apply_bgcc_augmentations` in train-phase branch of `__getitem__`
- `options/train/BGCC/bgcc_2x.yml` — add `bgcc_aug:` block with sensible starting-point values

---

## The 5 augmentations

| # | Name | Applies to | Rationale |
|---|---|---|---|
| 1 | Hue shift ± deg | LR + CC_HR (same value) | Diversifies target palette |
| 2 | Saturation ×[1−s, 1+s] | LR + CC_HR (same value) | Diversifies palette saturation |
| 3 | Gamma ±γ | HR only | Simulates HR mastering-curve variation |
| 4 | Per-channel RGB gain | HR only | Simulates white-balance shifts |
| 5 | Chroma bleed (Gaussian blur on Cb/Cr) | LR only | Simulates chroma-subsampling / bleed severity |

All augmentations are no-ops when their delta is 0 (default). Example YAML ships with modest starting values; users can tune.

---

## Task 1: BGCCAugOptions struct

**File:** `traiNNer/utils/redux_options.py`

- [ ] **Step 1.1: Add `BGCCAugOptions` struct near `DatasetOptions`**

```python
class BGCCAugOptions(StrictStruct):
    hue_shift_deg: Annotated[
        float,
        Meta(description="Max absolute hue shift in degrees applied identically to LR and CC_HR. 0 disables."),
    ] = 0.0
    saturation_delta: Annotated[
        float,
        Meta(description="Max fractional saturation scale applied identically to LR and CC_HR. 0 disables. Factor sampled from [1-delta, 1+delta]."),
    ] = 0.0
    hr_gamma_delta: Annotated[
        float,
        Meta(description="Max fractional gamma perturbation on HR only. 0 disables. Gamma sampled from [1-delta, 1+delta]."),
    ] = 0.0
    hr_rgb_gain_delta: Annotated[
        float,
        Meta(description="Max per-channel gain perturbation on HR only. 0 disables. Each channel independently scaled by a factor from [1-delta, 1+delta]."),
    ] = 0.0
    lr_chroma_bleed_sigma: Annotated[
        float,
        Meta(description="Max Gaussian sigma applied to Cb/Cr channels of LR (simulates chroma bleed). 0 disables. Sigma sampled uniformly from [0, this]."),
    ] = 0.0
```

- [ ] **Step 1.2: Add `bgcc_aug` field on `DatasetOptions`**

Near the existing `dataroot_hr` field:

```python
    bgcc_aug: BGCCAugOptions | None = None
```

---

## Task 2: Augmentation module + tests (TDD)

**Files:** `traiNNer/data/bgcc_augment.py` (new), `tests/test_data/test_bgcc_augment.py` (new)

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_data/test_bgcc_augment.py`:

```python
import random

import pytest
import torch


def _make_triplet(h: int = 32, w: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    lr = torch.rand(3, h // 2, w // 2)
    hr = torch.rand(3, h, w)
    cc_hr = torch.rand(3, h, w)
    return lr, hr, cc_hr


def test_zero_options_is_identity() -> None:
    from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
    from traiNNer.utils.redux_options import BGCCAugOptions

    lr, hr, cc_hr = _make_triplet()
    out_lr, out_hr, out_cc = apply_bgcc_augmentations(
        lr.clone(), hr.clone(), cc_hr.clone(), BGCCAugOptions()
    )
    assert torch.equal(out_lr, lr)
    assert torch.equal(out_hr, hr)
    assert torch.equal(out_cc, cc_hr)


def test_hue_shifts_lr_and_cchr_by_same_amount_and_leaves_hr_alone() -> None:
    from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
    from traiNNer.utils.redux_options import BGCCAugOptions

    lr, hr, cc_hr = _make_triplet()
    random.seed(123)
    opts = BGCCAugOptions(hue_shift_deg=30.0)
    out_lr, out_hr, out_cc = apply_bgcc_augmentations(
        lr.clone(), hr.clone(), cc_hr.clone(), opts
    )

    # HR unchanged
    assert torch.equal(out_hr, hr)
    # LR and CC_HR both changed
    assert not torch.equal(out_lr, lr)
    assert not torch.equal(out_cc, cc_hr)

    # Re-applying the same shift to matching inputs should yield matching outputs.
    # Use a simple check: hue is a channel-wise rotation in HSV; if LR and CC_HR
    # happened to have the same pixel values, their outputs would too.
    sentinel = torch.full_like(lr, 0.5)
    sentinel[0, 0, 0] = 0.7   # make it non-uniform so adjust_hue isn't a no-op
    random.seed(123)
    s_lr, _, s_cc = apply_bgcc_augmentations(
        sentinel.clone(), torch.zeros_like(hr), sentinel.clone(), opts
    )
    assert torch.allclose(s_lr, s_cc)


def test_saturation_changes_lr_and_cchr_not_hr() -> None:
    from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
    from traiNNer.utils.redux_options import BGCCAugOptions

    lr, hr, cc_hr = _make_triplet()
    random.seed(7)
    opts = BGCCAugOptions(saturation_delta=0.3)
    out_lr, out_hr, out_cc = apply_bgcc_augmentations(
        lr.clone(), hr.clone(), cc_hr.clone(), opts
    )
    assert torch.equal(out_hr, hr)
    assert not torch.equal(out_lr, lr)
    assert not torch.equal(out_cc, cc_hr)


def test_hr_gamma_changes_hr_not_lr_or_cchr() -> None:
    from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
    from traiNNer.utils.redux_options import BGCCAugOptions

    lr, hr, cc_hr = _make_triplet()
    random.seed(13)
    opts = BGCCAugOptions(hr_gamma_delta=0.3)
    out_lr, out_hr, out_cc = apply_bgcc_augmentations(
        lr.clone(), hr.clone(), cc_hr.clone(), opts
    )
    assert torch.equal(out_lr, lr)
    assert torch.equal(out_cc, cc_hr)
    assert not torch.equal(out_hr, hr)


def test_hr_rgb_gain_is_per_channel_and_affects_hr_only() -> None:
    from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
    from traiNNer.utils.redux_options import BGCCAugOptions

    lr, hr, cc_hr = _make_triplet()
    random.seed(21)
    opts = BGCCAugOptions(hr_rgb_gain_delta=0.2)
    out_lr, out_hr, out_cc = apply_bgcc_augmentations(
        lr.clone(), hr.clone(), cc_hr.clone(), opts
    )
    assert torch.equal(out_lr, lr)
    assert torch.equal(out_cc, cc_hr)

    # Per-channel ratios (ignoring clamped pixels) should be non-uniform across channels
    # unless the RNG unluckily drew three identical gains (unlikely at seed 21).
    with torch.no_grad():
        ratios = (out_hr + 1e-6) / (hr + 1e-6)
        channel_means = ratios.mean(dim=(1, 2))
    assert channel_means.std() > 1e-3, "RGB gain appears uniform across channels"


def test_lr_chroma_bleed_affects_lr_only() -> None:
    from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
    from traiNNer.utils.redux_options import BGCCAugOptions

    lr, hr, cc_hr = _make_triplet()
    random.seed(42)
    opts = BGCCAugOptions(lr_chroma_bleed_sigma=2.0)
    out_lr, out_hr, out_cc = apply_bgcc_augmentations(
        lr.clone(), hr.clone(), cc_hr.clone(), opts
    )
    assert torch.equal(out_hr, hr)
    assert torch.equal(out_cc, cc_hr)
    assert not torch.equal(out_lr, lr)


def test_output_stays_in_unit_range() -> None:
    from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
    from traiNNer.utils.redux_options import BGCCAugOptions

    lr, hr, cc_hr = _make_triplet()
    random.seed(99)
    opts = BGCCAugOptions(
        hue_shift_deg=30.0,
        saturation_delta=0.3,
        hr_gamma_delta=0.3,
        hr_rgb_gain_delta=0.2,
        lr_chroma_bleed_sigma=2.0,
    )
    out_lr, out_hr, out_cc = apply_bgcc_augmentations(
        lr.clone(), hr.clone(), cc_hr.clone(), opts
    )
    for t, name in [(out_lr, "lr"), (out_hr, "hr"), (out_cc, "cc_hr")]:
        assert t.min() >= 0.0, f"{name} went below 0"
        assert t.max() <= 1.0, f"{name} went above 1"
```

- [ ] **Step 2.2: Run tests, verify they fail (ImportError).**

- [ ] **Step 2.3: Implement `bgcc_augment.py`**

```python
from __future__ import annotations

import random

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision.transforms import functional as TF  # noqa: N812

from traiNNer.utils.redux_options import BGCCAugOptions


def _rgb_to_ycbcr(rgb: Tensor) -> Tensor:
    """BT.601 RGB -> YCbCr. Input (..., 3, H, W) in [0, 1]. Output same shape, Cb/Cr centered on 0.5."""
    r, g, b = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim=-3)


def _ycbcr_to_rgb(ycbcr: Tensor) -> Tensor:
    """Inverse of _rgb_to_ycbcr."""
    y = ycbcr[..., 0, :, :]
    cb = ycbcr[..., 1, :, :] - 0.5
    cr = ycbcr[..., 2, :, :] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=-3)


def _chroma_bleed(rgb: Tensor, sigma: float) -> Tensor:
    """Gaussian-blur Cb/Cr of an RGB image. Input (3, H, W) in [0, 1]."""
    if sigma <= 0:
        return rgb
    ycbcr = _rgb_to_ycbcr(rgb)
    # torchvision gaussian_blur expects (..., C, H, W); we blur Cb and Cr together.
    chroma = ycbcr[1:3].unsqueeze(0)   # (1, 2, H, W)
    # Kernel size ~ 6 sigma, odd.
    ks = max(3, int(2 * round(3 * sigma) + 1))
    if ks % 2 == 0:
        ks += 1
    chroma_blurred = TF.gaussian_blur(chroma, kernel_size=[ks, ks], sigma=[sigma, sigma])
    ycbcr_out = torch.stack([ycbcr[0], chroma_blurred[0, 0], chroma_blurred[0, 1]], dim=0)
    return _ycbcr_to_rgb(ycbcr_out).clamp(0.0, 1.0)


def apply_bgcc_augmentations(
    lr: Tensor,
    hr: Tensor,
    cc_hr: Tensor,
    opts: BGCCAugOptions,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply BGCC training-time augmentations with invariant discipline.

    (LR, CC_HR) share the target palette — palette perturbations apply identically.
    HR has the wrong colors — HR-side perturbations are independent.
    LR-only perturbations simulate degradation severity variation.
    """
    # 1. Hue on (LR, CC_HR)
    if opts.hue_shift_deg > 0:
        # torchvision adjust_hue factor is in [-0.5, 0.5] mapping to +/-180 deg
        factor = random.uniform(-opts.hue_shift_deg, opts.hue_shift_deg) / 360.0
        lr = TF.adjust_hue(lr, factor)
        cc_hr = TF.adjust_hue(cc_hr, factor)

    # 2. Saturation on (LR, CC_HR)
    if opts.saturation_delta > 0:
        factor = random.uniform(1.0 - opts.saturation_delta, 1.0 + opts.saturation_delta)
        lr = TF.adjust_saturation(lr, factor)
        cc_hr = TF.adjust_saturation(cc_hr, factor)

    # 3. Gamma on HR only
    if opts.hr_gamma_delta > 0:
        gamma = random.uniform(1.0 - opts.hr_gamma_delta, 1.0 + opts.hr_gamma_delta)
        hr = TF.adjust_gamma(hr, gamma)

    # 4. Per-channel RGB gain on HR only
    if opts.hr_rgb_gain_delta > 0:
        gains = torch.tensor(
            [
                random.uniform(1.0 - opts.hr_rgb_gain_delta, 1.0 + opts.hr_rgb_gain_delta)
                for _ in range(3)
            ],
            dtype=hr.dtype,
            device=hr.device,
        ).view(3, 1, 1)
        hr = (hr * gains).clamp(0.0, 1.0)

    # 5. Chroma bleed on LR only
    if opts.lr_chroma_bleed_sigma > 0:
        sigma = random.uniform(0.0, opts.lr_chroma_bleed_sigma)
        if sigma > 0:
            lr = _chroma_bleed(lr, sigma)

    return lr, hr, cc_hr
```

- [ ] **Step 2.4: Run tests, verify all 7 PASS.**

- [ ] **Step 2.5: Ruff + pyright clean on both files.**

- [ ] **Step 2.6: Commit**

```
git commit -m "BGCC: add training-time augmentation module (hue, saturation, gamma, RGB gain, chroma bleed)"
```

---

## Task 3: Wire augmentations into PairedCCDataset

**File:** `traiNNer/data/paired_cc_dataset.py`

- [ ] **Step 3.1: Add call in train-phase branch of `__getitem__`**

After the three `img2tensor` calls, before `normalize`:

```python
        # Training augmentations (BGCC-specific, invariant-respecting)
        if self.opt.phase == "train" and self.opt.bgcc_aug is not None:
            from traiNNer.data.bgcc_augment import apply_bgcc_augmentations
            lq_t, hr_t, gt_t = apply_bgcc_augmentations(
                lq_t, hr_t, gt_t, self.opt.bgcc_aug
            )
```

(Import at module top is also fine; inline is shown to reduce module load when augmentation is disabled.)

- [ ] **Step 3.2: Run the dataset test to confirm no regression**

```bash
pytest tests/test_data/test_paired_cc_dataset.py -v
pytest tests/test_data/test_bgcc_augment.py -v
ruff check traiNNer/data/paired_cc_dataset.py
```

All pass, ruff clean.

- [ ] **Step 3.3: Commit**

```
git commit -m "BGCC: wire training-time augmentations into PairedCCDataset"
```

---

## Task 4: Example YAML update

**File:** `options/train/BGCC/bgcc_2x.yml`

- [ ] **Step 4.1: Add `bgcc_aug` block to `datasets.train`**

Near the existing `use_hflip` / `use_rot` lines:

```yaml
    # BGCC training augmentations. All defaults 0 (disabled). Tuned values below
    # are conservative starting points for a small (4-show) dataset.
    bgcc_aug:
      hue_shift_deg: 5.0          # +/- degrees on (LR, CC_HR) — diversifies target palette
      saturation_delta: 0.10      # +/- 10% on (LR, CC_HR) — palette saturation variation
      hr_gamma_delta: 0.10        # HR gamma in [0.9, 1.1] — mastering curve variation
      hr_rgb_gain_delta: 0.05     # each HR channel x [0.95, 1.05] — white-balance drift
      lr_chroma_bleed_sigma: 0.8  # Gaussian sigma on Cb/Cr of LR — chroma bleed severity
```

Do NOT add the block to `datasets.val` — validation is unaugmented.

- [ ] **Step 4.2: Verify YAML parses**

```bash
python -c "
import yaml
with open('options/train/BGCC/bgcc_2x.yml') as f:
    d = yaml.safe_load(f)
print('bgcc_aug:', d['datasets']['train'].get('bgcc_aug'))
"
```

Expected: prints the dict.

- [ ] **Step 4.3: Commit**

```
git commit -m "BGCC: add bgcc_aug block with starting-point values to example YAML"
```

---

## Out of scope for v1

- Channel-permutation augmentation (deferred — optional per brainstorm, not in the shipped 5).
- Independent HR hue shift (deferred — optional per brainstorm; start without it and add if model underfits HR variety).
- JPEG re-compression augmentation (deferred — optional per brainstorm).
- Exposing per-augmentation enable flags (currently "delta = 0" disables each aug; additional flags would be YAGNI).

---

## Verification checklist

- [ ] All 7 BGCCAugOptions tests pass.
- [ ] PairedCCDataset tests still pass (no regression).
- [ ] Full BGCC test suite still green (15 previously passing + 7 new = 22).
- [ ] Example YAML parses.
- [ ] Ruff + pyright clean on all modified/created files.
- [ ] 4 focused commits (one per task).
