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


def _chroma_subsample_simulation(
    rgb: Tensor, factor: int, shift_dy: int, shift_dx: int
) -> Tensor:
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
    chroma = ycbcr[1:3].unsqueeze(0)  # (1, 2, H, W)
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
        factor = random.uniform(
            1.0 - opts.saturation_delta, 1.0 + opts.saturation_delta
        )
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
                random.uniform(
                    1.0 - opts.hr_rgb_gain_delta, 1.0 + opts.hr_rgb_gain_delta
                )
                for _ in range(3)
            ],
            dtype=hr.dtype,
            device=hr.device,
        ).view(3, 1, 1)
        hr = (hr * gains).clamp(0.0, 1.0)

    # 5. Chroma subsampling simulation on LR only (realistic DVD/LD bleed)
    if opts.lr_chroma_subsample_factor > 1:
        factor = random.randint(2, opts.lr_chroma_subsample_factor)
        # Optional random 1-pixel shift (50% chance) to simulate chroma misalignment
        shift_dy = random.choice([-1, 0, 1]) if random.random() < 0.5 else 0
        shift_dx = random.choice([-1, 0, 1]) if random.random() < 0.5 else 0
        lr = _chroma_subsample_simulation(lr, factor, shift_dy, shift_dx)

    return lr, hr, cc_hr
