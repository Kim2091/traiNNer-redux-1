from __future__ import annotations

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
        weight: Tensor | None = None,
        lq: Tensor | None = None,
        **kwargs,
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
        weight_map = 1.0 + self.focal_gain * bleed_mask  # (B, 1, H, W)
        return self.loss_weight * (abs_diff * weight_map).mean()
