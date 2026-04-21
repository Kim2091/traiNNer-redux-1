from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY  # noqa: F401


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
        bin_f = guidance.squeeze(1) * (d - 1)  # (B, H_hr, W_hr)
        bin_lo = bin_f.floor().clamp(0, d - 1).long()
        bin_hi = (bin_lo + 1).clamp(max=d - 1)
        w_hi = (bin_f - bin_lo.float()).unsqueeze(1)  # (B, 1, H_hr, W_hr)
        w_lo = 1.0 - w_hi

        # Gather along bin axis.
        # idx shape must match grid_up except on the gather axis:
        # grid_up is (B, C, D, H_hr, W_hr); idx must be (B, C, 1, H_hr, W_hr).
        idx_lo = bin_lo.unsqueeze(1).unsqueeze(1).expand(b, c, 1, h_hr, w_hr)
        idx_hi = bin_hi.unsqueeze(1).unsqueeze(1).expand(b, c, 1, h_hr, w_hr)
        m_lo = torch.gather(grid_up, 2, idx_lo).squeeze(2)  # (B, C, H_hr, W_hr)
        m_hi = torch.gather(grid_up, 2, idx_hi).squeeze(2)

        return w_lo * m_lo + w_hi * m_hi


# BGCC main module is added in a later task.
