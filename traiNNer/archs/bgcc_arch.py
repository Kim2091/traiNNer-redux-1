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
        x = torch.cat([lr, hr_ds], dim=1)  # (B, 6, H_lr, W_lr)

        # Encoder
        x = self.stem_act(self.stem(x))
        x = self.stage0(x)
        x = self.stage1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.stage3(self.down3(x))

        # Predict the bilateral grid: (B, 12*D, H_lr/8, W_lr/8)
        grid_flat = self.grid_head(x)
        # Reshape to (B, 12, D, H', W')
        grid = grid_flat.reshape(
            b, self.coeffs_per_voxel, self.d, *grid_flat.shape[-2:]
        )

        # Learned 1-ch guidance from HR
        guidance = self.guidance(hr)  # (B, 1, H_hr, W_hr)

        # Slice: per-HR-pixel 3x4 affine matrix coefficients.
        m = self.slicer(grid, guidance)  # (B, 12, H_hr, W_hr)

        # Apply per-pixel affine to HR.
        # M reshaped to (B, 3, 4, H_hr, W_hr); [R,G,B,1] to (B, 4, H_hr, W_hr).
        m = m.view(b, 3, 4, h_hr, w_hr)
        hr_aug = torch.cat([hr, torch.ones_like(hr[:, :1])], dim=1)  # (B, 4, H, W)
        out = (m * hr_aug.unsqueeze(1)).sum(dim=2)  # (B, 3, H_hr, W_hr)

        # Residual against HR for init stability (grid_head is zero-init -> out starts 0).
        return out + hr


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def bgcc(
    feat: int = 32,
    d: int = 8,
    n_blocks_per_stage: int = 2,
    guidance_hidden: int = 8,
    scale: int = 2,  # accepted for framework compatibility; not used architecturally
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
    scale: int = 2,  # accepted for framework compatibility; not used architecturally
) -> BGCC:
    return BGCC(
        feat=feat,
        d=d,
        n_blocks_per_stage=n_blocks_per_stage,
        guidance_hidden=guidance_hidden,
    )
