from __future__ import annotations

from typing import cast

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


class EdgeAwareGuidanceHead(nn.Module):
    """Predict a 1-channel bilateral guidance map from HR RGB and local edges."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, hidden, 3, padding=1, bias=True)
        self.act = nn.PReLU(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, hr: Tensor, edge: Tensor) -> Tensor:
        y = torch.cat([hr, edge], dim=1)
        y = self.act(self.conv1(y))
        y = self.act(self.conv2(y))
        y = self.conv3(y)
        return torch.sigmoid(y)


class HRRefineHead(nn.Module):
    """Predict a sharp HR residual after coarse bilateral color transfer."""

    def __init__(self, feat: int, n_blocks: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(8, feat, 3, padding=1, bias=True)
        self.act = nn.PReLU(feat)
        self.body = nn.Sequential(*[FastResBlock(feat) for _ in range(n_blocks)])
        self.conv_out = nn.Conv2d(feat, 3, 3, padding=1, bias=True)

        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def forward(
        self, coarse: Tensor, hr: Tensor, guidance: Tensor, edge: Tensor
    ) -> Tensor:
        x = torch.cat([coarse, hr, guidance, edge], dim=1)
        x = self.act(self.conv_in(x))
        x = self.body(x)
        return self.conv_out(x)


class BilateralSlicer(nn.Module):
    """Manual trilinear slice of a bilateral grid.

    Bilinear spatial upsample + linear bin interp via gather, so we stay within
    ONNX opset 16 (avoids 5D grid_sample).

    Inputs:
        grid: (B, C=12, D, H', W') of transform coefficients.
        guidance: (B, 1, H_hr, W_hr) in [0, 1].

    Output:
        M: (B, 12, H_hr, W_hr) — per-HR-pixel transform coefficients.
    """

    def forward(self, grid: Tensor, guidance: Tensor) -> Tensor:
        b, c, d, h_grid, w_grid = grid.shape
        _, _, h_hr, w_hr = guidance.shape

        # Fold D into channels so a single interpolate handles all bins.
        grid_flat = grid.reshape(b, c * d, h_grid, w_grid)
        grid_up = F.interpolate(
            grid_flat, size=(h_hr, w_hr), mode="bilinear", align_corners=False
        )
        grid_up = grid_up.reshape(b, c, d, h_hr, w_hr)

        bin_f = guidance.squeeze(1) * (d - 1)  # (B, H_hr, W_hr)
        bin_lo = bin_f.floor().clamp(0, d - 1).long()
        bin_hi = (bin_lo + 1).clamp(max=d - 1)
        w_hi = (bin_f - bin_lo.float()).unsqueeze(1)  # (B, 1, H_hr, W_hr)
        w_lo = 1.0 - w_hi

        # idx must broadcast to grid_up's shape except along the gather axis:
        # grid_up is (B, C, D, H_hr, W_hr), so idx must be (B, C, 1, H_hr, W_hr).
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
        refine_blocks: Number of FastResBlocks in the HR refinement head.
    """

    def __init__(
        self,
        feat: int = 32,
        d: int = 8,
        n_blocks_per_stage: int = 2,
        guidance_hidden: int = 8,
        refine_blocks: int = 2,
    ) -> None:
        super().__init__()
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

        # Fuse multiscale encoder features back to LR resolution before predicting
        # the bilateral grid so coefficients can change at every LR pixel.
        self.grid_proj0 = nn.Conv2d(feat, feat, 1, bias=False)
        self.grid_proj1 = nn.Conv2d(feat, feat, 1, bias=False)
        self.grid_proj2 = nn.Conv2d(feat, feat, 1, bias=False)
        self.grid_proj3 = nn.Conv2d(feat, feat, 1, bias=False)
        self.grid_fusion = nn.Sequential(
            nn.Conv2d(feat * 4, feat, 3, padding=1, bias=True),
            nn.PReLU(feat),
            FastResBlock(feat),
            nn.Conv2d(feat, feat, 3, padding=1, bias=True),
            nn.PReLU(feat),
        )
        self.grid_head = nn.Conv2d(feat, self.coeffs_per_voxel * d, 1, bias=True)

        # Zero-init the grid head so output = HR + 0 at init.
        nn.init.zeros_(self.grid_head.weight)
        if self.grid_head.bias is not None:
            nn.init.zeros_(self.grid_head.bias)

        self.guidance = EdgeAwareGuidanceHead(hidden=guidance_hidden)
        self.slicer = BilateralSlicer()
        self.refine = HRRefineHead(feat=feat, n_blocks=refine_blocks)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _edge_map(self, hr: Tensor) -> Tensor:
        # register_buffer returns Tensor | Module to pyright; cast for the conv call.
        sobel_x = cast(Tensor, self.sobel_x)
        sobel_y = cast(Tensor, self.sobel_y)
        luma = 0.299 * hr[:, 0:1] + 0.587 * hr[:, 1:2] + 0.114 * hr[:, 2:3]
        grad_x = F.conv2d(luma, sobel_x, padding=1)
        grad_y = F.conv2d(luma, sobel_y, padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)

    def forward(self, hr: Tensor, lr: Tensor) -> Tensor:
        b, _, h_hr, w_hr = hr.shape
        _, _, h_lr, w_lr = lr.shape

        # Downsample HR to LR size and concat with LR along channel dim.
        hr_ds = F.interpolate(
            hr, size=(h_lr, w_lr), mode="bilinear", align_corners=False
        )
        x = torch.cat([lr, hr_ds], dim=1)  # (B, 6, H_lr, W_lr)

        x = self.stem_act(self.stem(x))
        x0 = self.stage0(x)
        x1 = self.stage1(self.down1(x0))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))

        projs = (
            self.grid_proj0(x0),
            self.grid_proj1(x1),
            self.grid_proj2(x2),
            self.grid_proj3(x3),
        )
        grid_feat = torch.cat(
            [
                p
                if p.shape[-2:] == (h_lr, w_lr)
                else F.interpolate(
                    p, size=(h_lr, w_lr), mode="bilinear", align_corners=False
                )
                for p in projs
            ],
            dim=1,
        )
        grid_feat = self.grid_fusion(grid_feat)

        grid_flat = self.grid_head(grid_feat)
        grid = grid_flat.reshape(
            b, self.coeffs_per_voxel, self.d, *grid_flat.shape[-2:]
        )

        edge = self._edge_map(hr)
        guidance = self.guidance(hr, edge)  # (B, 1, H_hr, W_hr)

        m = self.slicer(grid, guidance)  # (B, 12, H_hr, W_hr)

        # Reshape M to (B, 3, 4, H, W) and pad HR with 1s to (B, 4, H, W) so the
        # per-pixel 3x4 affine applies as a pointwise multiply + sum along inputs.
        m = m.view(b, 3, 4, h_hr, w_hr)
        hr_aug = torch.cat([hr, torch.ones_like(hr[:, :1])], dim=1)
        out = (m * hr_aug.unsqueeze(1)).sum(dim=2)

        # Residual against HR for init stability (grid_head is zero-init -> out starts 0).
        coarse = out + hr
        refine = self.refine(coarse, hr, guidance, edge)
        return coarse + refine


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def bgcc(
    feat: int = 32,
    d: int = 8,
    n_blocks_per_stage: int = 2,
    guidance_hidden: int = 8,
    refine_blocks: int = 2,
    scale: int = 2,  # accepted for framework compatibility; not used architecturally
) -> BGCC:
    return BGCC(
        feat=feat,
        d=d,
        n_blocks_per_stage=n_blocks_per_stage,
        guidance_hidden=guidance_hidden,
        refine_blocks=refine_blocks,
    )


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def bgcc_tiny(
    feat: int = 16,
    d: int = 8,
    n_blocks_per_stage: int = 1,
    guidance_hidden: int = 8,
    refine_blocks: int = 1,
    scale: int = 2,  # accepted for framework compatibility; not used architecturally
) -> BGCC:
    return BGCC(
        feat=feat,
        d=d,
        n_blocks_per_stage=n_blocks_per_stage,
        guidance_hidden=guidance_hidden,
        refine_blocks=refine_blocks,
    )
