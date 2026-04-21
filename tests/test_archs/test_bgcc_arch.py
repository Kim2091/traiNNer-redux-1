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
