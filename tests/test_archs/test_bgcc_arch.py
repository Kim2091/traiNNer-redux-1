from pathlib import Path

import pytest
import torch


def test_fast_res_block_preserves_shape() -> None:
    from traiNNer.archs.bgcc_arch import FastResBlock

    block = FastResBlock(channels=16)
    x = torch.randn(2, 16, 32, 32)
    y = block(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_guidance_head_outputs_single_channel_in_range() -> None:
    from traiNNer.archs.bgcc_arch import EdgeAwareGuidanceHead

    head = EdgeAwareGuidanceHead(hidden=8)
    hr = torch.randn(2, 3, 64, 64).clamp(0, 1)
    edge = torch.rand(2, 1, 64, 64)
    g = head(hr, edge)
    assert g.shape == (2, 1, 64, 64)
    assert g.min() >= 0.0 and g.max() <= 1.0


def test_bilateral_slicer_output_shape() -> None:
    from traiNNer.archs.bgcc_arch import BilateralSlicer

    slicer = BilateralSlicer()
    # grid: (B=2, 12 coeffs, D=8 bins, H'=4, W'=4), guidance: (B=2, 1, H_hr=32, W_hr=32)
    grid = torch.randn(2, 12, 8, 4, 4)
    guidance = torch.rand(2, 1, 32, 32)
    m = slicer(grid, guidance)
    assert m.shape == (2, 12, 32, 32)


def test_bilateral_slicer_identity_grid_gives_identity_matrices() -> None:
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


def test_bilateral_slicer_interpolates_between_bins() -> None:
    """With guidance at 0.5 and D=2, output should be the mean of the two bins."""
    from traiNNer.archs.bgcc_arch import BilateralSlicer

    slicer = BilateralSlicer()
    grid = torch.zeros(1, 12, 2, 4, 4)
    grid[:, :, 0] = 1.0  # all of bin 0 is 1
    grid[:, :, 1] = 3.0  # all of bin 1 is 3
    guidance = torch.full((1, 1, 8, 8), 0.5)  # guidance * (D-1) = 0.5, midpoint
    m = slicer(grid, guidance)
    assert torch.allclose(m, torch.full_like(m, 2.0), atol=1e-5)


def test_bgcc_forward_shape_2x() -> None:
    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=32, d=8)
    hr = torch.randn(2, 3, 128, 128)
    lr = torch.randn(2, 3, 64, 64)
    out = model(hr, lr)
    assert out.shape == hr.shape


def test_bgcc_forward_shape_flexible_scale() -> None:
    """Model trained at 2x should still run at 3x at inference without code changes."""
    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=32, d=8)
    hr = torch.randn(1, 3, 192, 192)
    lr = torch.randn(1, 3, 64, 64)
    out = model(hr, lr)
    assert out.shape == hr.shape


def test_bgcc_initial_output_matches_hr_within_tolerance() -> None:
    """Zero-init + residual => output should equal hr at init (zero training)."""
    from traiNNer.archs.bgcc_arch import bgcc

    model = bgcc(feat=32, d=8).eval()
    hr = torch.randn(1, 3, 64, 64)
    lr = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(hr, lr)
    assert torch.allclose(out, hr, atol=1e-5)


def test_bgcc_gradients_flow_end_to_end() -> None:
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


def test_bgcc_onnx_export_round_trip(tmp_path: Path) -> None:
    """Export BGCC to ONNX opset 17 and compare against PyTorch forward."""
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
    ort_out = sess.run(None, {"hr": hr.numpy(), "lq": lr.numpy()})[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-3)


def test_bgcc_onnx_export_dynamic_scale_inference(tmp_path: Path) -> None:
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
        model,
        (hr_export, lr_export),
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

    rng = np.random.default_rng(0)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        None,
        {
            "hr": rng.standard_normal((1, 3, 72, 72)).astype(np.float32),
            "lq": rng.standard_normal((1, 3, 24, 24)).astype(np.float32),
        },
    )[0]
    assert ort_out.shape == (1, 3, 72, 72)
