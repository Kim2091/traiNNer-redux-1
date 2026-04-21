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
    lr[:, :, :4, :4] = 0.9  # "bled" region — upsamples to top-left 8x8 of HR

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
    import traiNNer.losses.bgcc_bleed_focal_loss  # noqa: F401
    from traiNNer.utils.registry import LOSS_REGISTRY

    assert "bleedfocall1loss" in LOSS_REGISTRY
