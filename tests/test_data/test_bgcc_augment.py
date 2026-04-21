import random

import torch


def _make_triplet(
    h: int = 32, w: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    sentinel[0, 0, 0] = 0.7  # make it non-uniform so adjust_hue isn't a no-op
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
