import numpy as np
import pyvips


def _make_vips_rgb(h: int, w: int, seed: int) -> pyvips.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return pyvips.Image.new_from_array(arr)


def test_triplet_random_crop_returns_aligned_regions() -> None:
    from traiNNer.data.transforms import paired_random_crop_triplet_vips

    img_lr = _make_vips_rgb(32, 32, seed=0)
    img_hr = _make_vips_rgb(64, 64, seed=1)
    img_cc = _make_vips_rgb(64, 64, seed=2)

    arr_hr, arr_cc, arr_lr = paired_random_crop_triplet_vips(
        img_hr_ref=img_hr,
        img_cc_hr=img_cc,
        img_lq=img_lr,
        gt_patch_size=32,
        scale=2,
    )
    assert arr_lr.shape == (16, 16, 3)
    assert arr_hr.shape == (32, 32, 3)
    assert arr_cc.shape == (32, 32, 3)


def test_triplet_augment_applies_identical_flip_rot_to_all_three() -> None:
    from traiNNer.data.transforms import augment_vips_triplet

    img_lr = _make_vips_rgb(32, 32, seed=0)
    img_hr = _make_vips_rgb(64, 64, seed=1)
    img_cc = _make_vips_rgb(64, 64, seed=2)

    out_hr, out_cc, out_lr = augment_vips_triplet(
        (img_hr, img_cc, img_lr),
        hflip=True,
        vflip=False,
        rot90=False,
        force_hflip=True,
        force_vflip=False,
        force_rot90=False,
    )
    expected_hr = np.flip(img_hr.numpy(), axis=1)
    expected_cc = np.flip(img_cc.numpy(), axis=1)
    expected_lr = np.flip(img_lr.numpy(), axis=1)
    assert np.array_equal(out_hr.numpy(), expected_hr)
    assert np.array_equal(out_cc.numpy(), expected_cc)
    assert np.array_equal(out_lr.numpy(), expected_lr)
