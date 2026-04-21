from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def triplet_dataset_root(tmp_path: Path) -> dict[str, str]:
    roots: dict[str, str] = {}
    rng = np.random.default_rng(42)
    for key, (h, w) in [
        ("lr", (32, 32)),
        ("hr", (64, 64)),
        ("cc_hr", (64, 64)),
    ]:
        d = tmp_path / key
        d.mkdir()
        for i in range(2):
            arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img{i}.png")
        roots[key] = str(d)
    return roots


def test_paired_cc_dataset_produces_triplets(
    triplet_dataset_root: dict[str, str],
) -> None:
    from traiNNer.data.paired_cc_dataset import PairedCCDataset
    from traiNNer.utils.redux_options import DatasetOptions

    opt = DatasetOptions(
        name="test_cc",
        type="PairedCCDataset",
        dataroot_gt=[triplet_dataset_root["cc_hr"]],
        dataroot_hr=triplet_dataset_root["hr"],
        dataroot_lq=[triplet_dataset_root["lr"]],
        scale=2,
        phase="train",
        gt_size=32,
        use_hflip=True,
        use_rot=True,
        io_backend={"type": "disk"},
    )
    ds = PairedCCDataset(opt)
    sample = ds[0]
    assert "lq" in sample
    assert "hr" in sample
    assert "gt" in sample
    assert sample["lq"].shape == (3, 16, 16)
    assert sample["hr"].shape == (3, 32, 32)
    assert sample["gt"].shape == (3, 32, 32)
