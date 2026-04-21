from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def tiny_triplet_root(tmp_path: Path) -> dict[str, str]:
    """4 fake triplets with consistent basenames; LR 16x16, HR/CC_HR 32x32."""
    roots = {}
    for key, (h, w) in [("lr", (16, 16)), ("hr", (32, 32)), ("cc_hr", (32, 32))]:
        d = tmp_path / key
        d.mkdir()
        for i in range(4):
            rng = np.random.default_rng(seed=i * 7 + hash(key) % 1000)
            arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img{i}.png")
        roots[key] = str(d)
    return roots


def test_bgcc_trains_loss_decreases(tiny_triplet_root: dict[str, str]) -> None:
    """50-iter smoke: verify loss decreases on a tiny batch."""
    from traiNNer.archs.bgcc_arch import bgcc
    from traiNNer.data.paired_cc_dataset import PairedCCDataset
    from traiNNer.utils.redux_options import DatasetOptions

    opt = DatasetOptions(
        name="smoke",
        type="PairedCCDataset",
        dataroot_lq=[tiny_triplet_root["lr"]],
        dataroot_hr=tiny_triplet_root["hr"],
        dataroot_gt=[tiny_triplet_root["cc_hr"]],
        scale=2,
        phase="train",
        gt_size=32,
        use_hflip=False,
        use_rot=False,
        io_backend={"type": "disk"},
    )
    ds = PairedCCDataset(opt)

    torch.manual_seed(0)
    model = bgcc(feat=16, d=4)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)

    initial_losses = []
    final_losses = []
    for step in range(50):
        idx = step % len(ds)
        sample = ds[idx]
        hr = sample["hr"].unsqueeze(0)
        lr = sample["lq"].unsqueeze(0)
        gt = sample["gt"].unsqueeze(0)
        out = model(hr, lr)
        loss = (out - gt).abs().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step < 5:
            initial_losses.append(loss.item())
        if step >= 45:
            final_losses.append(loss.item())

    assert np.mean(final_losses) < np.mean(initial_losses), (
        f"Loss did not decrease: initial={np.mean(initial_losses):.4f} "
        f"final={np.mean(final_losses):.4f}"
    )
