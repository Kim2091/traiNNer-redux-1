import torch


def _make_dummy_opt():  # noqa: ANN202
    """Minimal ReduxOptions with the fields SRModel/BGCCModel touch.

    ReduxOptions required positional fields: name, scale, num_gpu, path.
    All other fields have defaults.
    """
    from traiNNer.utils.redux_options import PathOptions, ReduxOptions

    opt = ReduxOptions(
        name="bgcc_unit",
        scale=2,
        num_gpu=0,
        path=PathOptions(),
        use_amp=False,
        use_channels_last=False,
        amp_bf16=False,
        use_compile=False,
        high_order_degradation=False,
        network_g={"type": "bgcc", "feat": 16, "d": 4, "n_blocks_per_stage": 1},
    )
    return opt


def test_bgcc_model_feed_data_pulls_hr() -> None:
    from traiNNer.models.bgcc_model import BGCCModel

    opt = _make_dummy_opt()
    model = BGCCModel(opt)
    batch = {
        "lq": torch.randn(1, 3, 32, 32),
        "hr": torch.randn(1, 3, 64, 64),
        "gt": torch.randn(1, 3, 64, 64),
    }
    model.feed_data(batch)
    assert model.hr is not None
    assert model.hr.shape == (1, 3, 64, 64)
    assert model.lq.shape == (1, 3, 32, 32)


def test_bgcc_model_forward_passes_both_inputs() -> None:
    from traiNNer.models.bgcc_model import BGCCModel

    opt = _make_dummy_opt()
    model = BGCCModel(opt)
    batch = {
        "lq": torch.randn(1, 3, 32, 32),
        "hr": torch.randn(1, 3, 64, 64),
        "gt": torch.randn(1, 3, 64, 64),
    }
    model.feed_data(batch)
    out = model._run_net(model.net_g, model.lq)  # noqa: SLF001
    assert out.shape == (1, 3, 64, 64)
