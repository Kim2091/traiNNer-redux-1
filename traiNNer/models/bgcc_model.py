from __future__ import annotations

from torch import Tensor, nn

from traiNNer.models.sr_model import SRModel
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


class BGCCModel(SRModel):
    """Dual-input color correction model.

    Extends SRModel to:
        - pull an additional `hr` tensor from each batch (the color-incorrect
          HR reference that the network should color-correct).
        - call `net_g(hr, lq)` instead of `net_g(lq)` via the `_run_net` hook.

    The supervision target stays in the `gt` key and is the CC_HR image.
    Existing SRModel loss/optimization/validation paths are unchanged.
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)
        self.hr: Tensor | None = None

    def feed_data(self, data: DataFeed) -> None:
        super().feed_data(data)
        assert "hr" in data, (
            "BGCCModel expects batches with an 'hr' key "
            "(the color-incorrect HR reference). Use PairedCCDataset."
        )
        self.hr = data["hr"].to(
            self.device, memory_format=self.memory_format, non_blocking=True
        )

    def _run_net(self, net: nn.Module, lq: Tensor) -> Tensor:
        assert self.hr is not None, (
            "BGCCModel._run_net called before feed_data — self.hr is None"
        )
        return net(self.hr, lq)
