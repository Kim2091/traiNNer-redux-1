import os

import numpy as np

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_util import paired_paths_from_folder
from traiNNer.data.transforms import (
    augment_vips_triplet,
    paired_random_crop_triplet_vips,
)
from traiNNer.utils import FileClient, img2tensor
from traiNNer.utils.img_util import img2rgb, vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class PairedCCDataset(BaseDataset):
    """Triplet dataset for BGCC color-correction training.

    Reads three folders:
        - dataroot_lq: LR images (DVD/LD, degraded, target colors).
        - dataroot_hr: HR reference images (BD, sharp structure, wrong colors).
        - dataroot_gt: CC_HR target images (manually color-corrected HR).
    All three folders must share image basenames.

    Returns dict with keys: "lq", "hr", "gt", plus "*_path".
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.mean = opt.mean
        self.std = opt.std

        assert isinstance(opt.dataroot_lq, list), "dataroot_lq must be a list"
        assert isinstance(opt.dataroot_gt, list), "dataroot_gt must be a list"
        assert opt.dataroot_hr is not None, "dataroot_hr required"

        self.lq_folder = opt.dataroot_lq
        self.gt_folder = opt.dataroot_gt
        self.hr_folder = (
            opt.dataroot_hr if isinstance(opt.dataroot_hr, list) else [opt.dataroot_hr]
        )
        self.filename_tmpl = opt.filename_tmpl

        lq_gt_paths = paired_paths_from_folder(
            (self.lq_folder, self.gt_folder), ("lq", "gt"), self.filename_tmpl
        )
        self.paths: list[dict[str, str]] = []
        for entry in lq_gt_paths:
            lq_path = entry["lq_path"]
            gt_path = entry["gt_path"]
            basename = os.path.basename(lq_path)
            hr_path = None
            for root in self.hr_folder:
                candidate = os.path.join(root, basename)
                if os.path.exists(candidate):
                    hr_path = candidate
                    break
            if hr_path is None:
                raise FileNotFoundError(
                    f"No HR reference found for {basename} in {self.hr_folder}"
                )
            self.paths.append(
                {"lq_path": lq_path, "gt_path": gt_path, "hr_path": hr_path}
            )

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt.scale
        assert scale is not None

        entry = self.paths[index]
        lq_path, gt_path, hr_path = (
            entry["lq_path"],
            entry["gt_path"],
            entry["hr_path"],
        )

        vips_lq = vipsimfrompath(lq_path)
        vips_hr = vipsimfrompath(hr_path)
        vips_gt = vipsimfrompath(gt_path)

        if self.opt.phase == "train":
            assert self.opt.gt_size is not None
            assert self.opt.use_hflip is not None
            assert self.opt.use_rot is not None

            vips_hr, vips_gt, vips_lq = augment_vips_triplet(
                (vips_hr, vips_gt, vips_lq),
                self.opt.use_hflip,
                self.opt.use_rot,
                self.opt.use_rot,
            )

            img_hr_raw, img_gt_raw, img_lq_raw = paired_random_crop_triplet_vips(
                img_hr_ref=vips_hr,
                img_cc_hr=vips_gt,
                img_lq=vips_lq,
                gt_patch_size=self.opt.gt_size,
                scale=scale,
            )

            img_lq = img2rgb(img_lq_raw)
            img_hr = img2rgb(img_hr_raw)
            img_gt = img2rgb(img_gt_raw)

            assert isinstance(img_lq, np.ndarray)
            assert isinstance(img_hr, np.ndarray)
            assert isinstance(img_gt, np.ndarray)
        else:
            img_lq = img2rgb(vips_lq.numpy())
            img_hr = img2rgb(vips_hr.numpy())
            img_gt = img2rgb(vips_gt.numpy())
            h_trim = img_lq.shape[0] * scale
            w_trim = img_lq.shape[1] * scale
            img_hr = img_hr[:h_trim, :w_trim, :]
            img_gt = img_gt[:h_trim, :w_trim, :]

        lq_t = img2tensor(img_lq, float32=True, from_bgr=False)
        hr_t = img2tensor(img_hr, float32=True, from_bgr=False)
        gt_t = img2tensor(img_gt, float32=True, from_bgr=False)

        if self.mean is not None and self.std is not None:
            from torchvision.transforms.functional import normalize

            normalize(lq_t, self.mean, self.std, inplace=True)
            normalize(hr_t, self.mean, self.std, inplace=True)
            normalize(gt_t, self.mean, self.std, inplace=True)

        return {
            "lq": lq_t,
            "hr": hr_t,
            "gt": gt_t,
            "lq_path": lq_path,
            "hr_path": hr_path,
            "gt_path": gt_path,
        }

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "triplet (lq, hr_ref, cc_hr) images"
