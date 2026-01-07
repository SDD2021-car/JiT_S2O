from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf

from .transforms import SARMultiscaleConfig, compute_sar_multiscale, normalize_opt, normalize_sar

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    files = [p for p in root_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    return sorted(files)


class SAROPTPairedDataset(Dataset):
    """Paired SAR/OPT dataset producing SAR raw + multiscale structure + OPT target.

    Output:
        sar_raw: (1, H, W) or (3, H, W) tensor
        sar_ms:  (C, H, W) tensor, C in [3, 6]
        opt_gt:  (3, H, W) tensor
    """

    def __init__(
        self,
        sar_root: str | Path,
        opt_root: str | Path,
        sar_clip_max: float = 2.0,
        sar_ms_config: Optional[SARMultiscaleConfig] = None,
        opt_norm_mode: str = "minus_one_one",
        opt_mean: Optional[Iterable[float]] = None,
        opt_std: Optional[Iterable[float]] = None,
        shared_transform: Optional[Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]] = None,
    ) -> None:
        self.sar_root = Path(sar_root)
        self.opt_root = Path(opt_root)
        self.sar_files = _list_images(self.sar_root)
        self.opt_files = _list_images(self.opt_root)
        if len(self.sar_files) != len(self.opt_files):
            raise ValueError("SAR and OPT datasets must be the same length.")
        for sar_path, opt_path in zip(self.sar_files, self.opt_files):
            if sar_path.name != opt_path.name:
                raise ValueError(f"Mismatched filenames: {sar_path.name} vs {opt_path.name}")
        self.sar_clip_max = sar_clip_max
        self.sar_ms_config = sar_ms_config or SARMultiscaleConfig()
        self.opt_norm_mode = opt_norm_mode
        self.opt_mean = opt_mean
        self.opt_std = opt_std
        self.shared_transform = shared_transform

    def __len__(self) -> int:
        return len(self.sar_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sar_path = self.sar_files[idx]
        opt_path = self.opt_files[idx]

        sar_img = Image.open(sar_path).convert("L")
        opt_img = Image.open(opt_path).convert("RGB")

        if self.shared_transform is not None:
            sar_img, opt_img = self.shared_transform(sar_img, opt_img)

        sar_tensor = tvf.to_tensor(sar_img)
        opt_tensor = tvf.to_tensor(opt_img)

        sar_raw = normalize_sar(sar_tensor, clip_max=self.sar_clip_max)
        sar_ms = compute_sar_multiscale(sar_raw, config=self.sar_ms_config)
        opt_gt = normalize_opt(opt_tensor, mode=self.opt_norm_mode, mean=self.opt_mean, std=self.opt_std)

        return {
            "sar_raw": sar_raw,
            "sar_ms": sar_ms,
            "opt_gt": opt_gt,
            "name": sar_path.name,
        }
