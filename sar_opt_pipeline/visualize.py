from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    return tensor.numpy()


def visualize_batch(
    sar_raw: torch.Tensor,
    sar_ms: torch.Tensor,
    opt_gt: torch.Tensor,
    save_path: str | Path,
    max_items: int = 4,
) -> Path:
    """Visualize a batch of SAR raw, SAR multiscale, and OPT GT.

    Args:
        sar_raw: (B, 1, H, W)
        sar_ms: (B, C, H, W)
        opt_gt: (B, 3, H, W)
    """
    save_path = Path(save_path)
    batch = min(sar_raw.shape[0], sar_ms.shape[0], opt_gt.shape[0], max_items)
    rows = batch
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = np.array([axes])

    for i in range(rows):
        sar_raw_img = _to_numpy_image(sar_raw[i])
        sar_ms_img = sar_ms[i]
        if sar_ms_img.shape[0] >= 3:
            sar_ms_img = sar_ms_img[:3]
        else:
            sar_ms_img = sar_ms_img.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        sar_ms_img = _to_numpy_image(sar_ms_img)
        opt_img = _to_numpy_image(opt_gt[i])

        axes[i, 0].imshow(np.clip(sar_raw_img.squeeze(), 0.0, 1.0), cmap="gray")
        axes[i, 0].set_title("SAR Raw")
        axes[i, 1].imshow(np.clip(sar_ms_img, 0.0, 1.0))
        axes[i, 1].set_title("SAR MS")
        axes[i, 2].imshow(np.clip((opt_img + 1.0) / 2.0, 0.0, 1.0))
        axes[i, 2].set_title("OPT GT")
        for j in range(cols):
            axes[i, j].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
