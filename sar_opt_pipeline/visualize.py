from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use("Agg")


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    return tensor.numpy()


def _normalize_feature_map(feature: torch.Tensor) -> np.ndarray:
    feature = feature.detach().float().cpu()
    if feature.dim() == 3:
        feature = feature.mean(dim=0)
    feature = feature.numpy()
    min_val = float(feature.min())
    max_val = float(feature.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(feature)
    return (feature - min_val) / (max_val - min_val)


def _progress_iter(total: int, desc: str) -> Iterable[int]:
    if total <= 0:
        return []
    for i in range(total):
        done = i + 1
        percent = done / total * 100
        bar_len = 20
        filled = int(bar_len * done / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r{desc}: [{bar}] {done}/{total} ({percent:5.1f}%)", end="")
        yield i
    print()


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


def save_sar_ms_channels(
    sar_ms: torch.Tensor,
    save_dir: str | Path,
    prefix: str = "sar_ms",
    max_items: int = 4,
) -> list[Path]:
    """Save each SAR MS channel as a separate image with distinct names.

    Args:
        sar_ms: (B, C, H, W)
        save_dir: directory to save images
        prefix: filename prefix
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    batch = min(sar_ms.shape[0], max_items)
    saved: list[Path] = []

    total = batch * sar_ms.shape[1]
    for idx in _progress_iter(total, desc="Saving SAR MS channels"):
        i = idx // sar_ms.shape[1]
        c = idx % sar_ms.shape[1]
        channel = _to_numpy_image(sar_ms[i, c : c + 1])
        filename = f"{prefix}_sample{i:03d}_ch{c:02d}.png"
        out_path = save_dir / filename
        plt.imsave(out_path, np.clip(channel.squeeze(), 0.0, 1.0), cmap="gray")
        saved.append(out_path)
    return saved


def visualize_sar_ms_channels(
    sar_ms: torch.Tensor,
    save_dir: str | Path,
    prefix: str = "sar_ms",
    max_items: int = 4,
    cols: int = 4,
) -> list[Path]:
    """Save a grid visualization per sample for SAR MS channels."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    batch = min(sar_ms.shape[0], max_items)
    saved: list[Path] = []
    num_channels = sar_ms.shape[1]
    cols = max(1, min(cols, num_channels))
    rows = int(np.ceil(num_channels / cols))

    for i in _progress_iter(batch, desc="Saving SAR MS grids"):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        for c in range(rows * cols):
            r = c // cols
            col = c % cols
            ax = axes[r, col]
            ax.axis("off")
            if c >= num_channels:
                continue
            channel = _to_numpy_image(sar_ms[i, c : c + 1])
            ax.imshow(np.clip(channel.squeeze(), 0.0, 1.0), cmap="gray")
            ax.set_title(f"ch{c:02d}")

        fig.tight_layout()
        out_path = save_dir / f"{prefix}_sample{i:03d}_grid.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
    return saved


def visualize_sar_encoder_layers(
    layers: dict[str, torch.Tensor],
    save_dir: str | Path,
    prefix: str = "sar_encoder",
    max_items: int = 4,
) -> list[Path]:
    """将 SAR encoder 各层特征输出可视化保存为图片。

    Args:
        layers: dict of (name -> feature map), each is (B, C, H, W)
        save_dir: output directory
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not layers:
        return []

    batch = min(next(iter(layers.values())).shape[0], max_items)
    saved: list[Path] = []
    total = batch * len(layers)

    layer_items = list(layers.items())
    for idx in _progress_iter(total, desc="Saving SAR encoder layers"):
        sample_idx = idx // len(layer_items)
        layer_idx = idx % len(layer_items)
        layer_name, feature = layer_items[layer_idx]
        fmap = _normalize_feature_map(feature[sample_idx])
        out_path = save_dir / f"{prefix}_sample{sample_idx:03d}_{layer_name}.png"
        plt.imsave(out_path, np.clip(fmap, 0.0, 1.0), cmap="viridis")
        saved.append(out_path)

    return saved
