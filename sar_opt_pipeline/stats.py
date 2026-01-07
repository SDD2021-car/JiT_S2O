from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from .dataset import SAROPTPairedDataset


@dataclass
class ChannelStatsAccumulator:
    """Accumulate mean/std for multi-channel tensors."""

    channels: int
    sum: torch.Tensor
    sumsq: torch.Tensor
    count: int

    @classmethod
    def create(cls, channels: int) -> "ChannelStatsAccumulator":
        return cls(channels=channels, sum=torch.zeros(channels), sumsq=torch.zeros(channels), count=0)

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.dim() != 4 or tensor.shape[1] != self.channels:
            raise ValueError("Expected tensor shape (B, C, H, W) with matching channels")
        batch = tensor.shape[0]
        flattened = tensor.permute(1, 0, 2, 3).reshape(self.channels, -1)
        self.sum += flattened.sum(dim=1)
        self.sumsq += (flattened**2).sum(dim=1)
        self.count += flattened.shape[1]

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.sum / max(self.count, 1)
        var = self.sumsq / max(self.count, 1) - mean**2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return mean, std


def compute_dataset_channel_stats(
    dataset: SAROPTPairedDataset,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute mean/std for sar_raw, sar_ms, opt_gt across dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    sar_raw_acc = ChannelStatsAccumulator.create(1)
    sar_ms_acc = None
    opt_acc = ChannelStatsAccumulator.create(3)

    for batch in loader:
        sar_raw = batch["sar_raw"]
        sar_ms = batch["sar_ms"]
        opt_gt = batch["opt_gt"]
        sar_raw_acc.update(sar_raw)
        if sar_ms_acc is None:
            sar_ms_acc = ChannelStatsAccumulator.create(sar_ms.shape[1])
        sar_ms_acc.update(sar_ms)
        opt_acc.update(opt_gt)

    sar_raw_stats = sar_raw_acc.finalize()
    sar_ms_stats = sar_ms_acc.finalize() if sar_ms_acc is not None else (torch.tensor([]), torch.tensor([]))
    opt_stats = opt_acc.finalize()
    return {
        "sar_raw": sar_raw_stats,
        "sar_ms": sar_ms_stats,
        "opt_gt": opt_stats,
    }