"""SAR-OPT paired dataset pipeline with multi-scale SAR structure features."""

from .dataset import SAROPTPairedDataset
from .stats import ChannelStatsAccumulator, compute_dataset_channel_stats
from .transforms import (
    SARMultiscaleConfig,
    compute_sar_multiscale,
    normalize_opt,
    normalize_sar,
)
from .visualize import save_sar_ms_channels, visualize_batch

__all__ = [
    "SAROPTPairedDataset",
    "SARMultiscaleConfig",
    "compute_sar_multiscale",
    "normalize_sar",
    "normalize_opt",
    "visualize_batch",
    "save_sar_ms_channels",
    "ChannelStatsAccumulator",
    "compute_dataset_channel_stats",
]
