"""SAR-OPT paired dataset pipeline with multi-scale SAR structure features."""

from .dataset import SAROPTPairedDataset
from .stats import ChannelStatsAccumulator, compute_dataset_channel_stats
from .transforms import (
    SARMultiscaleConfig,
    compute_sar_multiscale,
    compute_sar_multiscale_channels,
    normalize_opt,
    normalize_sar,
)
from .visualize import (
    save_sar_ms_channels,
    visualize_batch,
    visualize_sar_ms_channels,
    visualize_sar_encoder_layers,
)

__all__ = [
    "SAROPTPairedDataset",
    "SARMultiscaleConfig",
    "compute_sar_multiscale",
    "compute_sar_multiscale_channels",
    "normalize_sar",
    "normalize_opt",
    "visualize_batch",
    "save_sar_ms_channels",
    "visualize_sar_ms_channels",
    "visualize_sar_encoder_layers",
    "ChannelStatsAccumulator",
    "compute_dataset_channel_stats",
]
