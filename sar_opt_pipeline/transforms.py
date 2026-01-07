from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SARMultiscaleConfig:
    sobel: bool = True
    laplacian: bool = True
    local_variance: bool = True
    local_contrast: bool = True
    local_window: int = 7
    eps: float = 1e-6


def normalize_sar(sar: torch.Tensor, clip_max: float = 2.0) -> torch.Tensor:
    """Log + clip normalization for SAR (stable speckle).

    Args:
        sar: (1, H, W) or (3, H, W) tensor in [0, 1] or raw scale.
        clip_max: Upper bound after log1p.
    """
    sar = torch.log1p(torch.clamp(sar, min=0.0))
    sar = torch.clamp(sar, min=0.0, max=clip_max)
    return sar / clip_max


def normalize_opt(
    opt: torch.Tensor,
    mode: str = "minus_one_one",
    mean: Optional[Iterable[float]] = None,
    std: Optional[Iterable[float]] = None,
) -> torch.Tensor:
    """Normalize OPT to [-1, 1] or VAE-standardized inputs."""
    if mode == "minus_one_one":
        return opt * 2.0 - 1.0
    if mode == "vae":
        if mean is None or std is None:
            raise ValueError("mean/std must be provided when mode='vae'")
        mean_tensor = torch.as_tensor(mean, dtype=opt.dtype, device=opt.device)[:, None, None]
        std_tensor = torch.as_tensor(std, dtype=opt.dtype, device=opt.device)[:, None, None]
        return (opt - mean_tensor) / std_tensor
    raise ValueError(f"Unknown opt normalization mode: {mode}")


def _conv2d_single_channel(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 3:
        raise ValueError("Expected SAR tensor shape (C, H, W)")
    if tensor.shape[0] == 3:
        tensor = tensor.mean(dim=0, keepdim=True)
    elif tensor.shape[0] != 1:
        raise ValueError("Expected SAR tensor shape (1, H, W) or (3, H, W)")
    kernel = kernel.to(dtype=tensor.dtype, device=tensor.device)
    kernel = kernel[None, None, :, :]
    tensor = tensor[None, :, :, :]
    return F.conv2d(tensor, kernel, padding=kernel.shape[-1] // 2)[0]


def _local_stats(tensor: torch.Tensor, window: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = torch.ones((window, window), device=tensor.device, dtype=tensor.dtype)
    kernel = kernel / kernel.numel()
    kernel = kernel[None, None, :, :]
    tensor_batched = tensor[None, :, :, :]
    mean = F.conv2d(tensor_batched, kernel, padding=window // 2)[0]
    mean_sq = F.conv2d(tensor_batched**2, kernel, padding=window // 2)[0]
    var = torch.clamp(mean_sq - mean**2, min=0.0)
    std = torch.sqrt(var + eps)
    return mean, std


def compute_sar_multiscale_channels(
    sar: torch.Tensor, config: SARMultiscaleConfig
) -> dict[str, torch.Tensor]:
    """Compute multi-scale structure channels for SAR as a named dict."""
    channels: dict[str, torch.Tensor] = {}
    if sar.dim() != 3:
        raise ValueError("Expected SAR tensor shape (C, H, W)")
    sar_single = sar
    if sar_single.shape[0] == 3:
        sar_single = sar_single.mean(dim=0, keepdim=True)
    elif sar_single.shape[0] != 1:
        raise ValueError("Expected SAR tensor shape (1, H, W) or (3, H, W)")

    sobel_x_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=sar.dtype)
    sobel_y_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=sar.dtype)
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=sar.dtype)

    if config.sobel:
        channels["sobel_x"] = _conv2d_single_channel(sar, sobel_x_kernel)
        channels["sobel_y"] = _conv2d_single_channel(sar, sobel_y_kernel)
    if config.laplacian:
        channels["laplacian"] = _conv2d_single_channel(sar, laplacian_kernel)

    if config.local_variance or config.local_contrast:
        mean, std = _local_stats(sar_single, window=config.local_window, eps=config.eps)
        if config.local_variance:
            channels["local_variance"] = std**2
        if config.local_contrast:
            channels["local_contrast"] = std / (mean + config.eps)

    if not channels:
        raise ValueError("At least one SAR multiscale channel must be enabled.")
    return channels


def compute_sar_multiscale(sar: torch.Tensor, config: SARMultiscaleConfig) -> torch.Tensor:
    """Compute multi-scale structure channels for SAR.

    Outputs 3-6 channels: sobel_x/y, laplacian, local variance, local contrast.
    """
    channels = compute_sar_multiscale_channels(sar, config)
    return torch.cat(list(channels.values()), dim=0)
