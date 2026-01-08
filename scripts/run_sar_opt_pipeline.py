import torch
from torch.utils.data import DataLoader
from sar_opt_pipeline import (
    SAROPTPairedDataset,
    SARMultiscaleConfig,
    compute_sar_multiscale_channels,
    visualize_batch,
    compute_dataset_channel_stats,
    save_sar_ms_channels,
    visualize_sar_ms_channels,
    visualize_sar_encoder_layers,
)
from sar_encoder import SAREncoder


# 1) 配置多尺度通道（3~6 通道）
ms_cfg = SARMultiscaleConfig(
    sobel=True,
    laplacian=True,
    local_variance=True,
    local_contrast=True,
    local_window=7,
)

# 2) 构建数据集
dataset = SAROPTPairedDataset(
    sar_root="/data/hjf/Dataset/SEN12_Scene/trainA",
    opt_root="/data/hjf/Dataset/SEN12_Scene/trainB",
    sar_clip_max=2.0,          # SAR: log+clip
    sar_ms_config=ms_cfg,
    opt_norm_mode="minus_one_one",  # 或 "vae"
)

# 3) DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# 4) 可视化一个 batch
batch = next(iter(loader))
visualize_batch(
    sar_raw=batch["sar_raw"],
    sar_ms=batch["sar_ms"],
    opt_gt=batch["opt_gt"],
    save_path="outputs/overview.png",
)

# 4.1) 保存每个通道的 SAR MS 图片（带进度条）
save_sar_ms_channels(
    sar_ms=batch["sar_ms"],
    save_dir="outputs/sar_ms_channels",
    prefix="sar_ms",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4.2) 保存每个样本的 SAR MS 通道网格图（带进度条）
visualize_sar_ms_channels(
    sar_ms=batch["sar_ms"],
    save_dir="outputs/sar_ms_grids",
    prefix="sar_ms",
)

# 4.3) SAR encoder 方案 B：以 sar_ms 作为输入并输出分层可视化
sar_ms = batch["sar_ms"].to(device)
input_size = sar_ms.shape[-1]
patch_size = 16
token_count = (input_size // patch_size) ** 2
sar_encoder = SAREncoder(
    input_size=input_size,
    patch_size=patch_size,
    in_channels=sar_ms.shape[1],
    embed_dim=768,
    token_count=token_count,
    scheme="B",
).to(device)
with torch.no_grad():
    _tokens, pyramid = sar_encoder(sar_ms, return_pyramid=True)
visualize_sar_encoder_layers(
    layers=pyramid,
    save_dir="outputs/sar_encoder_layers",
    prefix="sar_encoder",
)

# 4.4) 检查每一种 SAR 特征的提取是否在 GPU 上
sar_raw_device = batch["sar_raw"].to(device)
channels = compute_sar_multiscale_channels(sar_raw_device, config=ms_cfg)
gpu_status = {name: tensor.is_cuda for name, tensor in channels.items()}
print("sar_ms channels on cuda:", gpu_status)

# 5) 统计均值方差
stats = compute_dataset_channel_stats(dataset, batch_size=4, num_workers=0)
print("sar_raw mean/std:", stats["sar_raw"])
print("sar_ms mean/std:", stats["sar_ms"])
print("opt_gt mean/std:", stats["opt_gt"])
