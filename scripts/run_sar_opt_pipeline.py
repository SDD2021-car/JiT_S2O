from torch.utils.data import DataLoader
from sar_opt_pipeline import (
    SAROPTPairedDataset,
    SARMultiscaleConfig,
    visualize_batch,
    compute_dataset_channel_stats,
)


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
    save_path="outputs",
)

# 5) 统计均值方差
stats = compute_dataset_channel_stats(dataset, batch_size=4, num_workers=0)
print("sar_raw mean/std:", stats["sar_raw"])
print("sar_ms mean/std:", stats["sar_ms"])
print("opt_gt mean/std:", stats["opt_gt"])
