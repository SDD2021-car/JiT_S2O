import math
from typing import Optional

import torch
import torch.nn as nn

from model_jit import BottleneckPatchEmbed


class SpeckleStabilizer(nn.Module):
    """简单相干斑稳定器：3x3 深度可分离卷积进行平滑。"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class SAREncoderBlock(nn.Module):
    """轻量 Transformer 编码层：MHSA + MLP。"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力特征更新
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop(attn_out)
        # MLP 进行通道混合
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ResidualConvBlock(nn.Module):
    """轻量级残差卷积块，用于 CNN 金字塔特征提取。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SAREncoder(nn.Module):
    """
    SAR 图像编码器：将输入 SAR 图像映射为 token 序列 (B, N, D)。
    支持两种方案：
    - scheme="A": 噪声稳定化 -> BottleneckPatchEmbed -> 轻量 Transformer
    - scheme="B": CNN 金字塔 -> 自适应池化 -> Token 投影
    """

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        token_count: int,
        scheme: str = "A",
        num_layers: int = 2,
        num_heads: int = 8,
        pca_dim: Optional[int] = None,
    ):
        super().__init__()
        scheme = scheme.upper()
        if scheme not in ("A", "B"):
            raise ValueError(f"Unsupported scheme: {scheme}")
        self.scheme = scheme
        self.embed_dim = embed_dim
        self.token_count = token_count

        if self.scheme == "A":
            if not (2 <= num_layers <= 4):
                raise ValueError("方案 A 的 Transformer 层数应为 2-4 层。")
            if pca_dim is None:
                pca_dim = embed_dim
            self.stabilizer = SpeckleStabilizer(in_channels)
            self.patch_embed = BottleneckPatchEmbed(
                img_size=input_size,
                patch_size=patch_size,
                in_chans=in_channels,
                pca_dim=pca_dim,
                embed_dim=embed_dim,
            )
            if self.patch_embed.num_patches != token_count:
                raise ValueError(
                    f"方案 A 的 token_count ({token_count}) 必须与 patch 数量 ({self.patch_embed.num_patches}) 一致。"
                )
            self.blocks = nn.ModuleList(
                [SAREncoderBlock(embed_dim, num_heads) for _ in range(num_layers)]
            )
        else:
            token_hw = int(math.sqrt(token_count))
            if token_hw * token_hw != token_count:
                raise ValueError("方案 B 需要 token_count 为完全平方数。")
            self.token_hw = token_hw
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.stage1 = ResidualConvBlock(32, 64, stride=2)
            self.stage2 = ResidualConvBlock(64, 128, stride=2)
            self.stage3 = ResidualConvBlock(128, 256, stride=2)
            self.pool = nn.AdaptiveAvgPool2d((token_hw, token_hw))
            self.proj = nn.Conv2d(256, embed_dim, kernel_size=1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        return_pyramid: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.scheme == "A":
            # 先进行相干斑噪声的稳定化，再提取 patch tokens
            x = self.stabilizer(x)
            tokens = self.patch_embed(x)
            for block in self.blocks:
                tokens = block(tokens)
            if return_pyramid:
                raise ValueError("方案 A 不提供 CNN 金字塔特征。")
            return tokens

        # 方案 B：多尺度 CNN 提取稳定空间特征，再对齐为 token 序列
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        pooled = self.pool(stage3)
        proj = self.proj(pooled)
        tokens = proj.flatten(2).transpose(1, 2)
        if tokens.size(1) != self.token_count:
            raise ValueError(f"Token 数量不匹配: {tokens.size(1)} != {self.token_count}")
        if return_pyramid:
            pyramid = {
                "stage1": stage1,
                "stage2": stage2,
                "stage3": stage3,
                "pooled": pooled,
                "proj": proj,
            }
            return tokens, pyramid
        return tokens
