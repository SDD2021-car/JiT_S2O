from typing import Optional

import torch
import torch.nn as nn


class SubspaceHead(nn.Module):
    """Subspace head for conditional low-rank basis prediction."""
    def __init__(self, embed_dim: int = 768, rank_k: int = 16, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim
        self.embed_dim = embed_dim
        self.rank_k = rank_k
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * rank_k),
        )

    def forward(self, sar_feat: torch.Tensor) -> torch.Tensor:
        """Predict normalized subspace basis Bmat with shape (B, D, k)."""
        if sar_feat.dim() != 4 or sar_feat.size(1) != self.embed_dim:
            raise ValueError(f"sar_feat should be [B, {self.embed_dim}, H, W], got {sar_feat.shape}")
        pooled = self.pool(sar_feat).flatten(1)
        basis = self.mlp(pooled).view(-1, self.embed_dim, self.rank_k)
        basis = torch.nn.functional.normalize(basis, dim=1, eps=1e-6)
        return basis

    def orthogonality_loss(self, basis: torch.Tensor) -> torch.Tensor:
        """Compute ||B^T B - I||_F^2 for each batch, averaged."""
        if basis.dim() != 3 or basis.size(1) != self.embed_dim:
            raise ValueError(f"basis should be [B, {self.embed_dim}, k], got {basis.shape}")
        btb = torch.matmul(basis.transpose(1, 2), basis)
        ident = torch.eye(btb.size(-1), device=btb.device, dtype=btb.dtype).unsqueeze(0)
        diff = btb - ident
        return (diff * diff).sum(dim=(1, 2)).mean()

    def project_tokens(
        self,
        x_tokens: torch.Tensor,
        basis: torch.Tensor,
        reg_lambda: float = 1e-4,
    ) -> torch.Tensor:
        """Project tokens to span(B) via B (B^T B + λI)^-1 B^T x."""
        if x_tokens.dim() != 3 or x_tokens.size(-1) != self.embed_dim:
            raise ValueError(f"x_tokens should be [B, T, {self.embed_dim}], got {x_tokens.shape}")
        if basis.dim() != 3 or basis.size(1) != self.embed_dim:
            raise ValueError(f"basis should be [B, {self.embed_dim}, k], got {basis.shape}")
        device_type = x_tokens.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            btb = torch.matmul(basis.float().transpose(1, 2), basis.float())
            k = btb.size(-1)
            ident = torch.eye(k, device=btb.device, dtype=btb.dtype).unsqueeze(0)
            btb = btb + reg_lambda * ident
            xt = x_tokens.float().transpose(1, 2)
            btx = torch.matmul(basis.float().transpose(1, 2), xt)
            coeff = torch.linalg.solve(btb, btx)
            proj = torch.matmul(basis.float(), coeff).transpose(1, 2)
        return proj
