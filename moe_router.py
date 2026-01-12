import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchwiseCondMoE(nn.Module):
    def __init__(
        self,
        router_dim: int,
        expert_dim: int,
        out_dim: int,
        num_experts: int = 4,
        router_hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1.")
        router_hidden_dim = max(1, int(router_dim * router_hidden_ratio))
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(router_dim, router_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, num_experts, bias=True),
        )
        self.experts = nn.ModuleList(
            [nn.Linear(expert_dim, out_dim, bias=True) for _ in range(num_experts)]
        )

    def initialize_weights(self) -> None:
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        for expert in self.experts:
            nn.init.xavier_uniform_(expert.weight)
            nn.init.constant_(expert.bias, 0)

    def forward(self, router_tokens: torch.Tensor, expert_tokens: torch.Tensor) -> torch.Tensor:
        if router_tokens.shape[:2] != expert_tokens.shape[:2]:
            raise ValueError("router_tokens and expert_tokens must share the same (B, N) shape.")
        weights = self.router(router_tokens)
        weights = F.softmax(weights, dim=-1)
        expert_outputs = torch.stack([expert(expert_tokens) for expert in self.experts], dim=-2)
        delta = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        return delta
