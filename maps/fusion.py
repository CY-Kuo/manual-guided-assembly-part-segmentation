"""Feature-fusion module used by the final MAPS training and inference paths."""

import torch
import torch.nn as nn


class StructureFusion(nn.Module):
    """Fuse student camera features with compact teacher/manual structure tokens."""

    def __init__(
        self,
        s_channels: int,
        t_channels: int,
        d: int = 128,
        num_tokens: int = 4,
        use_film: bool = True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.d = d
        self.use_film = use_film

        self.t_gap = nn.AdaptiveAvgPool2d(1)
        self.t_token = nn.Sequential(
            nn.Conv2d(t_channels, d * num_tokens, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.q_proj = nn.Conv2d(s_channels, d, 1, bias=True)
        self.out_proj = nn.Conv2d(d, s_channels, 1, bias=True)
        if use_film:
            self.film = nn.Sequential(
                nn.Conv2d(t_channels, s_channels * 2, 1, bias=True)
            )
        self.temperature = d ** 0.5

    def forward(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        batch, s_channels, height, width = s_feat.shape
        teacher_global = self.t_gap(t_feat)
        tokens = self.t_token(teacher_global).view(batch, self.num_tokens, self.d)

        query = self.q_proj(s_feat)
        query = query.view(batch, self.d, height * width).transpose(1, 2)
        attention = torch.softmax(
            torch.matmul(query, tokens.transpose(1, 2)) / self.temperature,
            dim=-1,
        )
        fused_delta = torch.matmul(attention, tokens)
        fused_delta = fused_delta.transpose(1, 2).view(batch, self.d, height, width)
        fused = s_feat + self.out_proj(fused_delta)

        if self.use_film:
            film_out = self.film(teacher_global)
            required = 2 * s_channels
            available = film_out.shape[1]
            if available < required:
                pad = torch.zeros(
                    film_out.shape[0],
                    required - available,
                    *film_out.shape[2:],
                    device=film_out.device,
                    dtype=film_out.dtype,
                )
                film_out = torch.cat([film_out, pad], dim=1)
            elif available > required:
                film_out = film_out[:, :required]
            gamma, beta = torch.split(film_out, [s_channels, s_channels], dim=1)
            fused = fused * torch.sigmoid(gamma) + beta

        return fused
