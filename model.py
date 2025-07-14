import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim // 2

    def forward(self, t):
        # t: [B, 1]
        freqs = 2 ** torch.arange(0, self.dim*2, device=t.device) * torch.pi
        sin_features = torch.sin(t * freqs)
        cos_features = torch.cos(t * freqs)
        return torch.cat([sin_features, cos_features], dim=-1)  # [B, dim]


class UNetMLP(nn.Module):
    def __init__(self, dim=256, num_classes=11, blocks=10):
        super().__init__()
        # time → film
        self.time_embed = FourierFeatures(dim=dim)
        # label → film
        self.label_embed = nn.Embedding(num_classes, dim)
        # label projection to match film dimension
        self.label_proj  = nn.Linear(dim, dim*2)

        # residual blocks
        layers = [ResBlock(dim)
                  for i in range(blocks)]
        self.net   = nn.Sequential(*layers)
        self.final = nn.Linear(dim, dim)

    def forward(self, x, y, t):
        # x: [B,256], t: [B,1], y: [B] (long)
        # 1) time embedding τ(t)
        film_time = self.time_embed(t) # [B,2*dim]
        # 2) label embedding
        y = self.label_embed(y).view(-1, self.label_embed.embedding_dim) # [B,dim]
        film_label= self.label_proj(y) # [B,2*dim]
        film = film_time + film_label          # [B,2*dim]
        # 3) pass through ResBlocks
        for block in self.net:
            x = block(x, film)
        return self.final(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, film):
        # film: [B,2*dim] → scale, shift of [B,dim]
        scale, shift = film.chunk(2, dim=-1)
        h = self.lin1(x)
        h = F.silu(h)
        h = self.lin2(h)
        # FiLM
        h = self.norm(h) * (1 + scale) + shift
        h = F.silu(h)
        return x + h


if __name__ == "__main__":
    model = UNetMLP()
    dummy_x = torch.randn(5, 256)  # Example input
    dummy_y = torch.randint(0, 11, (5, 1))     # Example input
    dummy_t = torch.randn(5, 1)     # Example input
    output = model(dummy_x, dummy_y, dummy_t)
    print(output.shape)