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


class MLPFiLM(nn.Module):
    def __init__(self, dim=512, num_classes=10, blocks=10):
        super().__init__()
        # time → film
        self.time_embed = FourierFeatures(dim=dim)
        # residual blocks
        layers = [ResBlock(dim)
                  for i in range(blocks)]
        self.net   = nn.Sequential(*layers)
        self.final = nn.Linear(dim, dim)


        # classifier
        self.classifier = Classifier(dim=dim, num_classes=num_classes)

    def forward(self, x, t):
        # x: [B,256], t: [B,1],
        # 1) time embedding τ(t)
        gen = x
        film = self.time_embed(t) # [B,2*dim]
        # 3) pass through ResBlocks
        for block in self.net:
            gen = block(gen, film)
        return self.final(gen), self.classifier(x, film)


class Classifier(nn.Module):
    def __init__(self, dim=512, num_classes=10):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, num_classes)
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
        return self.classifier(h)

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
    model = MLPFiLM()
    dummy_x = torch.randn(5, 256)  # Example input
    dummy_y = torch.randint(0, 11, (5, 1))     # Example input
    dummy_t = torch.randn(5, 1)     # Example input
    output = model(dummy_x, dummy_y, dummy_t)
    print(output.shape)