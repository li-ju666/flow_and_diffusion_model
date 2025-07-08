import torch

class Alpha:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        pass


class Beta:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        pass


class LinearAlpha(Alpha):
    def __call__(self, t):
        return t

    def dt(self, t):
        return torch.ones_like(t)

class LinearBeta(Beta):
    def __call__(self, t):
        return torch.ones_like(t) - t

    def dt(self, t):
        return -torch.ones_like(t)
