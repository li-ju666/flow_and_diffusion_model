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
        return t # batch size x 1

    def dt(self, t):
        return torch.ones_like(t) # batch size x 1

class LinearBeta(Beta):
    def __call__(self, t):
        return torch.ones_like(t) - t # batch size x 1

    def dt(self, t):
        return -torch.ones_like(t) # batch size x 1
