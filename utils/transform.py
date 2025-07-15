from torchvision import transforms
import torch


class MNISTTransform:
    def __init__(self):
        # mean = torch.tensor([0.1307]).view(1, 1, 1)  # Reshape for broadcasting
        # std = torch.tensor([0.3081]).view(1, 1, 1)
        self.normalizer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std)
        ])
        # mean, std = mean.unsqueeze(0), std.unsqueeze(0)
        # self.denormalizer = lambda x: x * std + mean
        self.denormalizer = lambda x: x
