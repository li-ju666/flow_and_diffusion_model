from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from autoencoder.model import Autoencoder
import torch


def undo_normalization(normalized_tensor):
    normalized_tensor = normalized_tensor.cpu()
    mean = torch.tensor([0.1307]).view(1, 1, 1, 1)  # Reshape for broadcasting
    std = torch.tensor([0.3081]).view(1, 1, 1, 1)   # Reshape for broadcasting
    original_tensor = normalized_tensor * std + mean
    return original_tensor


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST(
    root='/mimer/NOBACKUP/Datasets',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

dataloader = DataLoader(
    dataset, batch_size=512, num_workers=4, pin_memory=True
)

model = Autoencoder().to(DEVICE)
# load pretrained model if available
model.load_state_dict(torch.load('autoencoder/checkpoint.pth', map_location=DEVICE), strict=True)

# sample 5 images from the dataset
sample_images = [dataset[i][0] for i in range(5)]
sample_images = torch.stack(sample_images).to(DEVICE)

# compare original and reconstructed images
with torch.no_grad():
    model.eval()
    reconstructed_images = model(sample_images)
    # undo normalization for visualization
    reconstructed_images = undo_normalization(reconstructed_images)
    sample_images = undo_normalization(sample_images)

# visualize original and reconstructed images
import matplotlib.pyplot as plt

def show_images(original, reconstructed):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for i in range(5):
        # original images
        ax = axes[0, i]
        ax.imshow(original[i].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
    for i in range(5):
        # reconstructed images
        ax = axes[1, i]
        ax.imshow(reconstructed[i].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.savefig('autoencoder/reconstructed_images.png')
    plt.close()

show_images(sample_images, reconstructed_images)