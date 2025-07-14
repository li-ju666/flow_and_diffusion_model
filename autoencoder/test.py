from torchvision.datasets import MNIST
from utils.transform import MNISTTransform
from torch.utils.data import DataLoader
from autoencoder.model import Autoencoder
from utils.decode import decode_embeddings
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = MNISTTransform()
dataset = MNIST(
    root='/mimer/NOBACKUP/Datasets',
    train=True,
    transform=transformer.normalizer,
)

dataloader = DataLoader(
    dataset, batch_size=512, num_workers=4, pin_memory=True
)

model = Autoencoder().to(DEVICE)
# load pretrained model if available
model.load_state_dict(torch.load('checkpoints/autoencoder.pth', map_location=DEVICE), strict=True)

# sample 5 images from the dataset
num_samples = 5
sample_images = [dataset[i][0] for i in range(num_samples)]
sample_images = torch.stack(sample_images).to(DEVICE)

# encode the images
with torch.no_grad():
    encoded_images = model.encode(sample_images)

# decode the images
images = decode_embeddings(encoded_images, model)

# visualise the images versus the original images
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(sample_images[i].cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Original')

    plt.subplot(2, num_samples, i + 1 + num_samples)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
    plt.title('Decoded')

plt.savefig('autoencoder/sample_images.png')
plt.close()
