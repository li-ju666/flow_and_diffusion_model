from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from autoencoder.model import Autoencoder
import torch

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

# extract features
features = []
labels = []


with torch.no_grad():
    model.eval()
    for batch in dataloader:
        inputs, batch_labels = batch
        inputs = inputs.to(DEVICE)
        z = model.encode(inputs)
        features.append(z.cpu())
        labels.append(batch_labels.cpu())

features = torch.cat(features, dim=0)
labels = torch.cat(labels, dim=0)
# save features and labels
torch.save(features, 'autoencoder/features.pth')
torch.save(labels, 'autoencoder/labels.pth')
print(f'Extracted features shape: {features.shape}, Labels shape: {labels.shape}')