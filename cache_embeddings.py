from torchvision.datasets import MNIST
from utils.transform import MNISTTransform
from torch.utils.data import DataLoader
from autoencoder.model import Autoencoder
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST(
    root='/mimer/NOBACKUP/Datasets',
    train=True,
    transform=MNISTTransform().normalizer,
)

dataloader = DataLoader(
    dataset, batch_size=512, num_workers=4, pin_memory=True
)

model = Autoencoder().to(DEVICE)
# load pretrained model if available
model.load_state_dict(
    torch.load('checkpoints/autoencoder.pth', map_location=DEVICE), strict=True)

# extract embeddings
embeddings = []
labels = []

with torch.no_grad():
    model.eval()
    for batch in dataloader:
        inputs, batch_labels = batch
        inputs = inputs.to(DEVICE)
        z = model.encode(inputs)
        embeddings.append(z.cpu())
        labels.append(batch_labels.cpu())

embeddings = torch.cat(embeddings, dim=0)
labels = torch.cat(labels, dim=0)
# save embeddings and labels
torch.save(embeddings, 'checkpoints/embeddings.pth')
torch.save(labels, 'checkpoints/labels.pth')
print(f'Extracted embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}')