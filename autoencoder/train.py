from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from autoencoder.model import Autoencoder
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST(
    '/mimer/NOBACKUP/Datasets',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

dataloader = DataLoader(
    dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
)

num_epochs = 100

model = Autoencoder().to(DEVICE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(dataloader), eta_min=1e-6)


for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        inputs, _ = batch
        inputs = inputs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # save model checkpoint
    torch.save(model.state_dict(), 'autoencoder/checkpoint.pth')
