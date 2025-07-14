from torch.utils.data import DataLoader, TensorDataset
import torch
from prob_path import GaussianPath
from scheduler import LinearAlpha, LinearBeta
from model import UNetMLP


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(batch_size=512):
    feature_path = "autoencoder/features.pth"
    target_path = "autoencoder/labels.pth"

    features = torch.load(feature_path)
    labels = torch.load(target_path)
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    return dataloader


dataloader = load_data()
prob_path = GaussianPath(
    LinearAlpha(), LinearBeta(), dim=256)

model = UNetMLP().to(DEVICE)
num_epochs = 500
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(dataloader), eta_min=1e-6
)

for epoch in range(num_epochs):  # Example training loop
    for batch in dataloader:
        z, y = batch
        z = z.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()

        # sample t
        t = torch.rand(z.size(0), 1).to(DEVICE).view(-1, 1)  # Random time steps

        # randomly replace labels with 10.0 at 10% of the time
        masks = torch.rand(y.size(0)) < 0.1
        y[masks] = 10.0
        y = y.view(-1, 1)

        # sample x from the probability path
        x = prob_path.sample_x(z, t)

        # compute the model output
        output = model(x, y, t)

        # compute loss (example: MSE)
        target = prob_path.reference_vector_field(x, z, t)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # save model checkpoint
    torch.save(model.state_dict(), 'diffuser_checkpoint.pth')