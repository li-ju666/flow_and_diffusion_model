from torch.utils.data import DataLoader, TensorDataset
import torch
from core.ProbabilityPaths import GaussianPath
from core.Schedulers import LinearAlpha, LinearBeta
from models import MLPFiLM, MLPUnet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(batch_size=512):
    feature_path = "checkpoints/embeddings.pth"
    target_path = "checkpoints/labels.pth"

    features = torch.load(feature_path)
    labels = torch.load(target_path)
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    return dataloader

# model = MLPFiLM(dim=512).to(DEVICE)  # 10 classes + 1 for 10.0 label
model = MLPUnet(dim=512).to(DEVICE)  # 10 classes + 1 for 10.0 label

dataloader = load_data()
prob_path = GaussianPath(
    model, LinearAlpha(), LinearBeta(), dim=512)

num_epochs = 5000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(dataloader), eta_min=1e-5
)

for epoch in range(num_epochs):  # Example training loop
    for batch in dataloader:
        z, y = batch
        z = z.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()

        # sample t
        t = torch.rand(z.size(0), 1).to(DEVICE).view(-1, 1)  # Random time steps

        # randomly replace labels with 10.0 at 30% of the time
        masks = torch.rand(y.size(0)) < 0.3
        y[masks] = 10.0
        y = y.view(-1, 1)

        # sample x from the probability path
        xt = prob_path.sample_xt_cond_z(z, t)

        # compute the model output
        output = model(xt, y, t)

        # compute loss (example: MSE)
        target = prob_path.vector_field_cond_z(xt, z, t).to(DEVICE)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # save model checkpoint
    torch.save(model.state_dict(), 'checkpoints/vector_field_guided_y.pth')
