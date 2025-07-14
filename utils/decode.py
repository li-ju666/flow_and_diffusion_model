from utils.transform import MNISTTransform
from autoencoder.model import Autoencoder
import torch


def decode_embeddings(z, model=None):
    device = z.device
    if model is None:
        model = Autoencoder().to(device)
        model.load_state_dict(
            torch.load('checkpoints/autoencoder.pth', map_location=device), strict=True)
    else:
        model = model.to(device)

    with torch.no_grad():
        model.eval()
        reconstructed = model.decoder(z)

    # undo normalization for visualization
    transformer = MNISTTransform()
    reconstructed = transformer.denormalizer(reconstructed.cpu())
    return reconstructed.permute(0, 2, 3, 1).numpy()
