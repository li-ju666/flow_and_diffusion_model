from simulator import ODESimulator
import torch
from prob_path import GaussianPath
from scheduler import LinearAlpha, LinearBeta
from model import UNetMLP
from autoencoder.model import Autoencoder



diffuser = UNetMLP()
# load the model state if needed
diffuser.load_state_dict(torch.load('diffuser_checkpoint.pth', map_location='cpu'))
prob_path = GaussianPath(LinearAlpha(), LinearBeta(), dim=256)
simulator = ODESimulator(diffuser, prob_path)

decoder = Autoencoder()
# load the decoder state if needed
decoder.load_state_dict(torch.load('autoencoder/checkpoint.pth', map_location='cpu'))

def infer(y, num_steps=50):
    return simulator.simulate(y, num_steps)

def undo_normalization(normalized_tensor):
    normalized_tensor = normalized_tensor.cpu()
    mean = torch.tensor([0.1307]).view(1, 1, 1, 1)  # Reshape for broadcasting
    std = torch.tensor([0.3081]).view(1, 1, 1, 1)   # Reshape for broadcasting
    original_tensor = normalized_tensor * std + mean
    return original_tensor


if __name__ == "__main__":
    # Example usage
    batch_size = 5
    y = torch.arange(0, 11).view(-1, 1)
    y = y.repeat(batch_size, 1).view(-1, 1)  # Repeat labels for batch size of 5

    output = infer(y, num_steps=50)

    # decoder output
    z = output  # Assuming output is the latent variable
    with torch.no_grad():
        decoder.eval()
        decoded_output = decoder.decode(z)
    decoded_output = undo_normalization(decoded_output)

    # Visualize the 50 images with labels
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(batch_size, 11, figsize=(batch_size * 2, 8))
    for i in range(batch_size):
        for j in range(11):
            idx = i * 11 + j
            axes[i, j].imshow(decoded_output[idx].squeeze(), cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Label: {y[idx].item()}')
    plt.tight_layout()
    plt.savefig('inference_output.png')