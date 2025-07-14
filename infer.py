from core.Simulators import ODESimulator, SDESimulator
from core.ProbabilityPaths import GaussianPath
from core.Schedulers import LinearAlpha, LinearBeta
import torch
from model import MLPFiLM
from utils.decode import decode_embeddings



diffuser = MLPFiLM(dim=512)
# load the model state if needed
diffuser.load_state_dict(
    torch.load('checkpoints/vector_field_guided_y.pth', map_location='cpu'))
prob_path = GaussianPath(diffuser, LinearAlpha(), LinearBeta(), dim=512)
simulator = ODESimulator(prob_path)
# simulator = SDESimulator(prob_path, sigma=1e-4)


def infer(y, num_steps=100):
    z = simulator.simulate(y, num_steps)
    print(z.max(), z.min(), z.shape)
    return decode_embeddings(z)


if __name__ == "__main__":
    # Example usage
    batch_size = 5
    y = torch.arange(0, 11).view(-1, 1)
    y = y.repeat(batch_size, 1).view(-1, 1)  # Repeat labels for batch size of 5

    output = infer(y, num_steps=30)

    # Visualize the 30 images with labels
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(batch_size, 11, figsize=(batch_size * 2, 8))
    for i in range(batch_size):
        for j in range(11):
            idx = i * 11 + j
            axes[i, j].imshow(output[idx], cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Label: {y[idx].item()}')
    plt.savefig('inference_output.png')