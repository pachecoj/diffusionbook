import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np


# --- 1. Model Architecture ---
class ScoreNet(nn.Module):
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        # Simple U-Net-like structure
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)

        # Sigma conditioning (embedding)
        self.sigma_map = nn.Linear(1, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, sigma_idx):
        s = self.sigmas[sigma_idx].view(-1, 1, 1, 1)
        s_embed = self.sigma_map(s.view(-1, 1)).view(-1, 256, 1, 1)

        h1 = F.elu(self.conv1(x))
        h2 = F.elu(self.conv2(h1))
        h3 = F.elu(self.conv3(h2))

        # Inject sigma conditioning at the bottleneck
        h = F.elu(self.deconv1(h3 + s_embed))
        h = F.elu(self.deconv2(h + h2))
        out = self.conv_out(h + h1)

        # Scaling output by 1/sigma is crucial for the network to learn multiple scales
        return out / s


# --- 2. Sampling Logic (Annealed Langevin Dynamics) ---
@torch.no_grad()
def annealed_langevin_dynamics(model, sigmas, n_steps=100, eps=2e-5, device='cpu'):
    model.eval()
    # Start with pure noise
    x = torch.rand(16, 1, 28, 28).to(device)

    # Iterate from highest noise (sigma_max) to lowest (sigma_min)
    for i in range(len(sigmas)):
        sigma_idx = torch.full((16,), i, dtype=torch.long).to(device)
        step_size = eps * (sigmas[i] / sigmas[-1]) ** 2

        for _ in range(n_steps):
            noise = torch.randn_like(x)
            score = model(x, sigma_idx)
            x = x + 0.5 * step_size * score + torch.sqrt(step_size) * noise

    return x.clamp(0, 1)


# --- 3. Main Execution ---
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    sigma_min = 0.01
    sigma_max = 1.0
    num_scales = 10
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), num_scales)).to(device)

    # Data Loading
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize Model & Optimizer
    model = ScoreNet(sigmas).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    print("Starting Training...")
    for epoch in range(5):  # 5 epochs for demonstration
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()

            # Sample random sigma level for each image in batch
            idx = torch.randint(0, num_scales, (x.size(0),)).to(device)
            s = sigmas[idx].view(-1, 1, 1, 1)

            # Perturb data: x_tilde = x + sigma * noise
            noise = torch.randn_like(x)
            x_tilde = x + s * noise

            # Forward pass
            score_pred = model(x_tilde, idx)

            # Denoising Score Matching Loss
            # Target score is -(x_tilde - x)/sigma^2, which is -noise/sigma
            loss = 0.5 * ((score_pred * s + noise) ** 2).sum(dim=(1, 2, 3)).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")

    # Sampling
    print("Generating Samples...")
    samples = annealed_langevin_dynamics(model, sigmas, device=device)

    # Plotting
    grid = utils.make_grid(samples, nrow=4).cpu().numpy().transpose(1, 2, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid)
    plt.title("Generated MNIST Samples (NCSN)")
    plt.axis('off')
    plt.show()