import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class ScoreNet(nn.Module):
    def __init__(self, sigma_list):
        super().__init__()
        self.sigma_list = sigma_list
        # Simple CNN: In practice, use a U-Net with Skip Connections
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, sigma_idx):
        # We embed the noise level into the network
        # Here we just scale, but more complex models use conditional InstanceNorm
        used_sigmas = self.sigma_list[sigma_idx].view(-1, 1, 1, 1)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        score = self.conv4(h)

        # The output is normalized by sigma according to the NCSN paper
        return score / used_sigmas

def loss_fn(model, x, sigma_list):
    # Sample random noise scales for each batch element
    labels = torch.randint(0, len(sigma_list), (x.shape[0],), device=x.device)
    used_sigmas = sigma_list[labels].view(-1, 1, 1, 1)

    # Perturb data
    noise = torch.randn_like(x)
    perturbed_x = x + noise * used_sigmas

    # Predict score
    target = -noise / used_sigmas  # This is the ground truth score of the perturbation
    score = model(perturbed_x, labels)

    # Weight the loss by sigma^2 to balance different scales
    loss = 0.5 * ((score - target) ** 2).sum(dim=(1, 2, 3)) * (used_sigmas.squeeze() ** 2)
    return loss.mean()

@torch.no_grad()
def anneal_langevin_dynamics(model, sigma_list, n_steps=100, eps=0.00002):
    model.eval()
    # Start from pure white noise
    x = torch.rand(16, 1, 28, 28, device=sigma_list.device)

    for i, sigma in enumerate(sigma_list):
        # Step size for this specific noise scale
        alpha = eps * (sigma / sigma_list[-1]) ** 2

        for _ in range(n_steps):
            # L-steps of Langevin Dynamics
            z = torch.randn_like(x)
            score = model(x, torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
            x = x + alpha * score + torch.sqrt(2 * alpha) * z

    return x

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sigmas = torch.exp(torch.linspace(torch.log(torch.tensor(1.0)),
                                  torch.log(torch.tensor(0.01)), 10)).to(device)


model = ScoreNet(sigmas).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                          transform=transforms.ToTensor()), batch_size=128, shuffle=True)

# 2. Train
for epoch in range(100):
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, x, sigmas)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    if (epoch+1) % 10 == 0:
        samples = anneal_langevin_dynamics(model, sigmas)

        grid_img = make_grid(samples.cpu().detach(), nrow=4, padding=2)

        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis('off')
        plt.show()