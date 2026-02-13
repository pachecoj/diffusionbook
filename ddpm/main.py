import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class UNetWithTime(nn.Module):
    def __init__(self):
        super().__init__()
        # Time embedding layer
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.down1 = nn.Conv2d(1, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        # Embed time and reshape to match image dimensions
        t_emb = self.time_mlp(t.float().view(-1, 1)).view(-1, 64, 1, 1)

        h1 = F.relu(self.down1(x) + t_emb)  # Add time context
        h2 = F.relu(self.down2(h1))
        h3 = F.relu(self.up1(h2))
        return self.up2(h3 + h1)


# --- Configuration ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
n_steps = 1000
beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

model = UNetWithTime().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(datasets.MNIST('.', train=True, download=True,
                                   transform=transforms.ToTensor()), batch_size=128, shuffle=True)

# --- Training ---
model.train()
for epoch in range(10):
    for images, _ in loader:
        images = images.to(device)
        t = torch.randint(0, n_steps, (images.shape[0],)).to(device)

        # Add noise
        noise = torch.randn_like(images)
        sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
        sqrt_1m_ab = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]
        x_noisy = sqrt_ab * images + sqrt_1m_ab * noise

        # Predict & Optimize
        loss = F.mse_loss(model(x_noisy, t), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} complete | Loss: {loss.item()}")

# --- Generation (The Sampling Loop) ---
model.eval()
with torch.no_grad():
    # 1. Start with pure noise
    img = torch.randn(10, 1, 28, 28).to(device)

    # 2. Iteratively remove noise
    for i in reversed(range(n_steps)):
        t = torch.full((10,), i, dtype=torch.long).to(device)
        pred_noise = model(img, t)

        a_t = alpha[i]
        ab_t = alpha_bar[i]
        beta_t = beta[i]
        sig_t = torch.sqrt( (1-alpha_bar[i-1])/(1-alpha_bar[i])*beta_t)

        # DDPM Sampling Formula
        z = torch.randn_like(img) if i > 0 else 0
        img = (1 / torch.sqrt(a_t)) * (img - (beta_t / torch.sqrt(1 - ab_t)) * pred_noise) + sig_t * z

# --- Plotting Results ---
img_tmp = img.cpu().squeeze().numpy()
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i_prime in range(10):
    axes[i_prime].imshow(img_tmp[i_prime], cmap='gray')
    axes[i_prime].axis('off')
plt.show()