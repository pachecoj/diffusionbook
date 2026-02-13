import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Configuration and Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20
INPUT_DIM = 784  # 28x28 pixels
HIDDEN_DIM = 400
LATENT_DIM = 2  # Size of the 'z' vector
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# 2. Data Loading (MNIST)
# We apply a simple transform to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 3. Model Definition
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder Layers
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)  # Output: Mean
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)  # Output: Log Variance

        # Decoder Layers
        self.fc3 = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        # Reparameterization Trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        eps = torch.randn_like(std)  # Random noise from standard normal distribution
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # Sigmoid puts output in [0, 1] range for pixels

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, INPUT_DIM))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 4. Loss Function
# Loss = Reconstruction Loss + KL Divergence
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    # sums the error over all pixels
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_DIM), reduction='sum')

    # KL Divergence: Measures how much the learned distribution diverges from a standard normal
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# 5. Training Loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


# 6. Visualization
def visualize_results(model):
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        data, _ = next(iter(test_loader))
        data = data.to(DEVICE)

        # Reconstruct
        recon_batch, _, _ = model(data)

        # Move to CPU for plotting
        data = data.cpu()
        recon_batch = recon_batch.view(-1, 1, 28, 28).cpu()

        # Plot Original vs Reconstructed
        fig, axes = plt.subplots(2, 8, figsize=(10, 3))
        for i in range(8):
            # Original
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            # Reconstructed
            axes[1, i].imshow(recon_batch[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')

        axes[0, 0].set_title("Original Images")
        axes[1, 0].set_title("Reconstructed")
        plt.show()


# Run Training
for epoch in range(1, EPOCHS + 1):
    train(epoch)

# Show final results
visualize_results(model)