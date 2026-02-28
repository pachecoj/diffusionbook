import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from cond_refinenet_dilated import CondRefineNetDilated
import argparse
import os
import time
import yaml
import logging
import shutil

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--runner', type=str, default='AnnealRunner', help='The runner to execute')
    parser.add_argument('--config', type=str, default='anneal.yml',  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-o', '--image_folder', type=str, default='images', help="The directory of image outputs")

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])
    args.log = os.path.join(args.run, 'logs', args.doc)

    # parse config file
    if not args.test:
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
    else:
        with open(os.path.join(args.log, 'config.yml'), 'r') as f:
            config = yaml.load(f)
        new_config = config

    if not args.test:
        if not args.resume_training:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)

        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

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

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    return loader

def load_fashion_mnist():
    # 1. Define transformations
    # Common practice: Compose ToTensor with Normalize if your model
    # expects a specific range (e.g., mean 0.5, std 0.5 for [-1, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)) # Uncomment if using Tanh in your model
    ])

    # 2. Download and Load the Training Set
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 3. Download and Load the Test Set
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 4. Create DataLoaders
    loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return loader

# --- 3. Main Execution ---
if __name__ == "__main__":
    args, config = parse_args_and_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    lr = 0.001
    nepochs = 50
    plot_interval = 10
    sigma_min = 0.01
    sigma_max = 1.0
    num_scales = 10
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), num_scales)).to(device)

    # Data Loading
    loader = load_mnist()

    # Initialize Model & Optimizer
    # model = ScoreNet(sigmas).to(device)
    model = CondRefineNetDilated(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    print("Starting Training...")
    for epoch in range(nepochs):  # 5 epochs for demonstration
        total_loss = 0
        for x, _ in loader:

            x = x.to(device)
            x = x / 256. * 255. + torch.rand_like(x) / 256.
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
            # loss = 0.5 * ((score_pred * s + noise) ** 2).sum(dim=(1, 2, 3)).mean()
            loss = F.mse_loss(score_pred * s, -noise)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f}")

        # Sampling
        if (epoch + 1) % plot_interval == 0:
            print("Generating Samples...")
            samples = annealed_langevin_dynamics(model, sigmas, device=device)

            # Plotting
            grid = utils.make_grid(samples, nrow=4).cpu().numpy().transpose(1, 2, 0)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid)
            plt.title("Generated MNIST Samples (NCSN)")
            plt.axis('off')
            plt.show()