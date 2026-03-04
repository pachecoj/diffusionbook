import numpy as np
import matplotlib.pyplot as plt

# 1. Parameters
S0 = 100  # Initial stock price
mu = 0.05  # Annual drift (5%)
sigma = 0.2  # Annual volatility (20%)
T = 1.0  # Time horizon (1 year)
dt = 0.01  # Time step (e.g., ~2.5 days)
N = int(T / dt)  # Number of steps
t = np.linspace(0, T, N)

# 2. Simulate 5 different paths
np.random.seed(42)  # For reproducible results
paths = 5
S = np.zeros((N, paths))
S[0] = S0

for i in range(1, N):
    # Generate random noise for all paths at once
    Z = np.random.standard_normal(paths)

    # Euler-Maruyama update rule
    # Change = (Drift * dt) + (Diffusion * sqrt(dt) * Random Noise)
    S[i] = S[i - 1] + (mu * S[i - 1] * dt) + (sigma * S[i - 1] * np.sqrt(dt) * Z)

# 3. Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.title(f"Geometric Brownian Motion Simulation ({paths} paths)")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()
