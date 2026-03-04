# Simulation of the Weierstrass function because I wanted to see an example of a function that is continuous, but
# nowhere differentiable.  Try zooming in.

import numpy as np
import matplotlib.pyplot as plt

def weierstrass(x, a=0.5, b=3, iterations=100):
    """
    Computes the Weierstrass function: sum(a^n * cos(b^n * pi * x))
    """
    result = np.zeros_like(x)
    for n in range(iterations):
        result += (a**n) * np.cos((b**n) * np.pi * x)
    return result

# 1. Setup the x-axis
x = np.linspace(-2, 2, 5000000)
y = weierstrass(x)

# 2. Plotting
plt.figure(figsize=(12, 6))
plt.plot(x, y, color='darkblue', lw=0.5)
plt.title("The Weierstrass Function: Continuous but Nowhere Differentiable")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(alpha=0.3)
plt.show()