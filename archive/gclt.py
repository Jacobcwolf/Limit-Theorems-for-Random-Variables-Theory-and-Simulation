import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

# np.random.seed(42)

# 1. Setup Parameters
alpha = 1.5  # Tail index < 2 means INFINITE variance (Breaks standard CLT)
beta = 0     # Skewness (0 = symmetric)
n = 1000     # Sample size (number of variables in one sum)
simulations = 10000  # Number of times we repeat the sum to build the histogram

# 2. Generate i.i.d random variables from an alpha-stable distribution
# This acts as our heavy-tailed sequence (X_1, X_2, ..., X_n)
print("Simulating heavy-tailed variables... this takes a few seconds.")
rvs = levy_stable.rvs(alpha, beta, size=(simulations, n))

# Calculate the sums (X_1 + X_2 + ... + X_n)
sums = np.sum(rvs, axis=1)

# 3. Apply Standard CLT Scaling (a_n = 1 / sqrt(n))
# This is what you would normally do for finite variance. Let's watch it fail!
standard_scaling = 1 / np.sqrt(n)
standard_clt_result = sums * standard_scaling

# 4. Apply GCLT Scaling (a_n = 1 / n^(1/alpha))
# This is the correct scaling from your GCLT theorem image
gclt_scaling = 1 / (n ** (1 / alpha))
gclt_result = sums * gclt_scaling

# --- Plotting the Results ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Standard CLT Failure
# We limit the x-axis because the variance is infinite and outliers will ruin the plot
axes[0].hist(standard_clt_result, bins=200, range=(-20, 20), color='red', alpha=0.7, density=True)
axes[0].set_title(r"Standard CLT Scaling ($1/\sqrt{n}$): FAILED", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Density")
axes[0].text(0.05, 0.95, "Notice the heavy tails\nand lack of normality", 
             transform=axes[0].transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: GCLT Success
axes[1].hist(gclt_result, bins=200, range=(-10, 10), color='green', alpha=0.7, density=True)
axes[1].set_title(r"GCLT Scaling ($1/n^{1/\alpha}$): SUCCESS", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Value")
axes[1].text(0.05, 0.95, "Converges perfectly to an\n$\\alpha$-stable distribution", 
             transform=axes[1].transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()