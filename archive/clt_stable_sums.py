import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n = 500              # Sample size per simulation
simulations = 10000  # Number of sample means to calculate
standard_scaling = 1 / np.sqrt(n) # The standard n^(1/2) denominator

print("Simulating CLT Failure Case... this may take a moment.")

# 1. Generate data from an Alpha-Stable Distribution (Infinite Variance)
alpha = 1.5  # Alpha < 2 means infinite variance
beta = 0     # Symmetrical
alpha_data = levy_stable.rvs(alpha, beta, size=(simulations, n))

# 2. Calculate the sums and apply standard scaling
alpha_sums = np.sum(alpha_data, axis=1)
alpha_result = alpha_sums * standard_scaling

# 3. Plot the result
plt.figure(figsize=(8, 6))

# Note: We restrict the x-axis limits (range) because the infinite variance 
# creates massive outliers that would otherwise ruin the scale of the plot.
plt.hist(alpha_result, bins=200, range=(-15, 15), density=True, color='salmon', edgecolor='black', alpha=0.7)

plt.title(r"$\alpha$-Stable Distribution ($n^{1/2}$ Scaling): CLT Failure", fontsize=14, fontweight='bold')
plt.xlabel("Standardised Sample Sum")
plt.ylabel("Density")
plt.text(0.05, 0.95, "Alpha-Stable ($\\alpha=1.5$)\nVariance is infinite\nStandardisation collapses", 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()