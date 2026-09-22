import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n = 500              # Sample size per simulation
simulations = 10000  # Number of sample means to calculate
standard_scaling = 1 / np.sqrt(n) # The standard n^(1/2) denominator

print("Simulating CLT Success Case...")

# 1. Generate data from a Uniform Distribution (Finite Variance)
uniform_data = uniform.rvs(loc=-1, scale=2, size=(simulations, n))

# 2. Calculate the sums and apply standard scaling
uniform_sums = np.sum(uniform_data, axis=1)
uniform_result = uniform_sums * standard_scaling

# 3. Plot the result
plt.figure(figsize=(8, 6))
plt.hist(uniform_result, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7)

plt.title(r"Finite Variance ($n^{1/2}$ Scaling): CLT Success", fontsize=14, fontweight='bold')
plt.xlabel("Standardised Sample Sum")
plt.ylabel("Density")
plt.text(0.05, 0.95, "Uniform Distribution\nVariance is finite\nForms a normal curve", 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()