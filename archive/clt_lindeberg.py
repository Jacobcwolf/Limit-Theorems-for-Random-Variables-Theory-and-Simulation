import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n = 15              # We use a smaller n because the variance will grow exponentially
simulations = 10000 

print("Simulating Lindeberg Failure Case (INID)...")

# We will store the sum of our variables here
inid_sums = np.zeros(simulations)
total_variance = 0

# 1. Generate Independent but NOT Identically Distributed (INID) variables
for i in range(1, n + 1):
    bound = 2**i
    inid_sums += np.random.uniform(-bound, bound, simulations)
    
    # Calculate the variance of this specific Uniform distribution (a^2 / 3)
    total_variance += (bound**2) / 3

# 2. Apply standardisation
# Because they aren't identical, we divide by the exact standard deviation of the sum
true_std = np.sqrt(total_variance)
standardized_result = inid_sums / true_std

# 3. Plot the result
plt.figure(figsize=(10, 6))

# Add a subtle dotted grid behind the bars to match your other plots
plt.grid(True, linestyle=':', alpha=0.6, zorder=0)

# Plot the histogram in medium purple with black edges and a legend label
plt.hist(standardized_result, bins=60, density=True, color='mediumpurple', 
         edgecolor='black', alpha=0.7, label='Empirical Sample Means', zorder=3)

# Clean, professional title and labels
plt.title("Failure of the CLT: Lindeberg Condition (Finite Variance, Non-Identical)", fontsize=13)
plt.xlabel("Standardised Sample Sum")
plt.ylabel("Density")

# Add the legend to the top right
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()