import matplotlib
matplotlib.use('Agg') # Stops the macOS window crash
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------
# Parameters for the Simulation
# ---------------------------------------------------------
n_simulations = 10000
sample_size = 50

# Uniform Distribution Parameters U(0,1)
# Guarantees FINITE variance
mu = 0.5
sigma = np.sqrt(1/12)

# ---------------------------------------------------------
# Simulating the Dependent Random Variables
# ---------------------------------------------------------
# Draw one uniform value, and force all 50 variables to be identical (infinite memory)
sample_means = np.random.uniform(0, 1, n_simulations)

# ---------------------------------------------------------
# Applying the strict CLT Scaling Factor (Standardisation)
# ---------------------------------------------------------
# Z = (Sample Mean - True Mean) / (Standard Error)
z_scores = (sample_means - mu) / (sigma / np.sqrt(sample_size))

# ---------------------------------------------------------
# Plotting the Failure
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# 1. Plot the histogram of our Standardised Z-scores (NOW PURPLE)
plt.hist(z_scores, bins=50, density=True, alpha=0.7, color='purple', 
         edgecolor='black', label='Standardised Sample Means\n(Infinite Dependence)')

# 2. Plot the Standard Normal Curve N(0,1) (NOW A SOLID BLACK LINE)
x = np.linspace(-15, 15, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), color='black', linestyle='solid', linewidth=2.5, 
         label='Standard Normal $\mathcal{N}(0,1)$\n(Expected CLT Outcome)')

# Formatting the graph to academic standards
plt.title('Central Limit Theorem Failure: Infinite Dependence Axiom', fontsize=14, fontweight='bold')
plt.xlabel('Standardised $Z$-Score', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim(-15, 15)
plt.legend(fontsize=11, loc='upper right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Save the figure silently
plt.tight_layout()
plt.savefig('Figure_6_CLT_fail_fix.png', dpi=300)

print("Success! The new purple graph with a solid bell curve has been saved.")