import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42) # Uncomment to lock in the graph

def visualize_clt_failure(num_samples=10000, sample_size=50):
    """
    Visualises the failure of the CLT by plotting sample means from 
    a Pareto distribution with infinite variance (alpha = 1.5).
    """
    alpha = 1.5 # Finite mean, but infinite variance
    
    # Generate Pareto samples and calculate their means
    samples = np.random.pareto(alpha, (num_samples, sample_size))
    sample_means = np.mean(samples, axis=1)

    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    
    # We restrict the visual range to (0, 10) to clearly see the skewness.
    # Without this, infinite variance outliers would break the visual scale.
    plt.hist(sample_means, bins=50, range=(0, 10), density=True, 
             alpha=0.6, color='orange', edgecolor='black', 
             label='Empirical Sample Means')

    plt.title('Failure of the CLT: Sample Means of an Infinite Variance Pareto Distribution')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# Execute the visualization
visualize_clt_failure()