import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set a random seed for reproducibility
# np.random.seed(42)

def visualize_clt(num_samples=10000, sample_size=50):
    """
    Visualises the Central Limit Theorem by plotting the distribution 
    of sample means from a non-normal (Uniform) distribution.
    """
    # Generate samples from a Uniform(0,1) distribution and calculate their means
    sample_means = [np.mean(np.random.uniform(0, 1, sample_size)) for _ in range(num_samples)]
        
    # Plotting the histogram of sample means
    plt.figure(figsize=(10, 5))
    count, bins, ignored = plt.hist(sample_means, bins=40, density=True, 
                                    alpha=0.6, color='skyblue', edgecolor='black', 
                                    label='Empirical Sample Means')
    
    # Overlay the theoretical normal distribution curve
    mu = 0.5 # True mean of Uniform(0,1)
    sigma = np.sqrt((1/12) / sample_size) # Standard error of Uniform(0,1)
    x = np.linspace(min(bins), max(bins), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color='red', linewidth=2, 
             label='Theoretical Normal Curve')
    
    plt.title(f'Central Limit Theorem: Distribution of {num_samples} Sample Means')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Execute the visualization
visualize_clt()