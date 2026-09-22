import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42) # Uncomment to lock in the graph

def visualize_wlln_failure(max_n=3000, num_experiments=10000, epsilon=1.0):
    """
    Highly optimized visualization of the WLLN failing. 
    Plots the probability of error for a Cauchy distribution.
    """
    # Generate ALL Cauchy samples at once: 10,000 experiments x 3,000 rolls
    samples = np.random.standard_cauchy(size=(num_experiments, max_n))
    
    # Calculate the running sum for all 10,000 experiments simultaneously
    running_sums = np.cumsum(samples, axis=1)
    
    # Create an array of n values [1, 2, 3, ..., 3000]
    n_values = np.arange(1, max_n + 1)
    
    # Divide the running sums by n to get the running sample means
    running_means = running_sums / n_values
    
    # Check where the absolute difference from the median (0) is greater than epsilon
    deviations = np.abs(running_means) > epsilon
    
    # Calculate the empirical probability at each step 'n'
    probabilities = np.mean(deviations, axis=0)
    
    # Plotting (we plot every 10th point to make the graphing window render faster)
    plt.figure(figsize=(10, 5))
    plt.plot(n_values[9::10], probabilities[9::10], color='purple', alpha=0.8, linewidth=2, 
             label=f'Error > {epsilon}')
    plt.axhline(y=0, color='red', linestyle='--', label='Convergence Limit (0)')
    
    plt.title('Failure of the WLLN: Probability of Error (Cauchy Distribution)')
    plt.xlabel('Sample Size (n)')
    plt.ylabel(f'P(|Sample Mean| > {epsilon})')
    
    # Setting the y-axis limit slightly above 0.5 to clearly show the flatline
    plt.ylim(-0.05, 0.6)
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# Execute the visualization
visualize_wlln_failure()
