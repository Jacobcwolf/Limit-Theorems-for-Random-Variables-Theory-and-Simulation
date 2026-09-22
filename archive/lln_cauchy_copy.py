import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42) # Uncomment to lock in the graph

def visualize_lln_failure(num_trials=10000):
    """
    Visualises the failure of the LLN by plotting the running average 
    of a standard Cauchy distribution (which has no defined mean).
    """
    # Simulate sampling from a Standard Cauchy Distribution
    cauchy_samples = np.random.standard_cauchy(size=num_trials)
    
    # Calculate the cumulative running average
    running_averages = np.cumsum(cauchy_samples) / np.arange(1, num_trials + 1)

    # Plotting the failure
    plt.figure(figsize=(10, 5))
    plt.plot(running_averages, label='Cauchy Running Average', color='purple', alpha=0.8)
    plt.axhline(y=0, color='red', linestyle='--', label='Median (0)')
    
    # We strictly limit the y-axis because Cauchy outliers can easily reach +/- thousands
    plt.ylim(-20, 20) 
    
    plt.title('Failure of the LLN: Running Average of a Cauchy Distribution')
    plt.xlabel('Number of Trials (n)')
    plt.ylabel('Sample Mean')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# Execute the visualization
visualize_lln_failure()