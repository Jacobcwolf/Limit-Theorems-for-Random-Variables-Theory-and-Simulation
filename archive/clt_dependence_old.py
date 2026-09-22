import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42) # Uncomment to lock in the exact shape of the graph

def visualize_mixing_failure(num_samples=10000, sample_size=50):
    """
    Visualises the failure of the CLT when finite-variance data 
    possesses infinite dependence (violating strong mixing conditions).
    """
    # 1. Generate the shared underlying latent factor for each experiment
    # We use Uniform data (between 0 and 1), which strictly has a finite mean and variance.
    shared_factors = np.random.uniform(0, 1, num_samples)
    
    # 2. Break the independence (Infinite Memory):
    # We force all 50 variables in each sample to be perfectly correlated to the shared factor.
    # The correlation never decays, representing a complete collapse of "Strong Mixing".
    samples = np.tile(shared_factors, (sample_size, 1)).T
    
    # Calculate the sample means
    # Because every variable in a sample is identical, the mean is just the original variable!
    sample_means = np.mean(samples, axis=1)

    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    
    # We use 50 bins to give it a clean, blocky histogram look matching the other figures
    plt.hist(sample_means, bins=50, density=True, alpha=0.6, 
             color='#9B59B6', edgecolor='black', label='Empirical Sample Means')

    plt.title('Failure of the CLT: Infinite Dependence (Strong Mixing Collapse)')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    
    # We limit the x-axis to exactly 0 to 1 to show the flat uniform shape clearly
    plt.xlim(0, 1)
    
    # Add a theoretical line to show what it *should* look like if it hadn't failed
    # (Optional: you can remove this if you only want the purple blocks)
    # plt.axhline(1, color='red', linestyle='dashed', linewidth=2, label='Original Uniform Shape')
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# Execute the visualization
visualize_mixing_failure()