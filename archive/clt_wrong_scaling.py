import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42) # Uncomment to lock in the exact shape of the graph

def visualize_scaling_failure_histogram(num_samples=10000, sample_size=80):
    """
    Visualises what happens to standard finite-variance data when 
    scaled by the incorrect Generalized CLT factor (n^(1/alpha)).
    Keeps the visual style consistent with previous histogram figures.
    """
    # 1. Generate standard Uniform data (Finite Mean, Finite Variance)
    samples = np.random.uniform(-0.5, 0.5, (num_samples, sample_size))
    
    # Calculate the sum of the samples
    sample_sums = np.sum(samples, axis=1)

    # 2. Scale correctly using standard CLT (divide by n^(1/2))
    # Multiplied by sqrt(12) to normalize the uniform baseline variance
    correct_scaling = (sample_sums / np.sqrt(sample_size)) * np.sqrt(12)

    # 3. Scale incorrectly using Generalized CLT scaling (divide by n^(1/1.5))
    alpha = 1.5
    # We also multiply this by sqrt(12) so the ONLY mathematical difference 
    # between the two sets of data is the denominator scale. 
    incorrect_scaling = (sample_sums / (sample_size ** (1 / alpha))) * np.sqrt(12)

    # Plotting both histograms side-by-side
    plt.figure(figsize=(10, 5))
    
    # Create a shared set of 80 bins from -4 to 4. 
    # This ensures both histograms have identically sized bars and prevents the "barcode" look.
    shared_bins = np.linspace(-4, 4, 80)
    
    # Plot the correct bell curve (Blue, matching previous styles)
    plt.hist(correct_scaling, bins=shared_bins, density=True, alpha=0.6, 
             color='#4C72B0', edgecolor='black', label='Correct Scaling ($n^{1/2}$)')
    
    # Plot the collapsed spike (Red, with black edges for consistency)
    plt.hist(incorrect_scaling, bins=shared_bins, density=True, alpha=0.75, 
             color='#C44E52', edgecolor='black', label=f'Incorrect Scaling ($n^{{1/{alpha}}}$)')

    plt.title('Breakdown of Standardisation: The Effect of the $n^{1/\\alpha}$ Scaling Factor')
    plt.xlabel('Standardised Value')
    plt.ylabel('Density')
    
    # Restrict the x-axis to clearly see the contrast
    plt.xlim(-3.5, 3.5)
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6) # Matches your previous grids exactly
    plt.show()

# Execute the visualization
visualize_scaling_failure_histogram()