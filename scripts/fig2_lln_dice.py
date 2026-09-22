import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

def visualize_lln(num_trials=10000):
    """
    Visualises the Strong Law of Large Numbers using a running average of dice rolls.
    """
    true_mean = 3.5
    
    # Simulate rolling a 6-sided die 'num_trials' times
    rolls = np.random.randint(1, 7, size=num_trials)
    
    # Calculate the cumulative running average
    running_averages = np.cumsum(rolls) / np.arange(1, num_trials + 1)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(running_averages, label='Running Average', color='blue', alpha=0.8)
    plt.axhline(y=true_mean, color='red', linestyle='--', linewidth=2, label=f'True Mean ({true_mean})')
    
    plt.title('Law of Large Numbers: Running Average of Dice Rolls')
    plt.xlabel('Number of Trials (n)')
    plt.ylabel('Sample Mean')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# Execute the visualization with 10,000 trials
visualize_lln()