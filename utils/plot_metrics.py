import matplotlib.pyplot as plt
import os

def generate_comparison_chart():
    # These are the exact numbers from the terminal output!
    rounds = list(range(1, 26))
    
    fedavg_loss = [
        334.05, 315.38, 313.84, 296.28, 282.99, 265.53, 293.09, 277.83, 273.10, 
        306.12, 279.92, 295.14, 311.76, 308.21, 324.66, 308.34, 321.45, 333.74, 
        318.57, 294.33, 332.51, 310.83, 314.91, 294.10, 287.88
    ]
    
    fedprox_loss = [
        301.36, 254.44, 219.12, 196.51, 170.09, 153.02, 151.37, 136.24, 129.84, 
        127.55, 114.97, 116.74, 108.10, 110.09, 101.56, 105.46, 95.27, 98.79, 
        95.60, 96.85, 98.14, 93.68, 89.75, 88.66, 89.09
    ]

    # Set up the plot style for a professional academic look
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, fedavg_loss, marker='o', linestyle='-', color='#d9534f', linewidth=2, label='Standard FedAvg (Fails to Converge)')
    plt.plot(rounds, fedprox_loss, marker='s', linestyle='-', color='#5cb85c', linewidth=2, label='FedProx (Proposed Algorithm)')

    # Titles and Labels
    plt.title('Federated Learning Training Loss under Non-IID Data\n(FedAvg vs. FedProx over 25 Rounds)', fontsize=14, pad=15)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    
    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(50, 400) # Forces the top of the chart higher to create empty space
    plt.legend(fontsize=12, loc='upper right', framealpha=1.0) # Makes the box solid
    plt.xticks(range(1, 26, 2)) # Show tick marks every 2 rounds
    
    # Save the chart
    os.makedirs('results', exist_ok=True)
    save_path = 'results/federated_comparison_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Academic chart successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_comparison_chart()