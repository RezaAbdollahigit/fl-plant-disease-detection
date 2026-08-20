import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Dynamically add the project root to Python's path so it can find 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import load_federated_data

def plot_data_distribution():
    print("Generating Dirichlet data distribution proof...")
    
    # Keeping your 5-client architecture as requested!
    client_loaders, _, classes = load_federated_data(num_clients=5, batch_size=32, iid=False, alpha=0.1)
    
    num_clients = len(client_loaders)
    num_classes = len(classes)
    
    distribution = np.zeros((num_clients, num_classes))
    for cid, loader in enumerate(client_loaders):
        for _, labels in loader:
            for label in labels:
                distribution[cid][label.item()] += 1

    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(num_clients)
    client_labels = [f"Virtual Network {i+1}" for i in range(num_clients)]
    
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        class_counts = distribution[:, i]
        ax.bar(client_labels, class_counts, bottom=bottom, label=classes[i], color=colors[i], edgecolor='white')
        bottom += class_counts

    plt.title('Non-IID Data Distribution Across Virtual Networks\n(Severe Domain Shift via Dirichlet α = 0.1)', fontsize=14, pad=15)
    plt.xlabel('Edge Devices', fontsize=12)
    plt.ylabel('Total Images Allocated', fontsize=12)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title="Disease Classes")
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/data_distribution_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Data distribution visual successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    plot_data_distribution()