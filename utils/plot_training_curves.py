import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
FEDAVG_JSON = os.path.join(RESULTS_DIR, 'fedavg_metrics.json')
FEDPROX_JSON = os.path.join(RESULTS_DIR, 'fedprox_metrics.json')

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def main():
    if not os.path.exists(FEDAVG_JSON) or not os.path.exists(FEDPROX_JSON):
        print(f"Error: Could not find metrics JSON files in '{RESULTS_DIR}/'.")
        return

    fedavg_data = load_metrics(FEDAVG_JSON)
    fedprox_data = load_metrics(FEDPROX_JSON)

    rounds = list(range(1, len(fedavg_data['train_loss']) + 1))

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. Training Loss
    axes[0].plot(rounds, fedavg_data['train_loss'], label='FedAvg (Standard)', color='#e74c3c', marker='o', linewidth=2)
    axes[0].plot(rounds, fedprox_data['train_loss'], label=r'FedProx ($\mu=0.01$)', color='#2ecc71', marker='s', linewidth=2)
    axes[0].set_title('Global Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Communication Round', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Validation Loss
    axes[1].plot(rounds, fedavg_data['val_loss'], label='FedAvg (Standard)', color='#e74c3c', marker='o', linewidth=2)
    axes[1].plot(rounds, fedprox_data['val_loss'], label=r'FedProx ($\mu=0.01$)', color='#2ecc71', marker='s', linewidth=2)
    axes[1].set_title('Global Validation Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Communication Round', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Validation Accuracy
    avg_acc = [acc * 100 if acc <= 1.0 else acc for acc in fedavg_data['val_accuracy']]
    prox_acc = [acc * 100 if acc <= 1.0 else acc for acc in fedprox_data['val_accuracy']]

    axes[2].plot(rounds, avg_acc, label='FedAvg (Standard)', color='#e74c3c', marker='o', linewidth=2)
    axes[2].plot(rounds, prox_acc, label=r'FedProx ($\mu=0.01$)', color='#2ecc71', marker='s', linewidth=2)
    axes[2].set_title('Global Validation Accuracy (%)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Communication Round', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].legend(loc='lower right', frameon=True)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'dual_metrics_chart.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Convergence curves successfully saved to '{output_path}'")

if __name__ == '__main__':
    main()