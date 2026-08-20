import matplotlib.pyplot as plt
import json
import os

def load_metrics(filename):
    """Loads metrics from the JSON logs generated during simulation."""
    path = f"results/{filename}"
    if not os.path.exists(path):
        return None, None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("loss", []), data.get("accuracy", [])

def generate_dual_plots():
    avg_loss, avg_acc = load_metrics("fedavg_metrics.json")
    prox_loss, prox_acc = load_metrics("fedprox_metrics.json")

    if not avg_loss or not prox_loss:
        print("❌ Missing metric files. Please run both FedAvg and FedProx simulations first.")
        return

    rounds = list(range(1, len(avg_loss) + 1))

    # Setup dual subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 📉 Plot 1: Cross-Entropy Loss (Stability)
    ax1.plot(rounds, avg_loss, marker='o', color='#d9534f', linewidth=2, label='Standard FedAvg')
    ax1.plot(rounds, prox_loss, marker='s', color='#5cb85c', linewidth=2, label='FedProx (Proposed Algorithm)')
    ax1.set_title('Training Loss (Stability Proof)', fontsize=14, pad=15)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)

    # 📈 Plot 2: Accuracy (Convergence Speed)
    if avg_acc and prox_acc:
        avg_acc_perc = [a * 100 for a in avg_acc]
        prox_acc_perc = [a * 100 for a in prox_acc]
        
        ax2.plot(rounds, avg_acc_perc, marker='o', color='#d9534f', linewidth=2, label='Standard FedAvg')
        ax2.plot(rounds, prox_acc_perc, marker='s', color='#5cb85c', linewidth=2, label='FedProx (Proposed Algorithm)')
        ax2.set_title('Global Evaluation Accuracy (Convergence Proof)', fontsize=14, pad=15)
        ax2.set_xlabel('Communication Round', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=12)

    # Save the output
    os.makedirs('results', exist_ok=True)
    save_path = 'results/dual_metrics_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Dual metrics chart successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_dual_plots()