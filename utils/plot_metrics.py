import matplotlib.pyplot as plt
import json
import os

def load_metrics(filename):
    """Loads metrics from the JSON logs."""
    path = f"results/{filename}"
    if not os.path.exists(path):
        return None, None, None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("train_loss", []), data.get("val_loss", []), data.get("val_accuracy", [])

def generate_separated_plots():
    avg_t_loss, avg_v_loss, avg_v_acc = load_metrics("fedavg_metrics.json")
    prox_t_loss, prox_v_loss, prox_v_acc = load_metrics("fedprox_metrics.json")

    if not avg_t_loss or not prox_t_loss:
        print("❌ Missing metric files. Please ensure the JSON files are in the results folder.")
        return

    rounds = list(range(1, len(avg_t_loss) + 1))
    os.makedirs('results', exist_ok=True)

    # ==========================================
    # PLOT 1: TRAINING LOSS
    # ==========================================
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(rounds, avg_t_loss, marker='o', color='#d9534f', linewidth=2, markersize=6, label='Standard FedAvg')
    plt.plot(rounds, prox_t_loss, marker='s', color='#5cb85c', linewidth=2, markersize=6, label=r'FedProx ($\mu=0.1$)')
    
    plt.title('Global Training Loss Convergence (Cross-Entropy)', fontsize=16, pad=15, fontweight='bold')
    plt.xlabel('Federated Communication Round', fontsize=14)
    plt.ylabel('Training Loss (per batch)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    loss_path = 'results/training_loss_chart.png'
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Training loss chart saved: {loss_path}")

    # ==========================================
    # PLOT 2: VALIDATION ACCURACY
    # ==========================================
    if avg_v_acc and prox_v_acc:
        avg_acc_perc = [a * 100 for a in avg_v_acc]
        prox_acc_perc = [a * 100 for a in prox_v_acc]
        
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(rounds, avg_acc_perc, marker='o', color='#d9534f', linewidth=2, markersize=6, label='Standard FedAvg')
        plt.plot(rounds, prox_acc_perc, marker='s', color='#5cb85c', linewidth=2, markersize=6, label=r'FedProx ($\mu=0.1$)')
        
        plt.title('Global Validation Accuracy (Performance on Unseen Edge Data)', fontsize=16, pad=15, fontweight='bold')
        plt.xlabel('Federated Communication Round', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
        
        acc_path = 'results/validation_accuracy_chart.png'
        plt.savefig(acc_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Validation accuracy chart saved: {acc_path}")

    # ==========================================
    # PLOT 3: NORMALIZED VALIDATION LOSS
    # ==========================================
    if avg_v_loss and prox_v_loss:
        # Normalize accumulated sum over exactly 129 validation batches (4128 images / 32 batch size)
        num_val_batches = 129.0
        norm_avg_v_loss = [v / num_val_batches for v in avg_v_loss]
        norm_prox_v_loss = [v / num_val_batches for v in prox_v_loss]

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(rounds, norm_avg_v_loss, marker='o', color='#d9534f', linewidth=2, markersize=6, label='Standard FedAvg')
        plt.plot(rounds, norm_prox_v_loss, marker='s', color='#5cb85c', linewidth=2, markersize=6, label=r'FedProx ($\mu=0.1$)')
        
        plt.title('Global Validation Loss (Cross-Entropy on Unseen Data)', fontsize=16, pad=15, fontweight='bold')
        plt.xlabel('Federated Communication Round', fontsize=14)
        plt.ylabel('Cross-Entropy Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
        
        val_loss_path = 'results/validation_loss_chart.png'
        plt.savefig(val_loss_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Normalized validation loss chart saved: {val_loss_path}")

if __name__ == "__main__":
    generate_separated_plots()