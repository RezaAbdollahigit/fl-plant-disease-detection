import matplotlib.pyplot as plt
import json
import os

def load_metrics(filename):
    """Loads metrics from the new JSON logs."""
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
        print("❌ Missing metric files. Please ensure the new JSON files are in the results folder.")
        return

    rounds = list(range(1, len(avg_t_loss) + 1))
    os.makedirs('results', exist_ok=True)

    # ==========================================
    # PLOT 1: TRAINING LOSS
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, avg_t_loss, marker='o', color='#d9534f', linewidth=2, label='Standard FedAvg')
    plt.plot(rounds, prox_t_loss, marker='s', color='#5cb85c', linewidth=2, label='FedProx (Proposed Algorithm)')
    plt.title('Training Loss (Mathematical Stability Proof)', fontsize=16, pad=15)
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Cross-Entropy Training Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    loss_path = 'results/training_loss_chart.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training loss chart saved: {loss_path}")

    # ==========================================
    # PLOT 2: VALIDATION ACCURACY
    # ==========================================
    if avg_v_acc and prox_v_acc:
        avg_acc_perc = [a * 100 for a in avg_v_acc]
        prox_acc_perc = [a * 100 for a in prox_v_acc]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, avg_acc_perc, marker='o', color='#d9534f', linewidth=2, label='Standard FedAvg')
        plt.plot(rounds, prox_acc_perc, marker='s', color='#5cb85c', linewidth=2, label='FedProx (Proposed Algorithm)')
        plt.title('Global Validation Accuracy (Performance on Unseen Edge Data)', fontsize=16, pad=15)
        plt.xlabel('Communication Round', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        acc_path = 'results/validation_accuracy_chart.png'
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Validation accuracy chart saved: {acc_path}")

if __name__ == "__main__":
    generate_separated_plots()