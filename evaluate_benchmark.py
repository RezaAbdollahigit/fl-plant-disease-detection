import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from models.mobilenet import get_mobilenet
from utils.dataset import load_centralized_data

RESULTS_DIR = 'results'
DATA_DIR = os.path.join('data', 'PlantVillage')
BATCH_SIZE = 32

def evaluate_model(model_path, model, dataloader, device):
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return None, None
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # Use unified centralized validation loader
    _, test_loader, classes = load_centralized_data(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    num_classes = len(classes)

    models_to_eval = {
        'Centralized Baseline': os.path.join(RESULTS_DIR, 'baseline_model.pth'),
        'Standard FedAvg': os.path.join(RESULTS_DIR, 'fedavg_model_round_25.pth'),
        'FedProx (Proposed)': os.path.join(RESULTS_DIR, 'fedprox_model_round_25.pth')
    }

    results = {}
    f1_scores_per_class = {}

    for name, path in models_to_eval.items():
        print(f"Evaluating {name}...")
        model = get_mobilenet(num_classes=num_classes, pretrained=False)
        y_true, y_pred = evaluate_model(path, model, test_loader, device)
        
        if y_true is not None:
            acc = accuracy_score(y_true, y_pred) * 100
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            per_class_f1 = f1_score(y_true, y_pred, average=None)
            
            results[name] = {'Accuracy': acc, 'Macro_F1': macro_f1, 'y_true': y_true, 'y_pred': y_pred}
            f1_scores_per_class[name] = per_class_f1

    print("\n" + "="*65)
    print("📋 QUANTITATIVE BENCHMARK SUMMARY")
    print("="*65)
    print(f"{'Model':<25} | {'Global Accuracy (%)':<20} | {'Macro-F1 Score':<15}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<25} | {m['Accuracy']:<20.2f} | {m['Macro_F1']:<15.4f}")
    print("="*65 + "\n")

    # Plot F1 Breakdown Bar Chart
    clean_labels = [c.replace('___', ' - ').replace('__', ' ').replace('_', ' ') for c in classes]
    x = np.arange(len(clean_labels))
    width = 0.25

    plt.figure(figsize=(16, 7), dpi=300)
    if 'Centralized Baseline' in f1_scores_per_class:
        plt.bar(x - width, f1_scores_per_class['Centralized Baseline'], width, label='Baseline', color='#3498db')
    if 'Standard FedAvg' in f1_scores_per_class:
        plt.bar(x, f1_scores_per_class['Standard FedAvg'], width, label='FedAvg (Round 25)', color='#e74c3c')
    if 'FedProx (Proposed)' in f1_scores_per_class:
        plt.bar(x + width, f1_scores_per_class['FedProx (Proposed)'], width, label='FedProx (Round 25)', color='#2ecc71')

    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Per-Class F1 Score Comparison Across 15 Disease Categories', fontsize=14, fontweight='bold')
    plt.xticks(x, clean_labels, rotation=75, ha='right', fontsize=9)
    plt.legend(loc='lower right', frameon=True)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    f1_plot_path = os.path.join(RESULTS_DIR, 'dual_f1_breakdown.png')
    plt.savefig(f1_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ F1 Breakdown chart saved to: {f1_plot_path}")

    # Plot Side-by-Side Confusion Matrices (FedAvg vs FedProx)
    if 'Standard FedAvg' in results and 'FedProx (Proposed)' in results:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
        
        cm_avg = confusion_matrix(results['Standard FedAvg']['y_true'], results['Standard FedAvg']['y_pred'], normalize='true')
        cm_prox = confusion_matrix(results['FedProx (Proposed)']['y_true'], results['FedProx (Proposed)']['y_pred'], normalize='true')

        sns.heatmap(cm_avg, ax=axes[0], cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)
        axes[0].set_title('Standard FedAvg (Round 25) - Normalized CM', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')

        sns.heatmap(cm_prox, ax=axes[1], cmap='Greens', cbar=True, xticklabels=False, yticklabels=False)
        axes[1].set_title('FedProx (Proposed, Round 25) - Normalized CM', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')

        plt.tight_layout()
        cm_path = os.path.join(RESULTS_DIR, 'dual_confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Dual confusion matrix chart saved to: {cm_path}")

if __name__ == '__main__':
    main()