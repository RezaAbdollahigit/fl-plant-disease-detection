import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

def generate_individual_matrices():
    print("Loading data for Confusion Matrix generation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, classes = load_federated_data(num_clients=5, batch_size=32, iid=False, alpha=0.1)
    num_classes = len(classes)
    clean_classes = [c.replace('___', ' - ').replace('__', ' ').replace('_', ' ') for c in classes]
    
    models = {
        "fedavg": ("Standard FedAvg (Round 25)", "results/fedavg_model_round_25.pth"),
        "fedprox": ("FedProx Proposed Algorithm (Round 25)", "results/fedprox_model_round_25.pth")
    }
    
    os.makedirs('results', exist_ok=True)

    for algo_key, (title, path) in models.items():
        if not os.path.exists(path):
            print(f"❌ Error: Could not find {path}.")
            continue
            
        print(f"Generating matrix for {title}...")
        model = get_mobilenet(num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.to("cpu").numpy())
                
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(12, 10), dpi=300)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if algo_key == 'fedavg' else 'Greens',
                    xticklabels=clean_classes, yticklabels=clean_classes,
                    linewidths=.5, cbar=False)
        
        plt.title(f'{title}\nConfusion Matrix', fontsize=15, pad=15, fontweight='bold')
        plt.ylabel('True Disease Class', fontsize=12)
        plt.xlabel('Predicted Disease Class', fontsize=12)
        plt.xticks(rotation=75, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        save_path = f'results/{algo_key}_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved to: {save_path}")

if __name__ == "__main__":
    generate_individual_matrices()