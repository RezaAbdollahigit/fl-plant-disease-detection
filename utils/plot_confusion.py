import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Dynamically add the project root to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

def plot_dual_cm():
    print("Loading validation data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset to access the validation loader and class names
    _, val_loader, classes = load_federated_data(num_clients=5, batch_size=32, iid=False, alpha=0.1)
    num_classes = len(classes)
    
    models = {
        "Standard FedAvg": "results/fedavg_model.pth",
        "FedProx (Proposed Algorithm)": "results/fedprox_model.pth"
    }
    
    cms = {}
    
    # Run inference for both models
    for title, path in models.items():
        if not os.path.exists(path):
            print(f"❌ Error: Could not find {path}. Make sure it is in the results folder.")
            return
            
        print(f"Running inference for {title}...")
        model = get_mobilenet(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        cms[title] = confusion_matrix(all_labels, all_preds)
        
    print("Generating side-by-side heatmaps...")
    
    # Set up a wide figure for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    for ax, (title, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes, 
                    cbar=False, linewidths=0.5, ax=ax)
                    
        ax.set_title(f'{title}\nConfusion Matrix', fontsize=16, pad=20)
        ax.set_xlabel('Predicted Disease Class', fontsize=12, labelpad=10)
        ax.set_ylabel('True Disease Class', fontsize=12, labelpad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    save_path = "results/dual_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Dual Confusion Matrix successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    plot_dual_cm()