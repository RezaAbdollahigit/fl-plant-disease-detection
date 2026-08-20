import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Dynamically add the project root to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

def plot_dual_f1():
    print("Loading data and initializing comparative evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sticking with your 5-client architecture!
    _, val_loader, classes = load_federated_data(num_clients=5, batch_size=32, iid=False, alpha=0.1)
    num_classes = len(classes)
    
    models = {
        "Standard FedAvg": "results/fedavg_model.pth",
        "FedProx (Proposed Algorithm)": "results/fedprox_model.pth"
    }
    
    all_f1_scores = []
    
    for title, path in models.items():
        if not os.path.exists(path):
            print(f"❌ Error: Could not find {path}.")
            return
            
        print(f"Running inference for {title}...")
        model = get_mobilenet(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.to("cpu").numpy())
                
        report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
        
        for class_name in classes:
            all_f1_scores.append({
                "Algorithm": title,
                "Disease Category": class_name,
                "F1-Score": report[class_name]["f1-score"]
            })
            
    print("Generating grouped F1-Score chart...")
    df = pd.DataFrame(all_f1_scores)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df, x='F1-Score', y='Disease Category', hue='Algorithm', palette=['#d9534f', '#5cb85c'])
    plt.title('Comparative F1-Score per Disease Class (FedAvg vs. FedProx)', fontsize=16, pad=15)
    plt.xlabel('F1-Score (Higher is Better)', fontsize=12)
    plt.ylabel('Disease Category', fontsize=12)
    plt.legend(title='Aggregation Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/dual_f1_breakdown.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Dual F1-Score chart successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    plot_dual_f1()