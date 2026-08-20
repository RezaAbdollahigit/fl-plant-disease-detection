import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

def plot_f1():
    print("Loading data and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_loader, classes = load_federated_data(num_clients=3, batch_size=32, iid=False, alpha=0.1)
    num_classes = len(classes)
    
    model = get_mobilenet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("results/fedprox_model.pth", map_location=device))
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.to("cpu").numpy())
            
    print("Generating F1-Score chart...")
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    df = pd.DataFrame(report).transpose().iloc[:-3, :]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=df['f1-score'], y=df.index, palette="viridis")
    plt.title('Granular F1-Score per Disease Class (FedProx)', fontsize=16)
    plt.xlabel('F1-Score', fontsize=12)
    plt.ylabel('Disease Category', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/f1_breakdown.png', dpi=300, bbox_inches='tight')
    print("✅ F1-Score chart saved to results/f1_breakdown.png")

if __name__ == "__main__":
    plot_f1()