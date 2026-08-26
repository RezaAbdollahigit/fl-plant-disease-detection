import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import load_centralized_data
from models.mobilenet import get_mobilenet
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import time
import random

# 1. ENFORCE DETERMINISTIC SEEDS GLOBALLY
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train_baseline(epochs=5):
    # 2. Setup GPU Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on: {device}")

    # 3. Load Data
    print("Loading dataset...")
    train_loader, val_loader, classes = load_centralized_data(batch_size=32)
    
    # 4. Calculate Weighted Loss for Imbalanced Data
    print("Calculating class weights (Optimized)...")
    all_labels = [train_loader.dataset.dataset.targets[i] for i in train_loader.dataset.indices]
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # 5. Initialize Model, Loss, and Optimizer
    model = get_mobilenet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. The Training Loop
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f"Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Completed | Train Accuracy: {train_acc:.2f}% | Avg Loss: {epoch_loss:.4f}")

    # 7. Evaluate on Unseen Validation Data
    print("\n🔍 Evaluating Baseline on Validation Set...")
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    print(f"🏆 True Baseline Validation Accuracy: {val_acc:.2f}%")

    # 8. Save the Final Global Model
    os.makedirs('results', exist_ok=True)
    save_path = 'results/baseline_model.pth'
    torch.save(model.state_dict(), save_path)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 Baseline training complete in {total_time:.2f} minutes!")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train_baseline(epochs=5)