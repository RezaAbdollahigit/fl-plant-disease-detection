import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import load_centralized_data
from models.mobilenet import get_mobilenet
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import time

def train_baseline(epochs=5):
    # 1. Setup GPU Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on: {device}")

    # 2. Load Data
    print("Loading dataset...")
    train_loader, val_loader, classes = load_centralized_data(batch_size=32)
    
    # 3. Calculate Weighted Loss for Imbalanced Data
    print("Calculating class weights...")
    # Extracting labels from the dataset to find the exact distribution
    all_labels = [label for _, label in train_loader.dataset]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    # Move the weights to the GPU so PyTorch can use them during training
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # 4. Initialize Model, Loss (with weights), and Optimizer
    model = get_mobilenet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. The Training Loop
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the GPU
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 100 batches
            if (i+1) % 100 == 0:
                print(f"Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Completed | Accuracy: {train_acc:.2f}% | Avg Loss: {epoch_loss:.4f}")

    # 6. Save the Final Global Model
    os.makedirs('results', exist_ok=True)
    save_path = 'results/baseline_model.pth'
    torch.save(model.state_dict(), save_path)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 Baseline training complete in {total_time:.2f} minutes!")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    # 5 Epochs is enough to establish a solid baseline
    train_baseline(epochs=5)