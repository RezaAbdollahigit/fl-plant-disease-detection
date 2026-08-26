import os
import shutil
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

# 1. ENFORCE DETERMINISTIC SEEDS
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_transforms():
    """Standard image transformations for MobileNetV2."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_centralized_data(data_dir='data/PlantVillage', batch_size=32, train_split=0.8):
    """Loads and shuffles the full dataset for centralized baseline training."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}. Please check your path.")

    dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms())
    
    # 2. USE A SEEDED GENERATOR FOR THE SPLIT
    generator = torch.Generator().manual_seed(42)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 3. FREEZE THE TEST DATA TO THE HARD DRIVE FOR THE DASHBOARD DEMO
    holdout_dir = 'data/test_holdout'
    if not os.path.exists(holdout_dir):
        print(f"\n❄️ Freezing {val_size} test images to {holdout_dir} for the Streamlit dashboard...")
        os.makedirs(holdout_dir)
        for idx in val_dataset.indices:
            img_path, label = dataset.samples[idx]
            class_name = dataset.classes[label]
            
            class_dir = os.path.join(holdout_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            filename = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(class_dir, filename))
        print("✅ Test data successfully frozen!\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, dataset.classes

def load_federated_data(data_dir='data/PlantVillage', num_clients=5, batch_size=32, iid=False, alpha=0.1):
    """
    Partitions the dataset for Federated Learning.
    If iid=True: Random uniform split (Easy Mode).
    If iid=False: Non-IID split using Dirichlet distribution (Realistic Hard Mode).
    """
    train_loader, val_loader, classes = load_centralized_data(data_dir, batch_size=batch_size)
    
    base_dataset = train_loader.dataset.dataset
    train_indices = train_loader.dataset.indices
    
    client_loaders = []

    if iid:
        print("Dataset Mode: IID (Uniform distribution)")
        partition_size = len(train_indices) // num_clients
        lengths = [partition_size] * num_clients
        lengths[-1] += len(train_indices) % num_clients
        
        generator = torch.Generator().manual_seed(42)
        client_datasets = random_split(train_loader.dataset, lengths, generator=generator)
        
        for ds in client_datasets:
            client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2))
            
    else:
        print(f"Dataset Mode: Non-IID (Dirichlet Distribution, alpha={alpha})")
        class_to_indices = {i: [] for i in range(len(classes))}
        for idx in train_indices:
            label = base_dataset.targets[idx]
            class_to_indices[label].append(idx)
            
        client_indices = {i: [] for i in range(num_clients)}
        
        for c in range(len(classes)):
            c_indices = class_to_indices[c]
            np.random.shuffle(c_indices) 
            
            proportions = np.random.dirichlet([alpha] * num_clients)
            splits = (proportions * len(c_indices)).astype(int)
            
            start = 0
            for i in range(num_clients):
                end = start + splits[i]
                if i == num_clients - 1:  
                    end = len(c_indices)
                client_indices[i].extend(c_indices[start:end])
                start = end
                
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
            client_ds = Subset(base_dataset, client_indices[i])
            client_loaders.append(DataLoader(client_ds, batch_size=batch_size, shuffle=True, num_workers=2))

    return client_loaders, val_loader, classes