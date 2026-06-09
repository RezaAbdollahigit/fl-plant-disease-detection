import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

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
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, dataset.classes

def load_federated_data(data_dir='data/PlantVillage', num_clients=5, batch_size=32, iid=False):
    """
    Partitions the dataset for Federated Learning.
    If iid=True: Random uniform split (Easy Mode).
    If iid=False: Non-IID split where each client gets entirely different diseases (Hard Mode).
    """
    train_loader, val_loader, classes = load_centralized_data(data_dir, batch_size=batch_size)
    
    # We extract the underlying base dataset and the specific training indices
    base_dataset = train_loader.dataset.dataset
    train_indices = train_loader.dataset.indices
    
    client_loaders = []

    if iid:
        print("Dataset Mode: IID (Uniform distribution)")
        partition_size = len(train_indices) // num_clients
        lengths = [partition_size] * num_clients
        lengths[-1] += len(train_indices) % num_clients
        client_datasets = random_split(train_loader.dataset, lengths)
        
        for ds in client_datasets:
            client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2))
            
    else:
        print("Dataset Mode: Non-IID (Heterogeneous label distribution)")
        # 1. Map each class to the training images that belong to it
        class_to_indices = {i: [] for i in range(len(classes))}
        for idx in train_indices:
            label = base_dataset.targets[idx]
            class_to_indices[label].append(idx)
            
        # 2. Assign exactly 3 distinct disease classes to each of the 5 clients
        classes_per_client = len(classes) // num_clients
        
        for i in range(num_clients):
            client_specific_indices = []
            assigned_classes = range(i * classes_per_client, (i + 1) * classes_per_client)
            
            for c in assigned_classes:
                client_specific_indices.extend(class_to_indices[c])
                
            # Create a localized dataset just for this client
            client_ds = Subset(base_dataset, client_specific_indices)
            client_loaders.append(DataLoader(client_ds, batch_size=batch_size, shuffle=True, num_workers=2))

    return client_loaders, val_loader, classes

if __name__ == "__main__":
    # Test the new Non-IID distribution logic
    print("Testing Non-IID Federated Partitions...")
    client_loaders, _, classes = load_federated_data(num_clients=5, iid=False)
    
    print("\n--- Client Data Distribution ---")
    for i, loader in enumerate(client_loaders):
        # Extract labels to prove the Non-IID setup works
        labels = [loader.dataset.dataset.targets[idx] for idx in loader.dataset.indices]
        unique_classes = np.unique(labels)
        class_names = [classes[c] for c in unique_classes]
        print(f"Client {i+1} has {len(loader.dataset)} images handling diseases: {class_names}")