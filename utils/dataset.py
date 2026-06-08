import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms():
    """
    Standard image transformations required for pre-trained models
    like MobileNetV2 and EfficientNet.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_centralized_data(data_dir='data/PlantVillage', batch_size=32, train_split=0.8):
    """
    Loads the dataset for testing a standard, centralized baseline model.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}. Please check your path.")

    dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms())
    
    # Split into Train and Validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, dataset.classes

def load_federated_data(data_dir='data/PlantVillage', num_clients=5, batch_size=32):
    """
    Partitions the training dataset into IID slices for Federated Learning clients.
    Returns a list of train_loaders, one for each client, and a global val_loader.
    """
    train_loader, val_loader, classes = load_centralized_data(data_dir, batch_size=batch_size)
    
    # Get the raw training dataset from the centralized loader
    train_dataset = train_loader.dataset
    
    # Calculate partition sizes
    partition_size = len(train_dataset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(train_dataset) % num_clients # Add remainder to last client
    
    # Split the dataset
    client_datasets = random_split(train_dataset, lengths)
    
    # Create a DataLoader for each client
    client_loaders = []
    for ds in client_datasets:
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2))
        
    return client_loaders, val_loader, classes

if __name__ == "__main__":
    # Quick test to ensure everything works when you run this script directly
    print("Testing Centralized DataLoader...")
    train, val, classes = load_centralized_data()
    print(f"Classes found: {len(classes)}")
    print(f"Train batches: {len(train)} | Val batches: {len(val)}")
    
    print("\nTesting Federated Partitions...")
    client_loaders, global_val, _ = load_federated_data(num_clients=5)
    for i, loader in enumerate(client_loaders):
        print(f"Client {i+1} has {len(loader.dataset)} images.")