import flwr as fl
import torch
from client.client import PlantClient
from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

# 1. Global Setup
NUM_CLIENTS = 5
NUM_ROUNDS = 3  # We will do 3 rounds just to test the pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading partitioned datasets for virtual clients...")
client_loaders, val_loader, classes = load_federated_data(num_clients=NUM_CLIENTS, batch_size=32)

# 2. Virtual Client Generator
def client_fn(cid: str) -> fl.client.Client:
    """Creates a PlantClient instance on demand for the simulation."""
    # Load the empty architecture
    model = get_mobilenet(num_classes=len(classes)).to(device)
    
    # Get the specific data slice for this client ID
    train_loader = client_loaders[int(cid)]
    
    return PlantClient(model, train_loader, val_loader, device).to_client()

if __name__ == "__main__":
    print("\n🚀 Starting Federated Learning Simulation (FedAvg)...")
    
    # 3. Define the Aggregation Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,            # Sample 100% of available clients for training
        fraction_evaluate=1.0,       # Sample 100% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    
    # 4. Start the Virtual Engine
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        # This allocates the GPU sequentially to avoid Out-of-Memory crashes
        client_resources={"num_cpus": 2, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0},
    )