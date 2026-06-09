import flwr as fl
import torch
from client.client import PlantClient
from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

# ==========================================
# ⚙️ SIMULATION CONFIGURATION ⚙️
# ==========================================
# ALGORITHM = "FedProx"  # Options: "FedAvg" or "FedProx"
ALGORITHM = "FedAvg"  # Options: "FedAvg" or "FedProx"
PROXIMAL_MU = 0.1      # Only used if ALGORITHM is "FedProx"
NUM_CLIENTS = 5
NUM_ROUNDS = 3
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading partitioned datasets for {NUM_CLIENTS} virtual clients...")
# Running in HARD MODE (Non-IID) to show why FedProx is needed
client_loaders, val_loader, classes = load_federated_data(num_clients=NUM_CLIENTS, batch_size=32, iid=False)

def client_fn(cid: str) -> fl.client.Client:
    """Creates a PlantClient instance on demand for the simulation."""
    model = get_mobilenet(num_classes=len(classes)).to(device)
    train_loader = client_loaders[int(cid)]
    
    # Pass the penalty value to the client
    mu_value = PROXIMAL_MU if ALGORITHM == "FedProx" else 0.0
    
    return PlantClient(model, train_loader, val_loader, device, mu=mu_value).to_client()

if __name__ == "__main__":
    print(f"\n🚀 Starting Federated Learning Simulation using {ALGORITHM}...")
    print(f"Dataset Mode: Non-IID (Heterogeneous)")
    
    # Define the Strategy based on your configuration
    if ALGORITHM == "FedProx":
        strategy = fl.server.strategy.FedProx(
            fraction_fit=1.0,            
            fraction_evaluate=1.0,       
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            proximal_mu=PROXIMAL_MU,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,            
            fraction_evaluate=1.0,       
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )
    
    # Start the Virtual Engine
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0},
    )