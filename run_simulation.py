import flwr as fl
from flwr.common import parameters_to_ndarrays
import torch
import os
from collections import OrderedDict

from client.client import PlantClient
from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

# ==========================================
# ⚙️ SIMULATION CONFIGURATION ⚙️
# ==========================================
# ALGORITHM = "FedAvg"   # Other option "FedProx" !
ALGORITHM = "FedProx"   # Other option "FedAvg" !
PROXIMAL_MU = 0.1      # Only used if ALGORITHM is "FedProx"
NUM_CLIENTS = 5
NUM_ROUNDS = 25        # Increased to 25 so the models actually learn!
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading partitioned datasets for {NUM_CLIENTS} virtual clients...")
client_loaders, val_loader, classes = load_federated_data(num_clients=NUM_CLIENTS, batch_size=32, iid=False)
NUM_CLASSES = len(classes)

# --- 💾 CUSTOM MODEL SAVING LOGIC ---
def save_global_model(parameters, filename):
    """Converts Flower parameters to PyTorch and saves them to disk."""
    print(f"\n💾 Intercepting global weights and saving to results/{filename}...")
    ndarrays = parameters_to_ndarrays(parameters)
    
    # Load empty architecture
    model = get_mobilenet(num_classes=NUM_CLASSES)
    
    # Map NDArrays back to PyTorch state_dict
    params_dict = zip(model.state_dict().keys(), ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    # Save to disk
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), f'results/{filename}')
    print("✅ Save complete!\n")

# Custom Strategies to trigger the save function on the final round
class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None and server_round == NUM_ROUNDS:
            save_global_model(aggregated_parameters, "fedavg_model.pth")
        return aggregated_parameters, aggregated_metrics

class SaveModelFedProx(fl.server.strategy.FedProx):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None and server_round == NUM_ROUNDS:
            save_global_model(aggregated_parameters, "fedprox_model.pth")
        return aggregated_parameters, aggregated_metrics
# ------------------------------------

def client_fn(cid: str) -> fl.client.Client:
    model = get_mobilenet(num_classes=NUM_CLASSES).to(device)
    train_loader = client_loaders[int(cid)]
    mu_value = PROXIMAL_MU if ALGORITHM == "FedProx" else 0.0
    return PlantClient(model, train_loader, val_loader, device, mu=mu_value).to_client()

if __name__ == "__main__":
    print(f"\n🚀 Starting {NUM_ROUNDS} Rounds of Federated Learning using {ALGORITHM}...")
    
    # Use our custom saving strategies
    if ALGORITHM == "FedProx":
        strategy = SaveModelFedProx(
            fraction_fit=1.0,            
            fraction_evaluate=1.0,       
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            proximal_mu=PROXIMAL_MU,
        )
    else:
        strategy = SaveModelFedAvg(
            fraction_fit=1.0,            
            fraction_evaluate=1.0,       
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0},
    )