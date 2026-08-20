import os
import logging
import argparse
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'

import flwr as fl
from flwr.common import parameters_to_ndarrays
import torch
from collections import OrderedDict

logging.getLogger("flwr").setLevel(logging.ERROR)

from client.client import PlantClient
from utils.dataset import load_federated_data
from models.mobilenet import get_mobilenet

def save_global_model(parameters, filename, num_classes):
    print(f"\n💾 Intercepting global weights and saving to results/{filename}...")
    ndarrays = parameters_to_ndarrays(parameters)
    model = get_mobilenet(num_classes=num_classes)
    params_dict = zip(model.state_dict().keys(), ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), f'results/{filename}')
    print("✅ Save complete!\n")

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_custom_strategy(algo, num_rounds, num_clients, num_classes, mu):
    if algo == "fedprox":
        class SaveModelFedProx(fl.server.strategy.FedProx):
            def aggregate_fit(self, server_round, results, failures):
                print(f"\n==============================================")
                print(f"🌍 GLOBAL ROUND {server_round} / {num_rounds} COMPLETE")
                print(f"==============================================\n")
                agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
                if agg_params is not None and server_round == num_rounds:
                    save_global_model(agg_params, "fedprox_model.pth", num_classes)
                return agg_params, agg_metrics
        return SaveModelFedProx(
            fraction_fit=0.8, 
            fraction_evaluate=1.0, 
            min_fit_clients=4, 
            min_available_clients=num_clients, 
            proximal_mu=mu, 
            evaluate_metrics_aggregation_fn=weighted_average
        )
    else:
        class SaveModelFedAvg(fl.server.strategy.FedAvg):
            def aggregate_fit(self, server_round, results, failures):
                print(f"\n==============================================")
                print(f"🌍 GLOBAL ROUND {server_round} / {num_rounds} COMPLETE")
                print(f"==============================================\n")
                agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
                if agg_params is not None and server_round == num_rounds:
                    save_global_model(agg_params, "fedavg_model.pth", num_classes)
                return agg_params, agg_metrics
        return SaveModelFedAvg(
            fraction_fit=0.8, 
            fraction_evaluate=1.0, 
            min_fit_clients=4, 
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_average
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning CLI")
    parser.add_argument('--algo', type=str, choices=['fedavg', 'fedprox'], default='fedprox')
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--dp', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading partitioned datasets for {args.clients} virtual clients...")
    client_loaders, val_loader, classes = load_federated_data(
        num_clients=args.clients, 
        batch_size=32, 
        iid=False, 
        alpha=0.1
    )
    num_classes = len(classes)

    def client_fn(cid: str) -> fl.client.Client:
        model = get_mobilenet(num_classes=num_classes).to(device)
        train_loader = client_loaders[int(cid)]
        mu_value = args.mu if args.algo == "fedprox" else 0.0
        return PlantClient(model, train_loader, val_loader, device, mu=mu_value, dp_noise=args.dp).to_client()

    print(f"\n🚀 Starting {args.rounds} Rounds using {args.algo.upper()} (DP: {args.dp})...")
    strategy = get_custom_strategy(args.algo, args.rounds, args.clients, num_classes, args.mu)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0},
    )
    
    os.makedirs('results', exist_ok=True)
    metrics_path = f"results/{args.algo}_metrics.json"
    
    losses = [item[1] for item in history.losses_distributed] if history.losses_distributed else []
    accuracies = [item[1] for item in history.metrics_distributed.get("accuracy", [])] if "accuracy" in history.metrics_distributed else []
    
    with open(metrics_path, "w") as f:
        json.dump({"loss": losses, "accuracy": accuracies}, f)
    print(f"📊 Training metrics saved to {metrics_path}")