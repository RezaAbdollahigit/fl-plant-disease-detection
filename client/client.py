import torch
import flwr as fl
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class PlantClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        """Extracts the model's weights to send to the server."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Receives global weights from the server and applies them locally."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Trains the model locally on the edge device."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train for 1 local epoch per federated round
        for images, labels in self.trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            optimizer.step()
            
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluates the global model using local validation data."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total if total > 0 else 0.0
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}