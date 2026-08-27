import torch
import flwr as fl
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from tqdm import tqdm
import random

class PlantClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device, mu=0.0, dp_noise=0.0, local_epochs=5):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.mu = mu          
        self.dp_noise = dp_noise  
        self.local_epochs = local_epochs 
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_weights = [param.clone().detach() for param in self.model.parameters()]
        
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if random.random() < 0.2:
            work_fraction = random.uniform(0.1, 0.4) 
        else:
            work_fraction = 1.0 
            
        max_batches = max(1, int(len(self.trainloader) * work_fraction))
        
        total_train_loss = 0.0 
        total_steps = 0
        
        for epoch in range(self.local_epochs):
            progress_bar = tqdm(self.trainloader, desc=f"Local Epoch {epoch+1}/{self.local_epochs} (Work: {int(work_fraction*100)}%)", leave=False)
            
            for i, (images, labels) in enumerate(progress_bar):
                if i >= max_batches:
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(images)
                
                # 1. Calculate Pure Predictive Error (Cross-Entropy)
                ce_loss = self.criterion(outputs, labels)
                
                # 2. Add the Proximal Penalty for OPTIMIZATION ONLY
                if self.mu > 0.0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_weights):
                        proximal_term += ((local_param - global_param.to(self.device)) ** 2).sum()
                    optimization_loss = ce_loss + (self.mu / 2) * proximal_term
                else:
                    optimization_loss = ce_loss
                
                # 3. Backpropagate the penalized loss (The Engine)
                optimization_loss.backward()

                if self.dp_noise > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.normal(mean=0.0, std=self.dp_noise, size=param.grad.shape).to(self.device)
                            param.grad += noise
                
                optimizer.step()
                
                # 4. Log ONLY the pure Cross-Entropy (The Dashboard)
                total_train_loss += ce_loss.item() 
                total_steps += 1
                progress_bar.set_postfix(loss=f"{ce_loss.item():.4f}")
            
        actual_samples_processed = max_batches * self.trainloader.batch_size * self.local_epochs
        avg_train_loss = total_train_loss / total_steps if total_steps > 0 else 0.0 
        
        return self.get_parameters(config={}), actual_samples_processed, {"train_loss": avg_train_loss}

    def evaluate(self, parameters, config):
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