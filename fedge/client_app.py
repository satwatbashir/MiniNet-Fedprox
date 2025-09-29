# client_app.py
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

from flwr.client import ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters, NDArrays

from fedge.task import Net, load_data, set_weights, test, get_weights, DATA_FLAGS, set_global_seed
import time  # For computation and comms cost tracking
import pickle  # For measuring size of metrics payload


class FlowerClient(NumPyClient):
    def __init__(
        self,
        net: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        local_epochs: int,
        local_lr: float = 1e-2,
        client_id: int = 0,
        base_seed: int = 0,
    ):
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.client_id = client_id
        self.base_seed = base_seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Randomness control for reproducibility per client per run
        set_global_seed(self.base_seed + self.client_id)
        
        # Store global model parameters for proximal term
        self.global_params = None

    def _unwrap_parameters(
        self, parameters: Union[List[np.ndarray], Parameters]
    ) -> List[np.ndarray]:
        """Handle either a raw list of NDArrays or a Parameters object."""
        if isinstance(parameters, list):
            return parameters
        return parameters_to_ndarrays(parameters)

    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """FedProx training with proximal term."""
        # Start cost timer
        t_start = time.perf_counter()
        
        # Unwrap and load global parameters
        global_nd = self._unwrap_parameters(parameters)
        set_weights(self.net, global_nd)
        
        # Store global parameters for proximal term
        self.global_params = [param.clone().detach() for param in self.net.parameters()]
        
        # Get FedProx hyperparameters
        proximal_mu = config.get("proximal_mu", 0.01)
        local_epochs = config.get("local-epochs", self.local_epochs)
        
        # Create optimizer
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.local_lr)
        
        # Training metrics
        all_losses = []
        all_correct = 0
        all_total = 0
        
        # FedProx training
        self.net.train()
        for epoch in range(local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if y.ndim > 1:
                    y = y.squeeze()
                y = y.long()
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.net(x)
                loss = self.criterion(logits, y)
                
                # Add proximal term: μ/2 * ||w - w_global||²
                if proximal_mu > 0:
                    proximal_term = 0.0
                    for param, global_param in zip(self.net.parameters(), self.global_params):
                        proximal_term += torch.sum((param - global_param) ** 2)
                    loss += (proximal_mu / 2.0) * proximal_term
                
                # Backward pass and update
                loss.backward()
                optimizer.step()
                
                # Accumulate stats
                all_losses.append(loss.item())
                _, preds = torch.max(logits, 1)
                all_correct += (preds == y).sum().item()
                all_total += y.size(0)
        
        # Prepare results
        local_nd = get_weights(self.net)
        t_end = time.perf_counter()
        
        download_bytes = int(sum(arr.nbytes for arr in global_nd))
        upload_bytes = int(sum(arr.nbytes for arr in local_nd))
        self.comp_time_sec = t_end - t_start
        self.upload_bytes = upload_bytes
        self.download_bytes = download_bytes
        
        # Compute training metrics
        train_loss_mean = np.mean(all_losses) if all_losses else 0.0
        train_accuracy_mean = all_correct / max(1, all_total) if all_total > 0 else 0.0
        
        fit_metrics = {
            "comp_time_sec": self.comp_time_sec,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "total_train_samples": all_total,
        }
        return local_nd, len(self.trainloader.dataset), fit_metrics

 
    def evaluate(self, parameters, config):
        """Evaluate the received global model on THIS client's own shard only.

        Returns (loss, num_examples, metrics) with metrics keyed as
        'test_loss' and 'test_accuracy' to match the server aggregator.
        """
        t0 = time.perf_counter()

        # Load server-sent weights
        nds = self._unwrap_parameters(parameters)
        set_weights(self.net, nds)
        eval_download_bytes = int(sum(arr.nbytes for arr in nds))

        # Evaluate on client's validation/test shard
        loss, acc = test(self.net, self.valloader, self.device)

        metrics = {
            "cid": int(self.client_id), 
            "test_loss": float(loss),
            "test_accuracy": float(acc),
            "download_bytes": eval_download_bytes,
            "upload_bytes": 0,
            "comp_time_sec": time.perf_counter() - t0,
            "client_id": self.client_id,
            "n_val_examples": len(self.valloader.dataset),
        }
        return float(loss), len(self.valloader.dataset), metrics
    
        
def client_fn(context):
    # 1) Data + model
    dataset_flag = context.node_config.get("dataset_flag", "cifar10")
    base_seed = int(context.run_config.get("seed", 0))

    pid, num_parts = (
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
    )
    trainloader, valloader, n_classes = load_data(
        dataset_flag, pid, num_parts, 
        batch_size=context.run_config.get("batch_size", 32),
        alpha=context.run_config.get("dirichlet_alpha", 0.3),
        seed=base_seed,
    )
    sample, _ = next(iter(trainloader))
    _, c, h, w = sample.shape
    net = Net(in_ch=c, img_h=h, img_w=w, n_class=n_classes)

    # 2) Return NumPyClient
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs=context.run_config["local-epochs"],
        client_id=pid,  # For reproducibility
        base_seed=base_seed,
    ).to_client()

app = ClientApp(client_fn)
