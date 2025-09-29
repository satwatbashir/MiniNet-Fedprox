# client_app.py
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from flwr.client import ClientApp, NumPyClient
from flwr.common import Parameters

from fedge.task import (
    Net,
    load_data,
    set_weights,
    test,
    get_weights,
    set_global_seed,
)
import time
import subprocess
import re


def _read_link_state() -> Dict[str, float]:
    """
    Read current qdisc settings from 'tc qdisc show dev eth0' inside the Mininet host.
    Returns keys (when parse succeeds):
      - link_bw_mbit   : float (Mbit/s from TBF)
      - link_delay_ms  : float (ms from netem)
      - link_loss_pct  : float (% from netem)
    If parsing fails, returns {} (server-side code tolerates missing keys).
    """
    try:
        out = subprocess.check_output(["bash", "-lc", "tc qdisc show dev eth0"], text=True)
    except Exception:
        return {}

    bw_mbit = None
    delay_ms = None
    loss_pct = None

    # TBF example: ... tbf rate 12Mbit burst 32Kb lat 50.0ms ...
    m_rate = re.search(r"\btbf\b.*\brate\s+(\d+(?:\.\d+)?)\s*([kKmMgG]?[bB]it)", out)
    if m_rate:
        val, unit = m_rate.group(1), m_rate.group(2).lower()
        rate = float(val)
        if "kbit" in unit:
            bw_mbit = rate / 1000.0
        elif "mbit" in unit:
            bw_mbit = rate
        elif "gbit" in unit:
            bw_mbit = rate * 1000.0

    # netem example: ... netem delay 120ms loss 3% ...
    m_delay = re.search(r"\bnetem\b.*\bdelay\s+(\d+(?:\.\d+)?)\s*ms", out)
    if m_delay:
        delay_ms = float(m_delay.group(1))

    m_loss = re.search(r"\bnetem\b.*\bloss\s+(\d+(?:\.\d+)?)\s*%", out)
    if m_loss:
        loss_pct = float(m_loss.group(1))

    res: Dict[str, float] = {}
    if bw_mbit is not None:
        res["link_bw_mbit"] = float(bw_mbit)
    if delay_ms is not None:
        res["link_delay_ms"] = float(delay_ms)
    if loss_pct is not None:
        res["link_loss_pct"] = float(loss_pct)
    return res


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
        self.local_epochs = int(local_epochs)
        self.local_lr = float(local_lr)
        self.client_id = int(client_id)
        self.base_seed = int(base_seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Reproducibility per client
        set_global_seed(self.base_seed + self.client_id)

        # Cache of global params for FedProx term
        self.global_params: List[torch.Tensor] = []

    @staticmethod
    def _unwrap_parameters(
        parameters: Union[List[np.ndarray], Parameters]
    ) -> List[np.ndarray]:
        """Accept either a raw list of NDArrays or a Flower Parameters object."""
        from flwr.common import parameters_to_ndarrays  # lazy import

        if isinstance(parameters, list):
            return parameters
        return parameters_to_ndarrays(parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """FedProx local training with proximal term."""
        t_start = time.perf_counter()

        # Load global weights
        global_nd = self._unwrap_parameters(parameters)
        set_weights(self.net, global_nd)

        # Snapshot global params for proximal term (kept constant during local training)
        self.global_params = [p.detach().clone() for p in self.net.parameters()]

        # Hyperparams
        proximal_mu = float(config.get("proximal_mu", 0.01))
        local_epochs = int(config.get("local-epochs", self.local_epochs))

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.local_lr)

        # Training stats
        all_losses: List[float] = []
        correct = 0
        total = 0

        self.net.train()
        for _ in range(local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                # Ensure CE targets are 1D int64 even for batch_size==1
                if y.ndim != 1:
                    y = y.view(-1).long()
                else:
                    y = y.long()

                optimizer.zero_grad()
                logits = self.net(x)
                loss = self.criterion(logits, y)

                # Proximal term: (μ/2) * ||w - w_global||^2
                if proximal_mu > 0.0:
                    prox = 0.0
                    for p, gp in zip(self.net.parameters(), self.global_params):
                        prox = prox + torch.sum((p - gp) ** 2)
                    loss = loss + (proximal_mu / 2.0) * prox

                loss.backward()
                optimizer.step()

                all_losses.append(float(loss.item()))
                _, preds = torch.max(logits, 1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))

        # Collect results
        local_nd = get_weights(self.net)

        t_end = time.perf_counter()
        comp_time_sec = t_end - t_start
        download_bytes = int(sum(arr.nbytes for arr in global_nd))
        upload_bytes = int(sum(arr.nbytes for arr in local_nd))

        train_loss_mean = float(np.mean(all_losses)) if all_losses else 0.0
        train_accuracy_mean = float(correct / total) if total > 0 else 0.0

        # Attach actual link state (bw/delay/loss) from tc
        link_state = _read_link_state()

        fit_metrics: Dict[str, Any] = {
            "comp_time_sec": comp_time_sec,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "total_train_samples": total,
            "client_id": self.client_id,
        }
        fit_metrics.update(link_state)

        return local_nd, len(self.trainloader.dataset), fit_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the received global model on THIS client's own shard only."""
        t0 = time.perf_counter()

        nds = self._unwrap_parameters(parameters)
        set_weights(self.net, nds)
        eval_download_bytes = int(sum(arr.nbytes for arr in nds))

        loss, acc = test(self.net, self.valloader, self.device)

        metrics: Dict[str, Any] = {
            "test_loss": float(loss),
            "test_accuracy": float(acc),
            "download_bytes": eval_download_bytes,
            "upload_bytes": 0,
            "comp_time_sec": time.perf_counter() - t0,
            "client_id": self.client_id,
            "n_val_examples": len(self.valloader.dataset),
        }
        # Attach actual link state here too
        metrics.update(_read_link_state())

        return float(loss), len(self.valloader.dataset), metrics


def client_fn(context):
    # 1) Data + model
    dataset_flag = context.node_config.get("dataset_flag", "cifar10")
    base_seed = int(context.run_config.get("seed", 0))

    pid = int(context.node_config["partition-id"])
    num_parts = int(context.node_config["num-partitions"])

    trainloader, valloader, n_classes = load_data(
        dataset_flag,
        pid,
        num_parts,
        batch_size=int(context.run_config.get("batch_size", 32)),
        alpha=float(context.run_config.get("dirichlet_alpha", 0.5)),
        seed=base_seed,
    )

    # Infer input shape from a sample batch
    sample, _ = next(iter(trainloader))
    _, c, h, w = sample.shape
    net = Net(in_ch=int(c), img_h=int(h), img_w=int(w), n_class=int(n_classes))

    # 2) Return NumPyClient
    return FlowerClient(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=int(context.run_config["local-epochs"]),
        local_lr=float(context.run_config.get("local_lr", 1e-2)),
        client_id=pid,
        base_seed=base_seed,
    ).to_client()


app = ClientApp(client_fn)
