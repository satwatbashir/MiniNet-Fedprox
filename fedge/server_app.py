"""fedge: A Flower / PyTorch app."""
from typing import List, Tuple

import csv
import os
import time
import numpy as np
import torch
import json
import subprocess
import datetime
from pathlib import Path

from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx, Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import Scalar

from fedge.task import Net, get_weights, load_data, set_weights, test, set_global_seed

# Ensure top-level metrics dir exists (per-seed subdirs will be created later)
os.makedirs("metrics", exist_ok=True)

# Global counters for CSV "round" columns (Flower's agg hooks don't give the round)
dist_round_counter = {"value": 1}
fit_round_counter = {"value": 1}

# Global to store actual round timing from strategy (measured server-side)
actual_round_timing = {"round": 0, "time_sec": 0.0}


# ---- Convergence tracker for centralized eval ----
class ConvergenceTracker:
    def __init__(self):
        self.prev_loss = None
        self.prev_acc = None
        self.loss_changes = []
        self.acc_changes = []

    def update(self, round_num: int, loss: float, acc: float) -> dict:
        if np.isnan(loss) or np.isinf(loss):
            print(f"CONVERGENCE ERROR: loss={loss} at round {round_num}")
        if np.isnan(acc) or np.isinf(acc):
            print(f"CONVERGENCE ERROR: acc={acc} at round {round_num}")

        if self.prev_loss is None or round_num == 0:
            self.prev_loss, self.prev_acc = loss, acc
            return {
                "conv_loss_rate": 0.0,
                "conv_acc_rate": 0.0,
                "conv_loss_stability": 0.0,
                "conv_acc_stability": 0.0,
            }

        dl = float(loss - self.prev_loss)
        da = float(acc - self.prev_acc)

        if np.isnan(dl) or np.isinf(dl):
            print(f"CONVERGENCE ERROR: dl={dl} (loss={loss}, prev_loss={self.prev_loss})")
        if np.isnan(da) or np.isinf(da):
            print(f"CONVERGENCE ERROR: da={da} (acc={acc}, prev_acc={self.prev_acc})")

        self.loss_changes.append(dl)
        self.acc_changes.append(da)
        self.prev_loss, self.prev_acc = loss, acc

        loss_var = float(np.var(self.loss_changes)) if len(self.loss_changes) > 1 else 0.0
        acc_var = float(np.var(self.acc_changes)) if len(self.acc_changes) > 1 else 0.0

        if np.isnan(loss_var) or np.isinf(loss_var):
            print(f"CONVERGENCE ERROR: loss_var={loss_var}, loss_changes={self.loss_changes[-5:]}")
        if np.isnan(acc_var) or np.isinf(acc_var):
            print(f"CONVERGENCE ERROR: acc_var={acc_var}, acc_changes={self.acc_changes[-5:]}")

        return {
            "conv_loss_rate": dl,
            "conv_acc_rate": da,
            "conv_loss_stability": loss_var,
            "conv_acc_stability": acc_var,
        }


ctracker = ConvergenceTracker()


def _evaluate_and_log_central_impl(
    dataset_flag: str,
    round_num: int,
    parameters,
    config,
    metrics_dir: str = "metrics",
    seed: int = 0,
):
    # Full centralized eval on (train, test)
    trainloader, testloader, num_classes = load_data(
        dataset_flag, partition_id=0, num_partitions=1, seed=seed
    )

    sample, _ = next(iter(trainloader))
    if not isinstance(sample, torch.Tensor):
        raise ValueError(f"Sample is not a tensor (got {type(sample)}). Check transforms.")
    _, in_ch, H, W = sample.shape

    net = Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes)
    nds = parameters_to_ndarrays(parameters) if not isinstance(parameters, list) else parameters
    set_weights(net, nds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    train_loss, train_acc = test(net, trainloader, device)
    test_loss, test_acc = test(net, testloader, device)

    print(
        f"Round {round_num}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
    )

    rec = {
        "central_train_loss": float(train_loss),
        "central_train_accuracy": float(train_acc),
        "central_test_loss": float(test_loss),
        "central_test_accuracy": float(test_acc),
        "central_loss_gap": float(test_loss - train_loss),
        "central_accuracy_gap": float(train_acc - test_acc),
    }
    rec.update(ctracker.update(round_num, float(test_loss), float(test_acc)))

    # Write CSV
    fieldnames = [
        "round",
        "central_train_loss",
        "central_train_accuracy",
        "central_test_loss",
        "central_test_accuracy",
        "central_loss_gap",
        "central_accuracy_gap",
        "conv_loss_rate",
        "conv_acc_rate",
        "conv_loss_stability",
        "conv_acc_stability",
    ]
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, "centralized_metrics.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({"round": round_num, **rec})

    # Flower expects (loss, metrics)
    return float(test_loss), rec


# ---- Round ticker wrapper for round-driven jitter ----
class RoundTickerStrategy(Strategy):
    """Proxy strategy that writes the current server round to a tick file, then delegates."""

    def __init__(self, inner: Strategy, seed: int):
        self.inner = inner
        self.seed = seed
        self.tick_dir = Path(f"metrics/seed_{self.seed}")
        self.tick_dir.mkdir(parents=True, exist_ok=True)
        self.tick_file = self.tick_dir / "round_tick.txt"
        # Track actual round timing (server-side wall clock)
        self.round_start_times = {}

    # delegate to inner, with a hook in configure_fit
    def initialize_parameters(self, client_manager: ClientManager):
        return self.inner.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, dict]]:
        # Record round start time (before any network transfer)
        self.round_start_times[server_round] = time.perf_counter()

        try:
            # write round number BEFORE clients are configured
            self.tick_file.write_text(f"{server_round}\n", encoding="utf-8")
        except Exception as e:
            print(f"[RoundTicker] Failed to write round tick: {e}")
        return self.inner.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results, failures):
        global actual_round_timing
        # Calculate actual round time (after all network transfers complete)
        round_end_time = time.perf_counter()
        round_start_time = self.round_start_times.get(server_round, round_end_time)
        actual_time = round_end_time - round_start_time

        # Store for aggregation functions to use
        actual_round_timing = {"round": server_round, "time_sec": actual_time}

        return self.inner.aggregate_fit(server_round, results, failures)

    def configure_evaluate(self, server_round: int, parameters, client_manager: ClientManager):
        return self.inner.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round: int, results, failures):
        return self.inner.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters):
        return self.inner.evaluate(server_round, parameters)


def server_fn(context: Context):
    dataset_flag = context.node_config.get("dataset_flag", "cifar10")

    # Run config (respect your invariants)
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = float(context.run_config["fraction-fit"])
    min_available_clients = int(context.run_config.get("min_available_clients", 2))
    proximal_mu = float(context.run_config.get("proximal_mu", 0.01))
    dirichlet_alpha = float(context.run_config.get("dirichlet_alpha", 0.5))  # default = 0.5

    # Scenario-aware settings used for estimates/diagnostics
    assumed_mbps = float(context.run_config.get("assumed_mbps", 10.0))
    round_timeout_sec = float(context.run_config.get("round_timeout_sec", 300.0))

    # Seed
    seed = int(context.run_config.get("seed", 0))
    set_global_seed(seed)

    # Per-seed metrics dir (+ ensure tick file exists with 0)
    metrics_dir = os.path.join("metrics", f"seed_{seed}")
    os.makedirs(metrics_dir, exist_ok=True)
    Path(metrics_dir, "round_tick.txt").write_text("0\n", encoding="utf-8")

    # --- Manifest for reproducibility ---
    manifest = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "scenario": os.getenv("SCENARIO", "(unset)"),
        "node_config": context.node_config,
        "run_config": context.run_config,
        "env": {
            "HOSTNAME": os.getenv("HOSTNAME", ""),
            "PYTORCH_CUDA_AVAILABLE": torch.cuda.is_available(),
            "TORCH_VERSION": torch.__version__,
            "PY_VERSION": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        },
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode().strip()
        manifest["git_commit"] = commit
    except Exception:
        manifest["git_commit"] = "(unknown)"
    with open(os.path.join(metrics_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Build initial parameters from a shape-consistent Net
    trainloader, _, num_classes = load_data(
        dataset_flag, partition_id=0, num_partitions=1, batch_size=1, alpha=dirichlet_alpha, seed=seed
    )
    sample, _ = next(iter(trainloader))
    if not isinstance(sample, torch.Tensor):
        raise ValueError(f"Sample is not a tensor (got {type(sample)}). Check transforms.")
    _, in_ch, H, W = sample.shape
    initial_nd = get_weights(Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes))
    initial_params = ndarrays_to_parameters(initial_nd)

    # Centralized evaluation callback
    def eval_fn(round_num, parameters, config):
        return _evaluate_and_log_central_impl(
            dataset_flag, round_num, parameters, config, metrics_dir=metrics_dir, seed=seed
        )

    # Distributed evaluation aggregation/logging
    def aggregate_and_log_seeded(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
        """
        metrics_list: List of (num_examples, metrics) from clients' evaluate().
        Writes per-round aggregates to metrics/seed_{seed}/distributed_metrics.csv
        """
        global dist_round_counter
        round_num = dist_round_counter["value"]

        if not metrics_list:
            print(f"WARNING: No evaluation metrics received in round {round_num}")
            return {}

        print(f"\n=== DISTRIBUTED EVAL (Round {round_num}) ===")
        for idx, (_, m) in enumerate(metrics_list):
            print(
                f"Client {idx+1}: test_loss={m.get('test_loss', float('nan')):.4f}, "
                f"test_acc={m.get('test_accuracy', float('nan')):.4f}"
            )

        # --- Per-client EVAL CSV ---
        per_client_eval = os.path.join(metrics_dir, "per_client_eval.csv")
        pc_eval_header_needed = not os.path.exists(per_client_eval)
        with open(per_client_eval, "a", newline="") as pf:
            _fieldnames = [
                "round",
                "client_id",
                "num_examples",
                "test_loss",
                "test_accuracy",
                "comp_time_sec",
                "upload_bytes",
                "download_bytes",
                "assumed_mbps",
            ]
            w = csv.DictWriter(pf, fieldnames=_fieldnames)
            if pc_eval_header_needed:
                w.writeheader()
            for n, m in metrics_list:
                w.writerow(
                    {
                        "round": round_num,
                        "client_id": m.get("client_id"),
                        "num_examples": int(n),
                        "test_loss": float(m.get("test_loss", 0.0)),
                        "test_accuracy": float(m.get("test_accuracy", 0.0)),
                        "comp_time_sec": float(m.get("comp_time_sec", 0.0)),
                        "upload_bytes": int(m.get("upload_bytes", 0)),
                        "download_bytes": int(m.get("download_bytes", 0)),
                        "assumed_mbps": float(assumed_mbps),
                    }
                )

        # Aggregates
        n_total = sum(n for n, _ in metrics_list)
        accs = [float(m.get("test_accuracy", 0.0)) for _, m in metrics_list]
        losses = [float(m.get("test_loss", 0.0)) for _, m in metrics_list]

        avg_acc = float(sum(n * m.get("test_accuracy", 0.0) for n, m in metrics_list) / max(1, n_total))
        avg_loss = float(sum(n * m.get("test_loss", 0.0) for n, m in metrics_list) / max(1, n_total))

        acc_sd = float(np.std(accs)) if accs else 0.0
        loss_sd = float(np.std(losses)) if losses else 0.0
        n = max(1, len(accs))
        acc_se = acc_sd / (n ** 0.5)
        loss_se = loss_sd / (n ** 0.5)

        result = {
            "avg_accuracy": avg_acc,
            "avg_loss": avg_loss,
            "accuracy_std": acc_sd,
            "loss_std": loss_sd,
            "acc_ci95_lo": float(avg_acc - 1.96 * acc_se),
            "acc_ci95_hi": float(avg_acc + 1.96 * acc_se),
            "loss_ci95_lo": float(avg_loss - 1.96 * loss_se),
            "loss_ci95_hi": float(avg_loss + 1.96 * loss_se),
        }

        # Per-client values (accuracy/loss) for plotting
        for idx, v in enumerate(accs):
            result[f"client_{idx + 1}_accuracy"] = float(v)
        for idx, v in enumerate(losses):
            result[f"client_{idx + 1}_loss"] = float(v)

        # Efficiency/comm metrics (derived from client metrics)
        comp_times = [float(m.get("comp_time_sec", 0.0)) for _, m in metrics_list]
        up_bytes = [int(m.get("upload_bytes", 0)) for _, m in metrics_list]
        dn_bytes = [int(m.get("download_bytes", 0)) for _, m in metrics_list]

        # Estimated comm time (s) = bytes * 8 / (Mb/s * 1e6) - kept for comparison
        est_comm_times = [((up + dn) * 8.0) / (assumed_mbps * 1e6) for up, dn in zip(up_bytes, dn_bytes)]

        # Actual round time from server-side measurement
        global actual_round_timing
        actual_round_sec = actual_round_timing.get("time_sec", 0.0) if actual_round_timing.get("round") == round_num else 0.0
        max_comp_time = float(np.max(comp_times)) if comp_times else 0.0
        # Actual comm = round_time - slowest client's compute (parallel execution)
        actual_comm_sec = max(0.0, actual_round_sec - max_comp_time)

        result.update(
            {
                # Computation time
                "avg_comp_sec": float(np.mean(comp_times)) if comp_times else 0.0,
                "max_comp_sec": max_comp_time,
                "total_comp_sec": float(np.sum(comp_times)) if comp_times else 0.0,
                "std_comp_sec": float(np.std(comp_times)) if comp_times else 0.0,
                # Data transfer
                "avg_upload_MB": (float(np.mean(up_bytes)) / 1e6) if up_bytes else 0.0,
                "total_upload_MB": (float(np.sum(up_bytes)) / 1e6) if up_bytes else 0.0,
                "avg_download_MB": (float(np.mean(dn_bytes)) / 1e6) if dn_bytes else 0.0,
                "total_download_MB": (float(np.sum(dn_bytes)) / 1e6) if dn_bytes else 0.0,
                "total_comm_MB": (float(np.sum(up_bytes) + np.sum(dn_bytes)) / 1e6)
                if (up_bytes or dn_bytes)
                else 0.0,
                # Estimated communication (for comparison)
                "est_comm_sec": float(np.sum(est_comm_times)) if est_comm_times else 0.0,
                # Actual timing (measured server-side)
                "actual_round_sec": actual_round_sec,
                "actual_comm_sec": actual_comm_sec,
            }
        )

        csv_path = os.path.join(metrics_dir, "distributed_metrics.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            fieldnames = ["round"] + list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({"round": round_num, **result})

        dist_round_counter["value"] += 1
        return result

    # Fit metrics aggregation/logging
    def aggregate_fit_metrics_seeded(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
        """
        metrics_list: List of (num_examples, metrics) from clients' fit().
        Writes per-round aggregates to metrics/seed_{seed}/fit_metrics.csv
        """
        global fit_round_counter
        round_num = fit_round_counter["value"]

        if not metrics_list:
            print(f"WARNING: No fit metrics received in round {round_num}")
            return {}

        comp_times = [float(m.get("comp_time_sec", 0.0)) for _, m in metrics_list]
        up_bytes = [int(m.get("upload_bytes", 0)) for _, m in metrics_list]
        dn_bytes = [int(m.get("download_bytes", 0)) for _, m in metrics_list]
        train_losses = [float(m.get("train_loss_mean", 0.0)) for _, m in metrics_list]
        train_accs = [float(m.get("train_accuracy_mean", 0.0)) for _, m in metrics_list]
        inner_batches = [int(m.get("num_inner_batches", 0)) for _, m in metrics_list]  # schema stability
        train_samples = [int(m.get("total_train_samples", 0)) for _, m in metrics_list]

        # --- Per-client FIT CSV ---
        per_client_fit = os.path.join(metrics_dir, "per_client_fit.csv")
        pc_fit_header_needed = not os.path.exists(per_client_fit)
        with open(per_client_fit, "a", newline="") as pf:
            _fieldnames = [
                "round",
                "client_id",
                "train_loss_mean",
                "train_accuracy_mean",
                "total_train_samples",
                "comp_time_sec",
                "upload_bytes",
                "download_bytes",
                "assumed_mbps",
            ]
            w = csv.DictWriter(pf, fieldnames=_fieldnames)
            if pc_fit_header_needed:
                w.writeheader()
            for _, m in metrics_list:
                w.writerow(
                    {
                        "round": round_num,
                        "client_id": m.get("client_id"),
                        "train_loss_mean": float(m.get("train_loss_mean", 0.0)),
                        "train_accuracy_mean": float(m.get("train_accuracy_mean", 0.0)),
                        "total_train_samples": int(m.get("total_train_samples", 0)),
                        "comp_time_sec": float(m.get("comp_time_sec", 0.0)),
                        "upload_bytes": int(m.get("upload_bytes", 0)),
                        "download_bytes": int(m.get("download_bytes", 0)),
                        "assumed_mbps": float(assumed_mbps),
                    }
                )

        # Estimated comm time (for comparison)
        est_comm_times = [((up + dn) * 8.0) / (assumed_mbps * 1e6) for up, dn in zip(up_bytes, dn_bytes)]

        # Actual round time from server-side measurement
        global actual_round_timing
        actual_round_sec = actual_round_timing.get("time_sec", 0.0) if actual_round_timing.get("round") == round_num else 0.0
        max_comp_time = float(np.max(comp_times)) if comp_times else 0.0
        # Actual comm = round_time - slowest client's compute (parallel execution)
        actual_comm_sec = max(0.0, actual_round_sec - max_comp_time)

        result = {
            # Training metrics
            "avg_train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "std_train_loss": float(np.std(train_losses)) if train_losses else 0.0,
            "min_train_loss": float(np.min(train_losses)) if train_losses else 0.0,
            "max_train_loss": float(np.max(train_losses)) if train_losses else 0.0,
            "avg_train_acc": float(np.mean(train_accs)) if train_accs else 0.0,
            "std_train_acc": float(np.std(train_accs)) if train_accs else 0.0,
            "min_train_acc": float(np.min(train_accs)) if train_accs else 0.0,
            "max_train_acc": float(np.max(train_accs)) if train_accs else 0.0,
            # Computation time
            "avg_comp_sec": float(np.mean(comp_times)) if comp_times else 0.0,
            "std_comp_sec": float(np.std(comp_times)) if comp_times else 0.0,
            "max_comp_sec": max_comp_time,
            "total_comp_sec": float(np.sum(comp_times)) if comp_times else 0.0,
            # Data transfer
            "avg_upload_MB": float(np.mean(up_bytes)) / 1e6 if up_bytes else 0.0,
            "total_upload_MB": float(np.sum(up_bytes)) / 1e6 if up_bytes else 0.0,
            "avg_download_MB": float(np.mean(dn_bytes)) / 1e6 if dn_bytes else 0.0,
            "total_download_MB": float(np.sum(dn_bytes)) / 1e6 if dn_bytes else 0.0,
            "total_comm_MB": float(np.sum(up_bytes) + np.sum(dn_bytes)) / 1e6 if (up_bytes or dn_bytes) else 0.0,
            # Estimated communication (for comparison)
            "est_comm_sec": float(np.sum(est_comm_times)) if est_comm_times else 0.0,
            # Actual timing (measured server-side)
            "actual_round_sec": actual_round_sec,
            "actual_comm_sec": actual_comm_sec,
            # Other
            "avg_inner_batches": float(np.mean(inner_batches)) if inner_batches else 0.0,
            "total_train_samples": float(np.sum(train_samples)) if train_samples else 0.0,
        }

        path = os.path.join(metrics_dir, "fit_metrics.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            fieldnames = ["round"] + list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({"round": round_num, **result})

        fit_round_counter["value"] += 1
        return result

    # Base strategy
    base_strategy = FedProx(
        proximal_mu=proximal_mu,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=aggregate_fit_metrics_seeded,
        evaluate_metrics_aggregation_fn=aggregate_and_log_seeded,
        evaluate_fn=eval_fn,
    )

    # Wrap with round ticker
    strategy = RoundTickerStrategy(inner=base_strategy, seed=seed)

    config = ServerConfig(num_rounds=num_rounds, round_timeout=round_timeout_sec)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
