"""fedge: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from fedge.task import Net, get_weights, load_data, set_weights, test, set_global_seed
from typing import List, Tuple
import os, csv
import numpy as np
import torch
from fedge.task import DATA_FLAGS
# server_app.py — add this import at the top with the others
from flwr.common import parameters_to_ndarrays


# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)
strategy: FedProx  # will be set in server_fn below

# For tracking distributed metrics rounds
dist_round_counter = {"value": 1}
# For tracking fit metrics rounds
fit_round_counter = {"value": 1}
 

# Centralized convergence tracker
class ConvergenceTracker:
    def __init__(self):
        self.prev_loss = None
        self.prev_acc = None
        self.loss_changes = []
        self.acc_changes  = []
    def update(self, round_num, loss, acc) -> dict:
        # Debug convergence inputs
        if np.isnan(loss) or np.isinf(loss):
            print(f"CONVERGENCE ERROR: loss={loss} at round {round_num}")
        if np.isnan(acc) or np.isinf(acc):
            print(f"CONVERGENCE ERROR: acc={acc} at round {round_num}")
            
        if self.prev_loss is None or round_num == 0:
            self.prev_loss, self.prev_acc = loss, acc
            # Return default values for first round
            return {
                "conv_loss_rate": 0.0,
                "conv_acc_rate": 0.0,
                "conv_loss_stability": 0.0,
                "conv_acc_stability": 0.0,
            }
            
        dl = loss - self.prev_loss
        da = acc  - self.prev_acc
        
        # Debug convergence calculations
        if np.isnan(dl) or np.isinf(dl):
            print(f"CONVERGENCE ERROR: dl={dl} (loss={loss}, prev_loss={self.prev_loss})")
        if np.isnan(da) or np.isinf(da):
            print(f"CONVERGENCE ERROR: da={da} (acc={acc}, prev_acc={self.prev_acc})")
            
        self.loss_changes.append(dl)
        self.acc_changes.append(da)
        self.prev_loss, self.prev_acc = loss, acc
        
        # Calculate variance
        loss_var = float(np.var(self.loss_changes)) if len(self.loss_changes) > 1 else 0.0
        acc_var = float(np.var(self.acc_changes)) if len(self.acc_changes) > 1 else 0.0
        
        # Debug variance calculations
        if np.isnan(loss_var) or np.isinf(loss_var):
            print(f"CONVERGENCE ERROR: loss_var={loss_var}, loss_changes={self.loss_changes[-5:]}")
        if np.isnan(acc_var) or np.isinf(acc_var):
            print(f"CONVERGENCE ERROR: acc_var={acc_var}, acc_changes={self.acc_changes[-5:]}")
            
        return {
            "conv_loss_rate":      float(dl),
            "conv_acc_rate":       float(da),
            "conv_loss_stability": loss_var,
            "conv_acc_stability":  acc_var,
        }
ctracker = ConvergenceTracker()


 

def _evaluate_and_log_central_impl(dataset_flag: str, round_num: int, parameters, config, metrics_dir: str = "metrics", seed: int = 0):
    # 1) Load full dataset for centralized eval
    trainloader, testloader, num_classes = load_data(
        dataset_flag,
        partition_id=0,
        num_partitions=1,
        seed=seed,
    )

    # 2) Infer channels & dims
    sample, _ = next(iter(trainloader))
    _, in_ch, H, W = sample.shape

    # 3) Build model & load global params
    net = Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes)

    nds = parameters_to_ndarrays(parameters) if not isinstance(parameters, list) else parameters
    set_weights(net, nds)  # only this one



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 4) Centralized metrics
    train_loss, train_acc = test(net, trainloader, device)
    test_loss,  test_acc  = test(net, testloader, device)
    print(f"Round {round_num}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    rec = {
        "central_train_loss": float(train_loss),
        "central_train_accuracy": float(train_acc),
        "central_test_loss": float(test_loss),
        "central_test_accuracy": float(test_acc),
        "central_loss_gap": float(test_loss - train_loss),
        "central_accuracy_gap": float(train_acc - test_acc),
    }

    conv_metrics = ctracker.update(round_num, test_loss, test_acc)
    rec.update(conv_metrics)

    import os, csv
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
    return test_loss, rec

# Cluster metrics logger
def server_fn(context: Context):
    global strategy
    dataset_flag = context.node_config.get("dataset_flag", "cifar10")

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    min_available_clients = context.run_config.get("min_available_clients", 2)
    
    # FedProx hyperparameters
    proximal_mu = context.run_config.get("proximal_mu", 0.01)
    dirichlet_alpha = context.run_config.get("dirichlet_alpha", 0.3)

    # Seed handling for this run
    seed = int(context.run_config.get("seed", 0))
    set_global_seed(seed)

    # Per-seed metrics directory
    metrics_dir = os.path.join("metrics", f"seed_{seed}")
    os.makedirs(metrics_dir, exist_ok=True)

    trainloader, testloader, num_classes = load_data(dataset_flag, partition_id=0, num_partitions=1, batch_size=1, alpha=dirichlet_alpha, seed=seed)

    sample, _ = next(iter(trainloader))
    if isinstance(sample, torch.Tensor):
        _, in_ch, H, W = sample.shape
    else:
        print("Sample is NOT a tensor! Type:", type(sample))
        print("Sample content:", sample)
        raise ValueError("Sample is not a tensor. Did your transform apply?")

    ndarrays   = get_weights(Net(in_ch=in_ch, img_h=H, img_w=W, n_class=num_classes))
    parameters = ndarrays_to_parameters(ndarrays)

    def eval_fn(round_num, parameters, config):
        # Centralized eval; write to per-seed directory and ensure seeded data
        return _evaluate_and_log_central_impl(dataset_flag, round_num, parameters, config, metrics_dir=metrics_dir, seed=seed)

    # Aggregation callbacks with per-seed CSV routing
    def aggregate_and_log_seeded(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
        """
        Aggregation callback for distributed (per-client) evaluation.
        Logs standard accuracy/loss gap metrics.

        Appends all metrics to f"{metrics_dir}/distributed_metrics.csv".
        """
        global dist_round_counter
        round_num = dist_round_counter["value"]
        # Print distributed evaluation per-client metrics to console
        print(f"\n=== DISTRIBUTED EVAL (Round {round_num}) ===")
        for idx, (_, m) in enumerate(metrics_list):
            print(f"Client {idx+1}: test_loss={m['test_loss']:.4f}, test_acc={m['test_accuracy']:.4f}")

        # 1) Basic distributed metrics (clean version, no worst/best)
        n_total = sum(n for n, _ in metrics_list)
        accs   = [m["test_accuracy"] for _, m in metrics_list]
        losses = [m["test_loss"]     for _, m in metrics_list]

        n_total = sum(n for n, _ in metrics_list)
        avg_acc  = float(sum(n * m["test_accuracy"] for n, m in metrics_list) / max(1, n_total))
        avg_loss = float(sum(n * m["test_loss"]      for n, m in metrics_list) / max(1, n_total))

        acc_sd  = float(np.std(accs))  if accs   else 0.0
        loss_sd = float(np.std(losses)) if losses else 0.0
        n = max(1, len(accs))
        acc_se  = acc_sd  / (n ** 0.5)
        loss_se = loss_sd / (n ** 0.5)

        result = {
            "avg_accuracy":   avg_acc,
            "avg_loss":       avg_loss,
            "accuracy_std":   acc_sd,
            "loss_std":       loss_sd,
            "acc_ci95_lo":    float(avg_acc  - 1.96 * acc_se),
            "acc_ci95_hi":    float(avg_acc  + 1.96 * acc_se),
            "loss_ci95_lo":   float(avg_loss - 1.96 * loss_se),
            "loss_ci95_hi":   float(avg_loss + 1.96 * loss_se),
        }

        # Keep per-client values so you can plot them
        for idx, v in enumerate(accs):
            result[f"client_{idx + 1}_accuracy"] = float(v)
        for idx, v in enumerate(losses):
            result[f"client_{idx + 1}_loss"] = float(v)

        # 2a) Computation & communication cost (always include keys)
        comp_times = [m.get("comp_time_sec", 0.0) for _, m in metrics_list]
        up_bytes   = [m.get("upload_bytes", 0)   for _, m in metrics_list]
        dn_bytes   = [m.get("download_bytes", 0) for _, m in metrics_list]
        
        # Calculate wall clock time (computation + estimated communication time)
        # Assume 10 Mbps network speed for communication time estimation
        network_speed_mbps = 10.0
        comm_times = [(up + dn) / (network_speed_mbps * 1e6 / 8) for up, dn in zip(up_bytes, dn_bytes)]
        wall_clock_times = [comp + comm for comp, comm in zip(comp_times, comm_times)]

        result.update({
            # Computation metrics
            "avg_comp_time_sec":   float(np.mean(comp_times)) if comp_times else 0.0,
            "total_comp_time_sec": float(np.sum(comp_times))  if comp_times else 0.0,
            "std_comp_time_sec":   float(np.std(comp_times)) if comp_times else 0.0,
            
            # Communication volume metrics
            "avg_upload_MB":       (float(np.mean(up_bytes)) / 1e6) if up_bytes else 0.0,
            "total_upload_MB":     (float(np.sum(up_bytes)) / 1e6) if up_bytes else 0.0,
            "avg_download_MB":     (float(np.mean(dn_bytes)) / 1e6) if dn_bytes else 0.0,
            "total_download_MB":   (float(np.sum(dn_bytes)) / 1e6) if dn_bytes else 0.0,
            "total_communication_MB": (float(np.sum(up_bytes) + np.sum(dn_bytes)) / 1e6) if (up_bytes and dn_bytes) else 0.0,
            
            # Wall clock time metrics (computation + communication)
            "avg_wall_clock_sec":   float(np.mean(wall_clock_times)) if wall_clock_times else 0.0,
            "total_wall_clock_sec": float(np.sum(wall_clock_times)) if wall_clock_times else 0.0,
            "std_wall_clock_sec":   float(np.std(wall_clock_times)) if wall_clock_times else 0.0,
            "avg_comm_time_sec":    float(np.mean(comm_times)) if comm_times else 0.0,
            "total_comm_time_sec":  float(np.sum(comm_times)) if comm_times else 0.0,
        })

        # 4) Append to CSV in per-seed directory
        csv_path = os.path.join(metrics_dir, "distributed_metrics.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            fieldnames = ["round"] + list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({"round": round_num, **result})

        # Increment for next round
        dist_round_counter["value"] += 1

        return result

    def aggregate_fit_metrics_seeded(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate and log metrics returned by client `fit`, per-seed directory."""
        global fit_round_counter
        round_num = fit_round_counter["value"]

        comp_times = [m.get("comp_time_sec", 0.0) for _, m in metrics_list]
        up_bytes   = [m.get("upload_bytes", 0) for _, m in metrics_list]
        dn_bytes   = [m.get("download_bytes", 0) for _, m in metrics_list]
        train_losses = [m.get("train_loss_mean", 0.0) for _, m in metrics_list]
        train_accs   = [m.get("train_accuracy_mean", 0.0) for _, m in metrics_list]
        inner_batches = [m.get("num_inner_batches", 0) for _, m in metrics_list]
        train_samples = [m.get("total_train_samples", 0) for _, m in metrics_list]

        result = {
            # Training metrics with dispersion
            "avg_train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "std_train_loss": float(np.std(train_losses)) if train_losses else 0.0,
            "min_train_loss": float(np.min(train_losses)) if train_losses else 0.0,
            "max_train_loss": float(np.max(train_losses)) if train_losses else 0.0,
            
            "avg_train_accuracy": float(np.mean(train_accs)) if train_accs else 0.0,
            "std_train_accuracy": float(np.std(train_accs)) if train_accs else 0.0,
            "min_train_accuracy": float(np.min(train_accs)) if train_accs else 0.0,
            "max_train_accuracy": float(np.max(train_accs)) if train_accs else 0.0,
            
            # Computation metrics
            "avg_comp_time_sec": float(np.mean(comp_times)) if comp_times else 0.0,
            "std_comp_time_sec": float(np.std(comp_times)) if comp_times else 0.0,
            "total_comp_time_sec": float(np.sum(comp_times)) if comp_times else 0.0,
            
            # Communication metrics
            "avg_upload_MB": float(np.mean(up_bytes)) / 1e6 if up_bytes else 0.0,
            "total_upload_MB": float(np.sum(up_bytes)) / 1e6 if up_bytes else 0.0,
            "avg_download_MB": float(np.mean(dn_bytes)) / 1e6 if dn_bytes else 0.0,
            "total_download_MB": float(np.sum(dn_bytes)) / 1e6 if dn_bytes else 0.0,
            
            # pFedMe-specific metrics
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

    strategy = FedProx(
        proximal_mu=proximal_mu,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics_seeded,
        evaluate_metrics_aggregation_fn=aggregate_and_log_seeded,
        evaluate_fn=eval_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
