"""Generate summary plots for Federated Learning metrics.

This script reads CSVs from the `metrics` directory and produces a single
PNG file containing four sub-plots covering:
1. Accuracy curves
2. Loss curves
3. Communication (upload / download) per round
4. Wall-clock computation time per round

The resulting figure is saved in a `plots` directory which is created if it
does not already exist.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _read_csv(name: str) -> pd.DataFrame:
    """Read CSV `name` from the metrics directory if it exists, else empty df."""
    path = METRICS_DIR / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover – defensive guard
            print(f"Failed to read {path}: {exc}")
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

distributed_df = _read_csv("distributed_metrics.csv")
centralized_df = _read_csv("centralized_metrics.csv")
fit_df = _read_csv("fit_metrics.csv")

# Round index (distributed metrics only)
round_series = distributed_df["round"]

# ---------------------------------------------------------------------------
# Tick interval settings (customize)
# ---------------------------------------------------------------------------
X_TICK_INTERVAL = 10  # x-axis ticks interval (round)
Y_LOSS_MIN = 0.5      # y-axis minimum for loss plots
Y_LOSS_MAX = 4.0      # y-axis maximum for loss plots
Y_LOSS_TICKS = 5      # number of y-axis ticks for loss plots

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("FedProx FL Performance - Individual Client Metrics", fontsize=16)

# --------------------------------- Accuracy ---------------------------------
ax_acc = axes[0, 0]
# Plot individual client accuracies (main focus)
client_cols = [c for c in distributed_df.columns if c.startswith("client_") and c.endswith("_accuracy")]
colors = plt.cm.tab10(range(len(client_cols)))  # Use distinct colors
for i, col in enumerate(client_cols):
    ax_acc.plot(
        distributed_df["round"],
        distributed_df[col],
        color=colors[i],
        linewidth=2,
        label=col.replace("_accuracy", "").replace("_", " ").title(),
    )

ax_acc.set_title("Individual Client  Accuracy vs. Round")
ax_acc.set_xlabel("Round")
ax_acc.set_ylabel("Accuracy")
ax_acc.grid(True)
ax_acc.legend()

# ----------------------------------- Loss -----------------------------------
ax_loss = axes[0, 1]
# Plot individual client test losses (main focus)
client_loss_cols = [c for c in distributed_df.columns if c.startswith("client_") and c.endswith("_loss")]
colors = plt.cm.tab10(range(len(client_loss_cols)))  # Use distinct colors
for i, col in enumerate(client_loss_cols):
    ax_loss.plot(
        distributed_df["round"],
        distributed_df[col],
        color=colors[i],
        linewidth=2,
        label=col.replace("_loss", "").replace("_", " ").title(),
    )
ax_loss.set_title("Individual Client Loss vs. Round")
ax_loss.set_xlabel("Round")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)
ax_loss.legend()

# ------------------------------- Communication ------------------------------
ax_comm = axes[1, 0]
# Communication metrics (distributed only)
if "total_upload_MB" in distributed_df.columns:
    ax_comm.plot(distributed_df["round"], distributed_df["total_upload_MB"], label="Total Upload (MB)")
if "total_download_MB" in distributed_df.columns:
    ax_comm.plot(distributed_df["round"], distributed_df["total_download_MB"], label="Total Download (MB)")

ax_comm.set_title("Communication per Round")
ax_comm.set_xlabel("Round")
ax_comm.set_ylabel("Communication Bytes (MB)")
ax_comm.grid(True)
ax_comm.legend()

# ------------------------------ Wall-clock Time ------------------------------
ax_time = axes[1, 1]
if "total_comp_time_sec" in distributed_df.columns:
    ax_time.plot(distributed_df["round"], distributed_df["total_comp_time_sec"], label="Total Comp Time (s)")
    ax_time.set_ylabel("Seconds")

ax_time.set_title("Computation Time per Round")
ax_time.set_xlabel("Round")
ax_time.grid(True)
ax_time.legend()

# ---------------------------------------------------------------------------
# Finalise & save first figure
# ---------------------------------------------------------------------------
# Custom ticks for first figure
max_round = round_series.max()
xticks = np.arange(0, max_round+1, X_TICK_INTERVAL)
for a in [ax_acc, ax_loss, ax_comm, ax_time]:
    a.set_xticks(xticks)
    a.set_xlim(0, max_round + X_TICK_INTERVAL)
# Accuracy subplot y-axis (0 to 1)
ax_acc.set_ylim(0, 1)
ax_acc.set_yticks(np.arange(0, 1.01, 0.2))
# Communication subplot y-axis (-1 to 2 MB)
ax_comm.set_ylim(-1, 2)
ax_comm.set_yticks(np.arange(-0.5, 2.01, 0.5))

# Loss subplot y-axis (0 to 4)
ax_loss.set_ylim(0, 4)
ax_loss.set_yticks(np.arange(0, 4.1, 0.5))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
output_path = PLOTS_DIR / "metrics_summary.png"
fig.savefig(output_path, dpi=150)
print(f"Saved plots to {output_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Figure 2 – per-client metrics for clients 1-5 & 6-10
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Individual Client Metrics – Clients 1-5 & 6-10", fontsize=16)

max_round = round_series.max()
xticks = np.arange(0, max_round+1, X_TICK_INTERVAL)

# Accuracy (Clients 1-5)
ax_acc15 = axes2[0, 0]
for i in range(1, 6):
    col = f"client_{i}_accuracy"
    ax_acc15.plot(distributed_df["round"], distributed_df[col], linewidth=2, label=f"Client {i}")
ax_acc15.set_title("Client 1-5 Accuracy vs. Round")
ax_acc15.set_xlabel("Round")
ax_acc15.set_ylabel("Accuracy")
ax_acc15.set_xticks(xticks); ax_acc15.set_xlim(0, max_round + X_TICK_INTERVAL)
ax_acc15.set_ylim(0, 1); ax_acc15.set_yticks(np.arange(0, 1.01, 0.2))
ax_acc15.grid(True); ax_acc15.legend()

# Accuracy (Clients 6-10)
ax_acc610 = axes2[0, 1]
for i in range(6, 11):
    col = f"client_{i}_accuracy"
    ax_acc610.plot(distributed_df["round"], distributed_df[col], linewidth=2, label=f"Client {i}")
ax_acc610.set_title("Client 6-10 Accuracy vs. Round")
ax_acc610.set_xlabel("Round")
ax_acc610.set_ylabel("Accuracy")
ax_acc610.set_xticks(xticks); ax_acc610.set_xlim(0, max_round + X_TICK_INTERVAL)
ax_acc610.set_ylim(0, 1); ax_acc610.set_yticks(np.arange(0, 1.01, 0.2))
ax_acc610.grid(True); ax_acc610.legend()

# Loss (Clients 1-5)
ax_loss15 = axes2[1, 0]
for i in range(1, 6):
    col = f"client_{i}_loss"
    ax_loss15.plot(distributed_df["round"], distributed_df[col], linewidth=2, label=f"Client {i}")
ax_loss15.set_title("Client 1-5 Loss vs. Round")
ax_loss15.set_xlabel("Round")
ax_loss15.set_ylabel("Loss")
ax_loss15.set_xticks(xticks); ax_loss15.set_xlim(0, max_round + X_TICK_INTERVAL)
ax_loss15.set_ylim(0, 4); ax_loss15.set_yticks(np.arange(0, 4.1, 0.5))
ax_loss15.grid(True); ax_loss15.legend()

# Loss (Clients 6-10)
ax_loss610 = axes2[1, 1]
for i in range(6, 11):
    col = f"client_{i}_loss"
    ax_loss610.plot(distributed_df["round"], distributed_df[col], linewidth=2, label=f"Client {i}")
ax_loss610.set_title("Client 6-10 Loss vs. Round")
ax_loss610.set_xlabel("Round")
ax_loss610.set_ylabel("Loss")
ax_loss610.set_xticks(xticks); ax_loss610.set_xlim(0, max_round + X_TICK_INTERVAL)
ax_loss610.set_ylim(0, 4); ax_loss610.set_yticks(np.arange(0, 4.1, 0.5))
ax_loss610.grid(True); ax_loss610.legend()



# Finalise figure 2
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path2 = PLOTS_DIR / "metrics_summary_style2.png"
fig2.savefig(output_path2, dpi=150)
print(f"Saved additional styled plot to {output_path2.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Figure 3 	6 Server Metrics (Centralized)
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle("FedProx Server Metrics", fontsize=16)

# Server Accuracy (top-left)
ax_srv_acc = axes3[0, 0]
if not centralized_df.empty:
    ax_srv_acc.plot(centralized_df["round"], centralized_df["central_test_accuracy"], linewidth=2, label="Global Model Accuracy")
ax_srv_acc.set_title("Server Accuracy vs. Round")
ax_srv_acc.set_xlabel("Round")
ax_srv_acc.set_ylabel("Accuracy")
ax_srv_acc.grid(True)
ax_srv_acc.legend()

# Server Loss (top-right)
ax_srv_loss = axes3[0, 1]
if not centralized_df.empty:
    ax_srv_loss.plot(centralized_df["round"], centralized_df["central_test_loss"], linewidth=2, label="Global Model Loss")
ax_srv_loss.set_title("Server Loss vs. Round")
ax_srv_loss.set_xlabel("Round")
ax_srv_loss.set_ylabel("Loss")
ax_srv_loss.grid(True)
ax_srv_loss.legend()

# Average Client Accuracy (bottom-left)
ax_avg_acc = axes3[1, 0]
client_acc_cols = [c for c in distributed_df.columns if c.startswith("client_") and c.endswith("_accuracy")]
if client_acc_cols and not distributed_df.empty:
    avg_client_acc = distributed_df[client_acc_cols].mean(axis=1)
    ax_avg_acc.plot(distributed_df["round"], avg_client_acc, label="Average Client Accuracy", color='green', linewidth=2)
ax_avg_acc.set_title("Average Client Accuracy vs. Round")
ax_avg_acc.set_xlabel("Round")
ax_avg_acc.set_ylabel("Accuracy")
ax_avg_acc.grid(True)
ax_avg_acc.legend()

# Average Client Loss (bottom-right)
ax_avg_loss = axes3[1, 1]
client_loss_cols = [c for c in distributed_df.columns if c.startswith("client_") and c.endswith("_loss")]
if client_loss_cols and not distributed_df.empty:
    avg_client_loss = distributed_df[client_loss_cols].mean(axis=1)
    ax_avg_loss.plot(distributed_df["round"], avg_client_loss, label="Average Client Loss", color='orange', linewidth=2)
ax_avg_loss.set_title("Average Client Loss vs. Round")
ax_avg_loss.set_xlabel("Round")
ax_avg_loss.set_ylabel("Loss")
ax_avg_loss.grid(True)
ax_avg_loss.legend()

# Adjust ticks and limits for server metrics figure
max_round = round_series.max()
xticks = np.arange(0, max_round+1, X_TICK_INTERVAL)
for a in [ax_srv_acc, ax_srv_loss, ax_avg_acc, ax_avg_loss]:
    a.set_xticks(xticks)
    a.set_xlim(0, max_round)
# Accuracy plots y-axis
ax_srv_acc.set_ylim(0, 1)
ax_srv_acc.set_yticks(np.arange(0, 1.01, 0.2))
ax_avg_acc.set_ylim(0, 1)
ax_avg_acc.set_yticks(np.arange(0, 1.01, 0.2))
# Loss plots y-axis
ax_srv_loss.set_ylim(0, 4)
ax_srv_loss.set_yticks(np.arange(0, 4.1, 0.5))
ax_avg_loss.set_ylim(0, 4)
ax_avg_loss.set_yticks(np.arange(0, 4.1, 0.5))

# Finalise & save server metrics figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path3 = PLOTS_DIR / "server_metrics_summary.png"
fig3.savefig(output_path3, dpi=150)
print(f"Saved server metrics plot to {output_path3.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Figure 4 - Style 3: Server Metrics (top) + Communication/Computation (bottom)
# ---------------------------------------------------------------------------
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle("FedProx FL Performance - Server Metrics & Communication", fontsize=16)

# Server Test Accuracy (top-left)
ax_srv_test_acc = axes4[0, 0]
if not centralized_df.empty and "central_test_accuracy" in centralized_df.columns:
    ax_srv_test_acc.plot(centralized_df["round"], centralized_df["central_test_accuracy"], 
                        linewidth=2, label="Test Accuracy", color='blue')
if not centralized_df.empty and "central_train_accuracy" in centralized_df.columns:
    ax_srv_test_acc.plot(centralized_df["round"], centralized_df["central_train_accuracy"], 
                        linewidth=2, label="Train Accuracy", color='red')
# Add average client accuracy
client_acc_cols = [c for c in distributed_df.columns if c.startswith("client_") and c.endswith("_accuracy")]
if client_acc_cols and not distributed_df.empty:
    avg_client_acc = distributed_df[client_acc_cols].mean(axis=1)
    ax_srv_test_acc.plot(distributed_df["round"], avg_client_acc, 
                        linewidth=2, label="Average Client Accuracy", color='green')
ax_srv_test_acc.set_title("Server/Global Model & Average Client Accuracy")
ax_srv_test_acc.set_xlabel("Round")
ax_srv_test_acc.set_ylabel("Accuracy")
ax_srv_test_acc.grid(True)
ax_srv_test_acc.legend()

# Server Test Loss (top-right)
ax_srv_test_loss = axes4[0, 1]
if not centralized_df.empty and "central_test_loss" in centralized_df.columns:
    ax_srv_test_loss.plot(centralized_df["round"], centralized_df["central_test_loss"], 
                         linewidth=2, label="Test Loss", color='blue')
if not centralized_df.empty and "central_train_loss" in centralized_df.columns:
    ax_srv_test_loss.plot(centralized_df["round"], centralized_df["central_train_loss"], 
                         linewidth=2, label="Train Loss", color='red')
# Add average client loss
client_loss_cols = [c for c in distributed_df.columns if c.startswith("client_") and c.endswith("_loss")]
if client_loss_cols and not distributed_df.empty:
    avg_client_loss = distributed_df[client_loss_cols].mean(axis=1)
    ax_srv_test_loss.plot(distributed_df["round"], avg_client_loss, 
                         linewidth=2, label="Average Client Loss", color='green')
ax_srv_test_loss.set_title("Server/Global Model & Average Client Loss")
ax_srv_test_loss.set_xlabel("Round")
ax_srv_test_loss.set_ylabel("Loss")
ax_srv_test_loss.grid(True)
ax_srv_test_loss.legend()

# Communication (bottom-left) - Same as original metrics_summary
ax_comm4 = axes4[1, 0]
# Communication metrics (distributed only)
if "total_upload_MB" in distributed_df.columns:
    ax_comm4.plot(distributed_df["round"], distributed_df["total_upload_MB"], label="Total Upload (MB)")
if "total_download_MB" in distributed_df.columns:
    ax_comm4.plot(distributed_df["round"], distributed_df["total_download_MB"], label="Total Download (MB)")

ax_comm4.set_title("Communication per Round")
ax_comm4.set_xlabel("Round")
ax_comm4.set_ylabel("Communication Bytes (MB)")
ax_comm4.grid(True)
ax_comm4.legend()

# Computation (bottom-right) - Same as original metrics_summary
ax_comp4 = axes4[1, 1]
if "total_comp_time_sec" in distributed_df.columns:
    ax_comp4.plot(distributed_df["round"], distributed_df["total_comp_time_sec"], label="Total Comp Time (s)")
    ax_comp4.set_ylabel("Seconds")

ax_comp4.set_title("Computation Time per Round")
ax_comp4.set_xlabel("Round")
ax_comp4.grid(True)
ax_comp4.legend()

# Adjust ticks and limits for style 3 figure
max_round = round_series.max() if not round_series.empty else 100
xticks = np.arange(0, max_round+1, X_TICK_INTERVAL)
for a in [ax_srv_test_acc, ax_srv_test_loss, ax_comm4, ax_comp4]:
    a.set_xticks(xticks)
    a.set_xlim(0, max_round + X_TICK_INTERVAL)

# Set y-axis limits and ticks
# Accuracy subplot y-axis (0 to 1)
ax_srv_test_acc.set_ylim(0, 1)
ax_srv_test_acc.set_yticks(np.arange(0, 1.01, 0.2))
# Loss subplot y-axis (0 to 4)
ax_srv_test_loss.set_ylim(0, 4)
ax_srv_test_loss.set_yticks(np.arange(0, 4.1, 0.5))
# Communication subplot y-axis (-1 to 2 MB)
ax_comm4.set_ylim(-1, 2)
ax_comm4.set_yticks(np.arange(-0.5, 2.01, 0.5))
# Computation subplot y-axis (use default auto-scaling like original)

# Finalise & save style 3 figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path4 = PLOTS_DIR / "metrics_summary_style3.png"
fig4.savefig(output_path4, dpi=150)
print(f"Saved style 3 plot to {output_path4.relative_to(ROOT)}")
