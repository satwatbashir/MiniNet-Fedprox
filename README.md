# MiniNet-Fedprox

A Federated Learning framework implementing **FedProx** algorithm on **Mininet** network emulator using the **Flower** framework. This project enables realistic simulation of FL scenarios with configurable network conditions.

---

## Overview

This project simulates federated learning in realistic network environments by combining:
- **FedProx**: Federated optimization algorithm for heterogeneous networks
- **Mininet**: Network emulator for realistic bandwidth, latency, and packet loss simulation
- **Flower**: Production-ready federated learning framework

---

## Experiment Configuration

### Dataset
| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Training Samples | 50,000 |
| Test Samples | 10,000 |
| Classes | 10 |
| Input Shape | 3 x 32 x 32 (RGB) |

### Model Architecture (LeNet)
| Layer | Details |
|-------|---------|
| Conv1 | 3 → 6 channels, 5x5 kernel |
| Pool1 | MaxPool 2x2 |
| Conv2 | 6 → 16 channels, 5x5 kernel |
| Pool2 | MaxPool 2x2 |
| FC1 | → 120 units |
| FC2 | → 84 units |
| FC3 | → 10 classes |

### Federated Learning Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Rounds | 150 | Number of communication rounds |
| Fraction Fit | 0.5 | 50% of clients selected per round |
| Local Epochs | 1 | Training epochs per client per round |
| Batch Size | 32 | Mini-batch size |
| Learning Rate | 0.01 | SGD learning rate |
| Proximal Mu (μ) | 0.01 | FedProx regularization term |
| Min Clients | 3 | Minimum clients required to start |

### Data Heterogeneity (Non-IID)
| Parameter | Value | Description |
|-----------|-------|-------------|
| Dirichlet Alpha (α) | 0.5 | Controls data heterogeneity |

> **Note**: Lower α = more heterogeneous (non-IID). α=0.5 creates moderate heterogeneity where clients have imbalanced class distributions.

---

## Network Scenarios

Four network scenarios are implemented to simulate different real-world conditions:

### 1. Uniform Good (`uniform_good`)
| Link | Bandwidth | Delay | Loss |
|------|-----------|-------|------|
| Server | 100 Mbps | 5 ms | 0% |
| Clients | 50 Mbps | 10 ms | 0% |

*Ideal network conditions - fast, low-latency, no packet loss.*

### 2. Uniform Bad (`uniform_bad`)
| Link | Bandwidth | Delay | Loss |
|------|-----------|-------|------|
| Server | 5 Mbps | 200 ms | 5% |
| Clients | 2 Mbps | 200 ms | 5% |

*Poor network conditions - slow, high latency, with packet loss.*

### 3. Heterogeneous (`heterogeneous`)
| Link | Bandwidth | Delay | Loss |
|------|-----------|-------|------|
| Server | 50 Mbps | 20 ms | 0% |
| Fast Clients | 20 Mbps | 50 ms | 1% |
| Slow Clients | 5 Mbps | 150 ms | 3% |

*Mixed environment - some clients have good connectivity, others have poor.*

### 4. Jitter (`jitter`)
| Link | Bandwidth | Delay | Loss |
|------|-----------|-------|------|
| Server | 50 Mbps | 20 ms | 0% |
| Clients (base) | 20 Mbps | 50 ms | 1% |
| **Per-round fluctuation** | 1-50 Mbps | 10-300 ms | 0-10% |

*Dynamic network - client conditions change randomly each round.*

---

## Project Structure

```
MiniNet-Fedprox/
├── fedge/
│   ├── __init__.py
│   ├── client_app.py      # Flower client with FedProx training
│   ├── server_app.py      # Flower server with metrics aggregation
│   ├── task.py            # LeNet model & data loading
│   └── utils/
│       └── functional.py
├── tools/
│   └── net_topo.py        # Mininet network topology setup
├── run_server.py          # Server launcher
├── run_client.py          # Client launcher
├── scenarios.py           # Network scenario definitions
├── plot_metrics.py        # Metrics visualization
├── kill_bash.sh           # Cleanup utility
└── pyproject.toml         # Project configuration
```

---

## Requirements

- Python 3.8+
- Mininet
- PyTorch >= 2.2.1
- Flower >= 1.12.0
- CUDA (optional, for GPU acceleration)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/satwatbashir/MiniNet-Fedprox.git
cd MiniNet-Fedprox

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

---

## Usage

### Running an Experiment

```bash
# Set environment variables
export SCENARIO="uniform_good"    # or uniform_bad, heterogeneous, jitter
export NUM_CLIENTS=50
export ROUNDS=150
export FRACTION_FIT=0.5
export SEED=0

# Start the Mininet topology (requires sudo)
sudo -E python tools/net_topo.py
```

### Inside Mininet CLI
```bash
# Check connectivity
pingall

# Monitor server logs
srv tail -f server.log

# Monitor client logs
c1 tail -f client1.log

# Exit when done
exit
```

### Cleanup
```bash
# Kill all processes and clean up
./kill_bash.sh
```

---

## Metrics Collected

Each round collects:
- **Training**: Loss, Accuracy (per client and aggregated)
- **Evaluation**: Test loss, Test accuracy (centralized and distributed)
- **Timing**: Computation time, Communication time (actual server-side measurement)
- **Data Transfer**: Upload/Download bytes per client
- **Network State**: Bandwidth, delay, loss per client link

Metrics are saved to `metrics/seed_X/` directory.

---

## License

Apache-2.0
