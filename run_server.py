import os
from flwr.server import start_server
from fedge.server_app import server_fn
from flwr.server import ServerConfig

SCENARIO = os.getenv("SCENARIO", "uniform_good")
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", 10))
BIND_ADDR = os.getenv("BIND_ADDR", "0.0.0.0:8080")  # explicit Mininet srv IP



ASSUMED_MBPS_BY_SCENARIO = {
    "uniform_good": 50.0,
    "heterogeneous": 5.0,
    "dynamic": 10.0,
    "uniform_bad": 2.0,
    "jitter": 25.0,
}
ROUND_TIMEOUT_BY_SCENARIO = {
    "uniform_good": 300.0,
    "heterogeneous": 450.0,
    "dynamic": 600.0,
    "uniform_bad": 900.0,
    "jitter": 600.0,
}

assumed_mbps = float(os.getenv("ASSUMED_MBPS", ASSUMED_MBPS_BY_SCENARIO.get(SCENARIO, 10.0)))
round_timeout_sec = float(os.getenv("ROUND_TIMEOUT", ROUND_TIMEOUT_BY_SCENARIO.get(SCENARIO, 300.0)))

class Ctx:
    def __init__(self):
        self.node_config = {
            "dataset_flag": os.getenv("DATASET_FLAG", "cifar10"),
            "num-partitions": NUM_PARTITIONS,
        }
        self.run_config = {
            "num-server-rounds": int(os.getenv("ROUNDS", 5)),
            "fraction-fit": float(os.getenv("FRACTION_FIT", 0.5)),
            "min_available_clients": int(os.getenv("MIN_AVAIL", 10)),
            "proximal_mu": float(os.getenv("PROX_MU", 0.01)),
            "dirichlet_alpha": float(os.getenv("DIR_ALPHA", 0.5)),
            "batch_size": int(os.getenv("BATCH_SIZE", 32)),
            "local-epochs": int(os.getenv("LOCAL_EPOCHS", 1)),
            "seed": int(os.getenv("SEED", 0)),
            "assumed_mbps": assumed_mbps,
            "round_timeout_sec": round_timeout_sec,
        }

components = server_fn(Ctx())
server_cfg: ServerConfig = components.config

start_server(
    server_address=BIND_ADDR,
    config=server_cfg,
    strategy=components.strategy,
)
