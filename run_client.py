import os
from types import SimpleNamespace
from flwr.client import start_client
from fedge.client_app import client_fn as real_client_fn

SERVER_ADDR = os.getenv("SERVER_ADDR", "127.0.0.1:8080")
CID = int(os.getenv("CID", "1"))
PARTITION_ID = CID - 1 if CID > 0 else 0
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", 2))

def patched_client_fn(orig_ctx):
    base_run = dict(getattr(orig_ctx, "run_config", {}) or {})
    base_node = dict(getattr(orig_ctx, "node_config", {}) or {})

    node_cfg = {
        **base_node,
        "dataset_flag": os.getenv("DATASET_FLAG", base_node.get("dataset_flag", "cifar10")),
        "partition-id": base_node.get("partition-id", PARTITION_ID),
        "num-partitions": base_node.get("num-partitions", NUM_PARTITIONS),
    }
    run_cfg = {
        **base_run,
        "dirichlet_alpha": float(os.getenv("DIR_ALPHA", base_run.get("dirichlet_alpha", 0.5))),
        "batch_size": int(os.getenv("BATCH_SIZE", base_run.get("batch_size", 32))),
        "local-epochs": int(os.getenv("LOCAL_EPOCHS", base_run.get("local-epochs", 1))),
        "seed": int(os.getenv("SEED", base_run.get("seed", 0))),
        "local_lr": float(os.getenv("LOCAL_LR", base_run.get("local_lr", 1e-2))),
    }
    proxy_ctx = SimpleNamespace(run_config=run_cfg, node_config=node_cfg)
    return real_client_fn(proxy_ctx)

start_client(server_address=SERVER_ADDR, client_fn=patched_client_fn, max_retries=100, max_wait_time=60.0,insecure=True,)
