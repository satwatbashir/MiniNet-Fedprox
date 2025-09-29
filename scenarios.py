SCENARIOS = {
    "uniform_good": {
        "srv": {"bw": 100, "delay": "5ms",   "loss": 0},
        "client": {"bw": 50,  "delay": "10ms","loss": 0},
        "note": "All links are fast, low-latency, no loss.",
    },
    "uniform_bad": {
        "srv": {"bw": 5, "delay": "200ms", "loss": 5},
        "client": {"bw": 2, "delay": "200ms", "loss": 5},
        "note": "All links are slow, high latency, with packet loss.",
    },
    "heterogeneous": {
        "srv": {"bw": 50, "delay": "20ms", "loss": 0},
        "client_pattern": [
            {"bw": 20, "delay": "50ms",  "loss": 1},
            {"bw": 5,  "delay": "150ms", "loss": 3},
        ],
        "note": "Some clients fast/clean, others slow/lossy.",
    },
    "jitter": {
        "srv": {"bw": 50, "delay": "20ms", "loss": 0},
        "client": {"bw": 20, "delay": "50ms", "loss": 1},
        "client_jitter": {
            "period_sec": 30,
            "bw_mbit":  [1.0, 50.0],
            "delay_ms": [10, 300],
            "loss_pct": [0.0, 10.0],
        },
        "note": "Each client link fluctuates randomly per round.",
    },
}
