# tools/net_topo.py — VMware drop-in (thread-safe pexec + bind/all + long wait)
import os, sys, time, threading, random
from pathlib import Path
from typing import List

# Ensure imports work when run from tools/
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.cli import CLI as _CLI

from scenarios import SCENARIOS

# ---- FIXED PATHS (your VMware layout) ----
PROJECT = "/home/fedge/flproj/MiniNet-Fedprox"
VENV    = "/home/fedge/flproj/.venv"
PY      = "/home/fedge/flproj/.venv/bin/python"
if not os.path.exists(PY):
    print("*** WARN: venv python not found at", PY, "— falling back to /usr/bin/python3")
    PY = "/usr/bin/python3"

# Export into each namespace
ENV_PREFIX = (
    f"export VIRTUAL_ENV={VENV}; "
    f"export PATH=/usr/sbin:/sbin:{VENV}/bin:$PATH; "
    f"export PYTHONPATH={PROJECT}; "
    f"export MPLCONFIGDIR={PROJECT}/.mplcache; "
    f"export DATA_ROOT={os.getenv('DATA_ROOT','')}; "
    f"export DOWNLOAD_DATA={os.getenv('DOWNLOAD_DATA','')}; "
)

CMD_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
TRIPFILE = "/tmp/mininet_stop_trip.txt"
TC_PATH = {}  # host.name -> tc path

class SafeCLI(_CLI):
    def preloop(self):
        print("*** SafeCLI engaged: stop/start are disabled")
        super().preloop()
    def _log_block(self, line: str):
        try:
            with open(TRIPFILE, "a", encoding="utf-8") as f:
                f.write(f"BLOCKED at {time.time():.3f}: '{line}'\n")
        except Exception:
            pass
    def do_stop(self, line):
        self._log_block(f"stop {line}")
        print("*** stop disabled (ignored):", line)
    def default(self, line):
        s = (line or "").strip().lower()
        if s.startswith("stop") or s.startswith("st "):
            self._log_block(line)
            print("*** stop disabled (ignored):", line)
            return
        super().default(line)

def _link_kwargs_for_client(scenario: dict, idx: int) -> dict:
    if "client_pattern" in scenario:
        patterns = scenario["client_pattern"]
        return patterns[idx % len(patterns)]
    return scenario["client"]

def _apply_qdisc(host, rate_mbit: float, delay_ms: int, loss_pct: float) -> None:
    if STOP_EVENT.is_set():
        return
    dev = f"{host.name}-eth0"
    tc = TC_PATH.get(host.name, "tc")
    host.pexec(
        'bash','-lc',
        f'{tc} qdisc del dev {dev} root || true; '
        f'{tc} qdisc add dev {dev} root handle 1: tbf rate {rate_mbit}mbit burst 32kbit latency 50ms; '
        f'{tc} qdisc add dev {dev} parent 1: handle 10: netem delay {delay_ms}ms loss {loss_pct}%'
    )

# ---- thread-safe listening check (pexec) ----
def _is_listening(host, port: int) -> bool:
    out, err, _ = host.pexec(
        'bash','-lc',
        f'(ss -ltn 2>/dev/null | grep -q ":{port} ") || '
        f'(netstat -ltn 2>/dev/null | grep -q ":{port} ") || '
        f'(command -v lsof >/dev/null 2>&1 && lsof -iTCP:{port} -sTCP:LISTEN >/dev/null 2>&1) && echo YES || echo NO'
    )
    txt = (out or "") + (err or "")
    return "YES" in txt

def _watch_server_until_listen(srv, project_root: str, port: int, timeout_s: int = 120):
    start = time.time()
    last = ""
    print(f"*** Server watcher: streaming server.log while waiting for :{port} (max {timeout_s}s)...")
    while time.time() - start < timeout_s and not STOP_EVENT.is_set():
        if _is_listening(srv, port):
            print(f"*** Server watcher: :{port} is LISTENING.")
            return True
        out, err, _ = srv.pexec('bash','-lc', f'tail -n 15 {project_root}/server.log 2>/dev/null || true')
        txt = (out or "") + (err or "")
        if txt and txt != last:
            print("==> server.log <==")
            print(txt.rstrip())
            last = txt
        time.sleep(1.0)
    print(f"*** Server watcher: TIMEOUT waiting for :{port}; proceeding anyway.\n")
    out, err, _ = srv.pexec('bash','-lc', f'tail -n 60 {project_root}/server.log 2>/dev/null || echo "(no server.log yet)"')
    print(((out or "") + (err or "")).rstrip())
    return False

def _start_round_watcher(clients, jitter_cfg: dict, seed: int, tag: str, project_root: str):
    tick_file = Path(project_root) / f"metrics/seed_{seed}" / "round_tick.txt"
    bw_range = tuple(jitter_cfg.get("bw_mbit", [1.0, 50.0]))
    delay_range = tuple(jitter_cfg.get("delay_ms", [10, 300]))
    loss_range = tuple(jitter_cfg.get("loss_pct", [0.0, 10.0]))
    default_targets = ",".join([ch.name for ch in clients])
    targets = {t.strip() for t in os.getenv("JITTER_TARGETS", default_targets).split(",") if t.strip()}

    def read_tick(default=0):
        try:
            txt = tick_file.read_text(encoding="utf-8").strip()
            return int(txt) if txt else default
        except Exception:
            return default

    def apply_for_round(round_no: int):
        for idx, ch in enumerate(clients, start=1):
            if STOP_EVENT.is_set():
                return
            if ch.name not in targets:
                continue
            rng = random.Random(seed * 1_000_003 + round_no * 10_000_019 + idx)
            bw = rng.uniform(float(bw_range[0]), float(bw_range[1]))
            delay = rng.randint(int(delay_range[0]), int(delay_range[1]))
            loss = rng.uniform(float(loss_range[0]), float(loss_range[1]))
            print(f"*** {tag}: round={round_no} {ch.name} -> bw={bw:.2f} Mbit/s, delay={delay} ms, loss={loss:.2f}%")
            _apply_qdisc(ch, bw, delay, loss)

    def worker():
        last = -1
        print(f"*** {tag}: round-watcher started; waiting for {tick_file}")
        while not STOP_EVENT.is_set() and not tick_file.exists():
            time.sleep(0.5)
        if STOP_EVENT.is_set():
            return
        current = read_tick(default=0)
        if current != last and not STOP_EVENT.is_set():
            apply_for_round(current); last = current
        while not STOP_EVENT.is_set():
            time.sleep(0.5)
            current = read_tick(default=last)
            if current > last:
                apply_for_round(current); last = current

    threading.Thread(target=worker, daemon=True, name="round-watcher").start()

def run():
    scenario_name = os.getenv("SCENARIO", "uniform_good")
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown SCENARIO='{scenario_name}'. Options: {list(SCENARIOS.keys())}")
    scenario = SCENARIOS[scenario_name]
    print(f"*** Running scenario: {scenario_name}")
    print(f"*** Note: {scenario.get('note', '(no note)')}")

    NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "100"))
    num_parts_env = os.getenv("NUM_PARTITIONS")

    net = Mininet(controller=OVSController, link=TCLink, autoSetMacs=True, autoStaticArp=True)
    c0 = net.addController("c0")
    s1 = net.addSwitch("s1")
    srv = net.addHost("srv")
    clients: List = [net.addHost(f"c{i}") for i in range(1, NUM_CLIENTS + 1)]

    # Links
    net.addLink(srv, s1, **scenario["srv"])
    for i, ch in enumerate(clients):
        net.addLink(ch, s1, **_link_kwargs_for_client(scenario, i))
    net.start()

    # Cache tc path inside each namespace
    for h in [srv] + clients:
        out, err, _ = h.pexec('bash','-lc','command -v tc || echo /sbin/tc')
        TC_PATH[h.name] = ((out or err).strip() or "/sbin/tc")

    # Log scenario
    srv.pexec('bash','-lc', f'echo "Scenario: {scenario_name}" >> {PROJECT}/server.log')

    # Server envs
    rounds = os.getenv("ROUNDS", "50")
    fraction_fit = os.getenv("FRACTION_FIT", "0.1")
    min_avail = os.getenv("MIN_AVAIL", str(NUM_CLIENTS))
    num_parts = num_parts_env if num_parts_env is not None else str(NUM_CLIENTS)
    round_timeout = os.getenv("ROUND_TIMEOUT", "1200")
    dataset_flag = os.getenv("DATASET_FLAG", "cifar10")
    seed = os.getenv("SEED", "0")
    assumed = os.getenv("ASSUMED_MBPS", "")

    # Start server (bind on all v4; clients use srv_ip)
    srv_ip = srv.IP()  # e.g., 10.0.0.1
    srv.pexec(
        'bash','-lc',
        f'cd {PROJECT} && {ENV_PREFIX}'
        f'SCENARIO={scenario_name} NUM_PARTITIONS={num_parts} DATASET_FLAG={dataset_flag} '
        f'FRACTION_FIT={fraction_fit} MIN_AVAIL={min_avail} ROUNDS={rounds} ROUND_TIMEOUT={round_timeout} '
        f'SEED={seed} {"ASSUMED_MBPS="+assumed if assumed else ""} '
        f'BIND_ADDR=0.0.0.0:8080 '
        f'{PY} -u run_server.py > server.log 2>&1 &'
    )

    # Wait until port is actually listening (give Flower time)
    _watch_server_until_listen(srv, PROJECT, port=8080, timeout_s=120)

    # Start clients
    for idx, ch in enumerate(clients, start=1):
        ch.pexec(
            'bash','-lc',
            f'cd {PROJECT} && {ENV_PREFIX}'
            f'SERVER_ADDR={srv_ip}:8080 CID={idx} NUM_PARTITIONS={num_parts} DATASET_FLAG={dataset_flag} '
            f'{PY} -u run_client.py > client{idx}.log 2>&1 &'
        )
        time.sleep(0.05)

    if scenario_name == "jitter":
        _start_round_watcher(clients, scenario.get("client_jitter", {}), int(seed), "jitter:round", PROJECT)

    print("\n*** Mininet is running. Useful commands:")
    print("   pingall")
    print("   srv tail -f server.log")
    print("   c1  tail -f client1.log")
    print("   exit   # to stop and clean up\n")

    try:
        if os.getenv("USE_MININET_CLI", "1") == "1":
            SafeCLI(net)
        else:
            while True:
                time.sleep(3600)
    finally:
        STOP_EVENT.set()
        with CMD_LOCK:
            pass
        time.sleep(0.2)
        net.stop()

if __name__ == "__main__":
    setLogLevel("info")
    run()
