  GNU nano 4.8                                                                                   kill_fedge.sh                                                                                             
#!/bin/bash
set -e

echo "🔪 Killing all Fedge/Flower/Mininet processes..."

# Stop Python/Flower jobs cleanly
sudo pkill -f run_cloud.py || true
sudo pkill -f run_leaf_server.py || true
sudo pkill -f run_leaf_client.py || true
sudo pkill -f run_proxy.py || true
sudo pkill -f tools/net_topo.py || true
sudo pkill -f flower.server || true
sudo pkill -f python.*Mininet-Fedge || true

# Stop Mininet and remove stale namespaces, interfaces, and qdiscs
echo "🧹 Cleaning Mininet network..."
sudo mn -c > /dev/null 2>&1
for d in $(ls /sys/class/net | grep -E '^(s|c|veth|h|eth|enp|lo)'); do
  sudo tc qdisc del dev "$d" root 2>/dev/null || true
done

# Drop OS caches to actually release RAM
echo "🧠 Dropping Linux page caches..."
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# Clear your project’s transient folders
echo "🧾 Removing runtime folders..."
sudo rm -rf c*log server*.log proxy*.log cloud.log \
  metrics models signals runs rounds \
  /tmp/fedge_* /tmp/flwr_* 2>/dev/null || true

# Optional: show memory summary
echo "📊 Memory after cleanup:"
free -h

echo "✅ Environment reset complete."
