#!/bin/bash
# Runs the autoscaler against the REAL Train-Ticket cluster (another machine) after checking that everything it
# needs is reachable. Open the SSH tunnels first (see docs/TRAINTICKET.md).
#
#   bash scripts/19_trainticket_live.sh                 # shadow mode: observe and decide, change NOTHING
#   bash scripts/19_trainticket_live.sh act             # really scale (replicas only, see profiles/trainticket.env)
#   CONTROLLED_SERVICES=ts-order-service,ts-seat-service bash scripts/19_trainticket_live.sh act
#
# Dashboard: http://localhost:8080
set -e
cd "$(dirname "$0")/.."
MODE="${1:-shadow}"; PORT="${2:-8080}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/tt-agent-config}"; export KUBECONFIG
set -a; source profiles/trainticket.env; set +a
[ -f .venv/bin/activate ] && source .venv/bin/activate

fail() { echo "  FAIL  $1"; echo "        $2"; exit 1; }
ok()   { echo "  ok    $1"; }
echo "Preflight (Prometheus $PROMETHEUS_URL, Jaeger $JAEGER_QUERY_URL, kubeconfig $KUBECONFIG)"

[ -f "$KUBECONFIG" ] || fail "kubeconfig missing" "create it as described in docs/TRAINTICKET.md"
curl -fs -m 8 "$PROMETHEUS_URL/-/ready" >/dev/null || fail "Prometheus unreachable" "open the tunnel: ssh -fN -L 9090:127.0.0.1:9090 -L 16686:127.0.0.1:16686 <user>@<windows-tailscale-ip>"
ok "Prometheus"
curl -fs -m 8 "$JAEGER_QUERY_URL/api/services" >/dev/null || fail "Jaeger unreachable" "same tunnel as Prometheus (port 16686)"
ok "Jaeger"
SERVER=$(grep -m1 'server:' "$KUBECONFIG" | awk '{print $2}')
curl -fsk -m 8 "$SERVER/version" >/dev/null || fail "Kubernetes API unreachable at $SERVER" "minikube's API port changes on restart: kubectl config view --minify (on the cluster PC), re-tunnel, update 'server:' in $KUBECONFIG"
kubectl --request-timeout=10s auth can-i patch deployments >/dev/null || fail "service account cannot patch deployments" "token expired (24 h)? create a new one, see docs/TRAINTICKET.md"
ok "Kubernetes API ($SERVER), can patch deployments"

(cd autoscaler && python - <<'PY'
import sys, time, requests
from src.data_collection import jaeger_client
from src import config
skew = jaeger_client.clock_skew_s(0)
print(f"  {'ok  ' if abs(skew) < 5 else 'WARN'}  clock skew cluster PC - this Mac: {skew:+.0f} s" + ("" if abs(skew) < 5 else "  (handled for trace windows, but sync the cluster PC: w32tm /resync)"))
t = requests.get(f"{config.JAEGER_QUERY_URL}/api/traces", params={"service": config.ENTRY_SERVICE, "limit": 1}, timeout=15).json().get("data", [])
if not t:
    print("  WARN  no traces for", config.ENTRY_SERVICE, "- start the load generator (train-ticket-auto-query)")
else:
    age = time.time() + skew - max(s["startTime"] + s["duration"] for s in t[0]["spans"]) / 1e6
    print(f"  {'ok  ' if age < 120 else 'WARN'}  newest trace is {age:.0f} s old" + ("" if age < 120 else "  (no load running: the topology will freeze, the agent will hold)"))
PY
)

ARGS=(--mode live --port "$PORT" --interval "$STEP_SECONDS")
if [ "$MODE" = "act" ]; then
  echo; echo "ACT mode: replicas of ${CONTROLLED_SERVICES:-ALL discovered services} may really change (max $MAX_REPLICAS, resize=$ALLOW_RESIZE)."
else
  ARGS+=(--shadow); echo; echo "SHADOW mode: nothing will be changed."
fi
cd autoscaler && exec python -m src.dashboard.server "${ARGS[@]}"
