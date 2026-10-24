#!/bin/bash
# Runs the autoscaler against the REAL Train-Ticket cluster (another machine) after checking that everything it
# needs is reachable. Open the SSH tunnels first (see docs/TRAINTICKET.md).
#
#   bash scripts/19_trainticket_live.sh                 # shadow mode: observe and decide, change NOTHING
#   bash scripts/19_trainticket_live.sh act             # really scale (replicas only, see profiles/trainticket.env)
#   CONTROLLED_SERVICES=ts-order-service,ts-seat-service bash scripts/19_trainticket_live.sh act
#
# Options (environment variables):
#   ARM=mo_td3_trainticket_cpu     which checkpoint in autoscaler/models (default: the serving default; set it explicitly)
#   SLO_MS=3000                    absolute p95 target in ms (without it: a depth-based formula)
#   RESIZE=cpu | all               allow CPU changes (memory stays fixed) / CPU and memory; default none (replicas only). A resize restarts the pod.
#   NODE_RESERVE_MB=600            memory the node must keep free after a scale-up (profile: 768)
#
# History: with AUTOSCALER_DATABASE_URL set (bash scripts/22_db_up.sh prints it) everything the agent sees and does is stored, per
# cluster; AUTOSCALER_CLUSTER_NAME names this cluster (default: the cluster name in the kubeconfig).

# Dashboard: http://localhost:8080
set -e
cd "$(dirname "$0")/.."
MODE="${1:-shadow}"; PORT="${2:-8080}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/tt-agent-config}"; export KUBECONFIG
set -a; source profiles/trainticket.env; set +a
case "${RESIZE:-}" in cpu) export ALLOW_RESIZE=1 ALLOW_MEMORY_RESIZE=0;; all) export ALLOW_RESIZE=1;; "") ;; *) echo "RESIZE must be cpu or all"; exit 2;; esac
[ -n "$SLO_MS" ] && export SLO_LATENCY_MS="$SLO_MS"
[ -n "$NODE_RESERVE_MB" ] && export NODE_MEMORY_RESERVE_MB="$NODE_RESERVE_MB"
[ -n "$ARM" ] && ARGS_ARM=(--arm "$ARM") || ARGS_ARM=()
export AUTOSCALER_CLUSTER_NAME="${AUTOSCALER_CLUSTER_NAME:-$(kubectl config view --minify -o jsonpath='{.clusters[0].name}' 2>/dev/null)}"
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

if [ -n "$AUTOSCALER_DATABASE_URL" ]; then echo "  ok    history: stored in the database as cluster '${AUTOSCALER_CLUSTER_NAME:-?}'"; else echo "  --    history is off (AUTOSCALER_DATABASE_URL is not set; bash scripts/22_db_up.sh)"; fi
ARGS=(--mode live --port "$PORT" --interval "$STEP_SECONDS")
if [ "$MODE" = "act" ]; then
  echo; echo "ACT mode: replicas of ${CONTROLLED_SERVICES:-ALL discovered services} may really change (max $MAX_REPLICAS, resize=$ALLOW_RESIZE)."
else
  ARGS+=(--shadow); echo; echo "SHADOW mode: nothing will be changed."
fi
echo "model: ${ARM:-default checkpoint}   resize: ${RESIZE:-none (replicas only)}   SLO: ${SLO_LATENCY_MS:-formula}   controlled: ${CONTROLLED_SERVICES:-ALL}"
cd autoscaler && exec python -m src.dashboard.server "${ARGS[@]}" "${ARGS_ARM[@]}"
