#!/bin/bash
# Plug the autoscaler into ANY Kubernetes cluster: inspect it, derive the settings, start the control room.
#
#   bash scripts/20_autoscale_any_cluster.sh --namespace shop --prometheus http://localhost:9090 --jaeger http://localhost:16686
#   bash scripts/20_autoscale_any_cluster.sh ... --act                       # really scale (replicas only by default)
#   bash scripts/20_autoscale_any_cluster.sh ... --controlled svc-a,svc-b    # restrict what may be changed
#   bash scripts/20_autoscale_any_cluster.sh ... --entry web-frontend        # when traces do not reveal the entry service
#   bash scripts/20_autoscale_any_cluster.sh ... --slo 1400                  # p95 target in ms (the default is a depth-based formula, not measured)
#   bash scripts/20_autoscale_any_cluster.sh ... --arm mo_td3_trainticket_cap6   # which trained checkpoint in autoscaler/models
#   bash scripts/20_autoscale_any_cluster.sh ... --reserve 600               # MiB the node must keep free after a scale-up (default derived: 768)
#   bash scripts/20_autoscale_any_cluster.sh ... --resize cpu                # also let it change CPU (memory fixed); `--resize all` for CPU and memory. Default: replicas only
#   bash scripts/20_autoscale_any_cluster.sh ... --discover-only             # just print what it found, start nothing
#   bash scripts/20_autoscale_any_cluster.sh ... --database-url postgresql://...  --cluster-name shop-prod   # store the history (see docs/PERSISTENCE.md)
#
# Needs: a kubeconfig that can list Deployments/Pods (and patch them for --act), Prometheus with cAdvisor + kube-state-metrics,
# and OpenTelemetry traces in Jaeger for the call graph (see docs/TRAINTICKET.md). Starts in SHADOW mode: nothing is changed.
set -e
cd "$(dirname "$0")/.."
NS=default; PROM=http://localhost:9090; JAEGER=http://localhost:16686; PORT=8080; ACT=0; ONLY=0; ENTRY=""; CONTROLLED=""; SLO=""; ARM=""; RESERVE=""; DBURL=""; CNAME=""; RESIZE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --namespace) NS="$2"; shift 2;;      --prometheus) PROM="$2"; shift 2;;   --jaeger) JAEGER="$2"; shift 2;;
    --kubeconfig) export KUBECONFIG="$2"; shift 2;;   --port) PORT="$2"; shift 2;;   --entry) ENTRY="$2"; shift 2;;
    --controlled) CONTROLLED="$2"; shift 2;;   --slo) SLO="$2"; shift 2;;   --reserve) RESERVE="$2"; shift 2;;   --arm) ARM="$2"; shift 2;;   --resize) RESIZE="$2"; shift 2;;   --database-url) DBURL="$2"; shift 2;;   --cluster-name) CNAME="$2"; shift 2;;   --act) ACT=1; shift;;   --discover-only) ONLY=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
[ -f .venv/bin/activate ] && source .venv/bin/activate
OUT="profiles/auto-$NS.env"
cd autoscaler
echo "Inspecting namespace '$NS' (Prometheus $PROM, Jaeger $JAEGER)"; echo
python -m src.inference.auto_profile --namespace "$NS" --prometheus "$PROM" --jaeger "$JAEGER" ${ENTRY:+--entry "$ENTRY"} --out "../$OUT"
cd ..
[ "$ONLY" = 1 ] && exit 0
set -a; source "$OUT"; set +a
[ -n "$CONTROLLED" ] && export CONTROLLED_SERVICES="$CONTROLLED"
[ -n "$SLO" ] && export SLO_LATENCY_MS="$SLO"
[ -n "$RESERVE" ] && export NODE_MEMORY_RESERVE_MB="$RESERVE"
case "$RESIZE" in cpu) export ALLOW_RESIZE=1 ALLOW_MEMORY_RESIZE=0;; all) export ALLOW_RESIZE=1;; "") ;; *) echo "--resize must be cpu or all"; exit 2;; esac
[ -n "$DBURL" ] && export AUTOSCALER_DATABASE_URL="$DBURL"
[ -z "$CNAME" ] && CNAME="${AUTOSCALER_CLUSTER_NAME:-$(kubectl config view --minify -o jsonpath='{.clusters[0].name}' 2>/dev/null)}"
[ -n "$CNAME" ] && export AUTOSCALER_CLUSTER_NAME="$CNAME"
if [ -n "$AUTOSCALER_DATABASE_URL" ]; then echo "History: stored per cluster in the database, as '${AUTOSCALER_CLUSTER_NAME:-<unnamed: set --cluster-name>}'"; fi
[ -n "$ARM" ] && ARGS_ARM=(--arm "$ARM") || ARGS_ARM=()
if [ -z "$AUTOSCALER_ENTRY_SERVICE" ] && [ "$METRICS_SOURCE" = "otel" ]; then echo "no entry service found: pass --entry"; exit 1; fi
ARGS=(--mode live --port "$PORT" --interval "${STEP_SECONDS:-30}")
if [ "$ACT" = 1 ]; then
  echo; echo "ACT mode: replicas of ${CONTROLLED_SERVICES:-no service} may really change (max $MAX_REPLICAS, resize=$ALLOW_RESIZE)."
else
  ARGS+=(--shadow); echo; echo "SHADOW mode: observing and deciding, nothing is changed. Control room: http://localhost:$PORT"
fi
cd autoscaler && exec python -m src.dashboard.server "${ARGS[@]}" "${ARGS_ARM[@]}"
