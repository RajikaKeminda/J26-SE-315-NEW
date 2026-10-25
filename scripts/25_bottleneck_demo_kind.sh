#!/bin/bash
# Live "find the bottleneck wherever it is" demonstration on the local kind cluster (the 6-service demo app): make ANY one service the bottleneck, run the
# agent against the real cluster and watch it add capacity to that service. Everything is real Kubernetes: pods, CPU limits, traffic, Prometheus.
#
#   bash scripts/25_bottleneck_demo_kind.sh forward                  # port-forwards that do not clash with the Train-Ticket tunnels: Prometheus 19090, Jaeger 16687
#   bash scripts/25_bottleneck_demo_kind.sh reset                    # every service back to its manifest allocation (2 pods x 0.25 core)
#   bash scripts/25_bottleneck_demo_kind.sh stage pricing-service    # that service: 1 pod with a 0.025-core limit -> saturated at the demo load
#   bash scripts/25_bottleneck_demo_kind.sh load 12 600              # in-cluster steady load: 12 req/s for 600 s
#   bash scripts/25_bottleneck_demo_kind.sh agent [shadow]           # the agent (mo_td3_bn) on http://localhost:8090, acting; add "shadow" to only watch
#   bash scripts/25_bottleneck_demo_kind.sh status                   # pods, replicas and CPU limit of every service
#   bash scripts/25_bottleneck_demo_kind.sh stop                     # stop the port-forwards and the agent started by this script
set -e
cd "$(dirname "$0")/.."
NS=autoscaler-poc; CTX=kind-autoscaler; K="kubectl --context $CTX -n $NS"; PIDS=/tmp/bn_demo_pids
[ -f .venv/bin/activate ] && source .venv/bin/activate
case "${1:-status}" in
  forward)
    : > $PIDS
    $K port-forward svc/prometheus 19090:9090 >/dev/null 2>&1 & echo $! >> $PIDS
    $K port-forward svc/jaeger 16687:16686 >/dev/null 2>&1 & echo $! >> $PIDS
    sleep 3; echo "Prometheus http://localhost:19090   Jaeger http://localhost:16687";;
  reset)
    kubectl --context $CTX apply -f k8s/01-services/ >/dev/null
    for d in gateway catalog auth inventory pricing user; do
      $K patch deploy $d-service --type=json -p '[{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/cpu","value":"250m"},{"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/cpu","value":"250m"}]' >/dev/null 2>&1 || true
      $K scale deploy $d-service --replicas=2 >/dev/null
    done
    for d in gateway catalog auth inventory pricing user; do $K rollout status deploy/$d-service --timeout=120s >/dev/null; done
    echo "reset: every service 2 pods x 250m";;
  stage)
    SVC="${2:?service name, e.g. pricing-service}"
    $K patch deploy "$SVC" --type=json -p '[{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/cpu","value":"25m"},{"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/cpu","value":"25m"}]' >/dev/null
    $K scale deploy "$SVC" --replicas=1 >/dev/null
    $K rollout status deploy/"$SVC" --timeout=120s >/dev/null
    echo "bottleneck staged on $SVC: 1 pod, 25m";;
  load)
    RPS="${2:-12}"; SECS="${3:-600}"
    exec bash scripts/18_demo_traffic.sh steady "$RPS" "$SECS";;
  agent)
    export AUTOSCALER_NAMESPACE=$NS PROMETHEUS_URL=http://localhost:19090 JAEGER_QUERY_URL=http://localhost:16687 JAEGER_PUBLIC_URL=http://localhost:16687
    export ALLOW_MEMORY_RESIZE=0 SCALE_UP_MIN_UTIL="${SCALE_UP_MIN_UTIL:-0.5}" SCALE_DOWN_MAX_UTIL="${SCALE_DOWN_MAX_UTIL:-0.5}"
    export AUTOSCALER_CLUSTER_NAME="${AUTOSCALER_CLUSTER_NAME:-kind-demo}" KUBECONFIG="$HOME/.kube/config"
    kubectl config use-context $CTX >/dev/null
    ARGS=(--mode live --port 8090 --interval 10 --arm "${ARM:-mo_td3_bn}"); [ "$2" = shadow ] && ARGS+=(--shadow)
    echo "agent: ${ARM:-mo_td3_bn}, gate up>=$SCALE_UP_MIN_UTIL down<=$SCALE_DOWN_MAX_UTIL, ${2:-ACTING}; history -> ${AUTOSCALER_DATABASE_URL:+database}${AUTOSCALER_DATABASE_URL:-not stored}; http://localhost:8090"
    cd autoscaler && exec python -m src.dashboard.server "${ARGS[@]}";;
  status)
    $K get deploy -o 'custom-columns=NAME:.metadata.name,WANT:.spec.replicas,READY:.status.readyReplicas,CPU:.spec.template.spec.containers[0].resources.limits.cpu,MEM:.spec.template.spec.containers[0].resources.limits.memory' | grep -E "NAME|-service";;
  stop)
    [ -f $PIDS ] && xargs kill < $PIDS 2>/dev/null || true; rm -f $PIDS; echo "port-forwards stopped (stop the agent with Ctrl+C or kill the pid on port 8090)";;
  *) echo "usage: $0 forward|reset|stage <service>|load <rps> <secs>|agent [shadow]|status|stop"; exit 2;;
esac
