#!/bin/bash
# Live-demo load: open-loop traffic generated INSIDE the cluster (no port-forward in the path), so it keeps
# flowing while the autoscaler rolls pods.
#
#   bash scripts/18_demo_traffic.sh            # burst: 12 rps -> 40 rps -> 12 rps, ~7 minutes
#   bash scripts/18_demo_traffic.sh diurnal    # smooth day/night wave
#   bash scripts/18_demo_traffic.sh steady 12 300     # constant 12 rps for 300 s
#
# Watch it react: dashboard http://localhost:8080 (after scripts/05_port_forward.sh) and `kubectl get pods -n autoscaler-poc -w`.
set -e
cd "$(dirname "$0")/.."
NS=autoscaler-poc
PATTERN="${1:-burst}"
BASE="${2:-12}"
SECS="${3:-420}"

kubectl apply -f k8s/04-loadgen/loadgen.yaml >/dev/null
kubectl wait --for=condition=ready pod/loadgen -n $NS --timeout=120s >/dev/null
kubectl cp traffic-generator/generate_traffic.py $NS/loadgen:/tmp/generate_traffic.py

CURVE=$(python3 - "$PATTERN" "$BASE" "$SECS" <<'PY'
import math, sys
pattern, base, secs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
n = max(1, secs // 10)
if pattern == "burst":
    v = [base if (i < n // 3 or i >= n // 3 + n // 3) else base * 3.3 for i in range(n)]
elif pattern == "diurnal":
    v = [base * (1.5 + 1.2 * math.sin(2 * math.pi * i / n - math.pi / 2)) for i in range(n)]
else:
    v = [base] * n
print(",".join(f"{x:.1f}" for x in v))
PY
)
echo "Sending '$PATTERN' traffic (base $BASE rps, $SECS s) to the gateway from inside the cluster ..."
kubectl exec -n $NS loadgen -- python -u /tmp/generate_traffic.py \
  --url http://gateway-service.$NS.svc.cluster.local:8000 --curve "$CURVE" --step-seconds 10
