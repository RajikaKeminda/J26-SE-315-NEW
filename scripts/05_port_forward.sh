#!/bin/bash
# Opens port-forwards for everything the autoscaler, traffic generator, and
# you (for the dashboards) need to reach from outside the cluster:
#   gateway-service  -> localhost:8000   (traffic generator target)
#   prometheus       -> localhost:9090   (Prometheus UI + local autoscaler reads)
#   jaeger           -> localhost:16686  (Jaeger UI + local autoscaler reads)
#   autoscaler       -> localhost:8080   (control-room dashboard, if deployed in-cluster)
#
# Stays alive until you Ctrl+C, then cleans up every port-forward.
NAMESPACE="autoscaler-poc"

cleanup() {
  echo
  echo "Stopping port-forwards..."
  kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT

kubectl port-forward svc/gateway-service 8000:8000 -n "$NAMESPACE" &
kubectl port-forward svc/prometheus 9090:9090 -n "$NAMESPACE" &
kubectl port-forward svc/jaeger 16686:16686 -n "$NAMESPACE" &
if kubectl get svc autoscaler -n "$NAMESPACE" >/dev/null 2>&1; then
  kubectl port-forward svc/autoscaler 8080:8080 -n "$NAMESPACE" &
  echo "  Dashboard       -> http://localhost:8080  (in-cluster autoscaler)"
fi

echo "Port-forwards active:"
echo "  gateway-service -> http://localhost:8000"
echo "  Prometheus UI   -> http://localhost:9090"
echo "  Jaeger UI       -> http://localhost:16686"
echo
echo "Leave this running. Press Ctrl+C to stop all port-forwards."
wait
