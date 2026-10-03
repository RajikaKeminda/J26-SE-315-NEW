#!/bin/bash
# End-to-end smoke test against a RUNNING cluster (kind/minikube) that has had
# scripts/03_deploy_all.sh + 04_verify_deployment.sh applied. Verifies the whole chain:
#
#   real HTTP traffic -> services -> Prometheus metrics + Jaeger traces
#   -> autoscaler (in-cluster, RBAC-scoped) -> Deployments change (pods and/or CPU/memory)
#
# Exits non-zero with a message on the first failed check. Used by CI.
set -euo pipefail
NS=autoscaler-poc
cd "$(dirname "$0")/.."
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT
fail() { echo "FAIL: $*"; kubectl get pods -n $NS; kubectl logs -n $NS deploy/autoscaler --tail=60 || true; exit 1; }
ok() { echo "ok:   $*"; }

kubectl wait --for=condition=available --timeout=300s deployment --all -n $NS || fail "deployments not available"
ok "all deployments available"

# Traffic comes from INSIDE the cluster: a kubectl port-forward binds to ONE pod and dies when the
# autoscaler rolls that pod (it resizes Deployments, which is exactly what this test provokes).
kubectl apply -f k8s/04-loadgen/loadgen.yaml >/dev/null
kubectl wait --for=condition=ready pod/loadgen -n $NS --timeout=120s >/dev/null
kubectl cp traffic-generator/generate_traffic.py $NS/loadgen:/tmp/generate_traffic.py
GATEWAY=http://gateway-service.$NS.svc.cluster.local:8000
kubectl port-forward svc/prometheus 9090:9090 -n $NS >/dev/null 2>&1 & PIDS+=($!)
kubectl port-forward svc/jaeger 16686:16686 -n $NS >/dev/null 2>&1 & PIDS+=($!)
kubectl port-forward svc/autoscaler 8080:8080 -n $NS >/dev/null 2>&1 & PIDS+=($!)
sleep 5

# the autoscaler may be mid-rollout (it resizes Deployments), so allow a minute for the call chain to answer
answered=0
for i in $(seq 1 12); do
  if kubectl exec -n $NS loadgen -- python -c "import httpx,sys;r=httpx.get('$GATEWAY/work',timeout=10);sys.exit(0 if r.status_code==200 and r.json()['service']=='gateway-service' else 1)" 2>/dev/null; then answered=1; break; fi
  sleep 5
done
[ "$answered" = 1 ] || fail "gateway /work did not answer"
ok "gateway-service answers /work through the full call chain"

snapshot() {   # pods + CPU + memory of every service, as one comparable string
  kubectl get deploy -n $NS -l 'app in (gateway-service,catalog-service,auth-service,inventory-service,pricing-service,user-service)' \
    -o jsonpath='{range .items[*]}{.metadata.name}={.spec.replicas}/{.spec.template.spec.containers[0].resources.limits.cpu}/{.spec.template.spec.containers[0].resources.limits.memory}{"\n"}{end}' | sort
}
before=$(snapshot)

# a burst: 12 rps -> 40 rps -> 12 rps, one value per 10 s control step (open loop)
kubectl exec -n $NS loadgen -- python /tmp/generate_traffic.py --url $GATEWAY \
  --curve "${E2E_CURVE:-12,12,40,40,40,40,12,12}" --step-seconds 10 > /tmp/e2e_traffic.log 2>&1 || fail "traffic generator failed"
ok_requests=$(awk '/^step/ {for(i=1;i<=NF;i++) if($i=="ok") s+=$(i+1)} END {print s+0}' /tmp/e2e_traffic.log)
[ "$ok_requests" -gt 200 ] || fail "only $ok_requests requests succeeded - see /tmp/e2e_traffic.log"
ok "traffic generated ($(grep -c '^step' /tmp/e2e_traffic.log) control steps, $ok_requests requests served)"

q() { curl -fsSG --max-time 10 localhost:9090/api/v1/query --data-urlencode "query=$1" | python3 -c "import sys,json;r=json.load(sys.stdin)['data']['result'];print(float(r[0]['value'][1]) if r else 0)"; }
positive() { python3 -c "import sys;sys.exit(0 if float('$1')>0 else 1)"; }
positive "$(q 'sum(http_requests_total)')" || fail "Prometheus has no http_requests_total"
positive "$(q 'sum(rate(process_cpu_seconds_total[1m]))')" || fail "Prometheus has no process_cpu_seconds_total (real CPU signal)"
positive "$(q 'sum(service_own_duration_seconds_count)')" || fail "Prometheus has no service_own_duration_seconds (own-latency signal)"
ok "Prometheus scraped requests, CPU usage and own-latency histograms"

svcs=$(curl -fsS --max-time 10 localhost:16686/api/services | python3 -c "import sys,json;print(' '.join(sorted(json.load(sys.stdin)['data'])))")
echo "$svcs" | grep -q gateway-service || fail "Jaeger has no gateway-service traces ($svcs)"
ok "Jaeger traced: $svcs"

curl -fsS --max-time 10 localhost:8080/healthz | grep -q ok || fail "autoscaler dashboard not healthy"
state=$(curl -fsS --max-time 10 localhost:8080/api/state)
echo "$state" | python3 -c "import sys,json;s=json.load(sys.stdin);assert s['mode']=='live',s;assert not s['error'],s;print('ok:   autoscaler live, policy =',s['policy'],', controller =',s['controller'])" || fail "autoscaler state: $state"
curl -fsS --max-time 10 localhost:8080/api/calibration | python3 -c "import sys,json;assert json.load(sys.stdin)['available']" || fail "calibration report not served"

sleep 15
after=$(snapshot)
echo "before: $(echo $before | tr '\n' ' ')"
echo "after:  $(echo $after | tr '\n' ' ')"
[ "$after" != "$before" ] || fail "autoscaler never changed a replica count or a resource size"
ok "autoscaler acted on the cluster"
echo "E2E PASSED"
