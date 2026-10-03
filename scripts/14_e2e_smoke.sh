#!/bin/bash
# End-to-end smoke test against a RUNNING cluster (kind/minikube) that has had
# scripts/03_deploy_all.sh + 04_verify_deployment.sh applied. Verifies the whole chain:
#
#   real HTTP traffic -> services -> Prometheus metrics + Jaeger traces
#   -> autoscaler (in-cluster, RBAC-scoped) -> Deployment replicas change
#
# Exits non-zero with a message on the first failed check. Used by CI.
set -euo pipefail
NS=autoscaler-poc
cd "$(dirname "$0")/.."
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT
fail() { echo "FAIL: $*"; kubectl get pods -n $NS; kubectl logs -n $NS deploy/autoscaler --tail=40 || true; exit 1; }
ok() { echo "ok:   $*"; }

kubectl wait --for=condition=available --timeout=240s deployment --all -n $NS || fail "deployments not available"
ok "all deployments available"

kubectl port-forward svc/gateway-service 8000:8000 -n $NS >/dev/null 2>&1 & PIDS+=($!)
kubectl port-forward svc/prometheus 9090:9090 -n $NS >/dev/null 2>&1 & PIDS+=($!)
kubectl port-forward svc/jaeger 16686:16686 -n $NS >/dev/null 2>&1 & PIDS+=($!)
kubectl port-forward svc/autoscaler 8080:8080 -n $NS >/dev/null 2>&1 & PIDS+=($!)
sleep 5

curl -fsS localhost:8000/work | grep -q gateway-service || fail "gateway /work did not answer"
ok "gateway-service answers /work through the full call chain"

before=$(kubectl get deploy -n $NS -l 'app in (gateway-service,catalog-service,auth-service,inventory-service,pricing-service,user-service)' \
  -o jsonpath='{range .items[*]}{.spec.replicas}{"\n"}{end}' | awk '{s+=$1} END{print s}')

python3 -m pip install -q httpx numpy >/dev/null 2>&1 || true
python3 traffic-generator/generate_traffic.py --url http://localhost:8000 --pattern burst \
  --duration "${E2E_TRAFFIC_SECONDS:-75}" --base-rps "${E2E_BASE_RPS:-15}" > /tmp/e2e_traffic.log 2>&1 || fail "traffic generator failed"
ok "traffic generated ($(grep -c '^t=' /tmp/e2e_traffic.log) seconds)"

up=$(curl -fsS 'localhost:9090/api/v1/query?query=sum(http_requests_total)' | python3 -c "import sys,json;r=json.load(sys.stdin)['data']['result'];print(float(r[0]['value'][1]) if r else 0)")
python3 -c "import sys;sys.exit(0 if float('$up')>0 else 1)" || fail "Prometheus has no http_requests_total"
ok "Prometheus scraped $up requests"

svcs=$(curl -fsS localhost:16686/api/services | python3 -c "import sys,json;print(' '.join(sorted(json.load(sys.stdin)['data'])))")
echo "$svcs" | grep -q gateway-service || fail "Jaeger has no gateway-service traces ($svcs)"
ok "Jaeger traced: $svcs"

curl -fsS localhost:8080/healthz | grep -q ok || fail "autoscaler dashboard not healthy"
state=$(curl -fsS localhost:8080/api/state)
echo "$state" | python3 -c "import sys,json;s=json.load(sys.stdin);assert s['mode']=='live',s;print('ok:   autoscaler live, policy =',s['policy'],', error =',s['error'])" || fail "autoscaler state: $state"

sleep 20
after=$(kubectl get deploy -n $NS -l 'app in (gateway-service,catalog-service,auth-service,inventory-service,pricing-service,user-service)' \
  -o jsonpath='{range .items[*]}{.spec.replicas}{"\n"}{end}' | awk '{s+=$1} END{print s}')
echo "total replicas: before=$before after=$after"
[ "$after" != "$before" ] || fail "autoscaler never changed a replica count"
ok "autoscaler scaled the cluster ($before -> $after replicas)"
echo "E2E PASSED"
