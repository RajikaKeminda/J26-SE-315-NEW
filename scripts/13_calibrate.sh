#!/bin/bash
# Re-measures the real service on YOUR cluster and re-fits the simulator to it.
#
# Sweeps the CPU limit of a single leaf replica (0.1 / 0.25 / 0.5 / 1.0 core) and, at each, offers
# open-loop Poisson load from an in-cluster pod at 10-115 % of the expected capacity. The
# measurements land in calibration/, then src/simulation/calibration.py fits the service model
# and writes autoscaler/src/simulation/sim_params.json (+ a fit report).
#
# Needs: the cluster up and images loaded (scripts/00..02). Takes ~10 minutes.
set -e
cd "$(dirname "$0")/.."
NS=calib
kubectl create namespace $NS --dry-run=client -o yaml | kubectl apply -f - >/dev/null

kubectl apply -n $NS -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata: {name: leaf}
spec:
  replicas: 1
  selector: {matchLabels: {app: leaf}}
  template:
    metadata:
      labels: {app: leaf}
    spec:
      containers:
      - name: leaf
        image: k8s-autoscaler/generic-service:latest
        imagePullPolicy: IfNotPresent
        env:
        - {name: SERVICE_NAME, value: leaf}
        - {name: DOWNSTREAM, value: ""}
        - {name: WORK_MS, value: "15"}
        - {name: MAX_CONCURRENCY, value: "4"}
        - {name: TRACE_SAMPLE_RATIO, value: "0"}
        - {name: JAEGER_OTLP_ENDPOINT, value: "http://127.0.0.1:9/v1/traces"}
        resources:
          requests: {cpu: "250m", memory: "128Mi"}
          limits: {cpu: "250m", memory: "128Mi"}
        readinessProbe: {httpGet: {path: /health, port: 8000}, periodSeconds: 2}
---
apiVersion: v1
kind: Service
metadata: {name: leaf}
spec:
  selector: {app: leaf}
  ports: [{port: 8000}]
---
apiVersion: v1
kind: Pod
metadata: {name: loadgen}
spec:
  terminationGracePeriodSeconds: 1
  containers:
  - name: lg
    image: k8s-autoscaler/generic-service:latest
    imagePullPolicy: IfNotPresent
    command: ["sleep","infinity"]
    resources: {requests: {cpu: "500m", memory: "128Mi"}, limits: {cpu: "1500m", memory: "256Mi"}}
YAML
kubectl wait -n $NS --for=condition=ready pod/loadgen --timeout=120s
kubectl cp traffic-generator/calibrate.py $NS/loadgen:/tmp/calibrate.py
mkdir -p calibration

for cpu in 100m 250m 500m 1000m; do
  kubectl set resources -n $NS deploy/leaf --requests=cpu=$cpu,memory=128Mi --limits=cpu=$cpu,memory=128Mi
  kubectl rollout status -n $NS deploy/leaf --timeout=120s >/dev/null
  sleep 8
  c=$(python3 -c "print(float('${cpu%m}')/1000)")
  rates=$(python3 -c "c=$c;print(' '.join(f'{c*54*f:.1f}' for f in [0.1,0.25,0.4,0.55,0.7,0.8,0.9,1.0,1.15]))")
  echo "=== cpu limit $cpu ; offered rates: $rates"
  kubectl exec -n $NS loadgen -- python -u /tmp/calibrate.py --url http://leaf.$NS.svc:8000 \
      --rates $rates --duration 15 --label cpu$cpu --out /tmp/calib_$cpu.json
  kubectl cp $NS/loadgen:/tmp/calib_$cpu.json calibration/leaf_cpu$cpu.json
done
kubectl delete namespace $NS --wait=false >/dev/null

echo
cd autoscaler && python3 -m src.simulation.calibration --write
