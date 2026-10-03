#!/bin/bash
# Installs metrics-server (needed for the REAL Kubernetes HPA arm of the live experiment) and
# lets it talk to the kind/minikube kubelet, whose serving certificate is self-signed.
set -e
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl rollout status deployment/metrics-server -n kube-system --timeout=180s
echo "waiting for the metrics API ..."
for i in $(seq 1 30); do kubectl top nodes >/dev/null 2>&1 && break; sleep 5; done
kubectl top nodes
