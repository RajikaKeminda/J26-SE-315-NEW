"""
k8s_client.py

A real client for the Kubernetes API, using the official `kubernetes` Python
package. This reads REAL deployment state and applies REAL scaling actions -
`scale_replicas()` genuinely patches a Deployment's scale subresource, and the
Kubernetes control plane will genuinely create or terminate real pods.

Credentials: when running inside a pod (KUBERNETES_SERVICE_HOST is set) the
mounted ServiceAccount token is used; otherwise the local kubeconfig
(~/.kube/config - same credentials `kubectl` uses). See
k8s/03-autoscaler/ for the least-privilege Role the in-cluster mode needs.
"""

import os

from kubernetes import client, config as k8s_config

from src import config as app_config

_apps_v1 = None


def _load_client():
    """Returns a cached AppsV1Api, loading in-cluster or kubeconfig creds once."""
    global _apps_v1
    if _apps_v1 is None:
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            k8s_config.load_incluster_config()
        else:
            k8s_config.load_kube_config()
        _apps_v1 = client.AppsV1Api()
    return _apps_v1


def reset_client():
    """Drops the cached client (tests / credential rotation)."""
    global _apps_v1
    _apps_v1 = None


def get_replica_counts(services: list[str], namespace: str = None) -> dict[str, int]:
    """Real current replica count per Deployment, read straight from the K8s API.
    Services that cannot be read are omitted (caller decides how to treat them)."""
    namespace = namespace or app_config.NAMESPACE
    apps_v1 = _load_client()
    counts = {}
    for svc in services:
        try:
            dep = apps_v1.read_namespaced_deployment(name=svc, namespace=namespace)
            # .status.replicas can briefly be None right after a scaling event
            counts[svc] = dep.spec.replicas if dep.spec.replicas is not None else (dep.status.replicas or 0)
        except client.exceptions.ApiException as e:
            print(f"[k8s_client] failed to read {svc}: {e.status} {e.reason}")
    return counts


def scale_replicas(service: str, replica_delta: int, namespace: str = None) -> int:
    """
    Applies a REAL scaling action: reads the current desired replica count,
    adds replica_delta (clamped to [MIN_REPLICAS, MAX_REPLICAS]), and PATCHES
    the Deployment's scale subresource. Returns the new replica count.
    """
    namespace = namespace or app_config.NAMESPACE
    apps_v1 = _load_client()

    dep = apps_v1.read_namespaced_deployment(name=service, namespace=namespace)
    current = dep.spec.replicas if dep.spec.replicas is not None else 1
    new_count = max(app_config.MIN_REPLICAS, min(app_config.MAX_REPLICAS, current + replica_delta))

    if new_count == current:
        return current  # no-op, don't issue an empty patch

    patch = {"spec": {"replicas": new_count}}
    apps_v1.patch_namespaced_deployment_scale(name=service, namespace=namespace, body=patch)
    print(f"[k8s_client] scaled {service}: {current} -> {new_count} replicas")
    return new_count


if __name__ == "__main__":
    print(f"Reading live replica counts from namespace '{app_config.NAMESPACE}'...")
    counts = get_replica_counts(app_config.SERVICES)
    for svc, n in counts.items():
        print(f"  {svc:20s} {n} replicas")
