"""
k8s_client.py

A real client for the Kubernetes API, using the official `kubernetes` Python package.
It reads REAL Deployment state and applies REAL actions:

  * scale_replicas()  patches a Deployment's scale subresource;
  * set_resources()   patches the container's CPU/memory requests+limits in the pod
                      template, which makes the Deployment controller perform a rolling
                      update (maxSurge=1, maxUnavailable=0 - capacity is never reduced).
                      This is the "recreate" mode of the Vertical Pod Autoscaler; in-place
                      pod resize would avoid the restart but is not available on every
                      cluster version, and the template would still hold the old values.

Resources are applied with requests == limits (Guaranteed QoS), so "the allocation" is
one unambiguous number per resource.

Credentials: when running inside a pod (KUBERNETES_SERVICE_HOST is set) the mounted
ServiceAccount token is used; otherwise the local kubeconfig. See k8s/03-autoscaler/
for the least-privilege Role the in-cluster mode needs.
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


def parse_cpu(q) -> float:
    """'250m' -> 0.25, '1' -> 1.0, 0.5 -> 0.5"""
    if q is None:
        return 0.0
    q = str(q)
    return float(q[:-1]) / 1000.0 if q.endswith("m") else float(q)


def parse_mem_mb(q) -> float:
    """'128Mi' -> 128, '1Gi' -> 1024, '134217728' (bytes) -> 128"""
    if q is None:
        return 0.0
    q = str(q)
    units = {"Ki": 1 / 1024, "Mi": 1.0, "Gi": 1024.0, "Ti": 1024.0 ** 2,
             "K": 1000 / 2**20, "M": 1e6 / 2**20, "G": 1e9 / 2**20}
    for suffix, mult in units.items():
        if q.endswith(suffix):
            return float(q[:-len(suffix)]) * mult
    return float(q) / 2**20


def _container(dep, service):
    for c in dep.spec.template.spec.containers:
        if c.name == service:
            return c
    return dep.spec.template.spec.containers[0]


def get_state(services: list[str], namespace: str = None) -> dict[str, dict]:
    """Real state per Deployment: desired + ready replicas and the CPU/memory limits of
    the pod template (what a new pod will get). Services that cannot be read are omitted
    (the caller decides how to treat them)."""
    namespace = namespace or app_config.NAMESPACE
    apps_v1 = _load_client()
    state = {}
    for svc in services:
        try:
            dep = apps_v1.read_namespaced_deployment(name=svc, namespace=namespace)
        except client.exceptions.ApiException as e:
            print(f"[k8s_client] failed to read {svc}: {e.status} {e.reason}")
            continue
        limits = (_container(dep, svc).resources.limits or {}) if _container(dep, svc).resources else {}
        state[svc] = {
            "replicas": dep.spec.replicas if dep.spec.replicas is not None else (dep.status.replicas or 0),
            "ready": dep.status.ready_replicas or 0,
            "cpu": parse_cpu(limits.get("cpu")) or app_config.CPU_DEFAULT,
            "mem": parse_mem_mb(limits.get("memory")) or float(app_config.MEM_DEFAULT_MB),
        }
    return state


def rollout_complete(services: list[str], namespace: str = None) -> bool:
    """True when every Deployment has finished rolling out: the controller has seen the latest
    spec, all replicas are updated and ready, and no pod is unavailable (used to start a
    measurement from a settled cluster)."""
    namespace = namespace or app_config.NAMESPACE
    apps_v1 = _load_client()
    for svc in services:
        dep = apps_v1.read_namespaced_deployment(name=svc, namespace=namespace)
        want, st = dep.spec.replicas or 0, dep.status
        if ((st.observed_generation or 0) < (dep.metadata.generation or 0)
                or (st.updated_replicas or 0) != want or (st.ready_replicas or 0) != want
                or (st.replicas or 0) != want or (st.unavailable_replicas or 0) > 0):
            return False
    return True


def get_replica_counts(services: list[str], namespace: str = None) -> dict[str, int]:
    """Desired replica count per Deployment (kept for tools/tests)."""
    return {s: v["replicas"] for s, v in get_state(services, namespace).items()}


def scale_replicas(service: str, replica_delta: int, namespace: str = None) -> int:
    """Reads the current desired replica count, adds replica_delta (clamped to
    [MIN_REPLICAS, MAX_REPLICAS]) and PATCHES the scale subresource. Returns the new count."""
    namespace = namespace or app_config.NAMESPACE
    apps_v1 = _load_client()

    dep = apps_v1.read_namespaced_deployment(name=service, namespace=namespace)
    current = dep.spec.replicas if dep.spec.replicas is not None else 1
    new_count = max(app_config.MIN_REPLICAS, min(app_config.MAX_REPLICAS, current + replica_delta))
    if new_count == current:
        return current                       # no-op, don't issue an empty patch
    apps_v1.patch_namespaced_deployment_scale(
        name=service, namespace=namespace, body={"spec": {"replicas": new_count}})
    print(f"[k8s_client] scaled {service}: {current} -> {new_count} replicas")
    return new_count


def set_resources(service: str, cpu: float, mem_mb: float, namespace: str = None) -> bool:
    """Sets the container's CPU/memory (requests == limits). Returns True if a patch was
    issued. Triggers a rolling update, so callers must rate-limit (SafetyGuard does)."""
    namespace = namespace or app_config.NAMESPACE
    apps_v1 = _load_client()
    dep = apps_v1.read_namespaced_deployment(name=service, namespace=namespace)
    c = _container(dep, service)
    limits = (c.resources.limits or {}) if c.resources else {}
    cpu_m, mem_mi = int(round(cpu * 1000)), int(round(mem_mb))
    if (abs(parse_cpu(limits.get("cpu")) - cpu_m / 1000.0) < 1e-6
            and abs(parse_mem_mb(limits.get("memory")) - mem_mi) < 0.5):
        return False
    res = {"cpu": f"{cpu_m}m", "memory": f"{mem_mi}Mi"}
    patch = {"spec": {"template": {"spec": {"containers": [
        {"name": c.name, "resources": {"requests": res, "limits": res}}]}}}}
    apps_v1.patch_namespaced_deployment(name=service, namespace=namespace, body=patch)
    print(f"[k8s_client] resized {service}: cpu={cpu_m}m mem={mem_mi}Mi (rolling update)")
    return True


def apply_action(service: str, action: dict, namespace: str = None) -> dict:
    """Applies one guard-approved action {"dr", "cpu", "mem"}; returns what was done."""
    done = {"scaled": False, "resized": False}
    if action.get("dr"):
        before = get_state([service], namespace).get(service, {}).get("replicas")
        after = scale_replicas(service, action["dr"], namespace)
        done["scaled"] = before is None or after != before
    done["resized"] = set_resources(service, action["cpu"], action["mem"], namespace)
    return done


if __name__ == "__main__":
    print(f"Reading live Deployment state from namespace '{app_config.NAMESPACE}'...")
    for svc, st in get_state(app_config.SERVICES).items():
        print(f"  {svc:20s} replicas={st['replicas']} ready={st['ready']} "
              f"cpu={st['cpu']:.2f} mem={st['mem']:.0f}MiB")
