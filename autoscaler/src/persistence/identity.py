"""
identity.py

Which cluster is this? Resolved in the writer thread (it may call the cluster), in this order:
  1. AUTOSCALER_CLUSTER_NAME - what the operator calls it
  2. the UID of the kube-system namespace - a fingerprint no two clusters share (needs read access to that namespace; see
     k8s/03-autoscaler/rbac-cluster-identity.yaml). It is stored next to the name, and a name that later turns up with a different UID is refused
     (db.IdentityConflict) instead of silently mixing two clusters' history.
With neither, nothing is stored: the API server URL is NOT used, because over an SSH tunnel it is 127.0.0.1:<port> and the port changes
whenever the cluster restarts.
"""

from src import config
from src.persistence.db import IdentityError


def resolve(k8s=None, entry=None):
    info = {}
    if k8s is not None and hasattr(k8s, "cluster_identity"):
        try:
            info = k8s.cluster_identity() or {}
        except Exception:
            info = {}
    uid = info.get("kube_uid")
    name = config.CLUSTER_NAME or (f"cluster-{uid[:8]}" if uid else "")
    if not name:
        raise IdentityError("cluster identity unknown: set AUTOSCALER_CLUSTER_NAME (or --cluster-name), or allow reading the kube-system "
                           "namespace (k8s/03-autoscaler/rbac-cluster-identity.yaml)")
    return {"name": name, "namespace": config.NAMESPACE, "kube_uid": uid, "api_server": info.get("api_server"), "entry": entry}
