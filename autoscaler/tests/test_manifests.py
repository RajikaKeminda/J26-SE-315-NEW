"""The Kubernetes manifests: generated from the topology, safe, and consistent with the autoscaler."""

import glob
import os
import subprocess
import sys

import pytest
import yaml

from src import config
from src.topologies import DEMO6

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
K8S = os.path.join(ROOT, "k8s")


def docs(path):
    return [d for d in yaml.safe_load_all(open(path)) if d]


def test_service_manifests_match_the_generator():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_manifests.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_generator_can_emit_the_trainticket_topology(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_manifests.py"),
                          "--topology", "trainticket14", "--out", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert len(list(tmp_path.glob("*.yaml"))) == 14
    dep = docs(str(tmp_path / "ts-basic.yaml"))[0]
    env = {e["name"]: e["value"] for e in dep["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["WORK_MS"] == "22" and "ts-station:8000|0.5" in env["DOWNSTREAM"]


@pytest.mark.parametrize("svc", DEMO6.services)
def test_every_service_is_safe_to_resize_and_overload(svc):
    dep, service = docs(os.path.join(K8S, "01-services", f"{svc}.yaml"))
    spec = dep["spec"]
    c = spec["template"]["spec"]["containers"][0]
    assert c["resources"]["requests"] == c["resources"]["limits"], "Guaranteed QoS: allocation is one number"
    assert spec["strategy"]["rollingUpdate"] == {"maxSurge": 1, "maxUnavailable": 0}, "a resize must not drop capacity"
    assert c["livenessProbe"]["failureThreshold"] >= 5 and c["livenessProbe"]["timeoutSeconds"] >= 3
    assert c["readinessProbe"]["timeoutSeconds"] >= 3
    ann = spec["template"]["metadata"]["annotations"]
    assert ann["prometheus.io/scrape"] == "true" and ann["prometheus.io/path"] == "/metrics"
    assert dep["metadata"]["namespace"] == config.NAMESPACE == service["metadata"]["namespace"]
    assert service["spec"]["selector"] == {"app": svc}


def test_default_allocation_is_inside_the_agents_action_range():
    c = docs(os.path.join(K8S, "01-services", "gateway-service.yaml"))[0]["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["cpu"] == f"{int(config.CPU_DEFAULT * 1000)}m"
    assert c["resources"]["limits"]["memory"] == f"{config.MEM_DEFAULT_MB}Mi"
    assert docs(os.path.join(K8S, "01-services", "gateway-service.yaml"))[0]["spec"]["replicas"] == config.INITIAL_REPLICAS


def test_autoscaler_rbac_is_namespaced_and_least_privilege():
    kinds = {d["kind"]: d for d in docs(os.path.join(K8S, "03-autoscaler", "rbac.yaml"))}
    assert "ClusterRole" not in kinds and "ClusterRoleBinding" not in kinds
    rules = kinds["Role"]["rules"]
    allowed = {(r["resources"][0], tuple(sorted(r["verbs"]))) for r in rules}
    assert allowed == {("deployments", ("get", "list", "patch")),
                       ("deployments/scale", ("get", "patch", "update")),
                       ("pods", ("get", "list"))}                    # read-only: the guard reads the age of a service's newest pod
    core = [r for r in rules if r["apiGroups"] == [""]]
    assert [r["resources"] for r in core] == [["pods"]] and set(core[0]["verbs"]) <= {"get", "list"}, "nothing core except read-only pods"
    assert all(r["apiGroups"] in (["apps"], [""]) for r in rules) and not any("secrets" in r["resources"] for r in rules)


def test_profile_budget_scales_with_the_observed_services():
    """The cost counts every service shown; a fixed cluster budget sized for the traced ones made every scale-up exceed it."""
    prof = dict(l.split("#")[0].strip().split("=", 1) for l in open(os.path.join(os.path.dirname(K8S), "profiles", "trainticket.env"))
                if "=" in l and not l.lstrip().startswith("#"))
    assert prof.get("OBSERVE_UNTRACED") == "1" and "CLUSTER_BUDGET_CORES" not in prof and float(prof["BUDGET_PER_SERVICE"]) >= 1.0


def test_autoscaler_deployment_is_a_single_replica_non_root_with_probes():
    dep = docs(os.path.join(K8S, "03-autoscaler", "autoscaler.yaml"))[0]
    assert dep["spec"]["replicas"] == 1 and dep["spec"]["strategy"]["type"] == "Recreate"
    c = dep["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["PROMETHEUS_URL"].startswith("http://prometheus.") and env["AUTOSCALER_NAMESPACE"] == config.NAMESPACE
    assert c["args"][:2] == ["--mode", "live"] and c["readinessProbe"]["httpGet"]["path"] == "/healthz"
    assert dep["spec"]["template"]["spec"]["serviceAccountName"] == "autoscaler"


def test_prometheus_scrapes_the_annotated_pods_and_jaeger_is_bounded():
    cfg = yaml.safe_load(docs(os.path.join(K8S, "02-observability", "prometheus-configmap.yaml"))[0]
                         ["data"]["prometheus.yml"])
    assert cfg["global"]["scrape_interval"] == "2s"
    keep = cfg["scrape_configs"][0]["relabel_configs"][0]
    assert keep["action"] == "keep" and "prometheus_io_scrape" in keep["source_labels"][0]
    jaeger = docs(os.path.join(K8S, "02-observability", "jaeger.yaml"))[0]
    env = {e["name"]: e["value"] for e in jaeger["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert int(env["MEMORY_MAX_TRACES"]) > 0


def test_dockerfile_makes_the_app_readable_for_the_non_root_user():
    text = open(os.path.join(ROOT, "autoscaler", "Dockerfile")).read()
    assert "chmod -R a+rX /app" in text and text.index("chmod -R a+rX /app") < text.index("USER 10001")


def test_every_k8s_yaml_parses():
    files = glob.glob(os.path.join(K8S, "**", "*.yaml"), recursive=True)
    assert len(files) >= 13
    for f in files:
        assert docs(f), f


def test_docker_compose_mirrors_the_cluster_allocation_and_topology():
    compose = yaml.safe_load(open(os.path.join(ROOT, "docker-compose.yml")))
    for svc in DEMO6.services:
        s = compose["services"][svc]
        assert s["cpus"] == config.CPU_DEFAULT and s["mem_limit"] == f"{config.MEM_DEFAULT_MB}m"
        raw = s["environment"]["DOWNSTREAM"]
        got = [(p.split("|")[0], float(p.split("|")[1])) for p in raw.split(",") if p]
        assert got == [(f"http://{c}:8000", w) for c, w in DEMO6.call_graph[svc]]
        assert s["environment"]["SERVICE_NAME"] == svc
    prom = yaml.safe_load(open(os.path.join(ROOT, "deploy", "compose", "prometheus.yml")))
    targets = {t["targets"][0].split(":")[0] for t in prom["scrape_configs"][0]["static_configs"]}
    assert targets == set(DEMO6.services)
