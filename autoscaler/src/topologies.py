"""
topologies.py

Named service topologies the whole stack can run on. A topology is pure data:
which services exist, who calls whom (and what fraction of the caller's traffic
goes to each callee), and how much CPU time one request costs in each service.

  demo6          the 6-service application deployed under k8s/01-services/.
                 This is the topology measured on the real cluster.
  trainticket14  a 14-service application with the same call structure as the
                 core booking path of the Train-Ticket benchmark (the proposal's
                 benchmark): UI/gateway fan-out, a travel -> ticketinfo -> basic ->
                 station/price chain, a preserve (booking) subtree, and shared
                 callees (station is called from three places).

                 NOTE ON HONESTY: this is Train-Ticket's *shape*, run on the same
                 generic Python service image - NOT the ~40-service Java/Spring +
                 MongoDB/MySQL application, which needs far more memory than a
                 laptop-class cluster has. It is used (a) as an unseen, larger
                 topology to test that the shared policy generalises beyond the
                 6-service training distribution, and (b) can be deployed with
                 `python scripts/gen_manifests.py --topology trainticket14`.

All call graphs are acyclic by construction (the critical-path search and the
latency propagation both rely on that).
"""

from dataclasses import dataclass, field

DEFAULT_WORK_MS = 15.0


@dataclass(frozen=True)
class Topology:
    name: str
    services: list
    entry: str
    call_graph: dict                       # {caller: [(callee, traffic_share), ...]}
    work_ms: dict = field(default_factory=dict)   # {service: CPU ms per request}
    load_scale: float = 1.0                       # multiplies the base external traffic (bottleneck scenarios, see bottleneck_scenario)
    initial_cpu: dict = field(default_factory=dict)   # {service: starting CPU limit}; the others start at CONFIG.CPU_DEFAULT

    def work(self, service: str) -> float:
        return self.work_ms.get(service, DEFAULT_WORK_MS)


DEMO6 = Topology(
    name="demo6",
    services=["gateway-service", "catalog-service", "auth-service",
              "inventory-service", "pricing-service", "user-service"],
    entry="gateway-service",
    call_graph={
        "gateway-service":   [("catalog-service", 0.6), ("auth-service", 0.4)],
        "catalog-service":   [("inventory-service", 0.5), ("pricing-service", 0.5)],
        "auth-service":      [("user-service", 1.0)],
        "inventory-service": [],
        "pricing-service":   [],
        "user-service":      [],
    },
)

TRAINTICKET14 = Topology(
    name="trainticket14",
    services=["ts-gateway", "ts-travel", "ts-preserve", "ts-order", "ts-route",
              "ts-auth", "ts-ticketinfo", "ts-basic", "ts-station", "ts-price",
              "ts-contacts", "ts-seat", "ts-security", "ts-user"],
    entry="ts-gateway",
    call_graph={
        "ts-gateway":    [("ts-travel", 0.35), ("ts-preserve", 0.20), ("ts-order", 0.15),
                          ("ts-route", 0.10), ("ts-auth", 0.20)],
        "ts-travel":     [("ts-ticketinfo", 1.0)],
        "ts-ticketinfo": [("ts-basic", 1.0)],
        "ts-basic":      [("ts-station", 0.5), ("ts-price", 0.5)],
        "ts-preserve":   [("ts-contacts", 0.40), ("ts-seat", 0.35), ("ts-security", 0.25)],
        "ts-order":      [("ts-station", 1.0)],
        "ts-route":      [("ts-station", 1.0)],
        "ts-auth":       [("ts-user", 1.0)],
        "ts-station":    [], "ts-price": [], "ts-contacts": [],
        "ts-seat":       [], "ts-security": [], "ts-user": [],
    },
    work_ms={"ts-gateway": 10.0, "ts-travel": 18.0, "ts-preserve": 25.0, "ts-order": 15.0,
             "ts-route": 12.0, "ts-auth": 10.0, "ts-ticketinfo": 12.0, "ts-basic": 22.0,
             "ts-station": 12.0, "ts-price": 10.0, "ts-contacts": 12.0, "ts-seat": 20.0,
             "ts-security": 15.0, "ts-user": 10.0},
)

# The REAL Train-Ticket (46 Java/Spring services, minikube on a separate machine). NOTHING is declared except
# the entry point: the services and the call graph are learned from live Jaeger traces every control step
# (topology_intel/live_topology.py), and which of them the agent may act on is set by CONTROLLED_SERVICES.
TRAINTICKET_LIVE = Topology(
    name="trainticket-live",
    services=["ts-gateway-service"],
    entry="ts-gateway-service",
    call_graph={"ts-gateway-service": []},
    # CPU ms per request, MEASURED on the real cluster (2026-10-11, ~12 rps booking load, median of steady
    # windows of cAdvisor CPU / traced request rate; see docs/TRAINTICKET.md). They are priors only: the live
    # estimator refines them within 0.6x-1.8x. Services not listed use DEFAULT_WORK_MS.
    work_ms={
        "ts-basic-service": 17.1,
        "ts-contacts-service": 8.6,
        "ts-gateway-service": 14.4,
        "ts-order-service": 81.6,
        "ts-route-service": 13.0,
        "ts-seat-service": 18.4,
        "ts-station-service": 6.6,
        "ts-train-service": 10.0,
        "ts-travel-service": 24.9,
    },
)

# The same application as a FIXED topology for the simulator (training and evaluation of the policy): the call graph
# LEARNED from 253 real traces of the live cluster (tests/fixtures/trainticket_traces.json, 2026-10-11) and the CPU cost per
# request measured there. The simulator routes each external request along ONE chain chosen by these shares, which is exact
# for the gateway (one downstream per request type) and an approximation for services that call several in sequence.
TRAINTICKET_SIM = Topology(
    name="trainticket-sim",
    services=["ts-gateway-service", "ts-basic-service", "ts-contacts-service", "ts-order-service", "ts-preserve-service",
              "ts-route-service", "ts-seat-service", "ts-security-service", "ts-station-service", "ts-train-service",
              "ts-travel-service"],
    entry="ts-gateway-service",
    call_graph={
        "ts-gateway-service": [("ts-basic-service", 0.0738), ("ts-contacts-service", 0.2685), ("ts-order-service", 0.2953),
                               ("ts-preserve-service", 0.1074), ("ts-seat-service", 0.0738), ("ts-security-service", 0.0738),
                               ("ts-travel-service", 0.1074)],
        "ts-basic-service": [("ts-route-service", 0.2682), ("ts-station-service", 0.4636), ("ts-train-service", 0.2682)],
        "ts-preserve-service": [(c, 1 / 6) for c in ("ts-basic-service", "ts-contacts-service", "ts-order-service",
                                                     "ts-seat-service", "ts-security-service", "ts-travel-service")],
        "ts-seat-service": [("ts-order-service", 1.0)],
        "ts-security-service": [("ts-order-service", 1.0)],
        "ts-travel-service": [("ts-basic-service", 1 / 3), ("ts-seat-service", 2 / 3)],
        "ts-contacts-service": [], "ts-order-service": [], "ts-route-service": [], "ts-station-service": [], "ts-train-service": [],
    },
    work_ms={"ts-gateway-service": 14.4, "ts-basic-service": 17.1, "ts-contacts-service": 8.6, "ts-order-service": 81.6,
             "ts-preserve-service": 15.0, "ts-route-service": 13.0, "ts-seat-service": 18.4, "ts-security-service": 18.7,
             "ts-station-service": 6.6, "ts-train-service": 10.0, "ts-travel-service": 24.9},
)

def visit_ratios(topo: Topology) -> dict:
    """Expected calls per entry request, per service (the simulator forwards a caller's traffic to its callees in proportion to the shares)."""
    ratio = {s: 0.0 for s in topo.services}
    ratio[topo.entry] = 1.0
    indeg = {s: 0 for s in topo.services}
    for targets in topo.call_graph.values():
        for c, _ in targets:
            if c in indeg:
                indeg[c] += 1
    queue = [s for s, d in indeg.items() if d == 0]
    while queue:
        n = queue.pop()
        for c, share in topo.call_graph.get(n, []):
            if c in ratio:
                ratio[c] += ratio[n] * share
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
    return ratio


def shift_mix(topo: Topology, hot: str, k: float) -> Topology:
    """The same application under a different REQUEST MIX: every call that leads towards `hot` gets (1 + k) times its weight (the others are renormalised),
    so more of the traffic ends up there - the way a search-heavy or booking-heavy load moves the bottleneck from one service to another."""
    leads = {hot}
    changed = True
    while changed:
        changed = False
        for caller, targets in topo.call_graph.items():
            if caller not in leads and any(c in leads for c, _ in targets):
                leads.add(caller)
                changed = True
    graph = {}
    for caller, targets in topo.call_graph.items():
        if not targets or caller not in leads:
            graph[caller] = list(targets)
            continue
        w = [(c, share * ((1.0 + k) if c in leads else 1.0)) for c, share in targets]
        total = sum(x for _, x in w) / max(sum(share for _, share in targets), 1e-12)
        graph[caller] = [(c, x / total) for c, x in w]
    from dataclasses import replace
    return replace(topo, call_graph=graph)


def bottleneck_scenario(topo: Topology, rng, base_rps: float, hot=None, k_range=(3.0, 8.0), demand_range=(0.25, 1.6), max_scale=6.0,
                        others_max_demand=None, start_cpu=0.1, tries=30):
    """A scenario in which `hot` (one random service when not given) IS the bottleneck, whatever service that is: the request mix is shifted towards it,
    the traffic is scaled until it needs `demand_range` cores - but never so far that ANY other service would need more than `others_max_demand` cores (then
    that one would be the bottleneck) - and it starts with a small CPU limit (`start_cpu`), so it is saturated from the first step. The fix exists
    (up to 2 replicas of 1 core each cover the demand) and the other services stay well below their limits, so the agent can only be right by adding
    capacity to `hot`. -> (Topology, hot), or (None, hot) when this service cannot be made the only bottleneck."""
    from dataclasses import replace
    from src import config
    if others_max_demand is None:                       # 70 % of what every other service starts with (replicas x CPU limit)
        others_max_demand = 0.7 * config.INITIAL_REPLICAS * config.CPU_DEFAULT
    ratios0 = visit_ratios(topo)
    candidates = [s for s in topo.services if ratios0.get(s, 0.0) > 1e-6]
    for _ in range(1 if hot else tries):
        h = hot or candidates[int(rng.integers(len(candidates)))]
        shifted = shift_mix(topo, h, float(rng.uniform(*k_range)))
        v = visit_ratios(shifted)
        per_rps = {s: v[s] * shifted.work(s) / 1000.0 for s in shifted.services}          # cores each service needs per external request per second
        other_max = max(per_rps[s] for s in shifted.services if s != h)
        want = float(rng.uniform(*demand_range)) / max(base_rps * per_rps[h], 1e-12)
        scale = min(max(want, 1.0), max_scale, others_max_demand / max(base_rps * other_max, 1e-12))
        if base_rps * scale * per_rps[h] >= demand_range[0] and scale >= 0.2:
            return replace(shifted, name=f"{topo.name}+bn:{h}", load_scale=scale, initial_cpu={h: start_cpu}), h
        if hot:
            break
    return None, hot


def hot_variant(topo: Topology, rng, base_rps: float, demand_range=(0.7, 1.6), max_factor=8.0, hot=None):
    """The same application with ONE service turned into the bottleneck: its CPU cost per request is raised until, at `base_rps`, it needs
    `demand_range` cores (never lowered, never more than `max_factor` times); `hot` names the service, otherwise one is drawn. Training on these variants, with a different service each time,
    makes the agent read where the load is instead of memorising which service was heavy. -> (Topology, hot service)."""
    ratio = visit_ratios(topo)
    candidates = [s for s in topo.services if s != topo.entry and ratio.get(s, 0.0) > 1e-6]
    hot = hot or candidates[int(rng.integers(len(candidates)))]
    demand_now = base_rps * ratio[hot] * topo.work(hot) / 1000.0
    want = float(rng.uniform(*demand_range))
    factor = min(max(want / max(demand_now, 1e-9), 1.0), max_factor)
    work = {s: topo.work(s) for s in topo.services}
    work[hot] = work[hot] * factor
    from dataclasses import replace
    return replace(topo, name=f"{topo.name}+hot:{hot}", work_ms=work), hot


TOPOLOGIES = {t.name: t for t in (DEMO6, TRAINTICKET14, TRAINTICKET_LIVE, TRAINTICKET_SIM)}


def get_topology(name: str) -> Topology:
    try:
        return TOPOLOGIES[name]
    except KeyError:
        raise KeyError(f"unknown topology {name!r}; choose from {sorted(TOPOLOGIES)}") from None


def validate(topology: Topology) -> None:
    """Raises ValueError unless the topology is a well-formed DAG rooted at `entry`."""
    services = set(topology.services)
    if topology.entry not in services:
        raise ValueError("entry service is not in services")
    for caller, targets in topology.call_graph.items():
        if caller not in services:
            raise ValueError(f"unknown caller {caller}")
        for callee, share in targets:
            if callee not in services:
                raise ValueError(f"unknown callee {callee}")
            if not (0.0 < share <= 1.0):
                raise ValueError(f"bad share {caller}->{callee}: {share}")
        if targets and abs(sum(s for _, s in targets) - 1.0) > 1e-6:
            raise ValueError(f"{caller}: outgoing shares must sum to 1.0")
    # acyclic + everything reachable from the entry
    indeg = {s: 0 for s in services}
    for targets in topology.call_graph.values():
        for c, _ in targets:
            indeg[c] += 1
    queue, order = [s for s, d in indeg.items() if d == 0], []
    while queue:
        n = queue.pop()
        order.append(n)
        for c, _ in topology.call_graph.get(n, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != len(services):
        raise ValueError("call graph has a cycle")
    seen, stack = {topology.entry}, [topology.entry]
    while stack:
        for c, _ in topology.call_graph.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    if seen != services:
        raise ValueError(f"services unreachable from entry: {sorted(services - seen)}")
