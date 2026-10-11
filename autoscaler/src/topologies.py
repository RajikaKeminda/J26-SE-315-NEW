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
    work_ms={},                       # measured online by LiveStateBuilder; DEFAULT_WORK_MS until then
)

TOPOLOGIES = {t.name: t for t in (DEMO6, TRAINTICKET14, TRAINTICKET_LIVE)}


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
