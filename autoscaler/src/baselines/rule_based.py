"""
rule_based.py

Faithful reimplementation of Kubernetes HPA's actual scaling logic, used as
the primary comparison baseline in the ablation study.

Real HPA algorithm (from the Kubernetes docs):
    desiredReplicas = ceil(currentReplicas * currentMetric / targetMetric)
with a stabilization window to prevent thrashing on scale-down.

Crucially, HPA looks at ONE metric for ONE service in isolation - it has no
concept of the call graph, downstream dependencies, or end-to-end latency.
That isolation is precisely what this research argues is the limitation, so
reproducing it faithfully (rather than strawmanning it) matters for the
comparison to mean anything.
"""

import math

from src import config


class HPABaseline:
    """Per-service target-utilization autoscaler, matching real HPA behaviour."""

    def __init__(self, services, target_utilization=None, stabilization_steps=None):
        self.services = services
        self.target = target_utilization or config.HPA_TARGET_UTILIZATION
        self.stabilization = stabilization_steps or config.HPA_STABILIZATION_STEPS
        self.scale_down_history = {s: [] for s in services}

    def decide(self, metrics):
        """Returns {service: replica_delta}. Same interface as the RL agent."""
        deltas = {}
        for svc in self.services:
            m = metrics[svc]
            current = max(1, m["replicas"])
            util = m["utilization"]

            desired = math.ceil(current * (util / self.target)) if util > 0 else config.MIN_REPLICAS
            desired = max(config.MIN_REPLICAS, min(config.MAX_REPLICAS, desired))

            if desired > current:
                # scale up immediately (HPA does this)
                self.scale_down_history[svc].clear()
                deltas[svc] = min(1, desired - current)
            elif desired < current:
                # scale down only after the stabilization window agrees
                self.scale_down_history[svc].append(desired)
                if len(self.scale_down_history[svc]) >= self.stabilization:
                    deltas[svc] = -1
                    self.scale_down_history[svc].clear()
                else:
                    deltas[svc] = 0
            else:
                self.scale_down_history[svc].clear()
                deltas[svc] = 0
        return deltas

    def reset(self):
        self.scale_down_history = {s: [] for s in self.services}


class StaticBaseline:
    """No autoscaling at all - fixed replicas. Control condition."""

    def __init__(self, services):
        self.services = services

    def decide(self, metrics):
        return {s: 0 for s in self.services}

    def reset(self):
        pass
