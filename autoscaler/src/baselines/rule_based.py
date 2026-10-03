"""
rule_based.py

The conventional autoscalers the learned agent is compared against, implemented
faithfully rather than as straw men. All of them share one interface:

    controller.decide(metrics) -> {service: {"dr": replica_delta, "cpu": cores, "mem": MiB}}

(absolute resource targets, so "no resize" is cpu == current cpu) and see exactly the
per-service metrics the agent sees - but none of them sees the call graph, which is
precisely the limitation this research argues against.

  HPABaseline        Kubernetes Horizontal Pod Autoscaler: per-service
                         desired = ceil(current * utilisation / target)
                     with the 10 % tolerance band, the default scale-up policy
                     (up to max(4 pods, 100 %) per period) and a scale-down
                     stabilisation window that takes the HIGHEST recommendation seen
                     recently. Replicas only.
  VPABaseline        Kubernetes Vertical Pod Autoscaler (recommender + updater):
                     target = p90 of recent usage + safety margin, applied (a pod
                     restart) only when it differs from the current allocation by more
                     than a threshold. Resources only - it never changes replicas,
                     just like the real VPA. As in the real VPA, usage is capped by the
                     CPU limit, so a throttled pod can only be grown one margin at a time.
  ThresholdBaseline  The classic rule-based scaler: utilisation > 80 % -> +1 replica,
                     < 30 % -> -1 (with a cooldown).
  StaticBaseline     No autoscaling. Control condition.
"""

import math
from collections import deque

import numpy as np

from src import config


def _noop(m):
    return {"dr": 0, "cpu": m["cpu"], "mem": m["mem"]}


class HPABaseline:
    """Per-service target-utilisation autoscaler, matching real HPA behaviour."""

    name = "hpa"

    def __init__(self, services, target_utilization=None, stabilization_steps=None):
        self.services = list(services)
        self.target = target_utilization or config.HPA_TARGET_UTILIZATION
        self.stabilization = stabilization_steps or config.HPA_SCALE_DOWN_STABILIZATION_STEPS
        self.recommendations = {s: deque(maxlen=self.stabilization) for s in self.services}

    def decide(self, metrics):
        out = {}
        for svc in self.services:
            m = metrics[svc]
            current = max(1, m["replicas"])
            ratio = m["utilization"] / self.target
            if abs(ratio - 1.0) <= config.HPA_TOLERANCE:
                desired = current
            else:
                desired = math.ceil(current * ratio)
            desired = max(config.MIN_REPLICAS, min(config.MAX_REPLICAS, desired))

            self.recommendations[svc].append(desired)
            if desired > current:
                # default scale-up behaviour: at most max(4 pods, 100 % of current) per period
                desired = min(desired, current + max(config.HPA_SCALE_UP_MAX_PODS, current))
            else:
                # scale-down is stabilised: act on the highest recent recommendation
                desired = min(current, max(self.recommendations[svc]))
            act = _noop(m)
            act["dr"] = int(desired - current)
            out[svc] = act
        return out

    def reset(self):
        self.recommendations = {s: deque(maxlen=self.stabilization) for s in self.services}


class VPABaseline:
    name = "vpa"

    def __init__(self, services):
        self.services = list(services)
        self.cpu_hist = {s: deque(maxlen=config.VPA_WINDOW_STEPS) for s in self.services}
        self.mem_hist = {s: deque(maxlen=config.VPA_WINDOW_STEPS) for s in self.services}

    def decide(self, metrics):
        out = {}
        for svc in self.services:
            m = metrics[svc]
            self.cpu_hist[svc].append(m["utilization"] * m["cpu"])                 # cores / pod
            self.mem_hist[svc].append(min(m["mem_utilization"], 1.5) * m["mem"])   # MiB / pod
            act = _noop(m)
            if len(self.cpu_hist[svc]) >= 3:
                cpu_rec = float(np.quantile(self.cpu_hist[svc], config.VPA_TARGET_PERCENTILE)) \
                    * (1.0 + config.VPA_SAFETY_MARGIN)
                mem_rec = float(np.quantile(self.mem_hist[svc], 0.95)) * (1.0 + config.VPA_SAFETY_MARGIN)
                cpu_rec = min(config.CPU_MAX, max(config.CPU_MIN, cpu_rec))
                mem_rec = min(config.MEM_MAX_MB, max(config.MEM_MIN_MB, mem_rec))
                if abs(cpu_rec - m["cpu"]) / m["cpu"] > config.VPA_MIN_CHANGE:
                    act["cpu"] = cpu_rec
                if abs(mem_rec - m["mem"]) / m["mem"] > config.VPA_MIN_CHANGE:
                    act["mem"] = mem_rec
            out[svc] = act
        return out

    def reset(self):
        self.__init__(self.services)


class ThresholdBaseline:
    name = "threshold"

    def __init__(self, services, cooldown_steps=3):
        self.services = list(services)
        self.cooldown = cooldown_steps
        self.last_change = {s: -999 for s in self.services}
        self.t = 0

    def decide(self, metrics):
        out = {}
        for svc in self.services:
            m = metrics[svc]
            act = _noop(m)
            if m["utilization"] > config.THRESHOLD_UP:
                act["dr"] = 1
            elif m["utilization"] < config.THRESHOLD_DOWN and self.t - self.last_change[svc] >= self.cooldown:
                act["dr"] = -1
            if act["dr"]:
                self.last_change[svc] = self.t
            out[svc] = act
        self.t += 1
        return out

    def reset(self):
        self.__init__(self.services, self.cooldown)


class StaticBaseline:
    name = "static"

    def __init__(self, services):
        self.services = list(services)

    def decide(self, metrics):
        return {s: _noop(metrics[s]) for s in self.services}

    def reset(self):
        pass


BASELINES = {"static": StaticBaseline, "threshold": ThresholdBaseline,
             "hpa": HPABaseline, "vpa": VPABaseline}
