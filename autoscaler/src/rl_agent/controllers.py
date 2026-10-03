"""
controllers.py

Uniform "decide" wrappers so the evaluation, the live loop and the dashboard can drive
a learned agent and a rule-based baseline through one interface:

    controller.decide(obs, breaker_open=False) -> {svc: {"dr", "cpu", "mem"}}

`obs` is the observation dict from the environment (or from the live state builder);
rule-based controllers read obs["metrics"], the agent reads the graph tensors.
"""

import numpy as np

from src import config
from src.baselines.rule_based import BASELINES, HPABaseline
from src.rl_agent.env import decode_action


class RuleController:
    def __init__(self, name, services, label=None, initial=None, **kwargs):
        self.name = name
        self.label = label or name
        self.initial = initial        # manifest sizing the cluster starts from (cpu/mem/replicas)
        self.impl = BASELINES[name](services, **kwargs)

    def decide(self, obs, breaker_open=False):
        return self.impl.decide(obs["metrics"])

    def reset(self):
        self.impl.reset()


class RLController:
    """A trained MO-TD3 agent acting at preference `w`. If the circuit breaker is open
    (SLO breached for N consecutive windows despite acting) control passes to HPA rules
    until the system recovers."""

    def __init__(self, agent, services, w=config.DEFAULT_PREFERENCE, fallback=True, name="mo_td3"):
        self.name = name
        self.agent, self.services = agent, list(services)
        self.w = np.asarray(w, dtype=np.float32)
        self.fallback = HPABaseline(services) if fallback else None
        self.replica_only = agent.flags["replica_only"]
        self.last_used_fallback = False

    def decide(self, obs, breaker_open=False):
        self.last_used_fallback = False
        if breaker_open and self.fallback is not None:
            self.last_used_fallback = True
            return self.fallback.decide(obs["metrics"])
        idx, cont = self.agent.act(obs, self.w)
        return decode_action(self.services, obs["metrics"], idx, cont, self.replica_only)

    def reset(self):
        if self.fallback:
            self.fallback.reset()
