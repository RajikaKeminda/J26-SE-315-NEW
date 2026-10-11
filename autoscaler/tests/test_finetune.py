"""Fine-tuning: warm start from a checkpoint, wider topology range, target topologies, custom validation set."""

import numpy as np
import torch

from src.rl_agent.mo_td3 import MOTD3Agent
from src.rl_agent.trainer import train_agent
from src.topologies import TRAINTICKET_SIM, validate


def test_warm_start_copies_actor_critics_and_their_targets(tmp_path):
    src = MOTD3Agent(seed=1)
    path = str(tmp_path / "a.pt")
    src.save(path)
    dst = MOTD3Agent(seed=2)
    assert not torch.equal(next(iter(dst.actor.parameters())), next(iter(src.actor.parameters())))
    dst.warm_start(path)
    for net_a, net_b in ((dst.actor, src.actor_t), (dst.actor_t, src.actor_t), (dst.critic1, src.critic1),
                         (dst.critic1_t, src.critic1), (dst.critic2_t, src.critic2)):
        for p, q in zip(net_a.parameters(), net_b.parameters()):
            assert torch.equal(p, q)


def test_train_agent_fine_tunes_on_target_topologies_and_larger_graphs(tmp_path):
    src = MOTD3Agent(seed=0)
    path = str(tmp_path / "base.pt")
    src.save(path)
    agent, state, meta = train_agent("mo_td3_full", episodes=3, seed=3, demo_episodes=0, warmup_episodes=1,
                                     updates_per_episode=1, val_every=3, verbose=False, init_from=path,
                                     sizes=(6, 20), target_topologies=[TRAINTICKET_SIM], target_prob=1.0,
                                     val_topologies=[TRAINTICKET_SIM])
    assert meta["init_from"] == path and "6-20 services" in meta["topology_training"]
    assert "target topologies" in meta["topology_training"]
    assert meta["history"] and np.isfinite(meta["history"][-1]["val_scalarized"])


def test_trainticket_sim_topology_is_a_valid_dag_with_measured_costs():
    validate(TRAINTICKET_SIM)
    assert TRAINTICKET_SIM.work("ts-order-service") > 4 * TRAINTICKET_SIM.work("ts-station-service")


def test_simulator_accepts_per_service_initial_allocation_and_env_randomises_it():
    import numpy as np
    from src import config
    from src.rl_agent.env import AutoscalerEnv
    from src.simulation.simulator import Simulator
    sim = Simulator(TRAINTICKET_SIM, rng=np.random.default_rng(0))
    sim.reset(cpu={"ts-order-service": 0.3})
    assert sim.state["ts-order-service"].cpu == 0.3 and sim.state["ts-gateway-service"].cpu == config.CPU_DEFAULT
    env = AutoscalerEnv(topology=TRAINTICKET_SIM, seed=1)
    env.randomize_initial = True
    seen = set()
    for _ in range(12):
        env.reset("normal")
        seen |= {round(env.sim.state[s].cpu, 2) for s in env.services}
    assert len(seen) >= 3                                          # a heterogeneous cluster, not one value
    env.randomize_initial = False
    env.reset("normal")
    assert {env.sim.state[s].cpu for s in env.services} == {config.CPU_DEFAULT}
