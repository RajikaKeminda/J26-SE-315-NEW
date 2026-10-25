# Commits grouped by responsible person (the four components of the TAF, J26-SE-315)

**How to read this.** Every commit in the repository has the same git author (Rajika Keminda), so git itself says nothing about who is responsible for what. The grouping below is by **subject
matter**: which of the four components of the Topic Assessment Form a commit's code belongs to (the owners are those named in `docs/TAF_TRACEABILITY.md`). It is my reading of what each commit
changed, checked against the files it touched; where a commit spans components it is listed under each of them with the role it plays there ("main" = the bulk of the change, "part" = a smaller part).
A few commits belong to none of the four (the safety guard, the dashboard, the history database, demo scripts, documentation); they are listed last, because the TAF assigns them to nobody.
The merge commits (`db3ea4b`, `b1d46b8`, `6cfe308`) are not attributed.

| # | Component | Responsible | Where the code lives |
|---|---|---|---|
| 1 | Environment modeling and state representation | **Dilhara W. L.** | `simulation/`, `topologies.py`, `inference/live_state.py`, `auto_profile.py`, `data_collection/` (metrics), `k8s_action/`, `k8s/`, `services/`, `calibration/` |
| 2 | Topology feature extraction | **Upasena P. M. R. K.** | `topology_intel/` (graph builder, dual GCN, critical path, live topology, CRISP shares), `data_collection/otel_traces.py`, `evaluation/critical_path_eval.py` |
| 3 | Multi-objective DRL agent | **Perera H. A. T. J.** | `rl_agent/` (MO-TD3, trainer, replay, controllers), `models/`, `evaluation/train_agent.py`, fine-tuning |
| 4 | Reward engineering and evaluation | **Perera G. A. T. D.** | `rl_agent/rewards.py`, `baselines/`, `evaluation/` (ablation, Pareto, scale, regime, bottleneck evaluation), `results/`, figures |

## 1. Dilhara W. L. - environment modeling and state representation
| Commit | Role | What |
|---|---|---|
| `ae64529` first commit | main | the original stack: generic service, manifests, simulator, first autoscaler |
| `8b178e7` Make the real stack measurable and safe: calibrated service, probes, manifests, live tooling | main | calibrated service model, probes, manifests, live tooling |
| `4eaf811` Replace PPO with topology-aware MO-TD3 (hybrid actions) on a calibrated simulator | part | the calibrated simulator and the environment around the new agent |
| `d1fb836` Train-Ticket live path: OTel-span metrics source, profile, topology seed | main | the metrics source that turns spans and cAdvisor into the agent's state, the Train-Ticket profile |
| `81ac38c` Train-Ticket live run: safety for real actions, measured priors, preflight launcher, docs | main | measured CPU cost priors, pod ages, preflight, safety for real actions |
| `5b7cead` Train-Ticket calibration runs: recorder, results and findings | main | the recorder and the calibration of the simulator to the real cluster |
| `f0ace0b` Any-cluster auto-discovery, scale evaluation, fine-tuning support, replay mode, untraced observation | main | `auto_profile` (derives the settings of any cluster), observing untraced services, replay |
| `adc8a2e` Live rehearsal fixes: metric window vs scrape interval, guard stabilisation window, pod ages, regime evaluation | main | the metric window and pod-age fixes found live |
| `9108052` Live orchestration on Train-Ticket, regime training, scale results, interview runbook | part | the live orchestration and the real-cluster timing |
| `32ece53` Replicas + CPU model for the real Train-Ticket, live runs, and the training dataset in the database | part | `ALLOW_MEMORY_RESIZE`, the CPU-only Kubernetes patch, launcher options |
| `1b46886` Bottleneck-aware agent: scenarios, evaluation, demand gate, live demo on the demo app | part | `bottleneck_scenario` / `shift_mix` (in `topologies.py`), the kind demo script |

## 2. Upasena P. M. R. K. - topology feature extraction
| Commit | Role | What |
|---|---|---|
| `4eaf811` Replace PPO with topology-aware MO-TD3 | part | the two graphs (dependency graph, latency graph), the dual graph network input, the critical path |
| `d1fb836` Train-Ticket live path | part | the trace analysis (`otel_traces.py`) |
| `a7b4515` Live topology: dependency graph learned only from traces, shown in the dashboard | main | `live_topology.py` |
| `ceadf14` Critical path evidence: ground truth from real traces, evaluation, live dashboard panel | main | per-request ground truth, `critical_path_eval.py`, the proof panel |
| `6d97799` CRISP-style critical shares: per-request critical path aggregated per service, held-out comparison, live panel | main | `crisp.py` |
| `4f58e3b` Review fixes (10 findings) and the critical-set experiment | main | idle-expiry fix of the live topology, the critical-set experiment |
| `81ac38c` Train-Ticket live run | part | trace-derived call counts |

## 3. Perera H. A. T. J. - multi-objective DRL agent
| Commit | Role | What |
|---|---|---|
| `4eaf811` Replace PPO with topology-aware MO-TD3 (hybrid actions) on a calibrated simulator | main | MO-TD3, hybrid actions, trainer, replay, controllers |
| `fbfc15c` Results, trained models, real-cluster experiment, docs | part | the trained checkpoints |
| `f0ace0b` Any-cluster auto-discovery ... fine-tuning support | part | fine-tuning with a warm start |
| `9108052` Live orchestration on Train-Ticket, regime training, scale results | part | training in the Train-Ticket regime (`mo_td3_trainticket*`) |
| `4f58e3b` Review fixes and the critical-set experiment | part | the two fine-tuned agents of the experiment |
| `32ece53` Replicas + CPU model for the real Train-Ticket | main | `mo_td3_trainticket_cpu`, `scripts/23_train_trainticket_cpu.sh` |
| `1b46886` Bottleneck-aware agent | part | training on bottleneck scenarios (`hot_prob`), `mo_td3_bn`, `mo_td3_bn_tt` |

## 4. Perera G. A. T. D. - reward engineering and evaluation
| Commit | Role | What |
|---|---|---|
| `4eaf811` Replace PPO with topology-aware MO-TD3 | part | the reward vector, the baselines (HPA, VPA, threshold, static), the evaluation harness |
| `fbfc15c` Results, trained models, real-cluster experiment, docs | main | ablation, Pareto, the real-cluster experiment, the committed results and figures, README tables |
| `5b7cead` Train-Ticket calibration runs | part | the calibration results |
| `f0ace0b` Any-cluster auto-discovery, scale evaluation ... | part | `scale_eval.py` (8 to 100 services) |
| `adc8a2e` Live rehearsal fixes | part | `regime_eval.py` |
| `9108052` Live orchestration on Train-Ticket, regime training, scale results | main | regime and Pareto evaluation in the Train-Ticket regime, scale results |
| `f0a0677` Docs: README, traceability, Train-Ticket findings and interview runbook | main | `TAF_TRACEABILITY.md`, the findings, the runbook |
| `32ece53` Replicas + CPU model for the real Train-Ticket | part | the simulator comparison, the live results, the stored training dataset |
| `1b46886` Bottleneck-aware agent | part | `bottleneck_eval.py` (hit within 2 steps, wasted capacity, steps to SLO) |
| `ded3415` Bottleneck results on the real Train-Ticket and the demo app; resize safety limits; guides | main | `docs/BOTTLENECK_RESULTS.md`, the live results |

## Shared - not one of the four components (the TAF assigns them to nobody)
| Commit | What |
|---|---|
| `83090f2` Initial commit with project setup | scaffolding |
| `5594a96` Demo guide, live-demo traffic script; dashboard feed shows only meaningful events | demo tooling and dashboard |
| `1672932` Persistent per-cluster history in PostgreSQL | the history database (`persistence/`) |
| `d8d692e` Control room: Live and Analysis tabs, zoomable and expandable dependency graph | dashboard front end |
| `0ec1f19` Training dataset: no absolute paths of the machine; tests never write to a real history database | housekeeping |
| `0466bcc` Docs: full demonstration script | documentation |
| `2a7c333` History panel: real changes only (CPU before -> after), closed violation episodes, clearer wording | dashboard / history |
| `9662710` Idle reclaim: replicas come back after a load, without restarting anything | the safety guard (`safety/guards.py`) |
| `1cf0031` No scale-up without traffic (start-up CPU is not demand); full up-and-down cycle verified live | the safety guard and the live engine |
| `ded3415` (resize safety limits part) | the safety guard |

## Caveats
* The grouping is a judgment from commit subjects and touched files; the same commit often belongs to two or three components (the agent needs the graphs, the environment and the reward at once).
* A line count per component would be misleading: result and dataset files make "evaluation" look larger than it is, so they were left out when I checked the file evidence.
* If the people actually did different parts, the table is easy to change: the only input is which person owns which directory (the first table).
