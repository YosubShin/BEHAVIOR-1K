# 2025 BEHAVIOR Challenge — Literature Review (top contenders, approaches, compute)

Compiled 2026-07-08 from primary sources (team arXiv reports, code repos, official challenge pages). The 1st BEHAVIOR Challenge ran at **NeurIPS 2025** ("Foundation Models Meet Embodied Agents," San Diego, Dec 6–7 2025): 50 long-horizon household tasks, Galaxea R1 Pro, ~10k human teleop demos (200/task, 1,200+ hrs), BDDL partial-credit Q-score, 18 teams. **Note:** the official 2025 leaderboard URL now 404s (site rolled to the 2026 challenge); results below are reconstructed from participants' own papers/repos, which cross-corroborate (the 2nd-place NVIDIA team's paper independently confirms the 1st-place team).

## Leaderboard (reconstructed)

| Rank | Team | Affiliation | Track | Q-score | Full success |
|---|---|---|---|---|---|
| 1 | **Robot Learning Collective** | Independent (Larchenko, Zarin, Karnatak) | Standard | **0.260** | **12.4%** |
| 2 | **Comet ("OpenPI Comet")** | NVIDIA Research | Standard | 0.251 | 11.4% |
| 3 | SimpleAI Robot | Beijing Simple AI | Standard | 0.159 | 10.8% |
| 4 | The North Star | Huawei CRI EAI | Standard | 0.120 | 7.6% |
| 5 | Embodied Intelligence | Independent | Privileged | 0.095 | 5.2% |
| 6 | RAPPER | GIST | Privileged | N/A | 7.5% |
| 7 | tobi | Alzonova | Standard | 0.072 | 3.6% |
| 8 | MR | — | Privileged | 0.051 | 3.4% |
| 9–18 | RACΞL (CMU), Postech, Merlin Labs, LYQRobotics, ACT (Xiamen), StarVLA, … | various | mostly Standard | N/A | ≤1.4% |

Two tracks: **Standard** (RGB/onboard perception) and **Privileged** (extra sim info allowed). Top finishers were all Standard-track.

## What worked: methods (top teams)

**Both #1 and #2 fine-tuned π0.5 (Physical Intelligence / OpenPI)** — a VLA with SigLIP vision encoder + Gemma LLM backbone + flow-matching action head. **Pure imitation learning / behavior cloning. No RL-first, no TAMP, no LLM planner, no world model.**

**#1 Robot Learning Collective** (arXiv 2512.06951, code: IliaLarchenko/behavior-1k-solution):
- Base **π0.5**, trained on **successful demos only** (flow-matching BC).
- Key tricks: **correlated noise for flow matching** (+ correlation-aware inpainting → smoother action chunks), **learnable mixed-layer attention**, **"System 2" stage tracking** (resolves task-phase ambiguity), **multi-sample flow matching** (variance reduction), inference-time **action compression** + **challenge-specific correction rules**.
- Notable: **replaced the Gemma LLM with 50 trainable task embeddings** (one per task) instead of language conditioning.

**#2 Comet / NVIDIA Research** (arXiv 2512.10071, code: mli0603/openpi-comet):
- Base **π0.5**, systematic ablations on training techniques + data; found scaling gains in both pre- and post-training. Completed 22/50 tasks.
- (Post-challenge, they refined to a 0.345 *validation* Q-score — this is NOT the ranked 0.2514 leaderboard score; don't conflate.)

## Data collection

- **#1: NO additional data** — provided 10k demos only. Addressed the lack of *recovery* demos algorithmically (runtime heuristics, cross-task learning), not by collecting.
- **#2: DID augment** — added ~0.4K hrs (~3.6K trajectories) of **motion-planner demonstrations + offline-RL rollouts** (3 rounds, ~8,500 traj/round rejection/RL fine-tuning) on top of the 1.1K hrs of human demos.
- **No team collected real-robot data** — everything was in-sim. Notably, #1's pure-provided-data approach still beat #2's data augmentation.

## IL vs RL

**Overwhelmingly imitation learning.** Given ~10k near-optimal human demos + long-horizon sparse-reward tasks, IL on a pretrained VLA was the winning paradigm. RL appeared only as a *supplementary offline-RL data source* for #2. No online-RL, TAMP, or world-model entry placed top-3. (Makes sense: online RL on 6-minute, sparse-reward, house-scale tasks is brutal; the demos make BC the high-EV move.)

## Compute per approach (both top teams disclosed hardware)

| Approach | Hardware | Training | Cost | Inference |
|---|---|---|---|---|
| **#1 Robot Learning Collective** | **8× H200** | ~15 days non-stop (multi-task) + ~1 wk/group fine-tune; ~2 epochs total | **~$13k (~$10k GPU sponsored by Nebius)** | single **RTX 4090** (24 GB); eval on 20× RTX 4090 parallel |
| **#2 Comet (NVIDIA)** | **8× H200** | 50k pretrain steps + 15–20k finetune steps; per-device batch 64 | not stated | single 24 GB GPU |
| π0.5 baseline (official) | 1×/multi-GPU / SLURM | batch 64, 32-step horizon | — | single GPU |
| GR00T N1.7 baseline (official) | 8 GPUs | batch 2048, 150k steps, vision+LLM **frozen** | — | single GPU |

- Rough estimate: #1 ≈ **thousands of H200 GPU-hours** (8 × ~15 days ≈ 2,900 h main run + per-group fine-tunes → likely 5,000+ h).
- Reference points: model sizes ~3–3.3B (π0.5, GR00T). **GR00T-N1 *pretraining* alone ≈ 50,000 H100 GPU-hours** (up to 1,024 GPUs) — but *fine-tuning* a ~3B VLA on ~10k demos is only hundreds-to-low-thousands of GPU-hours, which is what the challenge entries cost.
- **Eval constraint:** policy must run on a **single 24 GB GPU** (RTX 4090-class in 2025; 3090/A5000/TitanRTX for 2026). Episode timeout = 1.5× mean human-demo length. VLA action-chunk latency ~60 ms/16-action chunk → real-time on one 24 GB card.

## Is it solved? SOTA comparison

**No — far from solved.** The *best* team clears only ~26% of BDDL subgoals (partial credit) and ~12% of full tasks. Across 18 teams nobody exceeded this. The ranking metric is deliberately partial-credit *because* binary success is so low. This is dramatically harder than manipulation benchmarks VLAs have largely **saturated** (LIBERO, CALVIN) — BEHAVIOR's long-horizon (multi-minute), multi-room, bimanual, object-state-change tasks are a different regime. (Cross-benchmark numeric comparison is qualitative — no source gave verified head-to-head numbers.)

## World models vs VLAs

- **In the 2025 challenge: zero world-model entries in the top 3.** VLA imitation learning dominated by a clear margin.
- **Broader field (late 2025 / early 2026):** world models are gaining real momentum — a wave of "**world-action models (WAMs)**" that pretrain a video/world predictor then fine-tune to act (NVIDIA framing), plus survey activity (e.g., Awesome-WAM, world-model surveys arXiv 2606.00113). Framed as an emerging *second* recipe alongside VLAs, with a hybrid future likely. **But as of this challenge they were not yet competitive** on long-horizon embodied manipulation — the VLA-BC recipe still wins in practice.

## Takeaways for our own effort
1. The bar to beat is **~0.26 Q / ~12%**, and the winning recipe is **π0.5 + BC + smart training/inference tricks** — exactly the stack we've stood up. Our provided-checkpoint baseline (~20% on the *single* task `turning_on_radio`) is consistent with a strong per-task policy; the challenge difficulty is the **breadth across 50–100 tasks**.
2. **Data augmentation didn't win** — #1 used only provided demos. Effort is better spent on training/inference tricks (correlated-noise flow matching, stage tracking, action smoothing, correction rules) and the **recovery-behavior gap** (policies fail by doing the wrong thing confidently — cf. our grab-and-wander observation).
3. **Compute is accessible-ish:** ~8× H200 for ~2 weeks (~$13k, much sponsorable) got 1st place. Not hyperscale.
4. **Task/language conditioning matters:** the winner ditched language for 50 learned task embeddings — a concrete, cheap idea.

### Primary sources
- 1st: arXiv **2512.06951** · github.com/IliaLarchenko/behavior-1k-solution · robot-learning-collective.github.io/winning-behavior-1k-challenge.html
- 2nd: arXiv **2512.10071** · github.com/mli0603/openpi-comet
- Official: behavior.stanford.edu/challenge (call_for_participation, baselines, evaluation — 2025)
- Compute refs: GR00T-N1 arXiv 2503.14734; world-model survey arXiv 2606.00113; NVIDIA "World-Action Models" blog (2026)
