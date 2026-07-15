# Online RL Fine-Tuning of a Flow-Matching VLA: the EXPO-FT Arc

**Project:** BEHAVIOR Challenge 2026, π0.5 baseline (turning_on_radio), EXPO-FT adoption
**Period:** 2026-07-05 → 2026-07-15 (runs v0–v52)
**TL;DR:** After a six-defect adoption debug, the only intervention that improved task success was an
inference-time change — executing the policy's full 32-action chunk instead of replanning halfway
(**15% → 54%** full-task success, generalizing to held-out instances). Every learning mechanism in the
EXPO-FT recipe was then tested to failure with an isolated cause: BC on own successes is flat; critic
selection is safe but capped by the policy's sample support; and residual edits — the recipe's only
support-escaping mechanism — are structurally adversarial in our regime and collapse rollouts regardless
of critic quality. Dense staged reward *does* fix critic learning; it does not rescue the improvement
operator. The thesis "commitment beats choice" summarizes the empirical content.

---

## 1. Setup

- **Base policy:** task-specialized π0.5 checkpoint (flow-matching VLA; 32-step action chunks at 30 Hz;
  delta actions for torso/arms, absolute for base/grippers; mean/std normalization).
- **Recipe:** [EXPO-FT](https://github.com/pd-perry/expo-ft) — online loop of {rollout with best-of-N
  candidate selection + residual edits} + {per-episode updates: REDQ-style critic (10 heads, min-of-2),
  success-only BC actor, SAC-style residual actor}.
- **Testbed:** `grasplift` mini-task (grasp the radio + raise it ≥ 0.15 m, sustained), carved out for a
  short-horizon 65–75% band; deterministic 20-start fixed-eval protocol (seeded pose/object jitter);
  sim on one node (eventually the remote RTX 5090/Windows box), JAX learner on a 96 GB workstation.

## 2. Adoption ledger: six silent defects

Each made the recipe look "just bad" without erroring; all were found by paired fixed-start evals and
video review, not loss curves.

| # | Defect | Symptom | Fix |
|---|--------|---------|-----|
| 1 | Replay-buffer chunk backfill re-anchored delta actions to the wrong state | ~75% of action signal erased 8 steps into chunks; critic trained on corrupted (s,a) | anchor to chunk-start raw state (`raw_state` field) |
| 2 | `actor_success_only` defaults False | BC on failures → policy imitates its own dithering | set True (reference config had it; ours didn't) |
| 3 | Output transforms fed dummy-zero state | Unnormalize(0) = mean state → arm fly-up / arm pinning | plumb `normalized_state` through |
| 4 | Receding-horizon replanning (execute 16 of 32) | intention resampling → dithering | replan-32 (see §3) |
| 5 | Actor lr 2.5e-5 (tuned for LoRA-on-generic-base) | graded erosion of the specialized checkpoint (~25 pts/600 updates) | 2.5e-6 |
| 6 | batch/UTD silently halved (32/10 vs reference 64/20) | ¼ critic gradient per datum | restored (needs sim off-GPU: update graph ~48 GB + 19 GB TD tensor) |

**Meta-lesson:** small-n unpaired evals are noise (same checkpoint read 38%/24%/8%); every conclusion
below is from paired fixed-start evals.

## 3. The positive result: commitment (replan-32)

Executing the full predicted 32-action chunk instead of re-inferring every 16 steps:

| eval | replan-16 | replan-32 |
|---|---|---|
| grasp subtask (paired 20 starts) | 12/20 | 18/20 |
| full task, train instances 0–4 | 3/20 (15%) | 12/30 (40%) |
| full task, held-out instances 5–9 | — | 15/20 (75%) |
| full task combined (10 instances) | — | 27/50 (54%) |

Interpretation: receding-horizon replanning makes the flow policy re-sample its intention mid-motion;
consecutive-chunk disagreement manifests as hesitation/dithering, the dominant failure mode. Full
commitment cures it. One line of inference config; generalizes across instances.

## 4. The negative results: every learning lever, isolated

| run | mechanism | result |
|---|---|---|
| v42 | RFT (BC on own successes), safe lr | flat: 27/50 post-update vs 65% band |
| v45 | best-of-8 Q-selection, frozen critic | flat: 12/20 vs 13/20 paired. Cause: critic has state-value structure (6σ success/fail separation) but ~no action discrimination (candidate spread 0.004 vs 0.012 state separation) |
| v46/v47 | full recipe at reference 64/20 | rollouts collapse to 30%; **weights intact** (v47 decomposition: same ckpt at N=1 → 67%) — harm is purely rollout-time |
| v49 | **oracle lookahead**: simulate all 8 candidates (sim = perfect model), rank by V | picks near-perfectly (grasping candidate chosen 56/57; lift-steering within ~1 cm of best) yet outcomes flat — **failures lie outside the policy's sample support** (lifts stall at 0.64–0.67 m vs the 0.684 m criterion) |
| v50 | **staged dense reward** (approach → +1 grasp step → linear lift height, potential-based) | phase 1: critic gains real action discrimination (4× spread, 10× grasp gap) with zero erosion; phase 2 (selection+edits on): collapse again, 2/9 |
| v51 | decomposition of v50 phase 2 | weights intact (12/22 ≈ band) |
| v52 | best-of-8 **plain** selection with the discriminating critic (no edits) | exactly neutral: 15/22; paired vs baseline 14 agree / 3 up / 3 down |

**The matrix has no exceptions: every collapse cell contains residual edits; every edit-free cell is
neutral, under both mush and discriminating critics.**

## 5. Mechanism: why edits are the poison

The residual actor trains by **gradient-ascending the critic's (mean-ensemble) Q with respect to its
736-dim action input** (`update_residual_actor`) — the canonical adversarial-example construction. Where
Q's genuine action-sensitivity is weak, the ascent direction is dominated by the critic's error surface,
so the residual actor learns perturbations that are *erroneously* scored high. At rollout, those
Q-adversarial chunks then win the argmax (76–89% edit-pick rate) because they were optimized to — the
robot executes adversarial noise against its own value function. Goodhart at 30 Hz.

Why this is lethal here and survivable in the reference's real-robot regime (three compounding axes):
1. **Coverage:** sim determinism + 20 fixed starts + one specialized policy = razor-thin Q data manifold;
   real-robot noise/demo diversity fattens it.
2. **Dimensionality:** our chunks are 32×23 = 736-dim (replan-32) vs ~60–130-dim reference chunks;
   off-manifold volume grows exponentially.
3. **Episode economics:** their ~11-decision episodes make early noise cheap exploration; our 19–28
   decision chains die from one bad chunk.

Standard mitigations the recipe omits: conservative/OOD-penalized critics, perturbation-robust Q
training, trust regions on the residual. (Also noted: shaped critics invert lookahead-style cross-state
ranking — V_shaped = V − coef·φ(s) — un-shape before ranking states.)

## 6. Corrected side-findings

- One of the 20 fixed starts spawns the radio on the floor (object jitter off the table edge) —
  permanently unwinnable; ~5% deflation of all historical rates; A/Bs unaffected (shared set).
- Q(s,a) here scores the **whole chunk** (736-dim flattened) with semi-MDP targets (γ³²); the critic
  never reasons below chunk granularity.
- Sim save/restore probes must also restore python-side env counters (`_current_step`), or probe steps
  silently advance Timeout truncation.
- Windows/5090 node: Isaac 5.1 needs driver R580 (610.x crashes rtx.scenedb), h5py==3.15.1,
  KMP_DUPLICATE_LIB_OK=TRUE, LongPathsEnabled, IPv4-explicit tunnels.

## 7. Implications & open directions

1. **Bank replan-32** for the challenge submission (validated, generalizing, free).
2. Selection-class methods are exhausted on this checkpoint: even oracle selection is support-capped.
3. Support escape without Q-ascent is the open research direction — e.g., offline DPO-style chunk
   preferences constructed from sim-lookahead pairs (the lookahead infra exists: `lookahead_probe` env op
   + `--lookahead_n`), which never lets a critic steer live rollouts.
4. Dense staged reward is a validated tool for critic quality (transferable to any future value-based
   attempt); the failure was never value learning after defect #1 was fixed.

## 8. Reproducibility map

Three repos are involved:
- **This repo (BEHAVIOR-1K fork), branch `challenge/pi05-baseline`:** OmniGibson + challenge eval harness
  + all experiment infrastructure under `challenge_experiments/` (env-ops server with subtask/fixed-eval/
  shaping/lookahead, probes, outcome records `v*_*.txt`, running log `EXPERIMENT_NOTES.md`).
- **Serving repo (wensi-ai/openpi fork, separate; local `/mnt/nvme/openpi`):** `scripts/b1k/serve_b1k.py`
  + `B1KPolicyWrapper`. The replan knob is `--action_horizon` ("actions to execute before replanning",
  default 16) — replan-32 reproduction is `--action_horizon 32` plus the standard eval harness from this
  repo. No code change required; exact commands in `REPRODUCE_REPLAN32.md`. (Not to be confused with the
  2025-solution stack at `behavior-1k-solution`, whose knob is `--actions_to_execute`.)
- **expo-ft fork (local, upstream pd-perry/expo-ft):** the recipe with our six fixes + lookahead/trace
  instrumentation; committed locally (no push rights on upstream).
Checkpoints/norm-stats are outside git (`/mnt/nvme/pi05_pretrained`, `/mnt/nvme/expoft_runs`).
