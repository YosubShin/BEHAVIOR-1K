# When Does Online RL Fine-Tuning Beat Interactive BC for Expressive VLAs?
## Findings and Paper Plan

**Working thesis:** The online-RL machinery in the EXPO/EXPO-FT family (best-of-N selection + a
Gaussian *edit* policy that ascends the critic) does **not** explore autonomously. Its advantage over
behavior cloning is **conditional on data coverage** that the method silently assumes — broad offline
data (EXPO's sim benchmarks) or a human in the loop (EXPO-FT's real robot). Strip that coverage and the
*exploration* component (the edit residual) degrades to **noise, and in high dimension to an adversarial
attack on the critic**; only the cheap parts (imitation, selection as a data-efficiency tool) retain value.
The negative result *is* the contribution: it delimits when the method helps.

---

## 1. The empirical arc (BEHAVIOR-1K, π0.5, `turning_on_radio` grasplift mini-task)

Regime: one task-specialized flow-matching VLA, ~20 human demos, deterministic 20-start fixed-eval,
sim-only, **no human interventions** — i.e. the EXPO-FT recipe with its coverage source removed. This is
the thin-support pole; every claim below is from paired fixed-start evals + video/oracle traces, never
loss curves.

### 1a. Adoption ledger: the null is not an implementation artifact (credibility armor)

Before any negative claim is admissible, we rule out that the recipe simply wasn't implemented correctly.
Six **silent** defects each made the recipe look "just bad" without erroring; all were found by paired
fixed-start evals + video review, *not* loss curves, and each has an isolated cause and fix. Only after
all six were corrected — and the recipe verified faithful to the reference config — do the C1–C9 negatives
stand. This ledger is itself a reproducibility contribution.

| # | Defect | Symptom | Fix |
|---|--------|---------|-----|
| 1 | Replay-buffer chunk backfill re-anchored **delta** actions to the wrong state | ~75% of action signal erased 8 steps into a chunk; critic trained on corrupted (s,a) | anchor to chunk-start raw state (`raw_state`) |
| 2 | `actor_success_only` defaulted False | BC on failures → policy imitates its own dithering | set True (reference had it; our config didn't) |
| 3 | Output transforms fed dummy-zero state | Unnormalize(0)=mean state → arm fly-up / pinning | plumb `normalized_state` through |
| 4 | Receding-horizon replanning (16 of 32) | intention resampling → dithering | replan-32 (the C1 positive) |
| 5 | Actor lr 2.5e-5 (tuned for LoRA-on-generic-base) | graded erosion of the specialized ckpt (~25 pts/600 upd) | 2.5e-6 |
| 6 | batch/UTD silently halved (32/10 vs reference 64/20) | ¼ critic gradient per datum | restored (needs sim off-GPU: 48GB graph + 19GB TD tensor) |

**Meta-lesson (goes in the paper):** small-n unpaired evals are pure noise here — the *same* checkpoint
read 38%/24%/8% across three runs. Every conclusion is from paired fixed-start evals; this is why the
negative is trustworthy where a loss-curve or unpaired-eval study would not be.

| # | Claim | Key evidence |
|---|---|---|
| C1 | The **only** intervention that improved task success was inference-time **commitment** (execute the full 32-chunk vs replan-16). | 15%→40% train, 75% held-out, 54% combined; grasp subtask 12/20→18/20. One CLI flag. |
| C2 | **BC on own successes (RFT)** is flat. | v42: 27/50 ≈ band. Imitating self-clones adds no new information. |
| C3 | **Critic selection is safe but support-capped**; even a *perfect* (sim-oracle) selector can't exceed the policy's sample support. | v45 flat; v49 oracle picks near-perfectly yet outcomes flat — failures lie *outside* the sample set. |
| C4 | **Residual edits are structurally adversarial** in high dim: the edit actor ascends mean-ensemble Q over a 736-dim action, manufacturing Q-high/reward-0 candidates that win the argmax. | v46/v50p2 collapse to 22–30%, 76–89% edit-pick; Q-vs-realized audit: chosen-Q +0.14 while realized ≈ 0. Harm is rollout-time only (weights intact). |
| C5 | Better critics (dense staged reward, consistent-diet demo relabeling) **attenuate but never eliminate** the edit harm. | v50/v53/v55: harm 36% at best, edit-pick 69%; the ascent attacks estimation error itself, which no diet removes. |
| C6 | **Dimensionality restriction cures the adversarial pathology but removes the teeth**: restricted to the reference's ~30-dim regime, edited candidates no longer carry inflated value. | v58: V(edit)≈V(plain), realized φ≈plain — direct interventional confirmation of the dimensionality axis. |
| C7 | The low-dim residual is **statistically indistinguishable from same-scale isotropic noise**. | v58d three-way (plain/edit/matched-noise, oracle-scored): win-shares edit 9% / noise 7%; paired best-φ diff +0.0009, n.s. Functionally a noise generator. |
| C8 | Even **exhaustive multi-chunk search** over the policy's samples cannot manufacture the missing behavior. | Depth-2 beam, 20 starts: opens a dead basin **iff** the policy already grasped (1/1 opened had a grasp root; 0/9 closed did — perfect separation). |
| C9 | The deficit is **one specific un-proposed action**: the grasp-close. | Dead basins never reach φ>1 (never grasp); lift is 2-chunk-reachable from *any* grasp (start 12: 0.55→0.73). Not a lift or reposition problem. |

**Synthesis:** every mechanism that *chooses among or locally perturbs the policy's own samples* is capped
at the band. Exceeding it requires *injecting experience the policy cannot currently produce* — which is
exactly what broad data (EXPO) or human interventions (EXPO-FT) supply, and what we withheld.

## 2. Mechanism: three compounding axes (why lethal here, survivable in the references)

1. **Coverage** — sim determinism + 20 fixed starts + one specialized policy = razor-thin Q manifold;
   real-robot noise / diverse D4RL data fatten it.
2. **Dimensionality** — 736-dim replan-32 chunks vs the reference's ~30–130-dim actions; off-manifold
   volume grows exponentially, and the same per-dim edit bound gives ~3× the perturbation norm.
3. **Episode economics** — ~11-decision reference episodes make early noise cheap exploration; our 19–28
   decision chains die from one bad chunk.

TD correction is **pointwise**; adversarial generation is **volumetric** — the critic patches one hole per
executed chunk while the ascent sources fresh holes from an exponential neighborhood, so it loses the arms
race. Grounded by the Q-vs-realized audit; the reference regimes sit on the benign side of all three axes.

## 3. What the method is actually worth (honest reframing)

RL's merit here is **not exploration** but **label efficiency + stitching**, both *multipliers on data,
not substitutes for it*:
- *Label efficiency*: a critic lets one success (human-provided or discovered) propagate via value backup
  to many similar states — this is why EXPO-FT's human-intervention rate anneals to zero. The human still
  supplies coverage; the critic makes each intervention go further.
- *Stitching*: a value function can recombine sub-behaviors across trajectories (offline-RL-beats-BC) —
  but only with support at the stitch points, precisely what thin regimes lack.

**Actionable decomposition:** drop the edit residual (the exploration component — noise/adversarial in
thin support); keep selection + critic only when interventions are expensive enough that value-propagation
pays for the added complexity and you can keep the critic in-support. Otherwise **BC / DAgger + a
failure-recovery collection pipeline** is the simpler, safer baseline — which is EXPO-FT with the
expensive/harmful part amputated and the interactive-collection part kept.

## 4. Paper plan

**Two-pronged, triangulated:**

**(A) Coverage ablation in EXPO's OWN sim benchmark** (keystone — proves the negative where the authors
claim the positive). Repo: `pd-perry/EXPO` (parent of our expo-ft fork), public D4RL/robomimic data,
MuJoCo/JAX, no Isaac.
- Conditions (one-flag each): **full EXPO** (`N=8, n_edit_samples=8`), **no-edit** (`n_edit_samples=0`,
  selection only), **base** (`N=1, n_edit_samples=0`).
- Coverage knob: subsample the D4RL offline dataset {100, 50, 25, 10, 5%} × diversity variants
  (antmaze play/diverse; robomimic ph/mh/mg). 3 seeds.
- **Metric = the gaps, not raw return** (controls for "less data hurts everything"):
  (EXPO − base) and (EXPO − no-edit) as functions of coverage. Prediction: both collapse to ≈0 (or
  negative) as coverage narrows; the edit's contribution flips from + to −.
- Cost: ~30 min/run, MLP policies, ~1 day for the full sweep on one GPU.
- First step: faithfully reproduce ONE positive (`antmaze-large-play-v2`) before ablating.

**(B) BEHAVIOR-1K thin-support VLA regime** (this arc) as the fresh-domain confirmation, and the sim
proxy for the EXPO-FT *intervention* ablation (not reproducible for us — no robot / released intervention
data). Deliver C1–C9 with the mechanism story of §2.

**Framing sentence:** *"Online RL fine-tuning beats interactive BC for expressive VLAs only under coverage
the base method silently assumes; its exploration component is inert-to-adversarial without it — shown by
coverage ablation in the method's own benchmark and a thin-support VLA regime where the edit degenerates
to noise."*

## 5. Open threads / caveats

- **Undertraining caveat (C7):** the low-dim residual had ~11 warm episodes / ~220 grad steps vs the
  reference's continuous training. Edit==noise + identical spread makes "more training rescues it"
  unlikely, but it's the one unfalsified thread; the EXPO ablation (continuous training) settles it cleanly.
- **Single task/checkpoint** for the BEHAVIOR arc — (A) generalizes it across 12+ standard tasks.
- **Positive control needed:** must reproduce an EXPO win first, or the ablation is unfalsifiable.
- The grasp-close deficit (C9) suggests the cheapest *fix* is one teleop grasp demo / grasp primitive per
  dead basin — orthogonal to the paper but validates the "inject experience" conclusion. (SpaceMouse
  inbound; local env-server teleop path scoped.)

## 6. Setup / infra status (as of 2026-07-20)

- Beam infra: `beam_save/load/exec/end` env ops + `--beam_depth2` harness (roots = top-K φ along greedy
  traj). Committed. Local env server is the robust path (Windows Session-0 blocks Isaac restarts).
- EXPO repo cloned `/mnt/nvme/expo_repo`; `expo` conda env (py3.8, jax 0.4.13) built; requirements + D4RL
  installed; MuJoCo 210 placed. **Pending:** `mujoco_py` first-compile (GL headers + gcc present; likely a
  patchelf/Cython detail) before the antmaze baseline can run.
- BEHAVIOR reproduction race (separate track): 3 lanes still queued on the clusters (Delta H200/A100,
  Koa) — for parity validation before `hanging_pictures`, unrelated to the paper.
