# Reproducing the challenge baseline with replan-32

**Result being reproduced** (see `EXPOFT_ARC_REPORT.md` §3): executing the π0.5 policy's full 32-action
chunk before re-inferring, instead of the default 16-of-32 receding horizon, raises full-task success on
`turning_on_radio` from **3/20 (15%)** to **12/30 (40%)** on train instances 0–4 and **15/20 (75%)** on
held-out instances 5–9 (27/50 combined). Paired grasp-subtask evidence: 12/20 → 18/20. Mechanism:
receding-horizon replanning makes the flow policy re-sample its intention mid-motion (dithering); full
commitment cures it.

The change is **one CLI flag on the policy server**. No code changes anywhere.

## Requirements

| piece | where |
|---|---|
| eval harness + sim | this repo (`omnigibson.eval.eval`), branch `challenge/pi05-baseline` |
| policy server | [wensi-ai/openpi](https://github.com/wensi-ai/openpi) fork (`scripts/b1k/serve_b1k.py`) — local: `/mnt/nvme/openpi` |
| checkpoint | `pi05TurningOnRadio.zip` (Drive id `1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2`) → unzip to e.g. `/mnt/nvme/pi05_pretrained/pi05_turn_on_the_radio` (12 GB params + bundled `assets/turning_on_radio/norm_stats.json`) |

Note: the openpi fork needs the `robot_r1` obs-key patch in `src/openpi/configs/robots/b1k.py`
(camera/proprio keys of the form `robot_r1::robot_r1:<link>:Camera:0::rgb`) so the server's expected keys
match what `eval/r1pro.yaml` emits — see EXPERIMENT_NOTES 2026-07-07 ("Obs-key mismatch RESOLVED").
Do **not** confuse this stack with `/mnt/nvme/behavior-1k-solution` (the 2025-challenge solution repo with
task embeddings / eval tricks; its replan knob is `--actions_to_execute` but it is a different baseline).

## 1. Serve the policy (replan-32)

```bash
cd /mnt/nvme/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 .venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio \
  --policy.config pi05_b1k --policy.dir /mnt/nvme/pi05_pretrained/pi05_turn_on_the_radio \
  --control_mode receding_horizon --action_horizon 32 --port 8010
```

`--action_horizon` is "number of actions to execute before replanning" (`serve_b1k.py`, default **16** =
the challenge baseline; **32** = full-chunk commitment). The wrapper's `act_receding_horizon`
(`src/openpi/shared/eval_b1k_wrapper.py:118`) infers every `action_horizon` steps and pops one 23-dim
action per env step, so the eval client below is unchanged. Mem fraction 0.4 leaves room for Isaac on the
same GPU; use a second GPU or machine otherwise.

## 2. Run the challenge eval (this repo)

```bash
OMNIGIBSON_HEADLESS=1 python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --robot-config OmniGibson/omnigibson/eval/r1pro.yaml \
  --mode train --instance-indices 0 1 2 3 4 \
  --host 127.0.0.1 --port 8010 \
  --output-dir outputs/replan32_eval --write-video
```

For the held-out read use `--instance-indices 5 6 7 8 9`; for the official split use
`--mode public_test` with instance ids 301+. Success/q_score land in per-rollout JSONs scored by
`omnigibson/eval/utils/score_utils.py`.

## 3. Expected numbers

At `--action_horizon 16`: ~15% (3/20 on instances 0–4; matches the original serve-parity reads of ~2/10).
At `--action_horizon 32`: ~40% on 0–4, ~75% on 5–9 (n=30/20; single-instance variance is large — use ≥20
episodes per condition and paired instances for comparisons). Successes finish in ~1200–1900 env steps;
failures typically run to the step cap.

## Caveats

- Fisher exact on the headline pair (12/30 vs 3/20) is p≈0.07 at these n; the grasp-subtask pair
  (18/20 vs 12/20, same fixed starts) and the held-out replication are the corroborating evidence.
- `--action_horizon 32` assumes the checkpoint's chunk length is 32 (it is for `pi05_b1k`); a value above
  the model's horizon would starve the action queue.
- Videos (`--write-video`) are the fastest sanity check: replan-16 failures show hesitation/dithering
  mid-reach; replan-32 failures are mostly clean misses or timeouts.
