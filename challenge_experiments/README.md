# BEHAVIOR 2026 Challenge — π0.5 baseline exploration

Reproducing and probing the provided **π0.5** baseline on the [2026 BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html), end-to-end on a single workstation (RTX PRO 6000, 96 GB). See [`EXPERIMENT_NOTES.md`](./EXPERIMENT_NOTES.md) for the full blow-by-blow log.

## TL;DR results — `turning_on_radio`, public instances 0–9

Provided pretrained π0.5 checkpoint, OpenPI [`behavior` fork](https://github.com/wensi-ai/openpi), served over the challenge websocket protocol; OmniGibson evaluator on the other end.

| Eval wrapper | Resolution to sim | Success | Successful instances |
|---|---|---|---|
| `RGBDFullResWrapper` (faithful / official) | head 720², wrist 480² + depth | **2/10 (20%)** | 306, 308 |
| `DefaultWrapper` (fast debug) | native 224² RGB | 1/10 (10%) | 310 |

**Key finding:** the two wrappers' successes are **disjoint** (306/308 vs 310). The wrapper doesn't uniformly degrade — it *shuffles which instances pass*. The provided baseline is **marginal/borderline** on this task, so a small input change (native-224 render vs 720→224 downsample) flips individual instances. → Use `RGBDFullResWrapper` for real numbers; `DefaultWrapper` only for coarse debugging. (n=1/instance, so wrapper-effect vs. sim-nondeterminism isn't fully separable without k-rollout repeats.)

The model **does** solve the task when it works (navigates → grasps → toggles the radio on, e.g. instance 308 in 1525 steps). The dominant failure mode: it **grasps the radio and carries it off** instead of toggling it — a policy-quality limitation, not a timeout (confirmed: still fails at 2× the time budget).

## What's here

| File | What |
|---|---|
| [`EXPERIMENT_NOTES.md`](./EXPERIMENT_NOTES.md) | Full running log: install, protocol trace, data shapes, every run + result |
| [`minimal_policy_server.py`](./minimal_policy_server.py) | Self-contained reference policy server speaking the challenge msgpack/websocket protocol (drop in your own `act()`) |
| [`mock_eval_client.py`](./mock_eval_client.py) | Stand-in evaluator client to smoke-test a policy server without OmniGibson |
| `*/json/` | Raw per-rollout result JSONs backing the tables above |

## Bug fix included

`OmniGibson/omnigibson/eval/wrappers/rgbd_full_res_wrapper.py` — the official RGB+depth wrapper **crashed on construction** (`AttributeError: 'NoneType' object has no attribute 'view'`): it called `env.load_observation_space()` (which reads robot joint positions for `proprioception_dim`) before the physics sim view existed. Fix: call `og.sim.update_handles()` first (same pattern the Evaluator uses). Without this, the faithful/official challenge-track wrapper is unusable.

## Reproduce

1. Install BEHAVIOR-1K (`./setup.sh --new-env behavior --omnigibson --bddl --joylo --dataset --eval`), point `OMNIGIBSON_DATA_PATH` at your data root.
2. Set up the OpenPI `behavior` fork (`uv sync`); the fork's `src/openpi/configs/robots/b1k.py` `R1Pro` config must use `name="robot_r1"` and matching `obs_key`s to align with the current `eval/r1pro.yaml`.
3. Serve: `serve_b1k.py --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio --policy.config pi05_b1k --policy.dir <ckpt> --port 8010`
4. Eval: `python -m omnigibson.eval.eval --task-name turning_on_radio --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --host 127.0.0.1 --port 8010 --mode public_test --instance-indices 0 1 2 3 4 5 6 7 8 9 --write-video`
