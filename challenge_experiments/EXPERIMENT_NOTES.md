# BEHAVIOR-1K 2026 Challenge — Experiment Notes

Running log of what we do to get onto the challenge. Newest work at the bottom.

- **Repo:** `/home/yosub/co/BEHAVIOR-1K` (branch `main`)
- **Challenge track:** single track — policy inputs restricted to **RGB + depth + proprioception**.
- **Ranking metric:** mean task success with BDDL **partial credit** = (goal predicates satisfied) / (total goal predicates), averaged over 100 tasks.
- **Report on:** `--mode public_test --instance-indices 0..9`, 1 rollout each, default `1.5x` mean-human timeout.

## Machine status (checked 2026-07-07)

| Thing | Status |
|---|---|
| GPU | ✅ RTX PRO 6000 Blackwell, 96 GB VRAM |
| `behavior` conda env | ❌ **not installed** (only `base`, `env_isaaclab`, `lerobot_alohamini`, `mcap`) |
| Demo dataset | ❌ not downloaded yet |
| `env_isaaclab` env | ✅ has `websockets 12.0` + `msgpack` + `torch 2.7.0+cu128` (used to smoke-test the policy server) |

> **Blocker for full eval:** OmniGibson isn't installed. Per repo `AGENTS.md` we don't self-install; the full `python -m omnigibson.eval.eval` loop needs the `behavior` env. See "Next steps".

---

## Item 1 — How eval + the websocket protocol works

Entry point: `python -m omnigibson.eval.eval` → `OmniGibson/omnigibson/eval/eval.py` → `Evaluator` (`evaluator.py`).

**Two-process design.** Your policy runs as a websocket **server**; OmniGibson is the **client** that drives it.

```
[your policy server :8000]  <--ws/msgpack-->  [omnigibson.eval.eval client]
   MyPolicy.act(obs)->action                     builds obs, steps sim, scores
```

**Per-step loop** (`Evaluator.step`, evaluator.py):
1. `action = policy.forward(obs)` → `WebsocketPolicy` (`policies.py`) sends obs, gets action.
2. `env.step(action)` advances the sim one control step (30 Hz action, 120 Hz physics).
3. Obs preprocessed (`_preprocess_obs`): dict flattened, `cam_rel_poses` + `task_id` appended.
4. Metrics updated; loop until `terminated` (task success) or `truncated` (timeout).

**Wire protocol** (`omnigibson/eval/utils/network_utils.py`, adapted from openpi):
- Transport: websocket, payloads = **msgpack** with a numpy/torch codec. Arrays travel as
  `{b"__ndarray__": True, b"data": bytes, b"dtype": str, b"shape": tuple}`.
- On connect, **server sends a metadata dict first**.
- Client health-checks `GET /healthz` (expects `200 OK`) before opening the ws.
- Each step: client sends `obs` dict → server replies `{"action": ndarray, "server_timing": {...}}`.
- Client sends `{"reset": True}` at rollout start → server calls `policy.reset()`, **no reply**.

**Obs dict the server receives** (flattened with `::`, default R1Pro + `RGBDFullResWrapper`):

| Key | Shape / dtype |
|---|---|
| `robot_r1::proprio` | `(61,)` float32 |
| `robot_r1::robot_r1:zed_link:Camera:0::rgb` (head) | `(720,720,4)` uint8 |
| `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb` | `(480,480,4)` uint8 |
| `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb` | `(480,480,4)` uint8 |
| `...::depth_linear` | `(H,W)` float32, meters `[0,10]` — **only** with `RGBDFullResWrapper` |
| `robot_r1::cam_rel_poses` | `(7*num_cams,)` float32 (cam→base rel pose) |
| `task_id` | `(1,)` int64 |

**Proprio layout** (`PROPRIOCEPTION_INDICES["R1Pro"]`, total 61):
`base_qvel[0:3] arm_left_qpos[3:10] arm_left_qvel[10:17] eef_left_pos[17:20] eef_left_quat[20:24] gripper_left_qpos[24:26] gripper_left_qvel[26:28] arm_right_qpos[28:35] arm_right_qvel[35:42] eef_right_pos[42:45] eef_right_quat[45:49] gripper_right_qpos[49:51] gripper_right_qvel[51:53] trunk_qpos[53:57] trunk_qvel[57:61]`

**Action the server returns** — a single 1-D vector of length `robot.action_dim`.
For the default `omnigibson/eval/r1pro.yaml` config, **action_dim = 23**:
`base[0:3]` (holonomic velocity) · `torso[3:7]` (trunk position) · `left_arm[7:14]` (position) · `left_gripper[14:15]` · `right_arm[15:22]` (position) · `right_gripper[22:23]`.

Note: the evaluator queries the server **every** sim step (no `need_new_action` gating in the current
`_preprocess_obs`), so if you use action chunking, buffer the chunk **inside the server** and pop one per call.

**Wrappers** (`--env-wrapper`, in `omnigibson/eval/wrappers/`):
- `DefaultWrapper` — RGB only, 224×224. Fast debugging, **no depth**.
- `RGBDFullResWrapper` — official: RGB+depth, head 720×720, wrist 480×480. **Use this for real eval.**

**Eval-only rules baked into `evaluator.py`:** base link mass forced to 250 kg (`EVAL_BASE_LINK_MASS`),
head camera horizontal aperture 40.0, `USE_GPU_DYNAMICS=False`, transition rules on. Robot start pose is
**overwritten** per instance from the task's `-tro_state` file. Test instance IDs are `301..340`
(first 20 = public, last 20 = hidden); `--mode public_test --instance-indices 0..9` maps to IDs `301..310`.

**Outputs:** `<output-dir>/json/<task>_<inst>_<rollout>.json` (has `success`, `q_score.final`, `time`,
`agent_distance`, normalized metrics) and, with `--write-video`, `<output-dir>/videos/*.mp4`
(composited head + L/R wrist). **Videos are required for submission.**

## Item 2 — Generating extra visual observations from raw demos

Script: `OmniGibson/scripts/learning/replay_obs.py` (needs OmniGibson → the `behavior` env).

- **Two datasets on HF:** `2026-challenge-rawdata` (1.44 TB, raw HDF5 replay states) and
  `2026-challenge-demos` (3.27 TB, ready-to-train LeRobot v3). Raw data is for **re-rendering your own obs**.
- Input path pattern: `<data_folder>/2026-challenge-rawdata/task-<NNNN>/episode_<demo_id:08d>.hdf5`.
  `task_id = demo_id // 10000` (so demo `0000XXXX` → task 0). Task↔name map lives in
  `2026-challenge-task-instances/metadata/B100_task_misc.csv`.
- It re-opens the exact scene + logged trajectory and **re-renders** `["proprio","rgb","depth_linear"]`
  at head 720×720 (40° aperture) / wrist 480×480, then writes either `hdf5` or a **LeRobot** dataset.
- Why you'd use it: render **extra modalities/resolutions** the released demos don't include (e.g. add
  depth, segmentation for *training-time* privileged signals, different camera settings) without re-teleoping.
  Reminder: privileged modalities are fine for training but **forbidden at eval**.

CLI:
```bash
python OmniGibson/scripts/learning/replay_obs.py \
  --data_folder <root> --demo_id <int> \
  --output_format lerobot   # or hdf5
  # optional: --use_longest_demo  --lerobot_root_dir <dir>  --lerobot_repo_id b1k/<task>  --resume_lerobot
```
Depth is log-quantized to 12-bit on write and dequantized on read (`obs_utils.quantize_depth` /
`dequantize_depth`, range `[0.01, 10.0] m`, `shift=3.5`) — mirror that if you decode videos yourself.

## Item 3 — Minimal policy server (built + tested on this machine ✅)

Files in this folder:
- `minimal_policy_server.py` — self-contained server speaking the exact protocol above. Replace
  `MyPolicy.act` with your model. Vendors the numpy/torch msgpack codec so it runs **without** OmniGibson;
  comments show how to swap in `omnigibson.eval.utils.network_utils.WebsocketPolicyServer` in the real env.
  Default policy = zero action (a valid smoke-test, equivalent to `LocalPolicy`).
- `mock_eval_client.py` — stand-in for OmniGibson's client: waits on `/healthz`, does the metadata
  handshake, sends R1Pro-shaped obs for N steps, asserts action shape `(23,)`, then tests a `reset` roundtrip.

**Test run (2026-07-07), `env_isaaclab`:**
```
python minimal_policy_server.py --host 127.0.0.1 --port 8991 --action-dim 23   # server
python mock_eval_client.py      --host 127.0.0.1 --port 8991 --steps 25        # client
```
Result — **PASS**:
```
[client] healthz OK (200)
[client] connected; server metadata = {'policy': 'minimal-zero'}
[client] step 0 action shape=(23,) dtype=torch.float32 timing={'infer_ms': ~5.3}
[client] ran 25 steps + reset roundtrip OK; action_dim=23
[client] PASS
```
This confirms the handshake, msgpack array (de)serialization, per-step act, and reset semantics are correct.
It does **not** exercise real OmniGibson obs or the physics/scoring — that needs the `behavior` env.

---

## Next steps (to run the *real* eval loop)

1. **Install the challenge env** (user action; not auto-run per repo policy):
   `./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval`
2. Sanity-check with the **built-in zero policy** — no server, no model needed:
   `conda run -n behavior python -m omnigibson.eval.eval --task-name turning_on_radio --policy local --mode public_test --instance-indices 0 --write-video`
   (drives the sim with zero actions; proves OmniGibson + assets + scoring work).
3. Point the **real evaluator** at `minimal_policy_server.py` (or a provided baseline checkpoint):
   start the server, then
   `conda run -n behavior python -m omnigibson.eval.eval --task-name turning_on_radio --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --host 127.0.0.1 --port 8000 --mode public_test --instance-indices 0 1 2 3 4 5 6 7 8 9 --write-video --output-dir eval_logs/turning_on_radio`
4. Download one task's demos and stand up a **baseline** (π0.5 or GR00T) per `docs/challenge/baselines.md`;
   swap it in behind the same websocket interface.

## Install run (started 2026-07-07)

Launched in the background (nohup), logging to `challenge_experiments/setup_install.log`:
```bash
OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data \
./setup.sh --new-env behavior --omnigibson --bddl --joylo --dataset --eval \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

**Storage map (this machine):**

| Data | Size | Destination | Mount |
|---|---|---|---|
| conda env (torch, Isaac Sim, OmniGibson) | ~20–25 GB | `~/miniconda3/envs/behavior` | boot SSD `/` (125 G free) |
| Sim assets + 2026 challenge task instances | tens of GB | `/mnt/nvme/behavior_data` (`OMNIGIBSON_DATA_PATH`) | 2 TB SSD `/mnt/nvme` (947 G) |
| Small demo subset (per-task, for training) | GBs/task | `/mnt/nvme/2026-challenge-demos` | 2 TB SSD `/mnt/nvme` |
| Full demo dataset (all 100 tasks) | 3.27 TB | `/mnt/nas/2026-challenge-demos` | NAS `/mnt/nas` (80 T) |

> ⚠️ **`OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data` must be exported for every eval/replay run too**, not just install — otherwise OmniGibson looks in the repo's `datasets/`. Consider adding it to `~/miniconda3/envs/behavior/etc/conda/activate.d/`.

Reversal: `conda env remove -n behavior` + `rm -rf /mnt/nvme/behavior_data`.

### Full demo dataset download (started 2026-07-07, parallel with install)

Public dataset (not gated, 18,154 files, ~3.27 TB). Launched detached to NAS, logging to
`challenge_experiments/demos_download.log`:
```bash
HF_HOME=/mnt/nvme/hf_home \
hf download behavior-1k/2026-challenge-demos --repo-type dataset \
  --local-dir /mnt/nas/2026-challenge-demos
# wrapped in `until ...; do sleep 30; done` so it auto-resumes on interruption (resumable by design)
```
`HF_HOME` points at the nvme so nothing stages on the 87 G-free boot drive. Monitor with
`du -sh /mnt/nas/2026-challenge-demos` and `tail -f challenge_experiments/demos_download.log`.
To resume manually after a stop, just re-run the same `hf download` command — it skips completed files.

## π0.5 pipeline (OpenPI fork, cloned to `/mnt/nvme/openpi`, branch `behavior`)

Cloned source-only (`GIT_LFS_SKIP_SMUDGE=1 git clone -b behavior --depth 1`). Dataset shapes
(from `/mnt/nas/2026-challenge-demos/meta/info.json`): LeRobot v3, R1Pro, 30 fps, 20k episodes.
`action`=[23], `observation.state`=[61], head RGB/depth 720², wrist 480², `robot2cam_pose.*`=[7].

**Key insight:** the model input state is **23-dim, not 61** — `b1k_policy.py:extract_state_from_proprio`
subsamples proprio via `R1Pro.proprio` indices and sums each 2-finger gripper to 1.
`61 → base_qvel(3)+trunk_qpos(4)+left_arm(7)+left_gripper(1)+right_arm(7)+right_gripper(1) = 23`.

Data prep → model → action, file:line map:
- `src/openpi/policies/b1k_policy.py` — `B1KInputs.__call__` (:61) builds state[23]+images(3×, →`base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`)+prompt; `B1KOutputs` (:110) truncates action to `[:23]`.
- `src/openpi/configs/robots/b1k.py:6` — `R1Pro` RobotConfig (observations/action/proprio index groups; `needs_delta_comp` on torso+arms).
- `src/openpi/configs/tasks/b1k.py` — `TASK_REGISTRY`: task → language prompt.
- `src/openpi/policies/policy.py:68` — `Policy.infer`: input_transform → `model.sample_actions` → output_transform, returns `{"actions":[T,23]}`.
- `src/openpi/shared/eval_b1k_wrapper.py` — `B1KPolicyWrapper`: `process_input` (:68, resize_with_pad 224²), `act` (:356) dispatch; `act_receding_horizon` (:118) infers every `action_horizon`(16) steps, pops one 23-dim action/step. temporal modes at :164/:276.
- `scripts/b1k/serve_b1k.py:44` — serve entry; wraps policy + serves `websocket_b1k_server.WebsocketPolicyServer` (openpi twin of this repo's `network_utils.py`).

Full loop: **`serve_b1k.py` (openpi) ⇄ `omnigibson.eval.eval` (this repo)**; `minimal_policy_server.py` stands in for serve_b1k.

✅ **Obs-key mismatch RESOLVED (Option B, 2026-07-07):** patched the fork's
`src/openpi/configs/robots/b1k.py` — `name="robot"→"robot_r1"` and the 3 `obs_key`s to the
`robot_r1::robot_r1:<link>:Camera:0::rgb` form, matching what the live `eval/r1pro.yaml`
(`name: robot_r1`) emits. `{self.robot.name}::proprio` in `eval_b1k_wrapper.py` now derives
`robot_r1::proprio` automatically; `dataset_key`s unchanged (training unaffected). Swept the whole
fork — `b1k.py` was the only place the naming was pinned. This keeps the canonical `r1pro.yaml`
(the config you submit) untouched. Not yet runtime-tested end-to-end (needs `behavior` env + a served ckpt).

## Smoke test — PASSED (2026-07-07)

Zero-policy eval end-to-end works. Command:
```bash
OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data OMNIGIBSON_HEADLESS=1 \
python -m omnigibson.eval.eval --task-name turning_on_radio --policy local \
  --mode public_test --instance-indices 0 --write-video --output-dir challenge_experiments/smoke_eval
```
Result: `SMOKE_EXIT=0`, instance 301, 3225 steps, success=False, **q_score.final=0.0** (correct for a
do-nothing policy — base moved 0.0004 m). JSON+MP4 written to `challenge_experiments/smoke_eval/`.
Confirms: Isaac Sim + `behavior` env + assets load, task instance resolves, rollout→BDDL scoring→outputs all work.
(The `[Error]` lines in the log are harmless gymnasium `UserWarning`s via Isaac stderr.)

## π0.5 pretrained pass (in progress)
- OpenPI JAX env: `uv sync` at `/mnt/nvme/openpi/.venv` (log `openpi_setup.log`).
- Pretrained ckpt: `pi05TurningOnRadio.zip` → `/mnt/nvme/pi05_pretrained/` via gdown (Drive id `1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2`).
- Serve: `serve_b1k.py --robot b1k/R1Pro --task b1k/turning_on_radio --policy.config pi05_b1k --policy.dir <ckpt>` (uses the `robot_r1` obs-key patch), then `eval.eval` against it. Watch: norm_stats must be bundled in the ckpt or pass `--repo-id`.

## π0.5 serving — working; RGBDFullResWrapper bug found (2026-07-07)

Pretrained ckpt served OK: `pi05_turn_on_the_radio` (12 GB params + bundled `assets/turning_on_radio/norm_stats.json`).
Serve cmd (port 8000 was taken by a pre-existing python pid 664036 → used **8010**):
```bash
cd /mnt/nvme/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 .venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio \
  --policy.config pi05_b1k --policy.dir /mnt/nvme/pi05_pretrained/pi05_turn_on_the_radio \
  --control_mode receding_horizon --action_horizon 16 --port 8010
```
Server loads params in ~8s, "Loaded norm stats from ...assets/turning_on_radio", healthz OK. GPU capped at 0.4 to share with Isaac Sim.

⚠️ **BUG in `omnigibson.eval.wrappers.RGBDFullResWrapper` (the official challenge-track wrapper):** it crashes at
construction. `RGBDFullResWrapper.__init__` calls `env.load_observation_space()` (rgbd_full_res_wrapper.py:41),
which computes `robot.proprioception_dim` → `get_joint_positions()` → `entity_prim.py:859
self._articulation_view.get_joint_positions().view(...)` → **`AttributeError: 'NoneType' has no attribute 'view'`**
because the physics sim view isn't created yet ("Physics Simulation View is not created yet"). The Evaluator only
calls `og.sim.update_handles()` *after* building the wrapper (evaluator.py __init__ order), so the full obs-space
reload runs too early. `DefaultWrapper` avoids it (reloads only per-camera spaces, never the proprio-dependent one).
→ For a working π0.5 eval we use `--env-wrapper omnigibson.eval.wrappers.DefaultWrapper` (224² RGB; π0.5 ignores
depth and resizes to 224 anyway). **RGBDFullResWrapper must be fixed for official submission** (candidate fix: defer
`env.load_observation_space()` until after physics handles exist, or call `og.sim.update_handles()` inside the wrapper
before reloading).

## π0.5 first result (instance 301, DefaultWrapper) — pipeline validated, q=0.0

| metric | zero policy | π0.5 |
|---|---|---|
| success | False | False |
| q_score.final | 0.0 | 0.0 |
| base dist | 0.0004 m | **2.09 m** |
| L/R arm dist | 0.58/0.58 | **2.75/3.73 m** |

Pipeline fully validated: server got the ws connection, inference ran clean for all 3225 steps (no errors),
actions applied, JSON+MP4 written. π0.5 produces **real motion** (navigates + moves arms) vs zero policy standing still.
But scored 0.0 on this single instance — task not completed. Two confounds: (1) `DefaultWrapper` renders native 224²
while π0.5 trained on 720/480→224 (distribution mismatch); (2) n=1. For a fair number: fix RGBDFullResWrapper + run
instances 0–9. Video: `challenge_experiments/pi05_eval/videos/turning_on_radio_301_0.mp4`.

## RGBDFullResWrapper FIX + faithful-res result (2026-07-08)

Fixed `rgbd_full_res_wrapper.py`: added `og.sim.update_handles()` before `env.load_observation_space()`
(+`import omnigibson as og`). Runtime-confirmed: wrapper instantiates, eval runs at 720/480+depth. **The fix works.**

Diagnostic run: RGBDFullResWrapper + `--max-steps 6000` (2×), instance 301:
`steps=6001, success=False, q_score=0.0` — ran the FULL 6000-step budget, still didn't turn on the radio.
Very active: base 5.65 m, arms 7.68/8.98 m (vs DefaultWrapper run base 2.09, arms 2.75/3.73).
→ **Not a timeout.** With 2× time it still fails → failure is in the final **manipulation** (actuating the radio
toggle), not the clock. Video: `challenge_experiments/pi05_eval_rgbd/videos/turning_on_radio_301_0.mp4`.
Caveat: still n=1; provided baseline ckpt may just have a modest per-task success rate. Real signal = aggregate over
instances 0–9 at the DEFAULT 1.5× timeout (submission protocol) with the now-working RGBDFullResWrapper.

## Failure mode (from video) + 0–9 run (2026-07-08)

Video of the full-res run: robot **grasps the radio and carries it to another part of the house instead of
toggling it on**. So the failure is policy behavior (assisted-grasp picks it up, then navigates away), not
timeout/resolution. Decision: keep the wrapper code fix + RGBDFullResWrapper (faithful, official), **revert the
timeout to default 1.5×** (just drop `--max-steps`; no code to revert). Running the real number:
```bash
python -m omnigibson.eval.eval --task-name turning_on_radio \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --host 127.0.0.1 --port 8010 \
  --mode public_test --instance-indices 0 1 2 3 4 5 6 7 8 9 --write-video --output-dir challenge_experiments/pi05_eval_0to9
```
→ aggregate success rate over public instances 0–9 = the leaderboard-style metric. Watcher `bljwavcyo`.

Resolution necessity: RGBDFullResWrapper is required for a faithful/submittable number (challenge track = RGB+depth;
π0.5 trained on 720/480→224 so native-224 DefaultWrapper is a train/test mismatch). Not the failure cause, but the
correct setup. DefaultWrapper only for fast debugging (~2× FPS).

## ✅ π0.5 BASELINE NUMBER — turning_on_radio, instances 0–9 (2026-07-08)

Faithful RGBDFullResWrapper, default 1.5× timeout (3225 steps), provided pretrained ckpt.
**Success 2/10 = 20% · mean q_score = 0.20.** Successes: inst 306 (2557 steps), inst 308 (1525 steps).
All 8 failures ran the full 3225 steps (grab-and-wander until timeout). q_score is binary (0/1) → single goal predicate.
Results: `challenge_experiments/pi05_eval_0to9/` (json + 10 videos). Success videos: 306, 308.

Conclusion: pipeline fully validated AND the model genuinely solves the task sometimes — it's instance-sensitive
(initial pose/state dependent). This matches the challenge design (average over many instances/tasks). The provided
π0.5 baseline is a real ~20% on this task, not a broken setup.

Resolution note (verified in code): model ALWAYS ingests 224² (`model.py:47 IMAGE_RESOLUTION`, `:164-166 resize_with_pad`
in shared preprocess_observation). Dataset is 720/480, downsampled to 224 by the model. Eval wrapper choice only changes
HOW the 224 is produced (RGBDFullRes = downsample-from-720 = training path; DefaultWrapper = native-224 render). Subtle
distribution nuance, NOT the failure cause (same behavior on both). Use RGBDFullRes for faithful/submittable numbers.

## Wrapper A/B: RGBDFullRes vs DefaultWrapper, turning_on_radio 0–9 (2026-07-08)

Same server/ckpt/timeout, only wrapper differs:
- **RGBDFullResWrapper: 2/10** — success on 306 (2557), 308 (1525).
- **DefaultWrapper: 1/10** — success on 310 (1221).
**Success sets are DISJOINT** (306,308 vs 310) — the wrapper doesn't uniformly degrade, it shuffles which instances
pass. → provided baseline is marginal/borderline; small input change (native-224 vs downsampled-from-720) flips
individual instances. DefaultWrapper is NOT a faithful proxy → keep RGBDFullResWrapper for real/submittable numbers;
DefaultWrapper only for coarse debugging. Caveat: n=1/instance, so wrapper-effect vs sim-nondeterminism not fully
separable without k-rollout repeats. Results: `pi05_eval_0to9_default/` vs `pi05_eval_0to9/`.

## Multi-process throughput benchmark (K=2, 2026-07-08)

`challenge_experiments/parallel_bench.sh` — DefaultWrapper, 2000 steps, load-subtracted via a calibration run.
**CAVEAT: GPU was already 100% util from a concurrent `lerobot-train` job (~/co/aic, 24.9 GB) + our π0.5 server
(39.5 GB), ~31 GB free.** So this is throughput-UNDER-LOAD, not clean scaling.

Results: load 110s; single-env stepping 144s = 13.9 steps/s; K=2 stepping 299s/env → aggregate 13.4 steps/s.
**Throughput benefit = 0.97× (≈ none).** Two envs each ran ~2× slower → time-sliced a saturated bottleneck, zero gain.
End-to-end 2 rollouts: serial 399s vs parallel 409s (0.98×). Min free GPU mem 9.8 GB (2 DefaultWrapper sims ≈ 21 GB
peak; K=3 would risk OOM here — guard capped at K=2). Training never at risk.

Interpretation: expected under a pre-saturated GPU (no spare compute to parallelize into). Does NOT prove multi-process
is useless on a FREE GPU. Two tangled bottlenecks this run can't separate: (1) GPU compute saturation from training
(gone on free GPU); (2) single π0.5 inference server serializing across sockets (persists even on free GPU → needs
batched inference or server replicas for real multi-process scaling). Clean test = re-run on an idle GPU + address (2).

## Repeatability / stochasticity (2026-07-08) — the 20% was noise

3 rollouts × 10 instances, RGBDFullResWrapper, provided ckpt. Confirmed STOCHASTIC (diffusion sampling + sim
nondeterminism): inst 301's 3 rollouts took different base-distance paths.
- Repeat: **2/30 = 7%** (inst 305 1/3, inst 308 1/3; all others 0/3). Per-pass: 1/10, 1/10, 0/10.
- Pooled with original 2/10 run → **4/40 ≈ 10% overall**; per-pass swung **0%→20%**. No instance reliably solved
  (308 ~2/4, 305/306 ~1/4, rest 0/4).
- **Takeaway: single 0–9 pass is far too noisy; provided π0.5 on turning_on_radio ≈ ~10% (CI ~3–24%), not 20%.**
  This is why the challenge averages 100 tasks × 10 instances = 1000 rollouts. Reinforces lit-review: winners used
  variance-reduction tricks (multi-sample flow matching, action smoothing, correction rules).
Cloned winning solutions for reference: `/mnt/nvme/behavior-1k-solution` (#1), `/mnt/nvme/openpi-comet` (#2).

## EXPO-FT adoption — Phase 0 setup (2026-07-08)

Direction chosen: sim↔real EXPO-FT bridge (BEHAVIOR sim as RL-dev lab; recipe transfers to aloha-mini).
Repo cloned: `/mnt/nvme/expo-ft` (+ its openpi fork at `expo_ft/agents/vla/openpi`, branch `expo_ft`; server venv via uv).

**How EXPO-FT actually works** (from `expo_ft/agents/alg/expo_ft.py`):
- Base π0.5 is NOT frozen — `update_actor` keeps BC-finetuning it on success episodes only (`actor_success_only=True`)
  → EXPO-FT literally contains RFT inside it (free baseline for ablations).
- Residual "edit" actor: small MLP (256×3, TanhNormal) over action CHUNKS (`full_action_dim=replan_steps×action_dim`),
  output scaled by `edit_scale=0.2` (stability = bounded correction). SAC-style update w/ auto temperature.
- Critic: REDQ ensemble (num_qs=10, min over random 2), own from-scratch ResNet encoder (latent 512), layer norm,
  TD on chunk-level transitions, discount^replan_steps.
- Policy = sample N=8 VLA chunks + 8 residual-edited → argmax target-Q over 16 candidates (rollout AND TD targets).
- Their real-world reward was a hand-coded pixel detector (light2.py: "≤400 yellow px, 5 frames") → BDDL predicates
  are strictly better supervision than what the method was built on.

**Env interface** (5 websocket ops, `expo_ft/env/env_client.py`): create_env / reset / step / get_observation /
get_info_for_step→(done, success, reward, mask). Obs dict: base_image, left_wrist_image, state, prompt.
Reward/done/mask logic lives server-side.

**Our adapter (scaffold, not yet run):** `challenge_experiments/expo_ft_behavior/behavior_env_server.py` —
wraps the challenge Evaluator; reward = dense BDDL partial-credit delta (sparse ablation flag); mask=0 on success /
1 on truncation; TRAIN instances only + optional Comet pose perturbation; 61→23 state extraction; one env per
process (OmniGibson singleton) → parallelism via multiple server processes/ports.

**Known risks:** long horizon (3-6k steps) is exactly where EXPO-FT is unproven (their tasks ~90 steps) — expect
Phase 0 vanilla to struggle; that's the motivating result. 16 VLA samples/replan is expensive → batched sampling +
maybe truncated task variant first. GPU now free (killed our pi05 server, was 39.5GB XLA prealloc; lerobot training done).

## EXPO-FT env adapter — VALIDATED (2026-07-09)

`expo_ft_behavior/behavior_env_server.py` + `smoke_client.py`. Smoke PASS: create_env 85s, reset 7s,
obs contract (state(23,), 3 RGB cams, prompt), dense reward exactly 0.0 under zero actions, mask semantics,
instance cycling + pose perturbation, **10.8 env-steps/s** (3 ws ops/step).

Architecture lesson (3 iterations): Isaac Sim (a) schedules its own asyncio work → can't live inside an async
ws server ("Cannot enter into task"), and (b) installs signal handlers → sim calls must run on the MAIN thread
("signal only works in main thread"). Final design: **ws handler threads do wire-I/O only; main thread owns the
sim and executes ops from a queue.** Also: reward now mirrors TaskMetric exactly (success⇒1.0; else max over
disjunctive goal options of newly-true fraction, initially-true predicates excluded) — my first naive
satisfied/total would have diverged from the leaderboard metric.
(Ops lesson: pkill -f from inside a launcher can match the launcher itself → v3's first "result" was v2's stale
log; watchers now verify a banner line before trusting logs.)

**Phase 0 wiring COMPLETE (2026-07-09), all validated:**
- Server create_env matches train_pi_robo's request ({example_action,env_usage,video_dir}); task via server CLI.
- `convert_demos_to_expoft.py`: LeRobot v3 → per-episode traj.hdf5. Validated: (T,224²,3) uint8 ×3 cams,
  state (T,23), action/joint_full(T,23)+gripper_empty(T,0) → loader concat (T,23) (no loader fork needed).
  ~435 MB/ep; radio eps ~2k steps. Full 20-ep set → /mnt/nvme/expoft_demos/turning_on_radio.
- openpi fork (/mnt/nvme/expo-ft/expo_ft/agents/vla/openpi): added `b1k_policy.py` (B1KInputs, 23-dim state as-is)
  + `B1KExpoDataConfig` + TrainConfig `expo_pi05_b1k_joint_state` (pi05, action_dim 32, horizon 32).
  **CPU-validated**: config resolves, provided ckpt's norm_stats load from bundled assets, fake transition composes
  through repack→B1KInputs→Normalize→model transforms → buffer schema (state(32,), actions(32,32), 3×224², prompt(200)).
- expo-ft configs: `configs/task/behavior_radio.py` (env_type sim, action_space joint_full/empty, control_hz 1000
  = no pacing sleep), `configs/model/expo_ft_b1k_config.py` (b1k TrainConfig name, ckpt params path, assets dir/id,
  freeze encoder).

## 🎉 RUN V0 LIVE (2026-07-09 06:09) — EXPO-FT training on BEHAVIOR

After 4 integration fixes, the full loop runs: **~9.7 env-steps/s** in the main rollout loop (incl. 16-candidate
π0.5 sampling every 8 steps), agent = π0.5-LoRA + residual actor + REDQ(10), GPU 83 GB (learner ~70 + sim ~13).
Debug ledger (each found by iterating, all now fixed):
1. checkpoint dir exists → `--overwrite` needed on fresh runs.
2. full-finetune π0.5 train state OOM @0.55×96GB → **LoRA TrainConfig** `expo_pi05_b1k_joint_state_lora`
   (mirrors their DROID reference; provided ckpt = frozen base) + XLA 0.7.
3. learner restart → 2nd create_env crashed singleton sim → **idempotent create_env** (reuses warm env; env
   `5dce4c4e` survived across learner restarts — scene load amortized).
4. offline demos lack `prompt` → repack KeyError (v0.3's "silent" death = same bug, traceback lost to my
   log rotation) → buffer `insert()` injects task_description as fallback prompt (avoids reconverting 8.6GB).
Ops lessons: watchers MUST treat process-death as terminal (pattern-only watchers sleep through silent kills);
pgrep patterns match wrapper shells — verify the actual python pid; PYTHONFAULTHANDLER=1 on every learner run.
Cadence: ~5.5 min/episode (3225 steps), updates gate at 10 online episodes (~1 hr), then 50 updates/episode.
Logs: `v0_learner.log`, `v0_env_server.log`. Checkpoints: `/mnt/nvme/expoft_runs/expoft_b1k_radio_v0/`.

**v0.6 STEADY STATE (2026-07-09 08:35):** survived the full cycle — rollout → 50 update rounds (critic TD +
π0.5-LoRA BC + residual SAC) → rollout resumed. Fix #5: update-phase OOM (20.5GB alloc; total batch 64×20=1280
materialized+augmented at once + π0.5 TD-target sampling inside critic update) → batch 32 × utd 10 (= 4× smaller
total, but NOTE: utd 20→10 is an ALGORITHMIC weakening vs repo default; batch_split/encode_batch_split=2 are the
memory-free knobs — for v1 restore utd 20 via bigger splits or per-minibatch augmentation).
VRAM ledger (measured): learner 73.6→88GB during updates (π0.5 weights ×3 copies ≈20GB: train/target/infer-cache;
critic weights <1GB — encoder shared, only MLP heads ×10; killer = update activations incl. VLA sampling inside
TD targets). Sim 12.2GB. Learner RSS 94GB system RAM (replay buffers ≈16GB×2 live in RAM). Sim CPU ~11 cores.
Rollout ~9.7 env-steps/s ≈ 3× slower than real time (episode 107.5s sim = ~5.5min wall).
Fast-obs change committed (native-224 render + 224 shipping, ~1.5-2× expected) — takes effect NEXT restart;
NOTE confound: native-224 vs downsampled-224 shifted eval successes in our wrapper A/B — fix one obs mode per
controlled experiment (--full-res flag exists).

## MINI-RADIO PLAN (v1 derisk, decided 2026-07-09)

Rationale (Yosub's call, correct): v0 conflates ~6 variables; first REPLICATE EXPO-FT's known-working regime
(short horizon, near-manipulation start, sparse success reward) in our stack, then extend horizon as the single
controlled variable. Bonus: radio's q_score is single-predicate → our "dense" reward is effectively sparse here
= exactly EXPO-FT's setting. No shorter task exists in B1K (radio ~shortest at ~72s human) → truncated-start is
the move. Target figure: SFT base success from near-radio starts vs EXPO-FT-trained — in an afternoon/day.

Infra (committed c22f1cd): server `--snapshot-record-dir` (buffer sim states every 30 steps; persist window
150-600 steps before success on successful episodes) + `--start-snapshot-dir` (mini-task: reset from pool +
5-step settle + re-snapshot initial predicates). Converter `--trim-last-steps 450` → mini demos
(`/mnt/nvme/expoft_demos/turning_on_radio_mini450`). Same snapshot machinery = reset-to-failure curricula later.

**Overnight (running now):** collection run at 224 obs + snapshot recording (server) + v0.6-config training
(learner `expoft_b1k_radio_v0_224snap`) — successes populate `/mnt/nvme/expoft_snapshots/turning_on_radio`.
UTD 20 deliberately DEFERRED to the 5090 async setup (solo it halves throughput; async hides it — see analysis).
Throughput analysis (measured): rollout 333s/ep full-res (9.7/s), updates ~220s/ep (utd10) → 6.5 eps/hr;
224 → ~8.6; +5090 async actor utd10 → ~16 (learner-bound); utd20 → ~8 but full sample-efficiency.
One async actor SATURATES the learner — learner-side speedups (per-minibatch augment, TD-sample reuse) before
more actors. 3090s: actors (barely, slow) or better as EVAL fleet; cannot host learner. Mini-task: 450-step
episodes → tens of eps/hr solo; EXPO-FT's ~60-130-episode regime = an afternoon.
**When 5090 SSH arrives:** async actor (train_pi_robo_async.py) = sim (~10GB) + π0.5 inference (~8GB) on 5090;
learner solo here w/ XLA 0.9 + utd 20.

## Overnight collection outcome + MINI-RADIO V1 LAUNCH (2026-07-09)

Collection run: **0 successes in ~13 full-length episodes** on TRAIN instances 0–4 (w/ pose perturbation) →
snapshot pool empty. Either bad luck (13 eps @10-15% → 12-25% chance of zero) or train instances are harder
than the test split we measured. Superseded by the deterministic fallback: **`--start-near-object`** (hand-placed
start: teleport base `start_distance`±0.05m in front of the named task-scope object, approach direction biased
to the original spawn side ±30° [avoids clipping into walls/furniture], facing jitter ±7.5°, settle+keep_still).
Added render ticks after both teleport paths — `step_physics()` doesn't render, first obs would ship stale
pre-teleport camera frames.

**Mini-radio v1 RUNNING:** server `--start-near-object radio --max-steps 450` (no perturb-pose — placement has
its own jitter; snapshot recording ON → successes now also bank pre-success states for reset-to-failure later).
Learner `expoft_b1k_miniradio_v1`: dataset = mini450 trimmed demos, **num_updates 15** (scaled from 50: sized
for 3225-step episodes; 15×10utd×32batch ≈ effective paper-UTD ~10 on 450-step episodes — closer to EXPO-FT's
regime; 50 would be UTD ~35, overhot + slow). First readout = base success rate over 20 mini-episodes.

## Mini-radio start-state debugging → EXACT STARTS (2026-07-09)

Mini-v1 (hand-placed, 0.6m, neutral pose): **0/20**. Visual probe vs demo T-450 frame showed why: demos at that
point are AT the table, trunk bent, arms raised over the radio; ours stood back with arms down — fully OOD.
Iterations: demo-posture joint pool (--start-joint-states) + 0.35m + render ticks (teleports leave temporal-AA
ghosting; step_physics doesn't render) → posture right but robot still not at a radio-on-table (instance-dependent
object placement suspected). Isaac swallows our logger.info after launch — debug via probe images, not logs.

**Solution: exact raw-state starts.** `2026-challenge-rawdata/task-0000` = 200 files × ~10MB, each with per-step
SERIALIZED SIM STATES (state: (T+1, 623)). `extract_raw_snapshots.py`: load state at T-450 into our env
(serialized load works despite partial scene — state covers task-relevant entities), verify episode mapping via
action-stream equality vs converted demos (**20/20 True**; raw file id ≈ (lerobot_idx+1)×10 with gaps), save as
dump_state dicts for --start-snapshot-dir. **Probe render ≈ demo frame (near pixel-identical).**
By construction: start distribution == trimmed-demo distribution. Also = replay-based reset-to-ANY-state infra
(reset-to-failure curricula now trivial: any timestep of any demo).

**Mini-v2 RUNNING** (`expoft_b1k_miniradio_v2_exact`): 20 exact starts, instance 0, max 450 steps, num_updates 15.
First readout: base success over 20 episodes.

**Training launch command (once conversion done; server first, then learner):**
```bash
# terminal 1 (behavior env):
OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data OMNIGIBSON_HEADLESS=1 \
python challenge_experiments/expo_ft_behavior/behavior_env_server.py --port 8102 \
  --task-name turning_on_radio --instance-ids 0 1 2 3 4 --perturb-pose
# terminal 2 (expo-ft venv, WANDB_MODE=offline unless logged in):
cd /mnt/nvme/expo-ft && .venv/bin/python train_pi_robo.py \
  --config configs/model/expo_ft_b1k_config.py --config_task configs/task/behavior_radio.py \
  --dataset_path /mnt/nvme/expoft_demos/turning_on_radio \
  --run_name expoft_b1k_radio_v0 --replan_steps 8 --batch_size 64 --utd_ratio 20
```

## Key file references

| Purpose | Path |
|---|---|
| Eval CLI entry | `OmniGibson/omnigibson/eval/eval.py` |
| Rollout loop / eval rules | `OmniGibson/omnigibson/eval/evaluator.py` |
| WS protocol + msgpack codec | `OmniGibson/omnigibson/eval/utils/network_utils.py` |
| Client policy wrapper | `OmniGibson/omnigibson/eval/policies.py` |
| Obs/proprio/action/camera consts | `OmniGibson/omnigibson/eval/utils/eval_utils.py` |
| Depth quantize + video I/O | `OmniGibson/omnigibson/eval/utils/obs_utils.py` |
| Eval wrappers | `OmniGibson/omnigibson/eval/wrappers/` |
| Default robot config (action_dim=23) | `OmniGibson/omnigibson/eval/r1pro.yaml` |
| Raw-demo re-render | `OmniGibson/scripts/learning/replay_obs.py` |

## 🔑 MINI-TASK WINDOW BUG FOUND (2026-07-09, Yosub's video observation) → v5

Yosub spot-checked the mini-demo videos: the last 450 steps show the human PUTTING THE RADIO BACK — no toggle!
Raw-reward analysis confirms: demo sequence = grasp (~T−730) → **toggle (~T−600)** → put back + idle (~600 steps).
So T−450 start states were POST-SUCCESS worlds (radio in-hand, already toggled) and the 450-step trims taught
put-down, not turn-on. 20/20 demos already grasping at T−450. Serialized restore evidently loses ToggledOn
(no insta-successes) and likely the assisted-grasp constraint. BDDL goal is ONLY `toggled_on`; put-back = style.
Explains v4/v4b zeros completely. LESSON: spot-check data windows visually before training on them.

**v5 (running):** starts = raw state at grasp_onset−60 (radio on table, off, grippers open — clean restore);
demos = core trims [onset−60, toggle+30]: 20 segments, len 128/224/427 — the skill is ~7.5 s median!
Episodes end AT the toggle (PredicateGoal terminates on success) so put-back never needed; max-steps 675.
Boundaries: grasp onset = left-gripper width (state dim 14) < 0.02; toggle = first raw reward > 0.5.
Artifacts: /mnt/nvme/expoft_snapshots/demo_starts_pregrasp + /mnt/nvme/expoft_demos/turning_on_radio_core.
Rollout videos now recorded per episode (/mnt/nvme/expoft_videos/miniradio, all successes + last 30 fails);
demo reference videos in /mnt/nvme/expoft_videos/demos. Docs' annotations folder is ABSENT upstream —
base_qvel/gripper boundaries are the label-free substitute.

## 🔑 THE PROMPT BUG + v8 pivot (2026-07-09 evening)

v7 keyframes: policy BACKS AWAY from the table and wanders — from verified-valid radio-on-table starts.
Root cause candidate found: **every env-server run ever (v0 collection AND all mini versions) sent the fallback
prompt "turning on radio"**; the checkpoint trained with "Turn on the radio receiver that's on the table in the
living room." (openpi TASK_REGISTRY). The only correct-prompt runs = serve_b1k evals = the only runs that ever
succeeded (10–20%). v0's 0/13 collection is thus explained by the prompt, not task difficulty.
v7b (correct prompt, same restored starts): STILL retreats — restored mid-demo states remain OOD for the policy
(also: all radiorest starts spawn mid-turn, |base_qvel|≈0.15, yaw −0.15 — operators never stop moving).
Replay probes: navend 1/5 (shortest horizon succeeded — execution layer validated end-to-end);
radiorest 0/5 open-loop (drift with horizon; OmniGibson's own replay uses STATES not actions for this reason).
Websocket probe hung twice (server main-thread wedge, unresolved); direct in-process probe works — use that.

**v8 (running): back to natural task resets (TRO init, official spawn — the KNOWN-GOOD config) + correct prompt +
snapshot recording → expect ~10–20% successes → snapshot pool from the POLICY'S OWN pre-success states
(in-distribution by construction) → rebuild mini-task from those. num_updates 30, full demos, default 1.5× timeout.**
Ops lesson of the day: pgrep/pkill/awk patterns inside launchers self-match their own cmdline (5+ incidents) —
ALWAYS: separate calls: (1) list pids w/ bracketed patterns, (2) kill by number, (3) launch. And disown launchers.

## 🔑 THE DELTA-ACTION BUG — checkpoint is ABSOLUTE (2026-07-09 night, v13→v14)

User artifact report survived TWO posture fixes (keep_still v11, drive-target reset v12→13): arms fly up + head
tilts down at every snapshot-start. That co-movement pattern = commanded, not passive.
**Root cause: my delta→absolute conversion itself.** Git dates settle it: the wensi-fork delta-action feature
(`extra_delta_transform`/`MappedDeltaActions`) merged **2026-06-28**; the provided π0.5 checkpoint was trained
**2026-06-25** — before the feature existed. **The checkpoint outputs ABSOLUTE joint targets.** My conversion
added current state onto absolute commands → arms commanded to ~2× angle (straight up), trunk over-lean (head dip).
Corollaries that now all fit:
- v11's real improvement was the **quantile→mean/std norm fix** (kept), not the delta plumbing.
- Navigation always looked fine: base dims 0:3 were never delta-mapped; only arms/torso were corrupted.
- Current-fork config.py's `extra_delta_transform=True` default describes the NEWER training recipe, not this ckpt.
Reverted: config delta push + expo_ft.py `_deltas_to_absolute` call sites (norm fix retained).
**v14 (running):** clean pipeline — absolute actions untouched, mean/std norm, correct prompt, radiorest starts,
drive-target fix at reset (still correct for snapshot loading). Watch: start posture stable + demo-like arm motion.
Lesson: when adopting a fork's data config for a pre-existing checkpoint, date-check every feature flag against
the checkpoint's training date (`git log -S <flag>`).

## v14 verdict + v15 pivot (2026-07-10 ~01:00)

**v14 (radiorest starts, fully-fixed pipeline): 0/25.** Frame analysis vs demo hdf5 ground truth:
- Demo at radiorest boundary: robot AT table, head down, right arm extended AT radio, near-still for 3s, then grasp.
- Snapshot pool is heterogeneous (grasp-while-driving): ~40% spawn hand-at-radio, ~30% mid-approach, ~30% facing away mid-turn.
- **Damning case ep032: spawns hand-at-radio → policy immediately turns AWAY and wanders to kitchen for 1200 steps.**
  Restored mid-demo states are behaviorally OOD even with correct actions/norm/prompt — v7's retreat, reproduced clean.
- Start-posture artifact: GONE in v14 (arms start down, rise into a forward reach — matches demo phase). Delta-revert confirmed.

**Reinterpretation: v8 (natural starts + correct prompt) failed only because of QUANTILE norm** — it predates the delta
plumbing, so its actions were already correctly absolute. Natural starts with today's fully-fixed pipeline have never
been tried → **v15 (running overnight): natural task resets, full demos dataset, num_updates 30, snapshot-record to
miniradio_own.** Expect serve-level ~10-20% successes → policy-own pre-success snapshot pool → rebuild mini-task
from states the policy actually visits.

## 🔑 SFT-drift hypothesis + v16 pristine probe (2026-07-10 ~03:00)

User caught in v15 videos: ep10 smooth; ep18/19 jerky head+arms THROUGHOUT; start-bounce present even on
NATURAL resets (ep10) → settle-to-quiescence now runs on every reset, not just snapshot restores.
"Different code path?" audit vs yesterday's smooth serve-eval found two real discrepancies:
1. **replan cadence**: serve executes 16 actions per plan (receding_horizon, action_horizon=16); we ran 8 →
   2× the chunk-boundary discontinuities. Now matched at 16.
2. **THE BIG ONE — policy weights are not static in our runs**: train_pi_robo gates updates with
   `can_update = ep_count >= 10`. Episodes 1-10 roll the PRISTINE checkpoint; ep11+ roll checkpoint + 30 LoRA
   SFT updates/episode. v15: ep10 smooth (pristine), ep18/19 jerky (~240 updates in) — **our SFT warm-up recipe
   DEGRADES the policy**; serve-eval never trains, hence always smooth.
**v16 probe (running): num_updates 0 (pristine forever) + replan 16 + settle-everywhere + 3-view videos,
natural resets.** If smooth + serve-level successes (~10-20%) over ~30 eps → entire expo adapter/inference
stack validated; isolated culprit = SFT recipe (suspects: LoRA lr, 20-demo overfit at 30 updates/ep,
action-padding/masking). If still jerky → residual inference-path bug.

## 🔑🔑 THE REAL DELTA STORY — shallow-clone archaeology burned us (2026-07-10 ~07:30)

v16 pristine probe: 0/30, wrist cams show ARMS NEVER LEAVE REST POSTURE all episode (base navigates fine).
That signature broke the case open:
- The "checkpoint predates delta feature" conclusion (yesterday) was WRONG — /mnt/nvme/openpi is a depth-1
  clone, so `git log -S` attributed the whole repo to the one visible merge commit. GitHub API on wensi-ai/openpi:
  the delta feature existed by cb7d32d0 (2026-06-08) and the checkpoint-training commit 03eaee33
  ("pi05 for 2026", 2026-06-25) has LeRobotB1KDataConfig extra_delta_transform=True.
  **The checkpoint IS delta-trained** (torso 3:7, arms 7:14/15:22 vs matching state slices; base+grippers absolute).
- Why v11-13 flew arms UP: expo's process_transformed_outputs fed a DUMMY ZERO state into the output chain
  [Unnormalize, MappedAbsoluteActions, B1KOutputs]. Unnormalize(0) = MEAN state, so the pipeline added the mean
  posture — then my call-site helper added the TRUE state again = DOUBLE-ADD ≈ 2× posture. Arm fly-up + head-down.
- Why v14-16 pinned arms at rest: my revert removed delta entirely → true deltas (≈0) executed as absolute
  position targets ≈ zero posture. Base dims are velocities (absolute) → nav OK, manipulation dead. Matches
  v15 ep18 (reaches table, arms never extend) and v16 wrist cams exactly.
FIX (three layers, mirrors serve exactly):
1. config.py: delta transforms restored (same mappings as 03eaee33).
2. pi05.py process_transformed_outputs(normalized_state=...): true normalized state instead of dummy zeros —
   Unnormalize restores raw state, MappedAbsoluteActions adds it ONCE.
3. expo_ft.py + bc.py call sites pass transformed_inputs["state"].
Norm fix (mean/std) unchanged. **v17 (running): pristine probe, replan 16, num_updates 0** — this is now a true
serve replica through the expo stack. Success bar: smooth + arms extend at the table + ~serve-level successes.
Lesson: NEVER date features with git -S in a shallow clone; check the GitHub API or unshallow first.

**Correction (user, 2026-07-10 morning): "restored states are OOD" is overstated.** All restored-start runs
(v7-v14) executed through the broken action pipeline (zero-pinned arms / mean-state posture corruption) — the
wander-away may have been the execution bug, not the states. Revisit demo-state starts with the fixed pipeline
after ≥1 policy success. Plan of record unchanged: v17 to 30-ep verdict → SFT back on → mini-task from
policy-own snapshots, THEN re-test demo-state starts as a controlled comparison.

## ✅ MILESTONE: expo stack validated against serve (2026-07-10)

User compared v17 failures vs yesterday's serve-eval failures (301/304/309): **"they look the same."**
Both reach the radio, extend the arm, and hover without committing the gripper. Action path audited bit-identical
(B1KOutputs = pure truncation in both forks; eval ws client passes raw actions; same env.step). Conclusion:
reach-and-dither is the CHECKPOINT's dominant failure mode (~80-90% on both stacks), not an adapter bug.
Consistent with demo structure: operators nudge the toggle, no decisive grasp — BC dithers at the commitment point.
The RL thesis writes itself from here: dense-reward RFT/residual should specifically fix the last-centimeter
commitment. Next: v17 to 30-ep verdict (expect ~1-3 successes on train instances) → pre-success snapshots →
v18 with SFT ON (now training deltas consistent with inference) → mini-task from policy-own snapshots →
residual+critic.

## ROADMAP (agreed 2026-07-10): prove EXPO-FT on a short-horizon subtask FIRST

**Now (plan of record):** demonstrate EXPO-FT works on ONE short-horizon target — the last-centimeter
grasp/toggle that both stacks fail 80-90% of the time. Sequence: v17 pristine verdict (30 eps) → harvest
pre-success snapshots on any success → v18 SFT-on (delta-consistent now) → mini-task from policy-own snapshots
(revisit demo-state starts as a controlled comparison — the OOD claim is unproven post-fix) → residual + critic.

**Future roadmap: semantic subtask decomposition** (navigate → reach → grasp → lift/toggle …). Rationale:
1. π0.5's native recipe is hierarchical (predict subtask text → act on it), but the challenge checkpoint was
   fine-tuned FLAT: dataset carries ONE task label per episode (verified: tasks.parquet = 100 whole-task rows,
   no per-frame skill/phase columns).
2. Externalized stage = fix for the Markovian limitation (can't tell "about to grasp" from "done"). 2025 #1
   (Robot Learning Collective) validated the concern with "System 2" stage tracking + inference-time correction
   rules — but via 50 learned task embeddings, NOT language subtasks. Language-subtask hierarchy = open territory.
3. Per-stage EXPO-FT: privileged-predicate stage machine in sim (near-table → EEF-near-radio → IsGrasping →
   ToggledOn), per-stage rewards, horizons of a few hundred steps — exactly EXPO-FT's regime. At eval (no
   privileged state): learned stage classifier from obs, or π0.5's own high-level inference.
**Cheap gating probe before investing:** `--prompt "pick up the radio receiver"` etc. for a few episodes —
does the flat-fine-tuned checkpoint still steer by novel language? If not, stage-conditioning must enter via
embeddings/per-stage fine-tunes instead.

## Findings ledger (as of 2026-07-10 morning)
1. Expo↔BEHAVIOR adapter VALIDATED end-to-end: v17 failures indistinguishable from serve-eval failures (user-confirmed).
2. Checkpoint is delta-trained (torso/arms) + mean/std norm; output chain needs the true normalized state
   (dummy-zero state silently becomes the MEAN state after Unnormalize).
3. Serve executes 16-step receding horizon; match replan_steps=16.
4. Dominant checkpoint failure mode: reaches radio, hovers, never commits the gripper (~80-90% both stacks) —
   the precise commitment gap RFT/residual should close; ideal RL target.
5. train_pi_robo only updates after ep 10 (eps 1-10 are always pristine) — v15's post-ep10 jerkiness says the
   OLD SFT recipe (delta-inconsistent then) degraded the policy; re-evaluate SFT with consistent pipeline in v18.
6. Settle-to-quiescence needed on every reset (head-bounce transient on natural resets too).
7. Restored demo-state starts: wander-away observed ONLY under the broken pipeline — OOD claim unproven, retest.

## 🎉 FIRST SUCCESS THROUGH OUR ENV STACK (2026-07-10 afternoon, parity probe)

**Parity probe ep5 = instance 305: SUCCESS at 2716 steps** (reference serve policy through our env server,
full-res, strict eval-parity resets). 15 pre-success snapshots auto-recorded to miniradio_own — the
policy-own mini-task pool EXISTS now.
The isolating evidence: every config with my reset extras went a POOLED 0/18 (native 0/12 + full-res 0/6);
removing settle/toggle-force converts within 5 episodes. **The settle loop was the suppressor** — the start
transient is in-distribution (reference evals + demos both have it), and freezing the robot for up to 8s +
re-pinning drive targets to a sagged posture pushed starts OFF-distribution. User's head-bounce report was
correct observation, wrong villain: the bounce belongs there.
Also: success on 305 (a serve-eval FAILURE instance) supports stochastic per-episode ~10-20% rather than
fixed instance difficulty.
Chain of custody for the env stack: obs (get_eval_obs) → serve policy → 23-dim actions → env.step, all
byte-equivalent to the challenge harness. Remaining parity probe episodes will give the quantitative rate.
NEXT: finish 20-ep parity probe → expo pristine full-res natural-reset (v18 probe) → if ~serve-level, SFT on.

**Parity probe final: 2/10 (eps 5=305 @2716, 10=310 @2212) — EXACT match to yesterday's serve-eval reference.**
Env stack quantitatively validated. Winning instances differ from yesterday (305/310 vs 306/308) → per-episode
outcomes are stochastic, not fixed instance difficulty. Snapshot pool: 30 pre-success states from 2 successes.
Cut probe at 10 eps (parity established; second pass low marginal value).
**v18 (running): expo pristine probe in the IDENTICAL config** (full-res, parity resets, 301-310, replan 16,
num_updates 0) — the last unvalidated piece is expo's inference internals. Bar: ~2/10-ish. Serve policy server
stopped (VRAM freed for the expo learner).

## 🎉🎉 EXPO STACK FIRST SUCCESS (2026-07-10 evening, v18 ep8)

**v18 ep8 = instance 308: SUCCESS at 1196 steps** — faster than any reference success (serve's best: 1221).
First success through the full expo inference path EVER (100+ episodes across v8-v17: zero). Also user-observed
in ep5 (305): first-ever expo grasp+lift (the commitment behavior that was missing all along).
Validation chain now CLOSED end-to-end:
  reference brain + our env = 2/10 (parity probe) ✓
  expo brain + our env = converts at reference-typical rate, running 1/8 ✓
The entire v8→v18 debugging arc reduced to four load-bearing fixes: (1) mean/std norm, (2) delta actions with
TRUE state through output transforms, (3) replan 16, (4) eval-parity resets (no settle) + full-res rendering.
Snapshots: pool now includes states from the EXPO policy's own successful trajectory.
NEXT: v18 to 20 eps for the rate → v19 = SFT ON (num_updates 30, delta-consistent training) → watch for
v15-style degradation (now diagnosable: pipeline is trusted) → mini-task from policy-own snapshots → residual.

## v19 MINI-TASK LAUNCH — the actual EXPO-FT experiment begins (2026-07-10 night)

Full-pool spawn-check (45 states, user-verified sheets): ep5/ep10 windows = all mid-carry (slow successes,
nav-end precedes the 600-step window); **ep20 b596-b506 = verified PRE-GRASP** (base parked at table, radio
upright, arm poised — user's requested regime). Contact at b476. Snapshot window widened 600→1500 for future
slow successes.
**v19 config:** starts = 4 pre-grasp states (miniradio_pregrasp_own, all from expo's own ep20 success on 308),
max_steps 1200 (~2x the successful remainder), full-res, parity resets for the snapshot path (settle+toggle-force
stay ON there — correct for restores), SFT ON (num_updates 30 after ep10), replan 16, N=1/n_edit 0 (residual off).
Success bar: pristine-policy successes from pre-grasp starts at ≥ full-task rate (~10-20%), then SFT should
push p(success) up — the first measurable EXPO-FT improvement. Watch: (a) AG not needed (pre-grasp avoids the
restore-grasp issue entirely), (b) v15-style SFT degradation after ep10 (now diagnosable on a trusted stack).
Pool caveat: single instance (308), single trajectory — diversity grows with each new success recorded.

**v19 pristine baseline (eps 1-6): 5/6 successes (~83%) from pre-grasp starts** — 658/908/736/748/751 steps.
The checkpoint's grasp→toggle skill was strong all along; full-task failures were navigation/approach-geometry.
Ideal RFT regime: success-rich buffer + measurable headroom. SFT engages at ep11 — metric of record: post-SFT
rate vs this baseline (and NO v15-style degradation).

## v20 jittered mini-task (2026-07-11): the calibrated experiment

User call: 8/10 pristine (v19) = too easy for RL. Difficulty knob = DART-style pose jitter applied AFTER
snapshot restore (was silently overwritten before — fixed), flags --perturb-xy/--perturb-yaw-deg.
Config: pre-grasp starts + jitter ±0.10m/±12°, max_steps 900 (speed: successes cluster 600-900).
**Jittered pristine baseline (eps 1-10): 6/10** — target band achieved. Failure mode: timeouts (reach
geometry unsolved from offset starts). SFT engaged ep11+; first SFT episode: success @599 (fastest yet).
Metric of record: SFT-phase success rate vs 6/10.

**v21 (jitter ±0.15m/±18°) pristine baseline: 4/10.** Difficulty ladder mapped: none=8/10, j10=6/10, j15=4/10 —
a clean monotone dial. SFT engaged ep11+ vs the 40% baseline (best-contrast test of the recipe so far).

## 🔑 THE SFT-DRIFT ROOT CAUSE: BC-ON-FAILURES (2026-07-11)

v21 SFT phase: 0/6 post-update, and USER video verdict: degraded — base stuck mid-turn, gripper opening/closing
in the air, right-gripper jerks (v15's disease, reproduced on the trusted stack with a 4/10 controlled baseline).
**Root cause: `actor_success_only` was never set in our b1k config → train_pi_robo defaults it FALSE → the
actor BC-trained on random replay minibatches = mostly FAILURE transitions.** Every update taught the policy
to imitate its own dithering. The reference config (expo_ft_pi_config.py) sets it True — EXPO-FT's RFT is
BC-on-successes BY DEFINITION; we were running an un-recipe.
This also retroactively explains v15's post-ep10 jerkiness (that run had ~0 successes in buffer → actor BC'd
on pure failure data + offline demos diluted 1:N).
Fix: config.actor_success_only = True. **v22 (running): identical v21 setup (jitter-15, 4/10 pristine
baseline) + success-only actor updates.** Metric: SFT-phase rate vs 4/10 with sane motion.
Lesson: when adopting a method's reference config, DIFF EVERY FIELD against your derived config — silent
defaults invert the algorithm.

## Frozen-eval bracket: the updates are the poison (2026-07-11)

Controlled comparison, same server/starts/harness:
- **ckpt-10000 (~100 success-only updates): 0/8 frozen** (+0/8 live on same weights = 0/16)
- **ckpt-5000 (pre-update = pristine): 5/7 frozen** (Fisher exact vs above: p≈0.007)
~100 gradient steps at WARMUP-scaled lr collapse a 45% policy → structural fault in update_actor's inputs or
step, not gradual overfit. v22's success-batch data path audited clean (demos marked is_success at insert;
sampler correct) → suspects narrowed to (a) replay-buffer re-chunking of ONLINE episodes corrupting BC targets
(demos arrive via a different, offline-converted path), or (b) update mechanics (my hand-authored TrainConfig
lr/schedule; flow-BC × LoRA interaction).
**v23 (running): BCLearner (dagger_b1k_config) — actor BC on DEMOS ONLY, no critic.** Healthy after 20 update
rounds → (a): fix buffer chunking. Degraded → (b): field-diff TrainConfig vs reference.

## 🔑🔑 THE BC-COLLAPSE ROOT CAUSE: chunk backfill breaks anchored deltas (2026-07-11)

v23 (BCLearner, "demos-only") ALSO degraded (1/7 post-update) — which exposed my wrong assumption: with
offline_ratio=0, demos are seeded into the replay buffer and re-chunked by ITS pipeline too. v23 didn't
isolate data source; it isolated the buffer. Config diff vs reference: clean (same lr 2.5e-5, NO warmup,
same LoRA/freeze).
**Bug (expo_ft/data/replay_buffer.py insert):** the streaming chunk backfill preprocesses each transition
ALONE — MappedDeltaActions anchors the action to its OWN state — then writes it into earlier rows' chunk
tails, where a correct target must be anchored to the CHUNK-START state. Every chunk position k>0 stored
a_{t+k}−state_{t+k} instead of a_{t+k}−state_t → targets biased toward zero ("stay put").
**Audit (real demo ep0):** at k=8 the buggy target is off by 0.53 normalized units vs a true signal of 0.70 —
~75% of the action signal erased 8 steps into the chunk, worse deeper. BC on this = trained hesitation;
collapse in ~100 updates at full lr (2.5e-5, warmup=0 — earlier "warmup-scaled" statements were wrong).
Why EXPO-FT never saw it: DROID uses velocity-style (anchor-free) actions — backfill of raw per-step actions
is valid there. Anchored-delta checkpoints (b1k π0.5) are the detonating case.
**Fix:** buffer stores each row's RAW state; backfill re-anchors the incoming raw action to the chunk-start
row's state (delta→normalize→pad, matching pipeline order). Auto-detects MappedDeltaActions in the pipeline
(None → old behavior, DROID semantics preserved). Verified: buffered chunk matches pipeline ground truth to
6e-8. Committed in the expo-ft repo.
**v24 (running): full recipe on the fixed buffer** — success-only RFT, jitter-15 starts, ~45% pristine
reference (9/20 pooled + 5/10 v23). This is attempt #4 at the SFT verdict, now with verified-correct BC targets.

## v24 RFT verdict: FLAT — ceiling = baseline (2026-07-11 night)

v24 final (jitter-15 pre-grasp mini-task, fixed buffer): pristine 5/10, post-update **8/20 (40%)** over 600
success-only BC updates. No collapse (fix holds), no lift. Reading: at this difficulty, failures are
geometry-dominated (hard jitter corners), not behavior-mode-dominated — self-imitation has no signal to
exploit. The RFT ceiling ≈ baseline, exactly the regime where EXPO-FT's actual mechanisms must earn their keep.
**v25 (running): FULL EXPO-FT** — N=8 best-of-N critic selection + residual edits (n_edit_samples=8, reference
scalar edit_scale 0.2 UNMODIFIED per user's parity argument), resumed from v24's ckpt-20000 (RFT'd actor +
~50-episodes-trained critic), --checkpoint_buffer added. Q-values of every candidate set now surface in
sample_info (committed earlier) for critic-quality traces.
Metric of record: success rate vs the 40-50% RFT ceiling. Secondary: does selection convert marginal starts
(watch which candidate idx wins); does the residual explore without destabilizing (user video review).
This run tests EXPO-FT's core claim on our stack for the first time.

## v25 full-recipe dip + Q-trace diagnosis → v26 critic bootstrap (2026-07-10 late night)

Full recipe (N=8 + residual, resumed actor+critic): **0/6.** Q-traces (new q_traces.jsonl logging) diagnose it:
- Critic UNINFORMATIVE: Qs tiny/negative (−0.08..−0.05 — impossible given non-negative rewards; ≈ init noise),
  candidate spread ~0.015, flat across episodes (no progress signal). 30 eps of sparse ToggledOn reward ≪ enough.
- Yet steering: **82% of executed candidates were residual-edited** — noise-ranked random edits displaced the
  competent base policy. Adverse selection = the drought mechanism. NOT plumbing (finite/smooth/varied values).
**Bootstrapping-order finding (paper-relevant): with sparse reward, untrained-critic+residual PREVENTS the
successes the critic needs.** Deadlock-breaker: v26 = selection-only (N=8, n_edit=0) — noise critic picking
among base-policy samples ≈ random good sample ≈ baseline regime → successes flow → critic learns real +1s.
Residual returns when Q-traces show structure (success/failure separation, positive values, rising in-episode).
Tomorrow w/ user: potential-based shaping reward (EEF→radio distance, training-only) to densify critic signal.

## Overnight pivot: RFT is a SLOW LEAK, not flat (2026-07-11 ~00:45)

v24 outcome sequence re-read: pristine 5/10 → update-half-1 5/10 → **update-half-2 3/10**; pooled evidence on
ckpt-20000 weights since: 0/6 (full recipe) + 0/4 (selection-only) → ~3/20 on late-update weights. The fixed-
chunk RFT still degrades, ~10x slower than the anchoring bug. v26 was compounding it (30 updates/ep) — stopped.
**v27 (running overnight): frozen eval of ckpt-10000** (~3 update rounds; ckpts 15000/20000 quarantined so
resume loads 10000). Brackets the leak's onset: ckpt-10000 ≈45% → damage is 10k-20k → suspects: lr 2.5e-5
no-warmup on LoRA (reference uses same, but their update:data ratio differs), online-success data mix, or
residual chunk-boundary artifacts in online success actions. ckpt-10000 low → leak starts immediately, points
harder at lr/optimizer.
Note: selection-only v26 (0/4) used the leaked actor — NOT evidence against selection itself. Redo selection
A/B from a healthy actor after the leak is plugged.

## Overnight bracket verdict: NO leak — SELECTION HARM is the whole story (2026-07-11 ~02:00)

Clean N=1 frozen evals: ckpt-10000 = 4/13 (31%), ckpt-20000 = 5/13 (38%), vs pristine ~48% pooled — all within
noise of each other. **Retraction: the "slow leak" was noise-fitting v24's 10-episode halves. RFT is safe and
flat, full stop.** The post-flip collapse (0/10 pooled under N=8/full recipe on the same weights, Fisher p≈0.02
vs 35%) is entirely SELECTION HARM: an untrained min-of-ensemble critic picks candidates systematically badly
(pessimism correlates with something anti-task — NOT a uniform random pick, my earlier assumption was wrong).
Paper-relevant finding: naive best-of-N under an undertrained critic is worse than no selection.
**v28 (running overnight): N=1 training resumed from ckpt-20000** — actor safely flat, critic ingesting ~40%-
success episodes (the reward events it was starved of). Morning agenda: (1) Q-trace structure check (positive
values? success/failure separation?) → selection A/B retry when structure appears; (2) discuss potential-based
shaping reward (EEF→radio) to accelerate critic; (3) transfer-eval design (user's overfitting critique).

## MORNING BRIEF — the night's findings, reframed by parameter forensics (2026-07-11 ~07:00)

**Headline: the actor never meaningfully trained, eval variance fooled us repeatedly, and the critic-side
machinery has a genuine math-defying anomaly.**
1. **LoRA params moved 0.2% over 450 updates** (norms 281.59→282.11 across ckpts 5000→20000). RFT at
   lr 2.5e-5 × this few steps barely dents the policy — "flat" was parameter-space truth, and ALL cross-
   checkpoint behavioral differences were EVAL VARIANCE (same ckpt-20000 read 38%/24%/8% across three N=1
   evals). My "slow leak" and "restore corruption" hypotheses: both retracted — noise-fitting, twice.
2. **Fixed-eval infrastructure works** (paired frames pixel-match: 2.5 codec noise vs 13-15 unpaired) —
   paired design validated; future claims need n≥40 paired starts for 20-point effects.
3. **Survivors — the real defects:**
   a. Full-recipe collapse 0/10 (untrained pessimistic critic + residual = adverse selection; Q-traces).
   b. **target_actor decay: 281.62 (=actor, ckpt-5000) → 208.74 (ckpt-20000) while the actor moved 0.2%.**
      optax.incremental_update(actor, target, tau) cannot produce this unless actor.get_params() returns
      something OTHER than the true actor params. → MORNING Q1: read Pi05Agent.get_params + actor_tau.
      Broken target policy = broken critic TD targets = uninformative critic (compounds sparse reward).
4. Checkpoint layout understood: params item = actor weights (healthy); agent item = optimizer state +
   target_actor + small nets. Save/restore code audited: correct for ema=None.
**Morning agenda:** (1) get_params/target-update audit → fix; (2) reward shaping (EEF→radio potential) for
critic; (3) RFT dose question — updates too weak to matter; decide higher lr/steps or accept BC-flat and lean
on selection+residual once critic is real; (4) all future evals: fixed-eval, n≥40; (5) 224 A/B + transfer
eval still queued.

**Target-decay forensics complete (07:15):** target lora norms 281.62 (5k) → 265.72 (10k) → 208.74 (20k) —
per-update multiplicative decay ≈0.999 toward ZERO. `optax.incremental_update(get_params(ts), target, tau=0.001)`
with a healthy 282-norm actor cannot do this → the actor tree actually fed to the soft update has lora ≈ 0.
get_params itself reads correct (params-or-ema). Suspect: tree-structure/trainable-split mismatch between
train_step's returned state and target tree (nnx 'value' wrappers / freeze-filter split). Q1 for interactive
morning session — one function chain to trace: train_step → new_train_state.params[lora] content at runtime.

## 🔑 TARGET-INIT BUG SOLVED — the anomaly was never the update (2026-07-11 morning)

Value-level check at ckpt-5000: actor vs target lora leaves have near-identical NORMS (7.689 vs 7.680) but
**cosine similarity 0.0014 — completely independent random draws.** My "impossible decay" was the norm of a
convex mix of near-orthogonal high-dim vectors shrinking at (1-tau)≈0.999/update — textbook, no update bug.
**Root cause: build_pi05's non-resume branch initializes target_actor_params via init_target_params = a FRESH
model with independently-random LoRA** instead of copying the actor. With tau=0.001, the target spends ~1000s
of updates as a random-LoRA policy — and the critic's TD targets sample next-actions from it → the critic has
been learning values of a NONSENSE policy in every run. Third load-bearing expo-ft bug (after chunk anchoring
and actor_success_only default); one-line fix (always copy), committed. Worth upstreaming all three.
Failure-chain now fully causally closed: sparse reward + random-LoRA target → empty/garbage critic →
adverse selection under best-of-N → full-recipe collapse. RFT side separately: dose too small to matter.
NEXT RUN (v31, user go): fixed pipeline + shaping reward decision + critic re-trained from scratch (old critic
learned garbage targets — must reinit), then selection A/B on fixed-eval set.

## v32 verdict + GRASP SUBTASK pivot (2026-07-11 midday, user-directed)

**v32 paired eval: 0/18 on the fixed set where pristine scored 9/20 — v31's update phase GENUINELY damaged
the actor** (despite 0.2% param-norm delta; norms don't bound behavioral change — inverse of the target-init
lesson). Update-recipe damage is now the one standing pipeline defect, reproducible and paired-measurable.
Also (user insight + their code): EXPO-FT's own rewards are BINARY (light2 = yellow-pixel ROI success DETECTOR
thresholded to 0/1; pick same; demos padded sparse-terminal). Their advantage = event density via HORIZON:
~90-step episodes ≈ 6 critic chunks vs our 900/56. Density-of-events, not gradedness.
**Pivot (user): grasp subtask** — start = pre-grasp pool (base parked), success = is_grasping(radio) sustained
15 steps (either arm), early-fail = radio falls (z drop >0.25, the observed sideways-knock), cap 300 steps.
~EXPO-FT's horizon regime; episodes 1-2 min → update-recipe A/Bs in ~30 min. New dials: object-pose DART
(--perturb-obj-xy/-yaw-deg) if the baseline is too easy (user's call).
**v33 (running): pristine baseline on 20 fixed starts, subtask=grasp, jitter-15, shaping 0.1.**

## Grasp-subtask testbed established (2026-07-11 afternoon)

User video review caught two detector defects (real grab scored fail; button-press scored success) → root cause:
goal-object selection picked the AGENT (robot_r1 passes a name-based "agent" filter; scope order:
[agent, radio, table, floor]) → detector/fall/shaping all watched the robot. Fixed: name-matched selection
(verified in logs: radio_89) + streak-only success (no BDDL passthrough). Also: logger.info is SUPPRESSED
after Isaac launch — module diagnostics must use logger.warning.
**Difficulty ladder (grasp, pre-grasp starts, fixed 20 starts, cap 600):**
- robot jitter 15/18° only: **17/20 (85%)**
- + radio jitter ±6cm/±30°: **12/20 (60%)** ✓ band
Reset artifact surfaced: occasional radio-knock during restore+jitter+settle → ~21-step auto-fail (~1/20).
**Control-frequency comparison (user Q):** EXPO-FT runs 10 Hz, 90-step ≈ 9s episodes ≈ 11 decisions; ours 30 Hz
(baked into checkpoint — retiming not viable), 600 steps ≈ 20s ≈ 37 decisions @replan16. Critic verified to do
proper semi-MDP per-chunk discounting (target = R_chunk + γ^replan·Q'; buffer sums γ^i rewards w/ masks) —
effective horizon = decision count. **v36 (running): replan-32 A/B on same fixed starts (halves decisions to
~19; commitment 1.07s vs EXPO-FT's 0.8s). Ref: 12/20.** Closer-starts option deferred (user: arm-out starts
likely awkward).

## 🔑🔑🔑 REPLAN-32 A/B: full-chunk execution CURES the commitment failure (2026-07-11)

**Paired verdict on identical fixed starts (grasp subtask, obj-jitter):**
- replan-16 (serve-standard receding horizon): **12/20 (60%)**
- replan-32 (execute full chunk, no mid-chunk replanning): **18/20 (90%)** — 17/19 excluding one
  radio-spawned-in-gripper freebie (ep40, 15-step success). p<0.01.
Successes also FASTER (median ~340 vs ~490 steps). Mechanism: each replan lets the flow model resample a
slightly different intention; consecutive-chunk disagreement = the hesitation/dithering that defined this
checkpoint's failure mode since the first serve evals. Committing to the full 32-step chunk (1.07s) removes
the re-decision points. Decision count per success ≈ 11 — EXPO-FT's regime achieved WITHOUT closer starts.
Implications: (1) replan-32 = new standard for the testbed; (2) re-check FULL-task rate at replan-32 later —
serve's 2/10 may be beatable with a one-flag change; (3) paper-worthy standalone finding (receding-horizon
replanning induces commitment failure in flow-matching VLAs).
NEXT: the standing defect — update-recipe A/B on this testbed (train ~15 eps full recipe @ replan-32 →
paired frozen eval vs the 18/20 pristine reference).

## v37 update-recipe A/B at replan-32: graded erosion confirmed (2026-07-11 evening)

v37 (full fixed recipe, lr 2.5e-5): baseline 10/10, **post-update 13/20 (65%)** vs pristine 18/20 (90%).
Data quality fully exonerated this time (buffer nearly all successes). With v23's demos-only damage (post-
buffer-fix), the suspect = UPDATE MECHANICS. Erosion is graded (~25 pts/600 updates), not collapse — fits
too-hot-lr on LoRA-over-specialized-base (their 2.5e-5 was tuned for LoRA on generic pi05_base).
**v38 (running): identical run at lr 2.5e-6** (new TrainConfig expo_pi05_b1k_joint_state_lora_lowlr +
configs/model/expo_ft_b1k_lowlr_config.py). Damage gone → tune lr upward for actual learning. Persists →
diff flow-loss/timestep-sampling vs wensi's original training code.

## ✅ PIPELINE-HEALTH ARC CLOSED: lr was the last defect (2026-07-11 night)

**v38 (lr 2.5e-6, all else = v37): baseline 6/10 (phase-shifted start subset), post-update 16/20 (80%)** —
statistically indistinguishable from pristine 18/20; vs hot-lr v37's 13/20. Single pairwise n=20 isn't
p<0.05 alone, but hot-lr damage replicated across v31/v32 (full mini-task) and v37 (grasp), and low-lr
matches pristine → practical verdict: **actor lr 2.5e-6 = safe default for LoRA on the task-specialized
checkpoint; 2.5e-5 (reference default, tuned for LoRA-on-generic-base) slowly erodes competence.**
Full defect ledger of the expo-ft adoption, all closed: (1) delta-anchor chunk backfill, (2) actor_success_only
default, (3) mean/std-vs-quantile + output-state plumbing, (4) replan-16 dithering (32 = cure + 90% baseline),
(5) actor lr. Whether 2.5e-6 can also LEARN (vs merely not-harm) = open; likely needs the restored-headroom
band to answer.
**5090 (new node)**: Windows 11 + RTX 5090 32GB; native-Windows install running (setup.ps1 -Eval -Dataset;
WSL2 ruled out for Isaac/Vulkan; llama-servers killed → 31.8GB VRAM freed). Role: full-task replan-32 eval,
then eval farm. Policy serving via SSH reverse tunnel from this box until WSL JAX is warranted.
NEXT: harder grasp band (radio jitter up) → RFT-lift attempt at safe lr → critic Q-structure gate → selection.

## Difficulty dial-sweep complete → grasplift 65% band → v42 RFT-LIFT ATTEMPT (2026-07-11 late)

Pose dials exhausted (policy at replan-32 visually re-solves any start: grasp 77-85% across jitter maxima).
**Criterion escalation restored the band: grasplift (hold + raise 0.15m sustained) = 13/20 (65%)** at full
jitter (robot ±0.22m/25°, radio ±0.10m/45°). Failure modes: drop-during-lift, hard-corner reaches — trainable.
**v42 (running, overnight): the actual RFT-lift experiment** — updates ON at safe lr 2.5e-6, 60 eps.
Baseline eps 1-10 pristine, updates from ep11. THE question: does BC-on-own-successes climb above 65%?
5090: setup mid-install (log growing; -JoyLo flag fixed the -Eval dependency).

## 5090 root cause found: driver 610.47 incompatible with Isaac 5.1 RTX renderer (2026-07-12 early)

Smoke test crashed 0xc0000139 + access violation in BOTH SYSTEM and user contexts; VC++ redist didn't fix.
Differential: bare kit (`SimulationApp({'headless': True})`, no OmniGibson) crashes identically → not our code.
Minidump (breakpad, parsed with python `minidump` locally): access violation in
**rtx.scenedb.plugin.dll +0xe533b** during RTX renderer bring-up (~5.4s in). AMD-iGPU masking
(VK_LOADER_DRIVERS_DISABLE) + multi_gpu:False did NOT help — not a device-selection issue.
**KNOWN upstream bug**: Isaac Sim 5.1 crashes in rtx.scenedb.plugin on driver branches newer than R580
(595.x, 610.x) — isaac-sim/IsaacSim#651, #517, NVIDIA forums (RTX 5060Ti/5080/5090 reports, Win+Linux).
Box runs 610.47; validated Windows driver = **580.88**. Fix in flight: download 580.88 → silent install
(-s -noreboot) → reboot (sshd is StartType=Automatic, survives) → re-run kit test → smoke test.
Box hardware note: Ryzen 5 9600X w/ AMD iGPU active alongside the 5090 (dual-adapter enumeration is benign).
v42 meanwhile: 30/48 (62.5%) vs 65% baseline — training safe at 2.5e-6, no lift signal so far.

## 5090 OPERATIONAL (2026-07-12 ~03:35)

Smoke test green (exit 0: empty-scene og.Environment + step + 3 renders, clean shutdown). Three stacked fixes:
1. **NVIDIA driver 610.47 → 580.88** (silent install -s -noreboot + reboot): cured rtx.scenedb.plugin access
   violation (known Isaac 5.1 x R590+/610 incompat, isaac-sim/IsaacSim#651).
2. **h5py 3.16.0 → 3.15.1**: cured 0xc0000139 on generic_mo_io.dll (omni.sensors.nv lidar/radar link HDF5;
   h5py 3.16's hdf5.dll poisons the process — IsaacLab discussion #5503).
3. **KMP_DUPLICATE_LIB_OK=TRUE**: cured OMP Error #15 (kit libomp.dll vs MKL libiomp5md.dll) — must be set
   in every launch script on this box.
NEXT on 5090: full-task replan-32 eval — Linux box serves policy (after v42 frees the GPU), reverse tunnel
`ssh -R 8010:localhost:8010`, challenge eval loop runs on 5090.

## v42 VERDICT: no RFT lift at safe lr (2026-07-12 ~05:30)

**v42 final: baseline 8/10, post-update 27/50 (54%)** vs v41 pristine 13/20 (65%), same fixed starts,
grasplift @ full jitter, replan-32, lr 2.5e-6, 50 update-episodes (~1500 grad steps). Flat-to-slightly-down;
z≈0.9, not a significant drop, but definitively NO CLIMB. Combined verdict across the lr axis: 2.5e-5 erodes
(v37), 2.5e-6 doesn't move (v38 no-harm, v42 no-lift). **BC-on-own-successes (RFT) is a dead end in this band
— the lift must come from the critic path** (Q-structure gate → selection A/B → residual edits), as queued.
Ordered outcomes: v42_rft_35of60.txt; videos archive_v42.

## v43 LAUNCHED: full-task replan-32 eval, sim on 5090 (2026-07-12 ~05:30)

5090 full stack proven: env server bound :8102, create_env loads turning_on_radio (task-instances dataset
downloaded after enabling Windows long paths — zipfile extract needs LongPathsEnabled=1), reset+get_observation
round-trip OK, head-cam renders the scene correctly. Windows gotcha: ssh -L must target 127.0.0.1 not
"localhost" (Windows resolves localhost→::1, server binds 0.0.0.0 IPv4 only).
**v43 = the queued challenge-relevant experiment**: full task (natural resets, train instances 0-4, no
subtask, no jitter, --full-res), pristine checkpoint (--num_updates 0), replan-32. Pairs against the ~2/10
replan-16 serve-parity reads. Learner on this box (4090, freed by v42 completion) ↔ tunnel ↔ 5090 sim.
If replan-32 lifts the full task like it lifted grasp (12→18/20), that's a free challenge-baseline boost.

## ★ FULL-TASK REPLAN-32 CONFIRMED: 15% → 48%, p=0.043 (2026-07-12 ~10:40)

Paired same-day eval, full task, natural resets, train instances 0-4, pristine checkpoint, no updates:
- **v43 replan-32 (sim on 5090, policy local via tunnel): 10/21 (47.6%)**, still accruing bonus episodes
- **v44 replan-16 control (all-local): 3/20 (15%)** — reproduces the historical ~2/10 serve-parity reads
- Fisher exact p=0.043. Successes finish in 1173-1940 steps; failures hit the 3225-step cap.
**The receding-horizon dithering disease is the dominant full-task failure mode, and full-chunk execution
cures ~2.5x of it — a one-line inference change, directly bankable for the challenge submission.**
This also replicates the grasp-subtask finding (12/20→18/20) at full-task scale: mini-task conclusions
transfer. Outcome records: v44_fulltask_replan16_3of20.txt, v43 list pending its stop.
Ops note: paired cross-machine eval works (v43 learner restarted mid-run @ XLA fraction 0.35 after v44 OOM;
create_env is idempotent so the 5090 env survived the learner swap; local env server needs
OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data explicitly).

## v45 Q-STRUCTURE GATE: PASS (2026-07-12 ~13:50)

Frozen probe: v42 checkpoint (first critic trained on correctly-anchored data), N=8 argmax-Q selection,
no edits, no updates, grasplift fixed starts. From 10 traced episodes:
- **Success/fail separation ~6σ**: success episode-mean Q −0.000±.002 vs fail −0.012±.001
- **In-episode rise**: +0.061 in successes vs +0.006 (flat) in fails — critic tracks progress
- **No selection degeneracy**: picks spread across all 8 candidate indices (v22 pathology absent)
- Caveats: Q magnitudes tiny (range −0.037..0.065, heavy pessimistic underestimation — ordering is what
  matters for selection, but calibration is poor); **within-replan candidate spread only ~0.004** — the
  critic discriminates states far better than actions. Selection lift, if any, will be modest until the
  critic sharpens on-policy.
Behavioral read so far (selection active): 8/10 vs 13/20 (65%) N=1 baseline on the same fixed-start set —
running to n=20 for the A/B verdict. This probe doubles as the selection arm.
v43 meanwhile: 12/29, stopping at 30.

## v45 SELECTION A/B: FLAT (2026-07-12 ~14:05)

N=8 argmax-Q selection (frozen v42 critic) vs v41 N=1, paired by start (uid%20): 15/20 starts agree,
2 flips up / 3 down, 12/20 vs 13/20. Selection is behaviorally NEUTRAL — adverse selection cured (v22:
0/10), but no lift. Cause per gate caveat: critic has state-value structure but ~no action discrimination
(candidate spread 0.004 << state separation 0.012). Corollary: outcomes in the 65% band are mostly
START-determined; all 8 candidates share the same fate at hard starts. Fixing those needs a better policy
(training), not better chunk-picking.
**Recommendation: run the full reference recipe end-to-end** — critic updates + N=8 selection + residual
edits ON, BC on successes, safe lr — and let the critic sharpen on-policy; re-read action-spread and the
frozen-eval curve as training progresses (eval farm on 5090). All components now individually validated:
clean buffer, safe lr, critic learns state values, selection harmless. Records: v45_selection_13of22.txt.

v43 FINAL at n=30: **12/30 (40%) vs replan-16 control 3/20 (15%), Fisher p=0.069** — the point estimate
softened from the n=21 read (48%) but the direction is unchanged; ~2.5x. Record: v43_fulltask_replan32_12of30.txt.
5090 env server left loaded (eval-farm role). Learner stopped, local GPU freed for the full-recipe run.

## v46 LAUNCHED: full reference recipe (2026-07-12 ~14:30)

Per lookahead-vs-training decision (lookahead parked as later diagnostic/ablation; checkpoint archived so
it stays runnable): **v46 = critic+actor updates (safe lr) + N=8 selection + n_edit_samples=8, reference
edit_scale**, resumed from v42 ckpt+buffer (copied to new run dir expoft_b1k_grasplift_v46_fullrecipe for
provenance). Grasplift fixed starts, replan-32, 60-ep target. Hypothesis: selection+edits live → action
contrast in the buffer → critic develops A(s,a). Metrics: (1) candidate Q-spread across checkpoints
(0.004 → toward 0.012 state-level = learning), (2) rolling success vs 65% band, (3) frozen 20-start evals
at checkpoint saves. Mem fractions: learner 0.5.

v46 RESTART at reference batch/utd (user-caught deviation, 2026-07-12 ~15:40): we had been running
batch 32/utd 10 since the early OOM era — reference is **64/20** (train_pi_robo flag defaults). That's 4x
less critic gradient per data than intended, plausibly co-responsible for the weak action discrimination.
96GB card + chunked batches + 0.5 fraction now afford parity. First 4 utd-10 episodes archived
(archive_v46_utd10, q_traces .utd10_first4eps). Expect ~30-40 min/ep; read curve at ~30 eps (~1 day).
Sixth adoption deviation for the ledger: batch/utd halving.

v46 OOM saga → SIM MOVED TO 5090 (2026-07-12 ~16:30): reference 64/20 update graph needs ~48GB live +
a 19GB TD-sampling tensor; OOM'd at 0.5, 0.68 (batch_split 2), and 0.72 (batch_split 4) with the local
sim's 14GB cohabiting. Fix: grasplift env server now runs ON the 5090 (:8103, snapshots copied to
C:\co\snapshots, videos to C:\co\videos_grasplift), learner gets 0.85×96GB locally. This is the
"5090 as actor" arrangement the user proposed — adopted for VRAM, not throughput. 8103 tunnel has
auto-heal in the progress monitor. batch_split=4 kept (math-preserving forward chunking).

## v46 STOPPED EARLY at 7/23 (user-approved) → v47 DECOMPOSITION EVAL (2026-07-13 ~10:10)

v46 final read: **7/23 (30%) vs 65% band — the full recipe HURT rollout performance.** Pre-registered
metrics both negative: candidate spread collapsed back to 0.004-0.008 (recalibration shock, no action
discrimination emerged from on-policy training at reference 64/20); success/fail Q-separation compressed
to noise. Rolling windows 3/10, 4/10, 3/10. Mechanism hypothesis: with a non-discriminating critic,
argmax-16 ≈ uniform draw and half the candidates are noise-edited chunks → ~every other executed chunk
is noised (mild v22 adverse selection that critic training never fixed).
**v47 (running): the decomposition** — v46 checkpoint step 15000, N=1, no edits, frozen, fixed 20 starts.
Reads: ~65% → weights intact, harm was rollout-time edit noise (improvement operator broken for this
checkpoint; crisp negative for the recipe). ~35% → training also damaged weights (reopens lr/critic-grad).
Records: v46_fullrecipe_7of23.txt; v46 train videos on 5090 C:\co\videos_v46train.

## ★ v47 DECOMPOSITION VERDICT: WEIGHTS INTACT — the harm was the improvement operator (2026-07-13 ~11:15)

**v47 (v46 ckpt step 15000, N=1, no edits, frozen, fixed starts): 14/21 (67%)** — statistically identical
to the 65% pristine band. v46's 30% was therefore ENTIRELY rollout-time: argmax-Q over 16 candidates with
a non-discriminating critic ≈ uniform draw, half the candidates are noise-edited chunks → ~every other
executed chunk was noised. Training at reference 64/20 + safe actor lr is weight-SAFE (no erosion) but
the recipe's improvement operator (critic selection + residual edits) is non-functional for this
checkpoint: the critic never develops action discrimination, even from on-policy contrast data.
Recipe status after the full arc (v42 RFT flat, v45 selection flat, v46 full recipe harmful-at-rollout,
v47 weights clean): **EXPO-FT's mechanisms do not lift this task-specialized flow-matching checkpoint in
this band. The only intervention that moved the number remains inference-time commitment (replan-32,
15%→40-48% on the full task).** Next-fork options for user: (a) sim-lookahead V-sufficiency diagnostic
(publishing ablation), (b) shaping/horizon axis (denser events), (c) bank replan-32 + focus on challenge
submission pipeline. Records: v47_decomp_14of21.txt.

## v49 LAUNCHED: sim-lookahead V-sufficiency diagnostic (2026-07-13 ~16:55, user picked fork 1)

The user's insight operationalized: enumerate actions + perfect model (the sim IS the model) + learned V.
New infra: env server op `lookahead_probe(actions)` — dump_state → execute candidate chunk → return
terminal obs + ORACLE signals (grasping bool, obj_z, phi) → load_state + drive-target re-pin (episode
untouched); agent methods `sample_candidate_actions` (N plain flow samples, no edits/selection) and
`estimate_state_value` (min-ensemble Q(s,a~pi), the TD-target estimator); train_pi_robo `--lookahead_n N`
branch + lookahead_traces.jsonl (per-replan V of all 8 candidates + chosen idx + oracle).
v49 config: v46 ckpt (weights=pristine-equivalent per v47; critic=trained), N=8 plain candidates,
replan-32, fixed 20 starts, frozen. ~10-15 min/ep (8x sim). Baselines: N=1 65% (v41/v47), Q-select flat
(v45). READS: lookahead >> 65% → V is control-sufficient, deficit was the missing implicit model (→
distill lookahead picks / train Q against probe outcomes). Lookahead ≈ 65% → V itself lacks action-relevant
signal → the critic axis is dead for this checkpoint; pivot to event-density (fork 2) or bank inference wins.
BONUS from oracle logs: correlation of V(s') with ground-truth grasp/obj_z per candidate = direct measure of
what V sees, independent of episode outcomes. v48 held-out full-task eval continues in parallel (5/7 so far).

v49 take-2 healthy (counter fix verified: 19 replans, 601-step episode). MAJOR side-discovery from oracle
logs: some fixed-eval starts have the radio ON THE FLOOR at spawn (obj_z=0.046 vs table 0.53) — the
±0.10m object jitter pushes it off the table edge for certain (snapshot, dx, dy) tuples → those starts are
UNWINNABLE for any policy. Likely explains the start-determinism (75% cross-arm agreement) and the ~65-75%
ceiling of every grasplift eval. TODO after v49: classify all 20 fixed starts by spawn obj_z (oracle now
logs it), recompute all historical rates on the winnable subset, and consider re-rolling the fixed-eval
tuples with a table-bounds check.

## v49 LOOKAHEAD VERDICT: V steers correctly; no outcome lift; failure = criterion-marginal lifts (2026-07-13 ~20:15)

**9/20 (45%; 9/19=47% on winnable table-starts) vs N=1 65%, Q-select 59% — no lift, mildly below.**
BUT the mechanism data is unambiguous and answers the fork-1 question:
- **V + perfect model = competent selector**: grasp-pick fidelity 56/57; during lifts the pick tracks the
  best-lifting candidate within ~1cm. The same critic that was useless for direct Q(s,a) ranking (v45
  spread 0.004) becomes accurate when the sim supplies the dynamics — the user's V-sufficiency conjecture
  CONFIRMED at the mechanism level; the Q-head's missing piece is exactly the implicit model.
- **Why no outcome lift**: greedy 1-chunk V-optimization walks into criterion-marginal states — failures
  lift to 0.637-0.673 vs 0.684 threshold and sag/time out (grip too weak to exceed +0.15m, or cap hits
  mid-climb). V has no preference gradient beyond the best of 8 samples, and horizon-32 greed doesn't plan
  grip quality for the LATER lift. Also one bug found+fixed en route: probe steps advanced the env's
  python-side episode counter → truncation at step 65 (counter now saved/restored in lookahead_probe).
- Interpretation for the recipe: selection CAN'T be rescued by a better selector — even oracle-model
  selection with honest V tops out at baseline. The band's residual failures are physics-marginal, not
  choice-limited. RL on this checkpoint needs either a different objective (grip quality / lift shaping)
  or a different criterion. Also 1/20 fixed starts confirmed floor-radio (unwinnable).
Records: v49_lookahead_9of20.txt; lookahead_traces.jsonl (+.buggy_counter).

## v48 HELD-OUT VERDICT: replan-32 generalizes — 15/20 (75%) on unseen instances (2026-07-13 ~23:20)

Full task, pristine ckpt, replan-32, natural resets, INSTANCES 5-9 (never used in any tuning/eval before):
**15/20 (75%)** vs 12/30 (40%) on instances 0-4. The replan-32 gain is not instance-overfit — if anything
0-4 are the hard draw. Combined full-task picture at replan-32: 27/50 (54%) across 10 instances vs
replan-16's 3/20 (15%) on 0-4. Addresses the user's overfitting critique for the inference-time win.
Record: v48_heldout_15of20.txt. All GPUs idle now — day closed with: replan-32 validated + generalizing;
EXPO-FT mechanisms exhausted (RFT/selection/edits/full-recipe/oracle-lookahead all flat-or-harmful with
weights intact); V-sufficiency confirmed mechanistically; fork-2 target sharpened to lift/grip shaping.

Floor-start census (2026-07-14 ~00:15, idle-time diagnostic via 1-noop lookahead_probe on all 20 fixed
starts): **1/20 spawns the radio on the floor** (z=0.095; one more borderline at 0.513 vs nominal 0.534).
Contamination = 5%: fixed-set ceiling ~19/20, the "65% band" ≈ 68% of winnable. Historical A/B comparisons
unaffected (shared set). Record: fixed_starts_floor_census.txt.

## v50 LAUNCHED: staged-shaping run (fork 2), phase 1 (2026-07-14 ~06:00)

New shaping mode in env server (--shaping-mode staged): phi = 1-tanh(dist) pre-grasp -> +1 step on grasp
-> 1 + clip(dz/0.15,0,1) while grasping. Potential-based (state-only, difference form) so optimal policy
unchanged; targets exactly the v49-diagnosed failure (lift stall + grip slip). Causal chain being tested:
staged phi -> critic learns action-relevant lift gradient -> (phase 2) residual actor ascends Q beyond the
base policy's sample support — the one mechanism v42-v49 never gave a fair shot.
v50 phase 1 (running): FRESH run (pristine actor, fresh critic — no old-shaping reward contamination in
the buffer), N=1 no edits (reference warm-up protocol; v8 lesson), 64/20 + batch_split 4, safe lr, staged
coef 0.1, ~45 min/ep, 25 eps (~19h) -> then flip config to N=8 + 8 edits (phase 2) and watch: rolling
success vs 65% band, candidate Q-spread, and (via any lookahead probes) max candidate lift height.

## v50 GATE PASSED → PHASE 2 LAUNCHED (2026-07-14 ~14:50)

Phase 1 final: 17/25 (68%), no erosion. Frozen lookahead probe (3 eps, v50 critic):
- **candidate V-spread 0.024 = 4x the terminal-reward critic** — action discrimination EMERGED from staged reward
- **grasp/non-grasp gap +0.072 vs +0.009 = 10x** the old 0.006
- corr(V, obj_z | grasping) = −0.61: NOT a defect — textbook potential-shaping offset (V_shaped = V − coef·φ(s));
  progress incentive lives in Q's action term where φ(s) cancels across candidates. LESSON: lookahead-style
  cross-state ranking with a shaped critic must un-shape first (rank V + coef·φ(s')). Probe metric corrected.
- Phase-1 actor already samples lifts to z=0.757 (> 0.684 threshold; pristine ceiling was 0.673) — BC on
  staged-era successes shifted the distribution upward. Probe eps went 2/3.
**Phase 2 running**: N=8 + 8 residual edits, staged reward, 64/20, resume from phase-1 ckpt+buffer.
Watch: rolling success vs 68%, q_traces spread, no v46-style edit-noise collapse (the phase-1-trained critic
is the difference vs v46). Verdict: frozen paired eval after ~30 phase-2 eps.

## v50 PHASE-2 KILLED at 2/9 [FSSFFFFFF] — v51 decomposition running (2026-07-14 ~21:15)

Kill rule (≤4/10) mathematically guaranteed at 2/9; six consecutive fails, edit-picks 76-89% throughout.
The v46 collapse pattern REPRODUCED even with a critic that provably discriminates actions (4x spread,
10x grasp gap from staged reward). Working hypothesis: max-over-16 harvests critic estimation noise
faster than real signal — the improvement operator is broken independent of critic quality on this
checkpoint. v51 (running): N=1 frozen on the phase-2 ckpt (step 15000), fixed starts — weights-intact
check. If intact (expected): the arc's final diagnosis = both components work in isolation (staged critic
learns actions; V+model selects correctly) but the recipe's rollout-time composition is inherently
destructive here. Video-count offset: v51 episodes = dir total − 36 (archive mv failed on ssh flake —
5090 connectivity intermittent tonight).

## v51 VERDICT: weights intact again — THE EXPO-FT ARC IS CLOSED (2026-07-15 ~23:20)

v51 (N=1 frozen, phase-2 ckpt step 15000, fixed starts): **12/22 (55%)** — statistically within the 65%
band (p=0.37), nothing like phase-2's 22% rollout rate. The phase-2 collapse was once again rollout-time.
FINAL DIAGNOSIS of the EXPO-FT program on this checkpoint, each link independently established:
1. Staged dense reward DOES teach the critic action discrimination (v50 phase 1: 4x spread, 10x grasp gap).
2. A good V + perfect model DOES select correctly (v49: 56/57 fidelity).
3. The rollout improvement operator (argmax-Q over half-noise-edited candidates) is destructive REGARDLESS
   — with a mush critic (v46) and with a discriminating critic (v50p2: 2/9, edit-picks 76-89%). Max-over-16
   harvests estimation noise faster than any learned signal accumulates.
4. Even oracle selection can't exceed the sample support (v49), and BC alone doesn't move the band (v42).
=> On a task-specialized flow checkpoint, EXPO-FT has no working lever: its evaluation machinery can be
made healthy, but its improvement machinery either does nothing (BC, selection) or harms (edits).
The program's only transferable win remains inference-time commitment (replan-32: 15%→54% full task,
generalizing). Records: v50 q_traces/lookahead_traces_gate, v51 count via offset (videos_grasplift 36→58).
GPUs idle. Candidate next directions (user call): write up the arc (paper-shaped: commitment vs choice in
flow-VLA fine-tuning), port replan-32 pipeline toward the challenge submission, or a fundamentally
different RL attack (e.g., DPO-style chunk preference on lookahead pairs — sidesteps rollout selection).

## v52 CLOSES THE MATRIX: EDITS CONCLUSIVELY THE POISON (2026-07-15 ~03:15)

v52 (N=8 PLAIN selection, staged critic, frozen, user-proposed missing cell): **15/22 (68%), paired
by-start vs v41: 14 agree / 3 up / 3 down — exactly neutral.** Full matrix: every collapse cell contains
residual edits (v46 30%, v50p2 22%); every edit-free cell is neutral (v45, v49, v52) regardless of critic.
MECHANISM (code-level): update_residual_actor trains by gradient-ascending the critic's MEAN-ensemble Q
w.r.t. the 736-dim action input — an adversarial-example generator; its attacks then win the rollout
argmax (76-89% edit-picks) because they were optimized to. Amplifiers (user's hypothesis, endorsed):
sim determinism + 20 fixed starts + specialized policy = razor-thin Q coverage; 736-dim chunks (replan-32)
= exponentially more off-manifold volume than the reference's ~60-130-dim real-robot chunks. Reference
regime sits on the benign side of all three axes.
ARC FULLY CLOSED with complete causal story. Next-direction options for user: writeup / submission
engineering on replan-32 / support-escape without Q-ascent (lookahead-pair DPO).

## REPRODUCTION TRACK: data pipeline validated BITWISE; Delta race queued (2026-07-15 ~13:10)

Plan (user): reproduce the radio fine-tune from pi05_base BEFORE training hanging_pictures (parity-first).
- Sliced per-task LeRobot repos from the NAS combined dataset (200 eps each, incl. depth videos — the
  loader requires all 6 video keys present). Fixed: loader's silent hub-fallback on missing files.
- **Norm-stats recomputed from our sliced radio repo == official checkpoint's bundled stats, 0.00e+00 error
  on all 23 dims × {state,actions} × {mean,std}. Data pipeline byte-equivalent to the organizers'.**
- Delta (NCSA) infra per user's m2sv pattern: uv-inside-apptainer, scratch layout /scratch/bchy/shin1/b1k,
  fork+data uploaded. Configs: pi05_b1k_delta (H200, replicated) + pi05_b1k_delta_a100 (fsdp_devices=4).
- TWO-QUEUE RACE submitted: 20234116 (gpuH200x8, ~430 chg-hrs, ~19h wall) vs 20234185 (gpuA100x4-preempt,
  ~220 chg-hrs, ~110h wall, --requeue + auto-resume). Race arbiter auto-cancels the loser on first RUNNING.
  Both est. start ~07-16 19:40. Budget 917 hrs. Mid-flight gates: eval 10K/20K ckpts locally during training.
- Koa checked: only H200-NVL x4/x2 nodes via kill-shared (free, preemptible) — currently fully occupied;
  held as backup. A100x8 ruled out (dominated: 660 chg-hrs, 50-60h, worst queue).

## Q-vs-REALIZED AUDIT (user-designed): adversarial edits CONFIRMED quantitatively; no reward hacking (2026-07-16)

From v50p2 buffer ckpt (per-episode realized shaped returns) x q_traces (chosen-candidate Q):
- Failures telescope to ~0 (-0.008..+0.034) = exactly the potential-shaping bound; successes +1.13-1.14
  (terminal 1.0 + net phi-gain ~0.14). NO realized-reward inflation across phase 2 => dense reward aligned,
  reward-hacking branch eliminated (telescoping property held empirically).
- Decoupling signature: failing eps 4/5 carried chosen-Q as high as the SUCCESSES (max +0.140/+0.152 vs
  +0.124/+0.154) and realized ~0 — the critic scored residual-actor candidates success-grade; reality
  disagreed by 1.1. Low-Q failures realized low => critic fooled specifically on manufactured candidates.
Chain now measured end-to-end: edits -> inflated Q on fakes -> argmax executes -> realized ~0.
Artifacts: v50_reward_audit.log, buffers ckpt @15000. (n=8 caveat; ordering-level violation = what argmax uses.)

Per-step reward sequences (buffer audit, addendum): the high-Q failing episodes' PEAK cumulative reward
was +0.01..+0.04 — below the +0.1 grasp-acquisition step — i.e. the edit-picked chunks never achieved any
real progress (no transient grasp-then-drop). Q max +0.14-0.15 (success-grade) vs peak reality +0.04.
Successes: clean monotone climb to +1.13/+1.14 with visible grasp/lift/terminal structure. The money table
for the writeup's mechanism section: Q and physics diverge maximally exactly on manufactured candidates.

## Demo-side wiring audit (user-requested, 2026-07-16)

- Demo rewards: process_droid_dataset synthesizes sparse terminal 1.0 (done=1, mask=0 at final step) —
  correctly encoded. Finding: MIXED reward semantics (demos terminal-only vs online staged-shaped, skew
  bounded ≤0.2) has existed since v41 — mildly underpays progress states in offline data.
- Demo-Q probe (v50 critic over a 1956-step full-task demo): shape CORRECT — flat ~0 through navigation
  (true 0.72^k ~ 0 there: correct calibration, not laziness), monotone rise over the final ~5 chunks
  (+0.023 -> +0.058). Magnitudes ~10x pessimistic near terminal (true 0.72 at 1 chunk vs 0.058) — the
  familiar min-ensemble + TD-horizon compression; ordering (what selection/TD use) intact.
Probe: scratchpad/demo_q_probe.py (standalone agent-loading harness — reusable for future offline Q audits).

## Koa lane VALIDATED with live throughput measurement (2026-07-16 evening)

Used the user's idle 2xH200 koa job (13942451, 48-min window) via srun --overlap: venv built, JAX sees
GPUs, base ckpt downloaded (12G, persists), and **555 real training steps at 1.9 it/s (batch 16, 2xH200)**.
Calibration: batch-64 projections => ~0.9-1.0 it/s on 4xH200 (koa job: 50K in ~14-16h) and ~2x that on
Delta's 8xH200 (~7-9h, ~200 chg-hrs — half the prior estimate; training is single-noise-level per sample,
much cheaper than inference-derived FLOPs). Three-way race: koa 13984075 (free, 2 rivals) vs Delta
20248299 (H200x8) vs 20257673 (A100x4-preempt); cross-cluster arbiter armed.
Also clarified (user q): offline_ratio=0 SEEDS demos into the online buffer (mechanism selector, not
amount) — critic diet was demo-dominated ~2.5:1; ep_count gate counts collected episodes only, so
"first 10 pristine" claims stand.

PARKED EXPERIMENT (user-designed, post-reproduction): "consistent-diet critic" — (1) recompute staged phi
for demo frames by decoding radio/EEF pose from the raw demos' per-step serialized sim states (exact, no
replay drift; needs one-time offset map); (2) filter demos to the grasp+lift segment (privileged boundaries;
check existing mini450/_core/_navend trims first). Then: retrain critic on consistent shaped rewards +
matched state distribution, re-test PLAIN selection (no edits). NOT applicable to reproduction runs (parity).

## v53 LAUNCHED: consistent-diet critic (user-designed, 2026-07-16 night)

Demo phi decoded from raw per-step sim states at chunk boundaries (telescoping => boundaries suffice;
20 eps in ~25 min). is_grasping doesn't survive cold state loads — replaced with the proximity proxy
(eef_dist<0.15 = hand-lock signature; validated against radio_z lift traces). Findings: 18/20 demos do
a full grasp+lift (peak phi=2.0); eps 2-3 toggle WITHOUT lifting (task doesn't require it!) and ep17
lifts only 0.14m — all 3 get terminal 0 under grasplift semantics (17/20 successes in the shaped set).
Datasets built: turning_on_radio_shaped (full, per-step stored rewards; chunk-exact phi lumps + criterion
terminal) and _shaped_trim (first_grasp-256 .. peak+2 chunks; 416-448 steps ~ band scale). Loader patched
to prefer stored "reward" (backwards-compatible). v53 = fresh critic, trim dataset, staged online shaping,
N=1 warm, 64/20. Payoff test after warm: PLAIN selection frozen A/B vs the 65% band.

## H200 run started; DATALOADER STARVATION found + zero-gap swap (2026-07-17 ~03:20)

20248299 started 02:53 on gpue05 (ffmpeg image OK, 8 devices, steps ticking) BUT 4.4s/it => 60h projection.
srun --overlap diagnosis: ALL 8 GPUs at 0%, CPUs 77% idle — the default num_workers=8 cannot feed batch-64
x 3-cam hevc decode. Fix (infra-only, no recipe change): --num_workers=48. Zero-gap swap: submitted
20262939 (exp_name radio_repro2, 48 workers) ALONGSIDE the running job (topped-up budget covers both
reservations); swap-arbiter cancels the slow job when the fast one starts. Lesson for the ledger: on
big-batch video-VLA training, ALWAYS check GPU util in the first 10 minutes — "training runs" != "GPUs fed".

## v53/v54 VERDICT: consistent diet does NOT unlock selection — support cap stands (2026-07-17 ~07:00)

v53 warm (fresh critic, shaped+trimmed demos, staged online rewards, 64/20): 17/25 (68%), no erosion.
v54 payoff (plain N=8 selection, frozen): **11/20 (55%)** [FSFFSSSSSSFFFFFSSSFS] — at/below the 65% band,
below v52's 15/22 (inconsistent-diet critic). Mechanism metrics: candidate spread 0.0136 (healthy, ~2x
v46-era but below the v50 gate's 0.024), success/fail separation +0.034/+0.011 (clean 3x). The critic is
fine; selection still can't beat the band. **The support-cap explanation survives its strongest test:
even ideal reward/state feeding doesn't make best-of-8 over on-policy samples lift.** The user's data
fixes are validated as critic-quality improvements (clean separation with HALF the training steps of v50)
— but the ceiling is the sample distribution, exactly as v49's oracle showed.
This closes the consistent-diet branch. RL-mechanism ledger final: commitment (replan-32) remains the
only lever that moves outcomes on this checkpoint.

## v55: edits STILL harmful with the consistent-diet critic — matrix complete (2026-07-17 ~08:00)

v55 (N=8+8 edits, consistent-diet critic + its residual actor, frozen): **5/14 (36%)** [FSFFFFSFFFFSSS],
edit-pick rate 69%. vs plain selection 55% (v54) and band 65-68%. Better critic => somewhat less
catastrophic than v46/v50p2 (22-30%) and lower edit-pick rate (69% vs 76-89%), but the harm direction is
unchanged: the adversarial-ascent mechanism survives ideal reward/state feeding, as predicted — it attacks
estimation error itself, which no diet eliminates. FINAL matrix: edits harmful in all 3 critic regimes
(mush / discriminating / consistent-diet); plain selection neutral in all; commitment (replan-32) remains
the only positive lever. The RL-mechanism investigation is now exhaustively closed.
Records: v54_payoff_11of20.jsonl, v55_edits_traces.jsonl.

## Reproduction 10K GATE: undertrained-not-broken — continue to 20K (2026-07-17 ~11:00)

10K ckpt gate eval (replan-32, instances 0-4): 0/4+ running, q=0.0. Video frames: coherent locomotion,
radio visibly in view at start, robot navigates AWAY — locomotion prior present, task grounding absent.
No pipeline red flags (sane actions, clean obs, no physics chaos). Call: NOT a kill — 10K/50K ≈ 1.5
epochs; re-gate at 20K (~10-19h on slow job, ~230 chg-hrs). Kill rule at 20K: still zero task-directedness
=> stop + investigate. Slow job at ~12.5K steps; 48-worker swap job still queued.

## 10K gate eval FINAL: 0/5, verdict recorded; serve stopped (2026-07-17 ~12:30)

Eval ran 1 ep per instance 0-4 and exited: 0/5, q=0.0 everywhere. Combined with frame read (coherent
locomotion, radio in view at start, robot navigates away) => undertrained-not-broken confirmed; Delta run
continues to the 20K gate (~19:15 at 4.9s/it). :8020 serve killed; local GPU free.

## 20K GATE: PASS — 2/5 successes; reproduction on-trajectory (2026-07-17 ~16:00)

20K ckpt (job 20248299), replan-32, instances 0-4: F F F S S — 2/5 with q=1.0 on both successes.
Task grounding emerged between 10K (0/5, navigates away from radio) and 20K. 40% matches the official
ckpt's replan-32 train-split rate (12/30). Verdict: continue to 50K; no kill. Serve torn down.
Chain fix: eval needs OMNIGIBSON_DATA_PATH=/mnt/nvme/behavior_data (gate_eval_ckpt.sh updated).
Slow job hits 24h wall ~01:00 at ~28K steps; swap job 20262939 (48 workers) still PENDING — if not
started by wall time, resubmit resume with --num_workers=48.

## Slow job KILLED per user; fast from-scratch job carries the reproduction (2026-07-17 ~16:45)

20248299 cancelled at 15h59m (~24K steps; 10K/20K ckpts kept on scratch, 20K local). Economics: ~24
chg-hrs/h at ~20% GPU efficiency vs the queued 48-worker twin doing the whole 50K in ~11h. 20262939
(radio_repro2, from scratch, --num_workers=48) kept queued — resubmitting as resume-from-20K would
reset ~14h of priority age to save ~120 chg-hrs; from-scratch also gives a seam-free reproduction and
the reusable template for hanging_pictures. First-10-minutes check when it starts: s/it + GPU util.

## hanging_pictures staging: latent slicer bug found+fixed; configs/norm-stats prepped (2026-07-17 ~17:30)

Prep for the next task while the fast repro job pends. Configs pi05_b1k_hang_delta/_local added (path-only
clones), train_hang_v1.slurm staged on Delta (48 workers; NOT submitted), config.py synced to Delta.
BUG: slice_task_repo.py zeroed videos/<k>/chunk_index only for k in dirs under SRC/videos at slice time —
depth dirs were staged later, so hanging_pictures metadata kept source chunk_index=34 (also
meta/episodes/chunk_index). Radio dodged it because task-0000's source chunk IS 000. Symptom: reader
try_load fails sufficiency (600 "missing" depth videos) -> silent HF hub 404. Fixed in-place (zeroed all
*/chunk_index cols), slicer hardened (zero every */chunk_index column present in metadata), fixed parquet
pushed to Delta copy. Local load verified: 200 eps / 477,483 frames. Norm stats recomputing.

## A100-preempt hedge lane submitted (user-approved); race arbiter armed (2026-07-17 ~23:00)

H200 queue estimate slipped to Jul 19 ~23:00 (253 pending in partition). Hedge: 20276044 on
gpuA100x4-preempt resumes radio from the 20K ckpt (42G full train_state copied to
ckpts/pi05_b1k_delta_a100/radio_repro_a100/20000; v3 script = v2 + --num_workers=40). Arbiter monitor:
H200 20262939 RUNNING => scancel A100 (fresh H200 run finishes faster than A100 remainder in ~all cases);
preemptions auto-requeue+resume; 48h wall may need one resubmit (~66h A100 grind for 30K steps).

## v49 failure taxonomy decoded (user asked to see videos, 2026-07-18 evening)

Per-episode peaks from lookahead_traces.jsonl (peak obj_z across ALL 8 candidates, oracle choosing;
criterion 0.684, table 0.534): 3 STALL failures (ep4 0.673 = 1.1cm short!, ep5 0.637, ep12 0.642),
7 NEVER-LIFT failures (z pinned at 0.534-0.546 — no candidate contains a lift at any decision),
1 floor-spawn (ep2). Refines the support-cap story: "1-2cm short" is only the stall subset.
Implication for noise+selection (candidate-jitter experiment): plausible ceiling ~9->12/20 (converts
stalls only) — just reaches the band; the never-lift majority needs policy change (DPO route) or noise
at the grasp-approach decision, not the lift. Frame the experiment as DIAGNOSTIC: measures shallow
(perturbation-reachable) vs deep support deficit. Videos pulled from 5090 box C:\co\videos_v49
(v49-v55 video archives all live on that box, videos_v<N> dirs).

## v56 VERDICT: temperature menu recovers TO the band, not above — support cap is DEEP (2026-07-19 ~02:00)

v56 (user-designed noise-proposes/selection-disposes): 8 candidates at temps [1,1,1.5,1.5,1.5,2,2,2],
ORACLE phi pick (sim-verified, --lookahead_oracle_pick), frozen v46 ckpt, same 20 fixed starts.
RESULT: 17/26 = 65% (first-20: 13/20) — exactly AT the N=1 band; vs v49 temp-1 45%. Reads:
(1) Shallow-deficit conversion CONFIRMED but small: 2 marginal successes at 0.698/0.702 (stall-class
    converts); successes lift higher overall (0.74-0.92 peaks).
(2) Never-lift class UNMOVED: 7/9 failures still z=0.51-0.54 with hot candidates winning ~85-90% of
    decisions — even temp-2.0 flow samples contain no lift from those states. +1 floor spawn, 1 stall
    (0.657) not converted.
(3) Safety confirmed: oracle-selected temperature sampling never fell below band (noise proposes,
    selection disposes = at-worst-neutral, as designed).
CONCLUSION: the support deficit is mostly DEEP — local perturbation along the policy manifold cannot
reach lifts from the never-lift basins; only policy change (lookahead-pair DPO route) can go above the
band on this checkpoint. Selection-class methods now FULLY exhausted (plain/critic/oracle/temp-menu).
Infra: temperature = scaled initial flow noise (sample_candidate_actions temps=, pi05 noise passthrough,
--lookahead_temps/--lookahead_oracle_pick); ~9 min/ep. Run: expoft_b1k_grasplift_v56_tempmenu.
