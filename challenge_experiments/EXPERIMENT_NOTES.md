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
