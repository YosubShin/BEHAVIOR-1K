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

Next (Phase 0 remaining): task config for train_pi_robo.py (mirror configs/task/light2.py → point at our server),
offline replay-buffer seeding from our LeRobot demos (their loop expects base_image/left_wrist_image/state/actions),
SFT init = our provided pi05 ckpt + its norm stats, then short vanilla run on turning_on_radio.

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
