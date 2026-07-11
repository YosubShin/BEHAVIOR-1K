"""BEHAVIOR-1K environment operations server for EXPO-FT.

Implements the 5-operation websocket protocol that EXPO-FT's learner speaks
(see /mnt/nvme/expo-ft/expo_ft/env/env_client.py):

    create_env       -> {env_id, task_description}
    reset            -> {observation, done}
    step             -> {action, action_type}
    get_observation  -> {observation}
    get_info_for_step-> {done, success, reward, mask}

This is the sim-side counterpart of their DROID client: it wraps OmniGibson
(reusing the challenge Evaluator machinery) so the EXPO-FT learner can do
online RL against BEHAVIOR tasks.

Observation contract (what train_pi_robo.py feeds the agent, see its lines
179-185 and 254-276):
    base_image        uint8 (H, W, 3)   -- we map: head camera RGB
    left_wrist_image  uint8 (H, W, 3)   -- left wrist RGB
    right_wrist_image uint8 (H, W, 3)   -- extra (BEHAVIOR has 3 cams; critic
                                           pixel_keys is configurable)
    state             float32 (23,)     -- extract_state_from_proprio(61->23),
                                           same mapping as the openpi b1k fork
    prompt            str               -- language instruction (π0.5 input)

Reward design (the part that is OURS, not EXPO-FT's):
    reward_t = q_score_t - q_score_{t-1}   (dense BDDL partial-credit delta)
    where q_score = fraction of satisfied goal predicates. Sparse-reward
    ablation: return the delta only at episode end (config flag).
    done    = task success OR step >= max_steps
    mask    = 0.0 if success (terminal state, no bootstrap)
              1.0 if timeout (truncation: bootstrap through)

Reset design:
    - cycles through TRAIN instance ids (never public_test 301-340!)
    - optional Comet-style pose perturbation (x,y ±0.15 m, yaw ±15°)
    - later (Phase 1): reset-to-failure-state curriculum hooks go here.

STATUS: scaffold — interfaces are grounded in the expo-ft code, but this file
has NOT been run against the sim yet. Run inside the `behavior` conda env with
OMNIGIBSON_DATA_PATH set. The msgpack codec comes from openpi_client
(pip-installable) or falls back to omnigibson's vendored equivalent.
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import traceback
import uuid

import numpy as np

logging.basicConfig(level=logging.INFO, format="[b1k-env-server] %(message)s")
logger = logging.getLogger("behavior_env_server")

# --- msgpack numpy codec (same wire format as expo-ft's openpi_client) -------
try:
    from openpi_client import msgpack_numpy

    Packer = msgpack_numpy.Packer
    unpackb = msgpack_numpy.unpackb
except ImportError:  # fall back to the identical codec vendored in OmniGibson
    from omnigibson.eval.utils.network_utils import Packer, unpackb  # noqa: F401


# 61-dim proprio -> 23-dim state, mirroring openpi's b1k RobotConfig
# (base_qvel 0:3, trunk_qpos 53:57, left_arm 3:10, left_grip sum(24:26),
#  right_arm 28:35, right_grip sum(49:51))
def extract_state_23(proprio: np.ndarray) -> np.ndarray:
    parts = [
        proprio[0:3],
        proprio[53:57],
        proprio[3:10],
        proprio[24:26].sum(keepdims=True),
        proprio[28:35],
        proprio[49:51].sum(keepdims=True),
    ]
    return np.concatenate(parts).astype(np.float32)


class BehaviorEnvOps:
    """One OmniGibson env exposed through the expo-ft ops protocol.

    NOTE: OmniGibson is a singleton simulator — one env per server process.
    Parallelism = run several server processes on different ports (this is
    also how expo-ft scales real robots).
    """

    def __init__(
        self,
        task_name: str,
        instance_ids: list[int],
        max_steps: int | None = None,
        dense_reward: bool = True,
        perturb_pose: bool = False,
        perturb_xy: float = 0.15,
        perturb_yaw_deg: float = 15.0,
        seed: int = 0,
        full_res: bool = False,
        snapshot_record_dir: str | None = None,
        snapshot_every: int = 30,
        snapshot_window: tuple[int, int] = (150, 1500),
        start_snapshot_dir: str | None = None,
        start_near_object: str | None = None,
        start_distance: float = 0.6,
        prompt: str | None = None,
        start_joint_states: str | None = None,
        fixed_eval_starts: int = 0,
        shaping_coef: float = 0.0,
        subtask: str | None = None,
        perturb_obj_xy: float = 0.0,
        perturb_obj_yaw_deg: float = 0.0,
    ):
        self.task_name = task_name
        self.instance_ids = instance_ids
        self._instance_cursor = 0
        self.dense_reward = dense_reward
        self.shaping_coef = shaping_coef
        # Subtask mode 'grasp': success = sustained is_grasping(goal obj) with either
        # arm; early FAIL when the object falls off the table (the observed sideways-
        # hit failure). Horizon ~EXPO-FT's regime (episodes end in seconds, not
        # minutes) -> fast paired A/Bs for pipeline-health debugging.
        self.subtask = subtask
        self.perturb_obj_xy = perturb_obj_xy
        self.perturb_obj_yaw_deg = perturb_obj_yaw_deg
        self._grasp_streak = 0
        self._goal_obj = None
        self._goal_obj_z0 = None
        self.perturb_pose = perturb_pose
        self.perturb_xy = perturb_xy
        self.perturb_yaw_deg = perturb_yaw_deg
        self._rng = np.random.default_rng(seed)

        # ---- launch sim (heavy). Reuses the challenge Evaluator config path.
        from omegaconf import OmegaConf

        from omnigibson.eval.evaluator import Evaluator

        # RL data collection defaults to native-224 RGB rendering (DefaultWrapper):
        # ~2x sim FPS vs full-res RGBD, and the policy consumes 224 anyway. Use
        # full_res only when eval-faithful rendering matters more than throughput.
        wrapper = (
            "omnigibson.eval.wrappers.RGBDFullResWrapper" if full_res else "omnigibson.eval.wrappers.DefaultWrapper"
        )
        cfg = OmegaConf.create(
            {
                "env_wrapper": {"_target_": wrapper},
                "policy_name": "local",
                "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
                "headless": True,
                "partial_scene_load": True,
                "max_steps": max_steps,
                "write_video": False,
                # instance ids 301-320 live in the public_test split; <301 are train.
                # (Comparison probes against serve-eval need the test split; RL runs
                # stay on train.) Mixed train/test id lists are not supported.
                "mode": "public_test" if min(int(i) for i in instance_ids) >= 301 else "train",
                "seed": seed,
                "task": {"name": task_name},
                "robot": None,
            }
        )
        self.evaluator = Evaluator(cfg)
        self.env = self.evaluator.env
        self.robot = self.evaluator.robot
        self._cam = self.evaluator.robot_camera_names  # {head,left_wrist,right_wrist}

        self.task_description = prompt or self._load_prompt(task_name)
        self._steps = 0
        self._prev_q = 0.0
        self._last_reward = 0.0
        self._prev_phi = None
        self._success = False
        self._done = False

        # --- sim-state snapshots (mini-task starts / reset-to-state curricula) ---
        # Record mode: buffer full sim states every `snapshot_every` steps; when an
        # episode SUCCEEDS, persist the ones `snapshot_window` steps before success
        # ("in front of the radio, pre-toggle" states from real successful rollouts).
        # Start mode: reset() loads a random saved snapshot -> short-horizon episodes.
        self._snap_record_dir = snapshot_record_dir
        self._snap_every = snapshot_every
        self._snap_window = snapshot_window
        self._snap_ring: list = []
        self._episode_uid = 0
        if snapshot_record_dir:
            import os as _os

            _os.makedirs(snapshot_record_dir, exist_ok=True)
        # Mini-task alternative: hand-placed start facing a task object (privileged
        # info at training-env setup time is challenge-legal). Deterministic and
        # independent of base-policy success rate, unlike the snapshot pool.
        self._start_near_object = start_near_object
        self._start_distance = start_distance
        # Optional pool of 23-dim demo start states: sampled at reset to put trunk/
        # arms/grippers into the demos' manipulation posture (neutral reset pose is
        # out-of-distribution for trimmed mini-demos -> policy never engages).
        self._start_joint_states = np.load(start_joint_states) if start_joint_states else None
        if self._start_joint_states is not None:
            logger.info(f"start joint-state pool: {self._start_joint_states.shape}")

        self._start_snaps: list[str] = []
        if start_snapshot_dir:
            import glob as _glob

            self._start_snaps = sorted(_glob.glob(f"{start_snapshot_dir}/*.pt"))
            if not self._start_snaps:
                raise FileNotFoundError(f"start_snapshot_dir has no .pt snapshots: {start_snapshot_dir}")
            logger.info(f"mini-task mode: {len(self._start_snaps)} start snapshots loaded")

        # Fixed-eval mode: a deterministic list of (snapshot_idx, dx, dy, dyaw)
        # cycled by episode index. Every evaluation scores the SAME start set —
        # removing jitter-draw sampling noise from checkpoint comparisons
        # (overnight 2026-07-11: n<=13 ad-hoc reads on shifting RNG streams
        # proved undecidable between 15% and 40%).
        self._fixed_eval = None
        if fixed_eval_starts and fixed_eval_starts > 0 and self._start_snaps:
            eval_rng = np.random.default_rng(12345)
            self._fixed_eval = [
                (
                    int(eval_rng.integers(0, len(self._start_snaps))),
                    float(eval_rng.uniform(-perturb_xy, perturb_xy)),
                    float(eval_rng.uniform(-perturb_xy, perturb_xy)),
                    float(eval_rng.uniform(-np.deg2rad(perturb_yaw_deg), np.deg2rad(perturb_yaw_deg))),
                )
                for _ in range(fixed_eval_starts)
            ]
            logger.info(f"FIXED-EVAL mode: {fixed_eval_starts} deterministic starts (seed 12345)")

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _load_prompt(task_name: str) -> str:
        """Language instruction; same source the openpi fork uses (TASK_REGISTRY)."""
        try:
            import json
            import os

            from omnigibson.macros import gm

            p = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "task_instructions.json")
            if os.path.exists(p):
                return json.load(open(p)).get(task_name, task_name.replace("_", " "))
        except Exception:
            pass
        return task_name.replace("_", " ")

    def _snapshot_initial_predicates(self) -> None:
        """Record which goal predicates are already true at episode start.

        Mirrors omnigibson/metrics/task_metric.py::TaskMetric.reset — the official
        q_score gives NO credit for predicates that were true before the robot acted.
        """
        task = self.env.task
        self._initial_predicate_states = [
            [pred.evaluate(task._evaluate_predicate) for pred in option] for option in task.ground_goal_state_options
        ]

    def _shaping_potential(self) -> float:
        """phi(s) = -distance(right EEF, first goal-scope object). Privileged sim
        info — legal at training time; never enters the observation."""
        try:
            import torch as th

            eef = self.robot.get_eef_position(arm="right")
            obj = next(
                e for e in self.env.task.object_scope.values()
                if e is not None and "agent" not in getattr(e, "name", "agent")
            )
            opos = obj.get_position_orientation()[0]
            return -float(th.linalg.norm(eef - opos))
        except Exception:
            return self._prev_phi if self._prev_phi is not None else 0.0

    def _q_score(self) -> float:
        """Official partial-credit q_score, computed every step.

        Mirrors TaskMetric._compute_episode_metrics exactly: 1.0 on success, else
        max over disjunctive goal options of the fraction of predicates that are
        newly true (true now AND not true at reset).
        """
        task = self.env.task
        if task.success:
            return 1.0
        return max(
            sum(
                int(not initially_true and pred.evaluate(task._evaluate_predicate))
                for pred, initially_true in zip(option, option_previous_state)
            )
            / len(option)
            for option, option_previous_state in zip(task.ground_goal_state_options, self._initial_predicate_states)
        )

    def _observation(self) -> dict:
        obs = self.evaluator.obs  # already flattened by evaluator._preprocess_obs
        rn = self.robot.name

        def rgb(cam_key):
            img = np.asarray(obs[self._cam[cam_key] + "::rgb"])[..., :3].astype(np.uint8)
            # Ship 224² regardless of render resolution: the learner resizes to 224
            # anyway, and full-res frames are ~10x the websocket payload.
            if img.shape[0] != 224 or img.shape[1] != 224:
                import cv2

                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            return img

        return {
            "base_image": rgb("head"),
            "left_wrist_image": rgb("left_wrist"),
            "right_wrist_image": rgb("right_wrist"),
            "state": extract_state_23(np.asarray(obs[f"{rn}::proprio"], dtype=np.float32)),
            "prompt": self.task_description,
        }

    def _settle_quiescent(self) -> None:
        """Step physics until the robot stops ringing, then refresh the obs."""
        import torch as th

        import omnigibson as og

        self.robot.keep_still()
        for i in range(240):
            og.sim.step_physics()
            if i >= 10 and th.max(th.abs(self.robot.get_joint_velocities())).item() < 0.02:
                break
        else:
            logger.warning(
                "settle: not quiescent after 240 steps (max |qvel|=%.3f)",
                th.max(th.abs(self.robot.get_joint_velocities())).item(),
            )
        # Drive targets may have been consumed during settling; re-pin them to
        # the settled posture so the episode starts from a held pose.
        self.robot.set_joint_positions(self.robot.get_joint_positions())
        # step_physics does NOT render — refresh frames or the first obs ships
        # stale pre-settle camera images.
        for _ in range(3):
            og.sim.render()
        obs, _ = self.env.get_obs()
        self.evaluator.obs = self.evaluator._preprocess_obs(obs)

    # ---- the 5 ops ---------------------------------------------------------
    def reset(self, snapshot_path: str | None = None) -> dict:
        instance_id = self.instance_ids[self._instance_cursor % len(self.instance_ids)]
        self._instance_cursor += 1

        self.evaluator.reset()
        self.evaluator.load_task_instance(int(instance_id))
        if self.perturb_pose:
            self._perturb_robot_pose()
        self.evaluator.reset()

        used_snapshot = bool(self._start_snaps or snapshot_path)
        # (snapshot restores overwrite the jitter above; re-applied post-restore below)
        if used_snapshot:
            import torch as th

            # snapshot_path (probe/debug): reset to a SPECIFIC snapshot instead of random.
            if snapshot_path:
                snap_path = snapshot_path
            elif self._fixed_eval is not None:
                snap_path = self._start_snaps[self._fixed_eval[self._episode_uid % len(self._fixed_eval)][0]]
            else:
                snap_path = self._start_snaps[int(self._rng.integers(len(self._start_snaps)))]
            state = th.load(snap_path, weights_only=False)
            import omnigibson as og

            if isinstance(state, dict) and "serialized_state" in state:
                # Raw-dataset demo state: flat vector from og.sim.dump_state(serialized=True)
                # (extracted at T-450 from task-0000 episodes = exact pre-success worlds).
                og.sim.load_state(state["serialized_state"], serialized=True)
            else:
                og.sim.load_state(state, serialized=False)
            # CRITICAL: load_state restores joint POSITIONS but not the position-
            # drive TARGETS, which still point at the reset posture — the drives
            # then drag the robot back (observed: arms lift + trunk pitches, head
            # dips, at episode start). Re-set targets to the restored positions.
            self.robot.set_joint_positions(self.robot.get_joint_positions())
            if self.perturb_pose:
                self._perturb_robot_pose()
            if self.perturb_obj_xy > 0.0 or self.perturb_obj_yaw_deg > 0.0:
                self._perturb_goal_object()

        if self._start_near_object:
            self._place_robot_near_object()

        # Settle to quiescence ONLY after snapshot restores. Natural resets must
        # match the challenge eval byte-for-byte: the reference success rates
        # (2/10 full-res) were measured WITH the start transient, and demos were
        # collected the same way — the bounce is in-distribution, and settling
        # (frozen robot + drive targets re-pinned to a sagged posture) makes our
        # starts LESS like training.
        if used_snapshot:
            self._settle_quiescent()

        self._grasp_streak = 0
        self._goal_obj = None
        self._goal_obj_z0 = None
        if self.subtask == "grasp":
            try:
                self._goal_obj = next(
                    e for e in self.env.task.object_scope.values()
                    if e is not None and "agent" not in getattr(e, "name", "agent")
                )
                self._goal_obj_z0 = float(self._goal_obj.get_position_orientation()[0][2])
            except Exception:
                logger.warning("grasp subtask: goal object lookup failed", exc_info=True)

        # Spawn-view debug: dump the head-camera view at t=0 (ring of last 40)
        # to compare against the mini-demos' first frames.
        try:
            import os as _os

            import cv2 as _cv2

            _os.makedirs("/mnt/nvme/expoft_debug/spawn_views", exist_ok=True)
            _img = self._observation()["base_image"]
            _cv2.imwrite(
                f"/mnt/nvme/expoft_debug/spawn_views/spawn_ep{self._episode_uid % 40:03d}.png",
                _cv2.cvtColor(_img, _cv2.COLOR_RGB2BGR),
            )
        except Exception:
            logger.warning("spawn-view dump failed", exc_info=True)

        # Force the goal object un-toggled after SNAPSHOT resets only: ToggledOn is
        # a functional state not covered by load_state (observed leak: instant
        # 'success' at step 1-2). Natural env.reset restores task init state itself,
        # and the challenge eval does not force it — parity requires we don't either.
        if used_snapshot:
            try:
                from omnigibson.object_states import ToggledOn

                for inst, entity in self.env.task.object_scope.items():
                    if entity is not None and hasattr(entity, "states") and ToggledOn in getattr(entity, "states", {}):
                        entity.states[ToggledOn].set_value(False)
            except Exception:
                logger.warning("toggled_on reset failed", exc_info=True)

        self._snap_ring = []
        self._ep_frames = []
        self._episode_uid += 1
        self._steps = 0
        self._snapshot_initial_predicates()
        self._prev_q = self._q_score()
        self._prev_phi = None
        self._last_reward = 0.0
        self._success = False
        self._done = False
        return {"observation": self._observation(), "done": False}

    def _perturb_robot_pose(self):
        """Comet-style start-pose jitter (DART augmentation): x,y ±perturb_xy m,
        yaw ±perturb_yaw_deg. For snapshot starts this is the mini-task difficulty
        knob: jitter from a pre-grasp state forces the policy to re-solve the reach
        geometry instead of replaying one memorized approach."""
        import omnigibson.utils.transform_utils as T
        import torch as th

        pos, quat = self.robot.get_position_orientation()
        if self._fixed_eval is not None:
            _, dx, dy, dyaw = self._fixed_eval[self._episode_uid % len(self._fixed_eval)]
        else:
            dx, dy = self._rng.uniform(-self.perturb_xy, self.perturb_xy, size=2)
            dyaw = self._rng.uniform(-np.deg2rad(self.perturb_yaw_deg), np.deg2rad(self.perturb_yaw_deg))
        yaw_q = T.euler2quat(th.tensor([0.0, 0.0, dyaw]))
        new_quat = T.quat_multiply(quat, yaw_q)
        new_pos = pos + th.tensor([dx, dy, 0.0])
        self.robot.set_position_orientation(new_pos, new_quat)

    def _perturb_goal_object(self) -> None:
        """Object-level DART: jitter the goal object's table pose (x, y, yaw).
        Small magnitudes — the settle loop resolves resulting contacts; the
        fall-detector (grasp subtask) or task predicates catch off-table slides."""
        import omnigibson.utils.transform_utils as T
        import torch as th

        try:
            obj = next(
                e for e in self.env.task.object_scope.values()
                if e is not None and "agent" not in getattr(e, "name", "agent")
            )
            pos, quat = obj.get_position_orientation()
            dx, dy = self._rng.uniform(-self.perturb_obj_xy, self.perturb_obj_xy, size=2)
            dyaw = self._rng.uniform(
                -np.deg2rad(self.perturb_obj_yaw_deg), np.deg2rad(self.perturb_obj_yaw_deg)
            )
            yaw_q = T.euler2quat(th.tensor([0.0, 0.0, dyaw]))
            obj.set_position_orientation(pos + th.tensor([dx, dy, 0.0]), T.quat_multiply(yaw_q, quat))
        except Exception:
            logger.warning("goal-object perturbation failed", exc_info=True)

    def _place_robot_near_object(self) -> None:
        """Teleport the robot base to `start_distance` m in front of the named task
        object, facing it (plus small jitter). Arms stay at reset pose."""
        import math

        import omnigibson as og
        import omnigibson.utils.transform_utils as T
        import torch as th

        target = None
        for inst, entity in self.env.task.object_scope.items():
            if self._start_near_object.lower() in inst.lower() and entity is not None:
                target = entity
                break
        if target is None:
            raise KeyError(
                f"start_near_object '{self._start_near_object}' not in task scope: "
                f"{list(self.env.task.object_scope.keys())}"
            )
        obj_pos, _ = target.get_position_orientation()
        robot_pos, _ = self.robot.get_position_orientation()
        logger.info(
            f"place_near: matched '{inst}' at {[round(float(x),2) for x in obj_pos]}, "
            f"robot spawn at {[round(float(x),2) for x in robot_pos]}"
        )

        d = self._start_distance + float(self._rng.uniform(-0.05, 0.05))
        # Approach from the side of the task's original robot spawn (guaranteed free
        # space) with ±30° jitter — a uniform direction would clip into the wall or
        # furniture the object sits against.
        theta = math.atan2(float(robot_pos[1]) - float(obj_pos[1]), float(robot_pos[0]) - float(obj_pos[0]))
        theta += float(self._rng.uniform(-math.pi / 6, math.pi / 6))
        base_x = float(obj_pos[0]) + d * math.cos(theta)
        base_y = float(obj_pos[1]) + d * math.sin(theta)
        yaw = math.atan2(float(obj_pos[1]) - base_y, float(obj_pos[0]) - base_x)
        yaw += float(self._rng.uniform(-math.pi / 24, math.pi / 24))  # ±7.5° facing jitter

        new_pos = th.tensor([base_x, base_y, float(robot_pos[2])])
        new_quat = T.euler2quat(th.tensor([0.0, 0.0, yaw]))
        self.robot.set_position_orientation(new_pos, new_quat)

        if self._start_joint_states is not None:
            # 23-dim state -> joint targets. State: [base_qvel 0:3, trunk 3:7,
            # left_arm 7:14, left_grip_width 14, right_arm 15:22, right_grip_width 22].
            # Robot joint vector (r1pro.yaml): 6 virtual base, 4 torso (6:10),
            # 7 left arm (10:17), 7 right arm (17:24), 2 L-fingers (24:26), 2 R (26:28).
            js = self._start_joint_states[int(self._rng.integers(len(self._start_joint_states)))]
            jp = self.robot.get_joint_positions()
            jp[6:10] = th.tensor(js[3:7])
            jp[10:17] = th.tensor(js[7:14])
            jp[17:24] = th.tensor(js[15:22])
            jp[24:26] = float(js[14]) / 2.0
            jp[26:28] = float(js[22]) / 2.0
            self.robot.set_joint_positions(jp)

        og.sim.update_handles()
        for _ in range(5):
            og.sim.step_physics()
            self.robot.keep_still()
        # 10+ renders: temporal AA/denoise needs several frames to flush ghosting
        # after a teleport; 3 was visibly insufficient.
        for _ in range(10):
            og.sim.render()
        obs, _ = self.env.get_obs()
        self.evaluator.obs = self.evaluator._preprocess_obs(obs)
        fp, _ = self.robot.get_position_orientation()
        logger.info(f"place_near: final robot pos {[round(float(x),2) for x in fp]} (target d={self._start_distance})")

    def step(self, action: list) -> dict:
        a = np.asarray(action, dtype=np.float32)
        import torch as th

        # Drive the sim directly (bypasses evaluator's policy — WE are the policy).
        obs, _, terminated, truncated, info = self.env.step(th.from_numpy(a), n_render_iterations=1)
        self.evaluator.obs = self.evaluator._preprocess_obs(obs)
        self._steps += 1

        q = self._q_score()
        self._last_reward = (q - self._prev_q) if self.dense_reward else 0.0
        self._prev_q = q
        # Potential-based shaping (training-time only, preserves optimal policy):
        # r += coef * (phi(s') - phi(s)) with phi = -dist(right EEF, goal object).
        # Densifies the critic's gradient — the BDDL q-score for this task is a
        # single ToggledOn predicate, i.e. effectively sparse-terminal.
        if self.shaping_coef > 0.0:
            phi = self._shaping_potential()
            if self._prev_phi is not None:
                self._last_reward += self.shaping_coef * (phi - self._prev_phi)
            self._prev_phi = phi
        self._success = bool(info["done"]["success"]) if "done" in info else bool(terminated and not truncated)
        self._done = bool(terminated or truncated)
        if self.subtask == "grasp" and self._goal_obj is not None:
            from omnigibson.controllers.controller_base import IsGraspingState

            grasping = any(
                self.robot.is_grasping(arm=a, candidate_obj=self._goal_obj) == IsGraspingState.TRUE
                for a in self.robot.arm_names
            )
            self._grasp_streak = self._grasp_streak + 1 if grasping else 0
            fell = float(self._goal_obj.get_position_orientation()[0][2]) < self._goal_obj_z0 - 0.25
            if self._grasp_streak >= 15:
                self._success = True
                self._done = True
                self._last_reward += 1.0
            elif fell:
                self._success = False
                self._done = True

        if self._done and not self.dense_reward:
            self._last_reward = q  # sparse variant: final partial credit only

        if self._snap_record_dir:
            if self._steps % self._snap_every == 0 and not self._done:
                import omnigibson as og

                self._snap_ring.append((self._steps, og.sim.dump_state(serialized=False)))
            if self._done and self._success:
                self._persist_snapshots()

        # Episode video (head + both wrist cams side-by-side, 672x224): buffer
        # every step, write on episode end. Wrist views are what expose gripper
        # mistakes (mis-grasp, premature close) the head cam hides.
        # Keeps ALL successes + the most recent failures (spot-checking rollouts).
        try:
            _o = self._observation()
            self._ep_frames.append(
                np.concatenate(
                    [_o["left_wrist_image"], _o["base_image"], _o["right_wrist_image"]], axis=1
                )
            )
            if self._done:
                self._write_episode_video()
        except Exception:
            logger.warning("episode video failed", exc_info=True)
        return {"action": a, "action_type": "policy"}

    _ep_frames: list = []
    _VIDEO_DIR = "/mnt/nvme/expoft_videos/miniradio"
    _KEEP_FAILS = 30

    def _write_episode_video(self) -> None:
        import glob
        import os

        import cv2

        # The learner sends 1-2 extra steps after done (its done-check lags one op);
        # each re-triggers 'done' — only write the first, real episode video.
        if len(self._ep_frames) < 5:
            self._ep_frames = []
            return
        os.makedirs(self._VIDEO_DIR, exist_ok=True)
        tag = "success" if self._success else "fail"
        path = f"{self._VIDEO_DIR}/ep{self._episode_uid:05d}_{tag}_{self._steps}steps.mp4"
        h, wpx = self._ep_frames[0].shape[:2]
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (wpx, h))
        for fr in self._ep_frames:
            w.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        w.release()
        self._ep_frames = []
        if self._success:
            logger.info(f"episode video (SUCCESS): {path}")
        else:  # prune old failure videos beyond the retention window
            fails = sorted(glob.glob(f"{self._VIDEO_DIR}/ep*_fail_*.mp4"))
            for old in fails[: -self._KEEP_FAILS]:
                os.remove(old)

    def _persist_snapshots(self) -> None:
        """On success, save the buffered states from `snapshot_window` steps before the end."""
        import torch as th

        lo, hi = self._snap_window
        kept = 0
        for step, state in self._snap_ring:
            back = self._steps - step
            if lo <= back <= hi:
                p = f"{self._snap_record_dir}/{self.task_name}_ep{self._episode_uid:05d}_t{step:05d}_b{back:04d}.pt"
                th.save(state, p)
                kept += 1
        logger.info(f"SUCCESS at step {self._steps}: persisted {kept} pre-success snapshots (window {lo}-{hi})")
        self._snap_ring = []

    def get_observation(self) -> dict:
        return {"observation": self._observation()}

    def get_eval_obs(self) -> dict:
        """The evaluator's preprocessed obs — byte-for-byte what the challenge
        eval client ships to a serve_b1k policy server. Lets an external probe
        drive THIS env with the reference serve policy (obs-provenance A/B)."""
        import torch as th

        def to_np(x):
            if isinstance(x, dict):
                return {k: to_np(v) for k, v in x.items()}
            if isinstance(x, th.Tensor):
                return x.detach().cpu().numpy()
            return x

        return {"eval_obs": to_np(dict(self.evaluator.obs))}

    def get_info_for_step(self) -> dict:
        # mask: 0 on true terminal (success), 1 on truncation (bootstrap through)
        mask = 0.0 if self._success else 1.0
        return {
            "done": self._done,
            "success": self._success,
            "reward": float(self._last_reward),
            "mask": float(mask),
        }


# ---- websocket plumbing: sim in MAIN thread, ws I/O in worker threads --------
# Two constraints force this architecture:
#   1. Isaac Sim schedules its own asyncio work -> can't host it inside an
#      async websockets server ("Cannot enter into task" conflicts).
#   2. OmniGibson's launch path installs signal handlers -> sim calls must run
#      in the MAIN thread ("signal only works in main thread").
# So: websockets.sync.server handler threads only do wire I/O and hand each
# request to the main thread over a queue; the main thread owns the sim and
# executes ops sequentially (RL rollout is sequential anyway).
def _execute_op(op: str, req: dict, state: dict, server_cfg: dict) -> dict:
    if op == "create_env":
        # OmniGibson is a singleton (one scene per process): create_env is
        # IDEMPOTENT. A learner restart reuses the warm env instead of trying
        # to load a second scene ("Simulator must be stopped before loading
        # scene!"). The learner resets right after create_env anyway.
        if state:
            env_id, env = next(iter(state.items()))
            logger.info(f"create_env: reusing existing env {env_id} (task={env.task_name})")
            return {"env_id": env_id, "task_description": env.task_description}
        # EXPO-FT's train_pi_robo.py sends {example_action, env_usage, video_dir}
        # (see its train_env_creation_request) — task selection is SERVER-side
        # (CLI args), mirroring how their DROID ops server is task-configured.
        # Request fields may override server defaults when present (smoke tests).
        env = BehaviorEnvOps(
            task_name=req.get("task_name", server_cfg["task_name"]),
            instance_ids=req.get("instance_ids", server_cfg["instance_ids"]),
            max_steps=req.get("max_steps", server_cfg["max_steps"]),
            dense_reward=req.get("dense_reward", server_cfg["dense_reward"]),
            perturb_pose=req.get("perturb_pose", server_cfg["perturb_pose"]),
            perturb_xy=server_cfg.get("perturb_xy", 0.15),
            perturb_yaw_deg=server_cfg.get("perturb_yaw_deg", 15.0),
            seed=req.get("seed", server_cfg["seed"]),
            full_res=req.get("full_res", server_cfg["full_res"]),
            snapshot_record_dir=server_cfg["snapshot_record_dir"],
            start_snapshot_dir=server_cfg["start_snapshot_dir"],
            start_near_object=server_cfg["start_near_object"],
            start_distance=server_cfg["start_distance"],
            prompt=server_cfg.get("prompt"),
            start_joint_states=server_cfg["start_joint_states"],
            fixed_eval_starts=server_cfg.get("fixed_eval_starts", 0),
            shaping_coef=server_cfg.get("shaping_coef", 0.0),
            subtask=server_cfg.get("subtask"),
            perturb_obj_xy=server_cfg.get("perturb_obj_xy", 0.0),
            perturb_obj_yaw_deg=server_cfg.get("perturb_obj_yaw_deg", 0.0),
        )
        env_id = str(uuid.uuid4())[:8]
        state[env_id] = env
        return {"env_id": env_id, "task_description": env.task_description}
    env = state[req["env_id"]]
    if op == "reset":
        if req.get("snapshot_path"):
            return env.reset(snapshot_path=req["snapshot_path"])
        return env.reset()
    if op == "step":
        return env.step(req["action"])
    if op == "get_observation":
        return env.get_observation()
    if op == "get_eval_obs":
        return env.get_eval_obs()
    if op == "get_info_for_step":
        return env.get_info_for_step()
    return {"status": "error", "message": f"unknown operation {op}"}


def _handle_connection(websocket, request_q: "queue.Queue"):
    """Runs in a websockets worker thread: wire I/O only, no sim calls."""
    packer = Packer()
    for message in websocket:
        req = unpackb(message)
        op = req.pop("operation")
        resp_q: "queue.Queue" = queue.Queue(maxsize=1)
        request_q.put((op, req, resp_q))
        resp = resp_q.get()  # blocks until the main thread has executed the op
        websocket.send(packer.pack(resp))


def main():
    from websockets.sync.server import serve

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--task-name", default="turning_on_radio")
    p.add_argument(
        "--prompt",
        default=None,
        help="exact language instruction (must match the checkpoint's training prompt; "
        "see openpi TASK_REGISTRY). Fallback derives from task name, which is OOD "
        "for the provided checkpoints.",
    )
    p.add_argument("--instance-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-steps", type=int, default=None, help="None = 1.5x mean human demo length")
    p.add_argument("--sparse-reward", action="store_true", help="final partial credit only (default: dense delta)")
    p.add_argument("--perturb-xy", type=float, default=0.15, help="pose jitter magnitude, meters")
    p.add_argument("--perturb-yaw-deg", type=float, default=15.0, help="pose jitter yaw, degrees")
    p.add_argument("--perturb-pose", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-res", action="store_true", help="render 720/480 RGBD (eval-faithful, ~2x slower)")
    p.add_argument("--snapshot-record-dir", default=None, help="record pre-success sim states here")
    p.add_argument("--start-snapshot-dir", default=None, help="mini-task mode: reset from these snapshots")
    p.add_argument("--start-near-object", default=None, help="teleport base in front of this task-scope object")
    p.add_argument("--start-distance", type=float, default=0.6)
    p.add_argument("--perturb-obj-xy", type=float, default=0.0, help="goal-object x/y jitter, meters")
    p.add_argument("--perturb-obj-yaw-deg", type=float, default=0.0, help="goal-object yaw jitter, degrees")
    p.add_argument("--subtask", default=None, choices=[None, "grasp"],
                   help="override success criterion: 'grasp' = sustained is_grasping(goal obj)")
    p.add_argument("--shaping-coef", type=float, default=0.0,
                   help="potential-based shaping coefficient (0 = off; phi = -dist(EEF, goal obj))")
    p.add_argument("--fixed-eval-starts", type=int, default=0,
                   help="deterministic eval start set size (0 = random training draws)")
    p.add_argument("--start-joint-states", default=None, help=".npy pool of 23-dim demo start states for trunk/arm/gripper pose")
    args = p.parse_args()
    server_cfg = {
        "task_name": args.task_name,
        "instance_ids": args.instance_ids,
        "max_steps": args.max_steps,
        "dense_reward": not args.sparse_reward,
        "perturb_pose": args.perturb_pose,
        "perturb_xy": args.perturb_xy,
        "perturb_yaw_deg": args.perturb_yaw_deg,
        "seed": args.seed,
        "full_res": args.full_res,
        "snapshot_record_dir": args.snapshot_record_dir,
        "start_snapshot_dir": args.start_snapshot_dir,
        "start_near_object": args.start_near_object,
        "start_distance": args.start_distance,
        "prompt": args.prompt,
        "start_joint_states": args.start_joint_states,
        "fixed_eval_starts": args.fixed_eval_starts,
        "shaping_coef": args.shaping_coef,
        "subtask": args.subtask,
        "perturb_obj_xy": args.perturb_obj_xy,
        "perturb_obj_yaw_deg": args.perturb_obj_yaw_deg,
    }

    request_q: "queue.Queue" = queue.Queue()

    # ping_interval=None: create_env/reset block for minutes (scene load);
    # keepalive pings would kill the connection mid-load.
    server = serve(
        lambda ws: _handle_connection(ws, request_q),
        args.host,
        args.port,
        compression=None,
        max_size=None,
        ping_interval=None,
    )
    threading.Thread(target=server.serve_forever, daemon=True, name="ws-accept").start()
    logger.info(f"BEHAVIOR env ops server on {args.host}:{args.port} (sim on main thread)")

    state: dict = {}
    while True:  # main thread: owns the simulator
        op, req, resp_q = request_q.get()
        try:
            resp = _execute_op(op, req, state, server_cfg)
        except Exception as e:
            logger.error(f"op '{op}' failed:\n{traceback.format_exc()}")
            resp = {"status": "error", "message": str(e)}
        resp_q.put(resp)


if __name__ == "__main__":
    main()
