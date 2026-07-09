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
        seed: int = 0,
        full_res: bool = False,
    ):
        self.task_name = task_name
        self.instance_ids = instance_ids
        self._instance_cursor = 0
        self.dense_reward = dense_reward
        self.perturb_pose = perturb_pose
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
                "mode": "train",  # TRAIN instances for RL — never the test split
                "seed": seed,
                "task": {"name": task_name},
                "robot": None,
            }
        )
        self.evaluator = Evaluator(cfg)
        self.env = self.evaluator.env
        self.robot = self.evaluator.robot
        self._cam = self.evaluator.robot_camera_names  # {head,left_wrist,right_wrist}

        self.task_description = self._load_prompt(task_name)
        self._steps = 0
        self._prev_q = 0.0
        self._last_reward = 0.0
        self._success = False
        self._done = False

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

    # ---- the 5 ops ---------------------------------------------------------
    def reset(self) -> dict:
        instance_id = self.instance_ids[self._instance_cursor % len(self.instance_ids)]
        self._instance_cursor += 1

        self.evaluator.reset()
        self.evaluator.load_task_instance(int(instance_id))
        if self.perturb_pose:
            self._perturb_robot_pose()
        self.evaluator.reset()

        self._steps = 0
        self._snapshot_initial_predicates()
        self._prev_q = self._q_score()
        self._last_reward = 0.0
        self._success = False
        self._done = False
        return {"observation": self._observation(), "done": False}

    def _perturb_robot_pose(self):
        """Comet-style start-pose jitter: x,y ±0.15 m, yaw ±15 deg."""
        import omnigibson.utils.transform_utils as T
        import torch as th

        pos, quat = self.robot.get_position_orientation()
        dx, dy = self._rng.uniform(-0.15, 0.15, size=2)
        dyaw = self._rng.uniform(-np.pi / 12, np.pi / 12)
        yaw_q = T.euler2quat(th.tensor([0.0, 0.0, dyaw]))
        new_quat = T.quat_multiply(quat, yaw_q)
        new_pos = pos + th.tensor([dx, dy, 0.0])
        self.robot.set_position_orientation(new_pos, new_quat)

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
        self._success = bool(info["done"]["success"]) if "done" in info else bool(terminated and not truncated)
        self._done = bool(terminated or truncated)
        if self._done and not self.dense_reward:
            self._last_reward = q  # sparse variant: final partial credit only
        return {"action": a, "action_type": "policy"}

    def get_observation(self) -> dict:
        return {"observation": self._observation()}

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
            seed=req.get("seed", server_cfg["seed"]),
            full_res=req.get("full_res", server_cfg["full_res"]),
        )
        env_id = str(uuid.uuid4())[:8]
        state[env_id] = env
        return {"env_id": env_id, "task_description": env.task_description}
    env = state[req["env_id"]]
    if op == "reset":
        return env.reset()
    if op == "step":
        return env.step(req["action"])
    if op == "get_observation":
        return env.get_observation()
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
    p.add_argument("--instance-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-steps", type=int, default=None, help="None = 1.5x mean human demo length")
    p.add_argument("--sparse-reward", action="store_true", help="final partial credit only (default: dense delta)")
    p.add_argument("--perturb-pose", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-res", action="store_true", help="render 720/480 RGBD (eval-faithful, ~2x slower)")
    args = p.parse_args()
    server_cfg = {
        "task_name": args.task_name,
        "instance_ids": args.instance_ids,
        "max_steps": args.max_steps,
        "dense_reward": not args.sparse_reward,
        "perturb_pose": args.perturb_pose,
        "seed": args.seed,
        "full_res": args.full_res,
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
