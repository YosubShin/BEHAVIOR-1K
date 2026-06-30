import csv
import cv2
import json
import logging
import os
import sys
import traceback
from signal import SIGINT, signal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

import omnigibson as og
import omnigibson.utils.transform_utils as T
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import (
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import create_module_logger


LIGHT_EVAL_TASKS = {"turning_out_all_lights_before_sleep"}
EVAL_BASE_LINK_MASS = 250.0
EVAL_HEAD_HORIZONTAL_APERTURE = 40.0
NUM_TEST_INSTANCES = 40

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def resolve_instance_ids(task_name: str, instance_indices: list[int]) -> list[int]:
    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, newline="", encoding="utf-8") as f:
        row = next(row for row in csv.DictReader(f) if row["Task"] == task_name)
    test_instances = [int(instance_id.strip()) for instance_id in row["Test Instance IDs"].split(",")]
    assert (
        len(test_instances) == NUM_TEST_INSTANCES
    ), f"Expected {NUM_TEST_INSTANCES} test instances for {task_name}, got {len(test_instances)}"
    assert set(instance_indices).issubset(
        set(range(NUM_TEST_INSTANCES))
    ), f"Instance indices must be in range({NUM_TEST_INSTANCES})"
    return [int(test_instances[i]) for i in instance_indices]


def evaluate_instances_batched(
    instances: Sequence,
    num_envs: int,
    load_fn: Callable[[Dict[int, object]], None],
    step_fn: Callable[[List[int]], "tuple[Sequence[bool], Sequence[bool]]"],
    record_fn: Callable[..., object],
    max_steps: Optional[int] = None,
) -> "Dict[object, object]":
    """
    Drive evaluation of @instances in groups of @num_envs. Pure orchestration: all sim work is
    delegated to the injected load_fn / step_fn / record_fn, and a slot that finishes early is dropped
    from the active list (frozen) until the whole group is done -- loading an instance settles physics
    for ALL scenes at once, so refilling one slot mid-group would disturb the slots still running.
    Returns {instance id -> record_fn's return}.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    results: Dict[object, object] = {}
    pending = list(instances)

    while pending:
        batch = pending[:num_envs]
        pending = pending[num_envs:]

        slot_to_instance: Dict[int, object] = {slot: inst for slot, inst in enumerate(batch)}
        load_fn(dict(slot_to_instance))

        active = {slot: True for slot in slot_to_instance}
        step = 0
        while any(active.values()):
            active_slots = sorted(slot for slot, is_active in active.items() if is_active)
            terminated, truncated = step_fn(active_slots)
            step += 1

            hit_cap = max_steps is not None and step >= max_steps
            for slot in active_slots:
                term = bool(terminated[slot])
                trunc = bool(truncated[slot]) or hit_cap
                if term or trunc:
                    results[slot_to_instance[slot]] = record_fn(
                        slot=slot,
                        instance=slot_to_instance[slot],
                        step=step,
                        terminated=term,
                        truncated=trunc,
                    )
                    active[slot] = False

    return results


class Evaluator:
    """
    Vectorized evaluator engine for BEHAVIOR tasks. Holds a single OmniGibson environment with
    ``num_envs`` parallel scene slots (one task instance per slot) and per-slot robots, policies,
    metrics, observations and video writers (index everything by ``env_idx``). ``num_envs=1``
    reproduces single-env evaluation. All sim interaction lives here; the offline driver (:meth:`run`)
    is a thin layer on top of :func:`evaluate_instances_batched`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0

        # Number of parallel env slots (instances evaluated concurrently). Defaults to 1.
        self.num_envs = int(cfg.get("num_envs", 1))
        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        assert (
            self.env.num_envs == self.num_envs
        ), f"Env created with num_envs={self.env.num_envs} but Evaluator expected {self.num_envs}."

        # Per-slot robots / policies / metrics / observations / video writers / light synchronizers.
        self.robots = self.load_robots()
        self._apply_robot_eval_settings()
        self.policies = self.load_policies()
        self.metrics = self.load_metrics()
        self.obs = [None] * self.num_envs
        self._video_writers = [None] * self.num_envs
        self.light_synchronizers = [None] * self.num_envs

        # Initialize the physics views so the first reset()/load can read/restore robot joint state.
        og.sim.update_handles()
        self.env._current_episode = 0

    @property
    def should_sync_lights(self) -> bool:
        return self.env.task.activity_name in LIGHT_EVAL_TASKS

    def _reset_light_synchronizer(self, env_idx: int) -> None:
        if self.should_sync_lights:
            synchronizer = LightToggleSynchronizer(self.env.scenes[env_idx])
            synchronizer.reset_from_current_state()
            self.light_synchronizers[env_idx] = synchronizer
        else:
            self.light_synchronizers[env_idx] = None

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for key in self.human_stats:
                    self.human_stats[key].append(episode[key])
        for key in self.human_stats:
            self.human_stats[key] = sum(self.human_stats[key]) / len(self.human_stats[key])

        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        robot_model = robot_type.lower()
        assert robot_model == "r1pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."

        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                robot_type=robot_model,
                robot_name="robot_r1",
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        cfg["robots"][0]["model"] = cfg["robots"][0].pop("type")
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

        # Camera resolution + modalities are decided by the chosen eval wrapper and baked into the
        # robot config HERE, before env creation. This is required (not just cleaner) because in
        # multi-env mode robot cameras are batched into a single TiledVisionSensor whose resolution is
        # fixed at creation -- a post-hoc per-sensor resize in the wrapper would be silently ignored.
        # Resolve the wrapper class (without instantiating it -- that needs the env) to read its spec.
        camera_spec = get_class(env_wrapper["_target_"]).camera_spec()
        cfg["robots"][0]["obs_modalities"] = ["proprio", *camera_spec["modalities"]]
        # Per-camera resolution via per-link sensor_config overrides. The link key is
        # "<link>:Camera:0" (see robots/robot.py), which takes precedence over the class-level
        # VisionSensor default; derive it from the robot-prefixed camera name in ROBOT_CAMERA_NAMES.
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            link_key = camera_name.split("::")[1].split(":", 1)[1]
            height, width = camera_spec["resolution"][camera_id]
            cfg["robots"][0]["sensor_config"][link_key] = {
                "sensor_kwargs": {"image_height": height, "image_width": width}
            }
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(
                OmegaConf.to_container(self.cfg.robot.controllers, resolve=True)
            )

        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False

        # Run num_envs instances of the same scene model + task in parallel slots.
        cfg.setdefault("env", {})
        cfg["env"]["num_envs"] = self.num_envs

        env = og.Environment(configs=cfg)
        return instantiate(env_wrapper, env=env)

    def load_robots(self) -> List[Robot]:
        # env.robots is list[list[Robot]] (one inner list per scene). The eval pipeline assumes a single
        # robot per scene (load_env configures exactly one "robot_r1"; _preprocess_obs hardcodes its
        # name) -- assert it so a multi-robot config fails loudly instead of silently using robot 0.
        for scene_robots in self.env.robots:
            assert len(scene_robots) == 1, f"Eval assumes one robot per scene, got {len(scene_robots)}."
        return [scene_robots[0] for scene_robots in self.env.robots]

    def _apply_robot_eval_settings(self) -> None:
        # Heavier base so contact with the (heavy) world does not shove the robot around during eval.
        # base mass writes require the sim stopped; do all robots inside a single stop/play (global op).
        if any(robot.model in ("r1", "r1pro") for robot in self.robots):
            og.sim.stop()
            for robot in self.robots:
                if robot.model in ("r1", "r1pro"):
                    robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
            og.sim.play()

        head_sensor_name = ROBOT_CAMERA_NAMES["R1Pro"]["head"].split("::")[1]
        for robot in self.robots:
            robot.sensors[head_sensor_name].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE

    def load_policies(self) -> List[Any]:
        # One independent (possibly stateful) policy instance per env slot.
        policies = []
        for _ in range(self.num_envs):
            policy = instantiate(self.cfg.model)
            if hasattr(policy, "set_action_dim"):
                policy.set_action_dim(self.robots[0].action_dim)
            policies.append(policy)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded {self.num_envs} policy instance(s): {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policies

    def load_metrics(self) -> List[List[MetricBase]]:
        return [
            [AgentMetric(self.human_stats, env_idx=i), TaskMetric(self.human_stats, env_idx=i)]
            for i in range(self.num_envs)
        ]

    def _apply_actions(self, actions: th.Tensor, active_slots: List[int]):
        """
        Step all env slots one step with the given ``(num_envs, action_dim)`` actions, then update the
        observations and metrics for the @active_slots only (frozen slots are still stepped by the
        shared simulator but their obs/metrics are not advanced). Returns ``(terminated, truncated,
        info)`` straight from the env (``terminated``/``truncated`` are ``(num_envs,)`` tensors, ``info``
        a per-slot list).
        """
        obs_list, _, terminated, truncated, info = self.env.step(actions, n_render_iterations=1)
        if self.should_sync_lights:
            # Re-derive visual light state (toggles drive lights that aren't directly serialized) for
            # the active slots, then re-read obs so the recorded frame matches the synced lights.
            for slot in active_slots:
                if self.light_synchronizers[slot] is not None:
                    self.light_synchronizers[slot].sync_from_current_state()
            for _ in range(3):
                og.sim.render()
            obs_list, _ = self.env.get_obs()
        for slot in active_slots:
            self.obs[slot] = self._preprocess_obs(obs_list[slot], env_idx=slot)
            for metric in self.metrics[slot]:
                metric.step(
                    self.env,
                    actions[slot],
                    obs_list[slot],
                    0.0,
                    bool(terminated[slot]),
                    bool(truncated[slot]),
                    info[slot],
                )
        return terminated, truncated, info

    def _step_fn(self, active_slots: List[int]) -> Tuple[th.Tensor, th.Tensor]:
        """
        ``step_fn`` consumed by :func:`evaluate_instances_batched`: drive active slots with their own
        policies (frozen slots get a zero action but are still stepped by the shared simulator).
        """
        action_dim = self.robots[0].action_dim
        actions = th.zeros((self.num_envs, action_dim), dtype=th.float32)
        for slot in active_slots:
            actions[slot] = self.policies[slot].forward(obs=self.obs[slot])
        terminated, truncated, _ = self._apply_actions(actions, active_slots)
        return terminated, truncated

    def _set_video_writer(self, env_idx: int, video_writer) -> None:
        existing = self._video_writers[env_idx]
        if existing is not None:
            container, stream = existing
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writers[env_idx] = video_writer

    def _load_instance_state(self, instance_id: int, env_idx: int) -> None:
        """
        Load a task instance's object/robot state into slot @env_idx. Does NOT settle physics or
        finalize the scene -- that happens once for the whole batch in :meth:`_settle_and_finalize` so
        settling does not disturb already-loaded sibling slots.
        """
        robot = self.robots[env_idx]
        scene = self.env.scenes[env_idx]
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(
                scene_model,
                f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
            )
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = {key.lower(): value for key, value in tro_state.items()}
                if "robot" in presampled_robot_poses:
                    available_poses = presampled_robot_poses["robot"]
                elif robot.model in presampled_robot_poses:
                    print("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_robot_poses[robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {robot.model}!")
                # Presampled poses are scene-relative; frame="scene" puts slot N's robot in its own
                # scene rather than env 0's geometry (cf. #2257).
                robot.set_position_orientation(
                    available_poses[0]["position"], available_poses[0]["orientation"], frame="scene"
                )
                scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[env_idx][tro_key].load_state(tro_state, serialized=False)

        if self.should_sync_lights:
            set_light_control_toggles(self.env.task.object_scope[env_idx].values(), True)
        og.sim.update_handles()

    def _settle_and_finalize(self, slots: List[int]) -> None:
        """
        Settle physics for all freshly-loaded @slots together and finalize each one's scene. Loading
        state can introduce jitter, so keep loaded task-relevant objects (not the robot) still for a
        few sub-steps before snapshotting each scene's initial state.
        """
        for _ in range(25):
            og.sim.step_physics()
            for slot in slots:
                for inst, entity in self.env.task.object_scope[slot].items():
                    if not is_system_bddl_inst(inst) and entity is not None and not isinstance(entity, Robot):
                        entity.keep_still()
        for slot in slots:
            self.env.scenes[slot].update_initial_file()
            self.env.scenes[slot].reset()

    def load_batch(
        self,
        slot_to_instance: dict,
        write_video: bool = False,
        video_path: str = None,
        rollout_id: int = 0,
    ) -> None:
        """
        Load a batch of instances (one per slot) into their slots, settle the whole batch together,
        then refresh observations and reset per-slot policy / metrics / light synchronizers (and video
        writers). Consumed by :meth:`run` via :func:`evaluate_instances_batched`.
        """
        ordered_slots = sorted(slot_to_instance)
        for slot in ordered_slots:
            self._load_instance_state(slot_to_instance[slot], env_idx=slot)
        self._settle_and_finalize(ordered_slots)
        obs_list, _ = self.env.reset(env_indices=th.tensor(ordered_slots, dtype=th.long))
        task_name = self.cfg.task.name
        for i, slot in enumerate(ordered_slots):
            self._reset_light_synchronizer(slot)
            self.obs[slot] = self._preprocess_obs(obs_list[i], env_idx=slot)
            self.policies[slot].reset()
            for metric in self.metrics[slot]:
                metric.reset(self.env)
            if write_video:
                video_name = os.path.join(video_path, f"{task_name}_{slot_to_instance[slot]}_{rollout_id}.mp4")
                self._set_video_writer(slot, create_video_writer(fpath=video_name, resolution=(448, 672)))

    def _preprocess_obs(self, obs: dict, env_idx: int = 0) -> dict:
        robot = self.robots[env_idx]
        obs = flatten_obs_dict(obs)
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        # The first camera-parameter query returns zeros; fall back to get_position_orientation() then.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self, env_idx: int) -> None:
        obs = self.obs[env_idx]
        if obs is None or ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in obs:
            return
        left_wrist_rgb = cv2.resize(obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(), (224, 224))
        right_wrist_rgb = cv2.resize(obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(), (224, 224))
        head_rgb = cv2.resize(obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(), (448, 448))
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self._video_writers[env_idx],
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """Generic reset of all slots to the currently-loaded (seed) instance + reset policies/metrics."""
        obs_list, _ = self.env.reset()
        for env_idx in range(self.num_envs):
            self._reset_light_synchronizer(env_idx)
            self.obs[env_idx] = self._preprocess_obs(obs_list[env_idx], env_idx=env_idx)
            for metric in self.metrics[env_idx]:
                metric.reset(self.env)
            self.policies[env_idx].reset()
        self.n_success_trials, self.n_trials = 0, 0

    def run(
        self,
        instances_to_run: List[int],
        write_video: bool = False,
        video_path: str = None,
        metrics_dir: str = None,
        rollout_id: int = 0,
    ) -> dict:
        """
        Offline driver: evaluate every instance in @instances_to_run, processing them ``num_envs`` at a
        time. The whole group finishes before the next group loads, and an early-finishing slot waits
        idle rather than getting a new instance. Writes one result JSON per instance to @metrics_dir if
        given. Returns {instance id -> result dict} (result is score_utils-compatible: q_score/time/
        agent_distance plus task/instance/success/steps).
        """
        task_name = self.cfg.task.name

        def load_fn(slot_to_instance):
            self.load_batch(
                slot_to_instance, write_video=write_video, video_path=video_path, rollout_id=rollout_id
            )

        def step_fn(active_slots):
            terminated, truncated = self._step_fn(active_slots)
            if write_video:
                for slot in active_slots:
                    self._write_video(slot)
            return terminated, truncated

        def record_fn(slot, instance, step, terminated, truncated):
            self.n_trials += 1
            success = bool(self.env.task.success[slot])
            if success:
                self.n_success_trials += 1
            result = {"task": task_name, "instance_id": int(instance), "rollout_id": rollout_id, "steps": step}
            result["success"] = success
            for metric in self.metrics[slot]:
                result.update(metric.aggregate(self.env))
            if metrics_dir is not None:
                with open(os.path.join(metrics_dir, f"{task_name}_{instance}_{rollout_id}.json"), "w") as f:
                    json.dump(result, f, indent=2, default=float)
            if write_video:
                self._set_video_writer(slot, None)
            q_score = result.get("q_score", {}).get("final")
            logger.info(
                f"Instance {instance} (slot {slot}) finished at step {step}: success={success} q_score={q_score}"
            )
            return result

        return evaluate_instances_batched(
            list(instances_to_run),
            self.num_envs,
            load_fn=load_fn,
            step_fn=step_fn,
            record_fn=record_fn,
        )

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        for env_idx in range(self.num_envs):
            self._set_video_writer(env_idx, None)
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


__all__ = ["Evaluator", "resolve_instance_ids", "evaluate_instances_batched"]
