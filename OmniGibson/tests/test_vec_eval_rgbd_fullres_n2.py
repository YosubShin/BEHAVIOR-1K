"""
Level-B (N=2, GPU) test that the eval camera profile is honored at env CREATION for the multi-env
tiled-rendering path, including PER-CAMERA resolutions and an extra modality.

RGBDFullResWrapper wants head=720x720, wrists=480x480, plus a depth modality. Because multi-env robot
cameras are batched into a single TiledVisionSensor whose resolution is fixed at creation, this can
only work if the wrapper's camera_spec is baked into the robot config before the env is built (a
post-hoc per-sensor resize is silently ignored by the tiled sensor). This test guards that path.

Requires the 2026-challenge-task-instances dataset (picking_up_trash / house_double_floor_lower).
"""

from omegaconf import OmegaConf

from omnigibson.macros import gm

NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"


def _make_cfg(num_envs):
    return OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBDFullResWrapper"},
            "policy_name": "local-zero",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy"},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": 3,
            "write_video": False,
            "num_envs": num_envs,
            "task": {"name": ACTIVITY_NAME},
            "robot": {"type": "R1Pro", "controllers": None},
        }
    )


def _camera_subdict(robot_obs, link_substr):
    key = next(k for k in robot_obs if link_substr in k)
    return robot_obs[key]


def test_rgbd_full_res_per_camera_resolution_on_tiled():
    gm.HEADLESS = True
    from omnigibson.eval.evaluator import Evaluator

    evaluator = Evaluator(_make_cfg(NUM_ENVS))
    try:
        obs_list, _ = evaluator.env.get_obs()
        assert len(obs_list) == NUM_ENVS
        for env_idx in range(NUM_ENVS):
            robot_obs = obs_list[env_idx]["robot_r1"]
            head = _camera_subdict(robot_obs, "zed_link")
            left_wrist = _camera_subdict(robot_obs, "left_realsense_link")
            right_wrist = _camera_subdict(robot_obs, "right_realsense_link")

            # Per-camera resolutions baked at creation (head 720, wrists 480) survive tiled rendering.
            assert tuple(head["rgb"].shape[:2]) == (720, 720), head["rgb"].shape
            assert tuple(left_wrist["rgb"].shape[:2]) == (480, 480), left_wrist["rgb"].shape
            assert tuple(right_wrist["rgb"].shape[:2]) == (480, 480), right_wrist["rgb"].shape

            # The extra depth modality from the wrapper spec is present.
            assert "depth_linear" in head, list(head.keys())
            assert tuple(head["depth_linear"].shape[:2]) == (720, 720), head["depth_linear"].shape
    finally:
        evaluator.env.close()
