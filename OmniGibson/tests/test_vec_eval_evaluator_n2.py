"""
Level-B (N=2, GPU) test for the vectorized eval engine: ``omnigibson.eval.evaluator.Evaluator`` and
``evaluate_instances_batched``.

Builds an Evaluator with num_envs=2 and a zero-action LocalPolicy, then evaluates two task instances
in a single batch (one per slot). Guards that the vectorized driver:
  - runs both slots concurrently and returns one result per instance,
  - produces independent, score_utils-compatible per-slot results (success + q_score in [0, 1]),
  - keeps num_envs=1's machinery generalized to N (per-slot robots/policies/metrics lists).

Uses a tiny max_steps so each rollout truncates quickly (the zero policy never completes the task).
Requires the 2026-challenge-task-instances dataset (picking_up_trash / house_double_floor_lower).
"""

from omegaconf import OmegaConf

import omnigibson as og
from omnigibson.macros import gm

NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"
INSTANCE_IDS = [0, 100]  # both have *-tro_state.json under the 2026 instances dir
MAX_STEPS = 3


def _make_cfg(num_envs):
    return OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
            "policy_name": "local-zero",
            # LocalPolicy with no inner policy => emits a zero delta action of the robot's action_dim.
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy"},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": MAX_STEPS,
            "write_video": False,
            "num_envs": num_envs,
            "task": {"name": ACTIVITY_NAME},
            "robot": {"type": "R1Pro", "controllers": None},
        }
    )


def test_vectorized_evaluator_runs_batch_of_two():
    gm.HEADLESS = True
    from omnigibson.eval.evaluator import Evaluator

    # NOTE: do NOT use `with Evaluator(...)` here -- its __exit__ calls og.shutdown(), which tears the
    # process down before any post-block assertions (or pytest's own summary) can run. Construct
    # directly, assert while the sim is live, and close the env in finally.
    evaluator = Evaluator(_make_cfg(NUM_ENVS))
    try:
        # Per-slot machinery is generalized to N.
        assert evaluator.num_envs == NUM_ENVS
        assert len(evaluator.robots) == NUM_ENVS
        assert len(evaluator.policies) == NUM_ENVS
        assert len(evaluator.metrics) == NUM_ENVS
        assert evaluator.env.num_envs == NUM_ENVS

        results = evaluator.run(INSTANCE_IDS)

        # One result per instance, batched together (num_envs == len(instances)).
        assert set(results.keys()) == set(INSTANCE_IDS), f"expected results for {INSTANCE_IDS}, got {list(results)}"

        for inst in INSTANCE_IDS:
            r = results[inst]
            assert r["instance_id"] == inst
            assert isinstance(r["success"], bool)
            # Zero policy can't finish the task -> truncates at the timeout (steps ~ MAX_STEPS).
            assert r["success"] is False
            assert 1 <= r["steps"] <= MAX_STEPS + 1
            # score_utils-compatible per-slot metrics.
            assert 0.0 <= r["q_score"]["final"] <= 1.0
            assert "time" in r
            assert {"base", "left", "right"}.issubset(r["normalized_agent_distance"].keys())

        # The two slots produced independent result dicts.
        assert results[INSTANCE_IDS[0]] is not results[INSTANCE_IDS[1]]
    finally:
        evaluator.env.close()
