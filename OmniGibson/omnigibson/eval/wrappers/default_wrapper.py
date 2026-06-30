from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import ROBOT_CAMERA_NAMES

EVAL_LOW_RES = (224, 224)  # (H, W)


class DefaultWrapper(EnvironmentWrapper):
    """
    Default eval wrapper: low-resolution (224x224) RGB observations.

    Camera resolution and modalities are declared via :meth:`camera_spec` and baked into the robot
    config at env CREATION by the Evaluator (see ``omnigibson.eval.evaluator.Evaluator.load_env``),
    not changed at runtime. This is required because in multi-env mode robot cameras are batched into
    a single ``TiledVisionSensor`` whose resolution is fixed at creation, so a post-hoc per-sensor
    resize would be silently ignored.

    Args:
        env (og.Environment): The environment to wrap.
    """

    @classmethod
    def camera_spec(cls) -> dict:
        """Returns {"modalities": [...], "resolution": {camera_id: (H, W)}} for this eval profile."""
        return {
            "modalities": ["rgb"],
            "resolution": {camera_id: EVAL_LOW_RES for camera_id in ROBOT_CAMERA_NAMES["R1Pro"]},
        }

    def __init__(self, env: Environment):
        super().__init__(env=env)
