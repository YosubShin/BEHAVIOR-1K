from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import HEAD_RESOLUTION, ROBOT_CAMERA_NAMES, WRIST_RESOLUTION


class RGBDFullResWrapper(EnvironmentWrapper):
    """
    Eval wrapper: full-resolution RGB-D observations (head at HEAD_RESOLUTION, wrists at
    WRIST_RESOLUTION) matching the data-collection cameras, plus a depth modality.

    As with :class:`~omnigibson.eval.wrappers.default_wrapper.DefaultWrapper`, camera resolution and
    modalities are declared via :meth:`camera_spec` and baked into the robot config at env CREATION
    (multi-env tiled rendering fixes resolution at creation, so a runtime resize would not apply).

    Args:
        env (og.Environment): The environment to wrap.
    """

    @classmethod
    def camera_spec(cls) -> dict:
        """Returns {"modalities": [...], "resolution": {camera_id: (H, W)}} for this eval profile."""
        resolution = {
            camera_id: (HEAD_RESOLUTION if camera_id == "head" else WRIST_RESOLUTION)
            for camera_id in ROBOT_CAMERA_NAMES["R1Pro"]
        }
        return {"modalities": ["rgb", "depth_linear"], "resolution": resolution}

    def __init__(self, env: Environment):
        super().__init__(env=env)
