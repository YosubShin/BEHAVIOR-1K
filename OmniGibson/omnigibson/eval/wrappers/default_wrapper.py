from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)


class DefaultWrapper(EnvironmentWrapper):
    """
    Default eval wrapper: low-resolution RGB observations.

    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # env.robots is list[list[Robot]] (one inner list per scene). All scenes share the same robot
        # config/names, but each scene has its own physical cameras -- set the eval resolution on every
        # one. The observation space is keyed once by robot.name (built from scene 0), so updating it
        # per robot is idempotent.
        for scene_robots in env.robots:
            for robot in scene_robots:
                for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
                    sensor_name = camera_name.split("::")[1]
                    sensor = robot.sensors[sensor_name]
                    sensor.image_height = 224
                    sensor.image_width = 224
                    sensor_space = sensor.load_observation_space()
                    if env.observation_space is not None and robot.name in env.observation_space.spaces:
                        env.observation_space.spaces[robot.name].spaces[sensor_name] = sensor_space
        logger.info("Reloaded camera observation spaces!")
