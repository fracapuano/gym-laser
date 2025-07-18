from gymnasium.envs.registration import register
from gym_laser.env_utils import EnvParametrization

default_dynamics = EnvParametrization().get_parametrization_dict()

register(
    id="LaserEnv",
    entry_point="gym_laser.LaserEnv:FROGLaserEnv",
    max_episode_steps=20,
    kwargs=default_dynamics
)

register(
    id="RandomLaserEnv",
    entry_point="gym_laser.RandomLaserEnv:RandomFROGLaserEnv",
    max_episode_steps=20,
    kwargs=default_dynamics
)