"""Pytest fixtures for gym-laser environment testing."""

import pytest
import torch
import numpy as np
from gym_laser.env_utils import EnvParametrization
from gym_laser.LaserEnv import FROGLaserEnv
from gym_laser.RandomLaserEnv import RandomFROGLaserEnv

from tests.constants import WINDOW_SIZE


@pytest.fixture
def default_params():
    """Default environment parametrization."""
    return EnvParametrization()


@pytest.fixture
def laser_env_params(default_params):
    """Parameters for LaserEnv initialization."""
    return {
        "compressor_params": default_params.compressor_params,
        "bounds": default_params.bounds,
        "B_integral": default_params.B_integral,
        "render_mode": "rgb_array",
        "device": "cpu",
        "window_size": WINDOW_SIZE,
        "env_kwargs": {"max_duration": 10, "max_steps": 5}
    }


@pytest.fixture
def random_laser_env_params(default_params):
    """Parameters for RandomLaserEnv initialization."""
    return {
        "compressor_params": default_params.compressor_params,
        "bounds": default_params.bounds,
        "B_integral": default_params.B_integral,
        "render_mode": "rgb_array",
        "device": "cpu",
        "window_size": WINDOW_SIZE,
        "env_kwargs": {"max_duration": 10, "max_steps": 5}
    }


@pytest.fixture
def laser_env(laser_env_params):
    """Instantiated LaserEnv for testing."""
    env = FROGLaserEnv(**laser_env_params)
    yield env
    env.close()


@pytest.fixture
def random_laser_env(random_laser_env_params):
    """Instantiated RandomLaserEnv for testing."""
    env = RandomFROGLaserEnv(**random_laser_env_params)
    yield env
    env.close()


@pytest.fixture
def sample_actions():
    """Sample actions for testing."""
    return {
        "zero_action": np.zeros(3, dtype=np.float32),
        "positive_action": np.array([0.1, 0.1, 0.1], dtype=np.float32),
        "negative_action": np.array([-0.1, -0.1, -0.1], dtype=np.float32),
        "large_action": np.array([0.5, -0.5, 0.3], dtype=np.float32),
        "boundary_action": np.array([1.0, -1.0, 0.0], dtype=np.float32)
    }


@pytest.fixture
def fixed_seed():
    """Fixed seed for reproducible testing."""
    return 42