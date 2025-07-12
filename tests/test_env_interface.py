"""Test gymnasium interface compliance for both LaserEnv and RandomLaserEnv."""

import pytest
import numpy as np
import torch
from gymnasium.spaces import Box, Dict
from gymnasium.utils.env_checker import check_env


class TestLaserEnvInterface:
    """Test LaserEnv gymnasium interface compliance."""
    
    def test_environment_initialization(self, laser_env):
        """Test that environment initializes correctly."""
        assert laser_env is not None
        assert hasattr(laser_env, 'observation_space')
        assert hasattr(laser_env, 'action_space')
        assert hasattr(laser_env, 'metadata')
        
    def test_observation_space_structure(self, laser_env):
        """Test observation space structure."""
        obs_space = laser_env.observation_space
        assert isinstance(obs_space, Dict)
        
        # Check required keys
        required_keys = {'frog_trace', 'psi', 'action'}
        assert set(obs_space.spaces.keys()) == required_keys
        
        # Check frog_trace space
        frog_space = obs_space.spaces['frog_trace']
        assert isinstance(frog_space, Box)
        assert frog_space.shape == (1, 32, 32)  # window_size=32 from fixtures
        assert frog_space.dtype == np.uint8
        assert frog_space.low == 0
        assert frog_space.high == 255
        
        # Check psi space
        psi_space = obs_space.spaces['psi']
        assert isinstance(psi_space, Box)
        assert psi_space.shape == (3,)
        assert psi_space.dtype == np.float32
        assert np.all(psi_space.low == 0.0)
        assert np.all(psi_space.high == 1.0)
        
        # Check action space
        action_space = obs_space.spaces['action']
        assert isinstance(action_space, Box)
        assert action_space.shape == (3,)
        assert action_space.dtype == np.float32
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
        
    def test_action_space_structure(self, laser_env):
        """Test action space structure."""
        action_space = laser_env.action_space
        assert isinstance(action_space, Box)
        assert action_space.shape == (3,)
        assert action_space.dtype == np.float32
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
        
    def test_reset_returns_correct_format(self, laser_env, fixed_seed):
        """Test that reset returns observation and info in correct format."""
        obs, info = laser_env.reset(seed=fixed_seed)
        
        # Check observation format
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {'frog_trace', 'psi', 'action'}
        
        # Check observation values
        assert obs['frog_trace'].shape == (1, 32, 32)
        assert obs['frog_trace'].dtype == np.uint8
        assert obs['psi'].shape == (3,)
        assert obs['psi'].dtype == np.float32
        assert obs['action'].shape == (3,)
        assert obs['action'].dtype == np.float32
        
        # Check info
        assert isinstance(info, dict)
        required_info_keys = {
            'current_control', 'current_control (picoseconds)',
            'current FWHM (ps)', 'current Peak Intensity (TW/m^2)',
            'x_t(perc)', 'TL-L1Loss', 'FWHM-failure', 'Timesteps-failure',
            'B-value'
        }
        assert set(info.keys()).issuperset(required_info_keys)
        
    def test_step_returns_correct_format(self, laser_env, sample_actions, fixed_seed):
        """Test that step returns observation, reward, terminated, truncated, info."""
        laser_env.reset(seed=fixed_seed)
        
        for action_name, action in sample_actions.items():
            obs, reward, terminated, truncated, info = laser_env.step(action)
            
            # Check observation format
            assert isinstance(obs, dict)
            assert set(obs.keys()) == {'frog_trace', 'psi', 'action'}
            
            # Check reward
            assert isinstance(reward, (float, np.floating))
            assert np.isfinite(reward)
            
            # Check terminated/truncated
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            
            # Check info
            assert isinstance(info, dict)
            
            # Reset for next test
            laser_env.reset(seed=fixed_seed)
            
    def test_observation_space_compliance(self, laser_env, fixed_seed):
        """Test that observations comply with observation space."""
        obs, _ = laser_env.reset(seed=fixed_seed)
        assert laser_env.observation_space.contains(obs)
        
        # Test multiple steps
        for _ in range(3):
            action = laser_env.action_space.sample()
            obs, _, _, _, _ = laser_env.step(action)
            assert laser_env.observation_space.contains(obs)
            
    def test_action_space_compliance(self, laser_env, sample_actions, fixed_seed):
        """Test that actions comply with action space."""
        laser_env.reset(seed=fixed_seed)
        
        for action_name, action in sample_actions.items():
            # All sample actions should be valid
            assert laser_env.action_space.contains(action)
            
            # Should be able to step with valid actions
            obs, reward, terminated, truncated, info = laser_env.step(action)
            assert obs is not None
            
            # Reset for next test
            laser_env.reset(seed=fixed_seed)
            
    def test_deterministic_reset(self, laser_env, fixed_seed):
        """Test that reset is deterministic with fixed seed."""
        obs1, info1 = laser_env.reset(seed=fixed_seed)
        obs2, info2 = laser_env.reset(seed=fixed_seed)
        
        # Observations should be identical
        assert np.array_equal(obs1['frog_trace'], obs2['frog_trace'])
        assert np.array_equal(obs1['psi'], obs2['psi'])
        assert np.array_equal(obs1['action'], obs2['action'])
        
    def test_step_count_tracking(self, laser_env, fixed_seed):
        """Test that step count is tracked correctly."""
        laser_env.reset(seed=fixed_seed)
        
        assert laser_env.n_steps == 0
        
        for i in range(3):
            action = laser_env.action_space.sample()
            laser_env.step(action)
            assert laser_env.n_steps == i + 1
            
    def test_episode_termination(self, laser_env, fixed_seed):
        """Test that episode terminates correctly."""
        laser_env.reset(seed=fixed_seed)
        
        # Test that episode terminates when max steps reached
        for i in range(laser_env.MAX_STEPS):
            action = laser_env.action_space.sample()
            obs, reward, terminated, truncated, info = laser_env.step(action)
            
            if i == laser_env.MAX_STEPS - 1:
                assert truncated  # Should be truncated at max steps
            else:
                assert not truncated
                
    def test_rendering_modes(self, laser_env):
        """Test that rendering modes work correctly."""
        laser_env.reset()
        
        # Test rgb_array mode
        rgb_array = laser_env.render()
        assert isinstance(rgb_array, np.ndarray)
        assert rgb_array.ndim == 3
        assert rgb_array.shape[2] == 3  # RGB channels
        
        # Test that multiple renders work
        rgb_array2 = laser_env.render()
        assert rgb_array2.shape == rgb_array.shape


class TestRandomLaserEnvInterface:
    """Test RandomLaserEnv gymnasium interface compliance."""
    
    def test_environment_initialization(self, random_laser_env):
        """Test that environment initializes correctly."""
        assert random_laser_env is not None
        assert hasattr(random_laser_env, 'observation_space')
        assert hasattr(random_laser_env, 'action_space')
        assert hasattr(random_laser_env, 'metadata')
        
    def test_observation_space_structure(self, random_laser_env):
        """Test observation space structure."""
        obs_space = random_laser_env.observation_space
        assert isinstance(obs_space, Dict)
        
        # Check required keys (RandomLaserEnv has additional keys)
        required_keys = {'frog_trace', 'psi', 'action', 'B_integral', 'compressor_GDD'}
        assert set(obs_space.spaces.keys()) == required_keys
        
        # Check additional spaces specific to RandomLaserEnv
        b_space = obs_space.spaces['B_integral']
        assert isinstance(b_space, Box)
        assert b_space.shape == (1,)
        assert b_space.dtype == np.float32
        assert b_space.low == 1.0
        assert b_space.high == 3.5
        
        gdd_space = obs_space.spaces['compressor_GDD']
        assert isinstance(gdd_space, Box)
        assert gdd_space.shape == (1,)
        assert gdd_space.dtype == np.float32
        assert gdd_space.low == 2.47
        assert gdd_space.high == 2.87
        
    def test_reset_returns_correct_format(self, random_laser_env, fixed_seed):
        """Test that reset returns observation and info in correct format."""
        obs, info = random_laser_env.reset(seed=fixed_seed)
        
        # Check observation format
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {'frog_trace', 'psi', 'action', 'B_integral', 'compressor_GDD'}
        
        # Check additional observation values
        assert obs['B_integral'].shape == (1,)
        assert obs['B_integral'].dtype == np.float32
        assert obs['compressor_GDD'].shape == (1,)
        assert obs['compressor_GDD'].dtype == np.float32
        
    def test_domain_randomization_interface(self, random_laser_env):
        """Test domain randomization interface."""
        # Test that RandomLaserEnv has the required DR methods
        assert hasattr(random_laser_env, 'set_dr_training')
        assert hasattr(random_laser_env, 'get_dr_training')
        assert hasattr(random_laser_env, 'set_task')
        assert hasattr(random_laser_env, 'get_task')
        assert hasattr(random_laser_env, 'get_default_task')
        
        # Test DR training flag
        assert random_laser_env.get_dr_training() is False  # Default
        random_laser_env.set_dr_training(True)
        assert random_laser_env.get_dr_training() is True
        
        # Test task interface
        default_task = random_laser_env.get_default_task()
        assert isinstance(default_task, np.ndarray)
        assert default_task.shape == (2,)  # B_integral and compressor_GDD
        
        current_task = random_laser_env.get_task()
        assert isinstance(current_task, np.ndarray)
        assert current_task.shape == (2,)
        
    def test_psi_property_type(self, random_laser_env, fixed_seed):
        """Test that psi property returns numpy array (not tensor)."""
        random_laser_env.reset(seed=fixed_seed)
        
        psi = random_laser_env.psi
        assert isinstance(psi, np.ndarray)
        assert psi.dtype == np.float32
        assert psi.shape == (3,)
        
    def test_observation_space_compliance(self, random_laser_env, fixed_seed):
        """Test that observations comply with observation space."""
        obs, _ = random_laser_env.reset(seed=fixed_seed)
        assert random_laser_env.observation_space.contains(obs)
        
        # Test multiple steps
        for _ in range(3):
            action = random_laser_env.action_space.sample()
            obs, _, _, _, _ = random_laser_env.step(action)
            assert random_laser_env.observation_space.contains(obs)


class TestEnvironmentCheckerCompliance:
    """Test environments against gymnasium's environment checker."""
    
    def test_laser_env_checker(self, laser_env_params):
        """Test LaserEnv with gymnasium's environment checker."""
        from gym_laser.LaserEnv import FROGLaserEnv
        
        # Create a fresh environment for the checker
        env = FROGLaserEnv(**laser_env_params)
        
        # This will raise an exception if the environment is not compliant
        try:
            check_env(env)
        except Exception as e:
            pytest.fail(f"LaserEnv failed gymnasium compliance check: {e}")
        finally:
            env.close()
            
    def test_random_laser_env_checker(self, random_laser_env_params):
        """Test RandomLaserEnv with gymnasium's environment checker."""
        from gym_laser.RandomLaserEnv import RandomFROGLaserEnv
        
        # Create a fresh environment for the checker
        env = RandomFROGLaserEnv(**random_laser_env_params)
        
        # This will raise an exception if the environment is not compliant
        try:
            check_env(env)
        except Exception as e:
            pytest.fail(f"RandomLaserEnv failed gymnasium compliance check: {e}")
        finally:
            env.close() 