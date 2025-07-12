"""Test environment semantics for both LaserEnv and RandomLaserEnv."""

import pytest
import numpy as np
import torch


class TestLaserEnvSemantics:
    """Test LaserEnv semantics and behavior."""
    
    def test_state_changes_with_actions(self, laser_env, sample_actions, fixed_seed):
        """Test that actions actually change the environment state."""
        laser_env.reset(seed=fixed_seed)
        
        # Get initial state
        initial_psi = laser_env.psi.clone()
        initial_obs, _ = laser_env.reset(seed=fixed_seed)
        
        # Apply action and check state change
        action = sample_actions['positive_action']
        new_obs, reward, terminated, truncated, info = laser_env.step(action)
        
        # State should have changed
        assert not torch.allclose(laser_env.psi, initial_psi)
        assert not np.array_equal(new_obs['psi'], initial_obs['psi'])
        assert np.array_equal(new_obs['action'], action)
        
    def test_psi_bounds_enforcement(self, laser_env, fixed_seed):
        """Test that psi values are kept within [0, 1] bounds."""
        laser_env.reset(seed=fixed_seed)
        
        # Test with large positive action
        large_positive = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        laser_env.step(large_positive)
        assert np.all(laser_env.psi >= 0.0)
        assert np.all(laser_env.psi <= 1.0)
        
        # Test with large negative action
        large_negative = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        laser_env.step(large_negative)
        assert np.all(laser_env.psi >= 0.0)
        assert np.all(laser_env.psi <= 1.0)
        
    def test_action_remapping(self, laser_env, fixed_seed):
        """Test that actions are correctly remapped from [-1, 1] to action bounds."""
        laser_env.reset(seed=fixed_seed)
        
        # Test boundary actions
        boundary_action = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        remapped = laser_env.remap_action(boundary_action)
        
        # Should be within action bounds
        assert np.all(remapped >= laser_env.action_lower_bound)
        assert np.all(remapped <= laser_env.action_upper_bound)
        
        # Test that zero action maps to middle of range
        zero_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        remapped_zero = laser_env.remap_action(zero_action)
        expected_middle = (laser_env.action_lower_bound + laser_env.action_upper_bound) / 2
        assert np.allclose(remapped_zero, expected_middle)
        
    def test_reward_calculation_consistency(self, laser_env, fixed_seed):
        """Test that reward calculation is consistent."""
        laser_env.reset(seed=fixed_seed)
        
        # Test multiple steps with same action
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        rewards = []
        for _ in range(3):
            laser_env.reset(seed=fixed_seed)
            _, reward, _, _, _ = laser_env.step(action)
            rewards.append(reward)
            
        # Rewards should be identical for identical conditions
        assert np.allclose(rewards, rewards[0])
        
    def test_reward_components(self, laser_env, fixed_seed):
        """Test that reward components are properly calculated."""
        laser_env.reset(seed=fixed_seed)
        
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, _, _, info = laser_env.step(action)
        
        # Check that reward components exist
        assert 'alive_component' in info
        assert 'intensity_component' in info
        assert 'duration_component' in info
        assert 'final_reward' in info
        
        # Check that components are numeric
        assert isinstance(info['alive_component'], (int, float))
        assert isinstance(info['intensity_component'], (int, float))
        assert isinstance(info['duration_component'], (int, float))
        assert isinstance(info['final_reward'], (int, float))
        
        # Check that final reward matches step reward
        assert np.isclose(info['final_reward'], reward)
        
    def test_termination_conditions(self, laser_env, fixed_seed):
        """Test termination conditions."""
        laser_env.reset(seed=fixed_seed)
        
        # Test that termination is initially False
        assert not laser_env.is_terminated()
        
        # Test that truncation occurs at max steps
        for i in range(laser_env.MAX_STEPS):
            action = laser_env.action_space.sample()
            obs, reward, terminated, truncated, info = laser_env.step(action)
            
            if i == laser_env.MAX_STEPS - 1:
                assert truncated
            else:
                assert not truncated
                
    def test_step_count_increment(self, laser_env, fixed_seed):
        """Test that step count increments correctly."""
        laser_env.reset(seed=fixed_seed)
        
        initial_steps = laser_env.n_steps
        assert initial_steps == 0
        
        for i in range(3):
            action = laser_env.action_space.sample()
            laser_env.step(action)
            assert laser_env.n_steps == i + 1
            
    def test_controls_buffer_management(self, laser_env, fixed_seed):
        """Test that controls buffer is managed correctly."""
        laser_env.reset(seed=fixed_seed)
        
        # Buffer should have initial control
        assert len(laser_env.controls_buffer) == 1
        
        # Add actions and check buffer
        for i in range(3):
            action = laser_env.action_space.sample()
            laser_env.step(action)
            assert len(laser_env.controls_buffer) == i + 2  # initial + i+1 steps
            
        # Buffer should respect maxlen
        for i in range(10):
            action = laser_env.action_space.sample()
            laser_env.step(action)
            assert len(laser_env.controls_buffer) <= 5  # maxlen=5
            
    def test_physics_properties_consistency(self, laser_env, fixed_seed):
        """Test that physics properties are consistent."""
        laser_env.reset(seed=fixed_seed)
        
        # Test that properties exist and are numeric
        assert hasattr(laser_env, 'pulse_FWHM')
        assert hasattr(laser_env, 'peak_intensity')
        assert hasattr(laser_env, 'frog')
        assert hasattr(laser_env, 'pulse')
        
        fwhm = laser_env.pulse_FWHM
        intensity = laser_env.peak_intensity
        frog = laser_env.frog
        pulse = laser_env.pulse
        
        # Check types and values
        assert isinstance(fwhm, (int, float, torch.Tensor))
        assert isinstance(intensity, (int, float, torch.Tensor))
        assert isinstance(frog, torch.Tensor)
        assert isinstance(pulse, tuple)
        assert len(pulse) == 2  # (time, intensity)
        
        # Check that values are reasonable
        assert float(fwhm) > 0
        assert float(intensity) >= 0
        assert frog.ndim == 2  # 2D FROG trace
        
    def test_transform_limited_regret(self, laser_env, fixed_seed):
        """Test transform limited regret calculation."""
        laser_env.reset(seed=fixed_seed)
        
        regret = laser_env.transform_limited_regret()
        assert isinstance(regret, (int, float))
        assert regret >= 0  # Regret should be non-negative
        
    def test_reset_consistency(self, laser_env, fixed_seed):
        """Test that reset brings environment to consistent state."""
        laser_env.reset(seed=fixed_seed)
        
        # Take some steps
        for _ in range(3):
            action = laser_env.action_space.sample()
            laser_env.step(action)
            
        # Reset and check state
        obs, info = laser_env.reset(seed=fixed_seed)
        
        assert laser_env.n_steps == 0
        assert len(laser_env.controls_buffer) == 1
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        
    def test_rendering_consistency(self, laser_env, fixed_seed):
        """Test that rendering is consistent."""
        laser_env.reset(seed=fixed_seed)
        
        # Test that rendering works at reset
        rgb1 = laser_env.render()
        assert isinstance(rgb1, np.ndarray)
        
        # Test that rendering works after steps
        action = laser_env.action_space.sample()
        laser_env.step(action)
        rgb2 = laser_env.render()
        assert isinstance(rgb2, np.ndarray)
        assert rgb2.shape == rgb1.shape


class TestRandomLaserEnvSemantics:
    """Test RandomLaserEnv semantics and behavior."""
    
    def test_domain_randomization_functionality(self, random_laser_env, fixed_seed):
        """Test domain randomization functionality."""
        # Test without DR
        random_laser_env.set_dr_training(False)
        task1 = random_laser_env.get_task()
        random_laser_env.reset(seed=fixed_seed)
        task2 = random_laser_env.get_task()
        assert np.array_equal(task1, task2)  # Should be same without DR
        
        # Test with DR
        random_laser_env.set_dr_training(True)
        tasks = []
        for _ in range(5):
            random_laser_env.reset(seed=None)  # Different seed each time
            task = random_laser_env.get_task()
            tasks.append(task)
            
        # With DR, tasks should vary (though this is probabilistic)
        tasks_array = np.array(tasks)
        assert not np.all(np.array_equal(tasks_array[0], tasks_array[i]) for i in range(1, len(tasks)))
        
    def test_task_setting_and_getting(self, random_laser_env):
        """Test task setting and getting functionality."""
        # Get default task
        default_task = random_laser_env.get_default_task()
        assert isinstance(default_task, np.ndarray)
        assert default_task.shape == (2,)
        
        # Set a custom task
        custom_task = np.array([2.5, 2.6])
        random_laser_env.set_task(custom_task[0], custom_task[1])
        
        # Get current task
        current_task = random_laser_env.get_task()
        assert np.allclose(current_task, custom_task)
        
        # Test that laser parameters actually changed
        assert np.isclose(random_laser_env.laser.B, custom_task[0])
        
    def test_task_bounds_compliance(self, random_laser_env):
        """Test that task bounds are respected."""
        # Test task bounds
        for i in range(2):
            lower = random_laser_env.get_task_lower_bound(i)
            upper = random_laser_env.get_task_upper_bound(i)
            assert lower < upper
            
        # Test search bounds
        bounds = random_laser_env.get_search_bounds_mean(0)
        assert len(bounds) == 2
        assert bounds[0] < bounds[1]
        
    def test_observation_includes_dynamics(self, random_laser_env, fixed_seed):
        """Test that observations include dynamics information."""
        obs, _ = random_laser_env.reset(seed=fixed_seed)
        
        # Should have dynamics info
        assert 'B_integral' in obs
        assert 'compressor_GDD' in obs
        
        # Values should be reasonable
        assert obs['B_integral'].shape == (1,)
        assert obs['compressor_GDD'].shape == (1,)
        assert 1.0 <= obs['B_integral'][0] <= 3.5
        assert 2.47 <= obs['compressor_GDD'][0] <= 2.87
        
    def test_psi_property_numpy_consistency(self, random_laser_env, fixed_seed):
        """Test that psi property maintains numpy type consistently."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Initial psi should be numpy
        assert isinstance(random_laser_env.psi, np.ndarray)
        
        # After steps, should still be numpy
        for _ in range(3):
            action = random_laser_env.action_space.sample()
            random_laser_env.step(action)
            assert isinstance(random_laser_env.psi, np.ndarray)
            
    def test_dispersion_coefficient_calculation(self, random_laser_env, fixed_seed):
        """Test dispersion coefficient calculation for dynamics."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Test dispersion coefficient calculation
        coeff = random_laser_env.dispersion_coefficient_for_dynamics(0)
        assert isinstance(coeff, (int, float, torch.Tensor))
        
        # Should be within reasonable bounds
        assert 2.47 <= float(coeff) <= 2.87
        
    def test_success_metric_tracking(self, random_laser_env, fixed_seed):
        """Test that success metric is tracked correctly."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Step and check success metric
        action = random_laser_env.action_space.sample()
        random_laser_env.step(action)
        
        # Should have success metric
        assert hasattr(random_laser_env, 'success_metric')
        assert random_laser_env.success_metric == 'peak_intensity_ratio'
        
        # Should have the metric value
        assert hasattr(random_laser_env, 'peak_intensity_ratio')
        metric_value = random_laser_env.peak_intensity_ratio
        assert isinstance(metric_value, (int, float, torch.Tensor))
        assert float(metric_value) >= 0
        
    def test_controls_buffer_numpy_compatibility(self, random_laser_env, fixed_seed):
        """Test that controls buffer works with numpy arrays."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Buffer should contain numpy arrays
        assert len(random_laser_env.controls_buffer) == 1
        assert isinstance(random_laser_env.controls_buffer[0], np.ndarray)
        
        # After steps, should still contain numpy arrays
        for _ in range(3):
            action = random_laser_env.action_space.sample()
            random_laser_env.step(action)
            for control in random_laser_env.controls_buffer:
                assert isinstance(control, np.ndarray)


class TestEnvironmentComparison:
    """Test that both environments behave consistently where they should."""
    
    def test_common_interface_consistency(self, laser_env, random_laser_env, fixed_seed):
        """Test that common interface elements are consistent."""
        # Both should have same action space
        assert laser_env.action_space.shape == random_laser_env.action_space.shape
        assert laser_env.action_space.dtype == random_laser_env.action_space.dtype
        
        # Both should have same basic observation structure
        laser_obs, _ = laser_env.reset(seed=fixed_seed)
        random_obs, _ = random_laser_env.reset(seed=fixed_seed)
        
        # Common keys should exist
        common_keys = {'frog_trace', 'psi', 'action'}
        assert common_keys.issubset(set(laser_obs.keys()))
        assert common_keys.issubset(set(random_obs.keys()))
        
        # Common keys should have same shapes
        for key in common_keys:
            assert laser_obs[key].shape == random_obs[key].shape
            assert laser_obs[key].dtype == random_obs[key].dtype
            
    def test_reward_calculation_similarity(self, laser_env, random_laser_env, fixed_seed):
        """Test that reward calculations are similar (should use same logic)."""
        # Set same initial conditions
        laser_env.reset(seed=fixed_seed)
        random_laser_env.reset(seed=fixed_seed)
        
        # Apply same action
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        _, laser_reward, _, _, laser_info = laser_env.step(action)
        _, random_reward, _, _, random_info = random_laser_env.step(action)
        
        # Rewards should be of same type and finite
        assert isinstance(laser_reward, (int, float, np.floating))
        assert isinstance(random_reward, (int, float, np.floating))
        assert np.isfinite(laser_reward)
        assert np.isfinite(random_reward)
        
        # Both should have reward components
        assert 'alive_component' in laser_info
        assert 'alive_component' in random_info
        
    def test_physics_properties_similarity(self, laser_env, random_laser_env, fixed_seed):
        """Test that physics properties are calculated similarly."""
        # Set same conditions
        laser_env.reset(seed=fixed_seed)
        random_laser_env.reset(seed=fixed_seed)
        
        # Both should have same physics properties
        laser_fwhm = laser_env.pulse_FWHM
        random_fwhm = random_laser_env.pulse_FWHM
        
        # Should be of same type
        assert type(laser_fwhm) == type(random_fwhm)
        assert float(laser_fwhm) > 0
        assert float(random_fwhm) > 0 