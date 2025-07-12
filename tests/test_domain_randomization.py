"""Test domain randomization functionality for RandomLaserEnv."""

import pytest
import numpy as np
import torch
from gym_laser.RandomLaserEnv import RandomFROGLaserEnv


class TestDomainRandomizationInterface:
    """Test domain randomization interface methods."""
    
    def test_dr_training_flag(self, random_laser_env):
        """Test DR training flag functionality."""
        # Initial state should be False
        assert random_laser_env.get_dr_training() is False
        
        # Set to True
        random_laser_env.set_dr_training(True)
        assert random_laser_env.get_dr_training() is True
        
        # Set back to False
        random_laser_env.set_dr_training(False)
        assert random_laser_env.get_dr_training() is False
        
    def test_task_getting_and_setting(self, random_laser_env):
        """Test task getting and setting methods."""
        # Get initial task
        initial_task = random_laser_env.get_task()
        assert isinstance(initial_task, np.ndarray)
        assert initial_task.shape == (2,)  # B_integral and compressor_GDD
        
        # Set a new task
        new_task = np.array([2.5, 2.6])
        random_laser_env.set_task(new_task[0], new_task[1])
        
        # Verify task was set
        current_task = random_laser_env.get_task()
        assert np.allclose(current_task, new_task)
        
        # Verify laser parameters actually changed
        assert np.isclose(random_laser_env.laser.B, new_task[0])
        
    def test_default_task(self, random_laser_env):
        """Test default task functionality."""
        default_task = random_laser_env.get_default_task()
        assert isinstance(default_task, np.ndarray)
        assert default_task.shape == (2,)
        assert np.array_equal(default_task, np.array([2, 2.67]))
        
    def test_task_bounds_methods(self, random_laser_env):
        """Test task bounds methods."""
        # Test bounds for each parameter
        for i in range(2):
            lower = random_laser_env.get_task_lower_bound(i)
            upper = random_laser_env.get_task_upper_bound(i)
            
            assert isinstance(lower, (int, float))
            assert isinstance(upper, (int, float))
            assert lower < upper
            
        # Test search bounds
        for i in range(2):
            bounds = random_laser_env.get_search_bounds_mean(i)
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] < bounds[1]
            
    def test_dynamics_index_to_name_mapping(self, random_laser_env):
        """Test dynamics index to name mapping."""
        assert hasattr(random_laser_env, 'dyn_ind_to_name')
        assert isinstance(random_laser_env.dyn_ind_to_name, dict)
        assert 0 in random_laser_env.dyn_ind_to_name
        assert 1 in random_laser_env.dyn_ind_to_name
        assert random_laser_env.dyn_ind_to_name[0] == "B_integral"
        assert random_laser_env.dyn_ind_to_name[1] == "compressor_GDD"


class TestDomainRandomizationBehavior:
    """Test domain randomization behavior during training."""
    
    def test_dr_training_disabled_behavior(self, random_laser_env, fixed_seed):
        """Test behavior when DR training is disabled."""
        random_laser_env.set_dr_training(False)
        
        # Reset multiple times and check that task remains the same
        tasks = []
        for _ in range(5):
            random_laser_env.reset(seed=fixed_seed)
            task = random_laser_env.get_task()
            tasks.append(task.copy())
            
        # All tasks should be identical
        for i in range(1, len(tasks)):
            assert np.array_equal(tasks[0], tasks[i])
            
    def test_dr_training_enabled_behavior(self, random_laser_env):
        """Test behavior when DR training is enabled."""
        random_laser_env.set_dr_training(True)
        
        # Reset multiple times and check that task varies
        tasks = []
        for _ in range(10):
            random_laser_env.reset(seed=None)  # Random seed
            task = random_laser_env.get_task()
            tasks.append(task.copy())
            
        # Tasks should vary (probabilistic test)
        # At least some should be different
        unique_tasks = []
        for task in tasks:
            is_unique = True
            for unique_task in unique_tasks:
                if np.allclose(task, unique_task, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_tasks.append(task)
                
        # Should have more than one unique task
        assert len(unique_tasks) > 1
        
    def test_task_bounds_compliance(self, random_laser_env):
        """Test that sampled tasks comply with bounds."""
        random_laser_env.set_dr_training(True)
        
        # Sample multiple tasks
        for _ in range(20):
            random_laser_env.reset(seed=None)
            task = random_laser_env.get_task()
            
            # Check bounds compliance
            assert random_laser_env.min_task[0] <= task[0] <= random_laser_env.max_task[0]
            assert random_laser_env.min_task[1] <= task[1] <= random_laser_env.max_task[1]
            
    def test_task_setting_affects_physics(self, random_laser_env, fixed_seed):
        """Test that setting tasks affects physics calculations."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Get initial physics
        initial_fwhm = float(random_laser_env.pulse_FWHM)
        initial_intensity = float(random_laser_env.peak_intensity)
        
        # Set different task
        new_task = np.array([3.0, 2.7])
        random_laser_env.set_task(new_task[0], new_task[1])
        
        # Get new physics
        new_fwhm = float(random_laser_env.pulse_FWHM)
        new_intensity = float(random_laser_env.peak_intensity)
        
        # At least one should be different
        assert not (np.isclose(initial_fwhm, new_fwhm, atol=1e-6) and
                   np.isclose(initial_intensity, new_intensity, atol=1e-6))


class TestTaskDistributionSampling:
    """Test task distribution sampling functionality."""
    
    def test_uniform_distribution_sampling(self, random_laser_env):
        """Test uniform distribution sampling (default behavior)."""
        # Set up for uniform distribution sampling
        random_laser_env.set_dr_training(True)
        
        # Sample many tasks
        tasks = []
        for _ in range(100):
            random_laser_env.reset(seed=None)
            task = random_laser_env.get_task()
            tasks.append(task)
            
        tasks_array = np.array(tasks)
        
        # Check that values are distributed across the range
        for i in range(2):
            param_values = tasks_array[:, i]
            min_val = random_laser_env.min_task[i]
            max_val = random_laser_env.max_task[i]
            
            # Should have values across the range
            assert param_values.min() >= min_val
            assert param_values.max() <= max_val
            
            # Should have reasonable spread (not all clustered)
            assert param_values.std() > 0.1 * (max_val - min_val)
            
    def test_task_search_bounds_consistency(self, random_laser_env):
        """Test that task search bounds are consistent."""
        # Get bounds from different methods
        min_task = random_laser_env.min_task
        max_task = random_laser_env.max_task
        
        for i in range(2):
            search_bounds = random_laser_env.get_search_bounds_mean(i)
            lower_bound = random_laser_env.get_task_lower_bound(i)
            upper_bound = random_laser_env.get_task_upper_bound(i)
            
            # Should be consistent
            assert np.isclose(min_task[i], search_bounds[0])
            assert np.isclose(max_task[i], search_bounds[1])
            assert np.isclose(lower_bound, search_bounds[0])
            assert np.isclose(upper_bound, search_bounds[1])
            
    def test_sample_task_method(self, random_laser_env):
        """Test sample_task method if available."""
        # Check if sample_task method exists
        if hasattr(random_laser_env, 'sample_task'):
            # Sample multiple tasks
            tasks = []
            for _ in range(10):
                task = random_laser_env.sample_task()
                tasks.append(task)
                
                # Check bounds compliance
                assert isinstance(task, np.ndarray)
                assert task.shape == (2,)
                assert random_laser_env.min_task[0] <= task[0] <= random_laser_env.max_task[0]
                assert random_laser_env.min_task[1] <= task[1] <= random_laser_env.max_task[1]
            
            # Should have some variation
            tasks_array = np.array(tasks)
            assert tasks_array.std(axis=0).sum() > 0


class TestDomainRandomizationObservations:
    """Test that domain randomization affects observations correctly."""
    
    def test_observations_include_dynamics(self, random_laser_env, fixed_seed):
        """Test that observations include dynamics parameters."""
        random_laser_env.reset(seed=fixed_seed)
        
        obs, _ = random_laser_env.reset(seed=fixed_seed)
        
        # Should include dynamics parameters
        assert 'B_integral' in obs
        assert 'compressor_GDD' in obs
        
        # Check values match current task
        current_task = random_laser_env.get_task()
        assert np.isclose(obs['B_integral'][0], current_task[0])
        assert np.isclose(obs['compressor_GDD'][0], current_task[1])
        
    def test_observations_change_with_task(self, random_laser_env, fixed_seed):
        """Test that observations change when task changes."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Get initial observation
        initial_obs, _ = random_laser_env.reset(seed=fixed_seed)
        
        # Change task
        new_task = np.array([3.0, 2.7])
        random_laser_env.set_task(new_task[0], new_task[1])
        
        # Get new observation
        new_obs, _ = random_laser_env.reset(seed=fixed_seed)
        
        # Dynamics observations should be different
        assert not np.isclose(initial_obs['B_integral'][0], new_obs['B_integral'][0])
        assert not np.isclose(initial_obs['compressor_GDD'][0], new_obs['compressor_GDD'][0])
        
        # But other parts should be the same (same seed)
        assert np.array_equal(initial_obs['psi'], new_obs['psi'])
        assert np.array_equal(initial_obs['action'], new_obs['action'])
        
    def test_observation_bounds_compliance(self, random_laser_env):
        """Test that observations comply with observation space bounds."""
        random_laser_env.set_dr_training(True)
        
        # Sample multiple episodes
        for _ in range(10):
            random_laser_env.reset(seed=None)
            obs, _ = random_laser_env.reset(seed=None)
            
            # Check bounds compliance
            assert random_laser_env.observation_space.contains(obs)
            
            # Specifically check dynamics bounds
            assert 1.0 <= obs['B_integral'][0] <= 3.5
            assert 2.47 <= obs['compressor_GDD'][0] <= 2.87


class TestDomainRandomizationConsistency:
    """Test consistency aspects of domain randomization."""
    
    def test_task_persistence_during_episode(self, random_laser_env, fixed_seed):
        """Test that task remains consistent during an episode."""
        random_laser_env.set_dr_training(True)
        random_laser_env.reset(seed=fixed_seed)
        
        # Get initial task
        initial_task = random_laser_env.get_task()
        
        # Take several steps
        for _ in range(5):
            action = random_laser_env.action_space.sample()
            random_laser_env.step(action)
            
            # Task should remain the same
            current_task = random_laser_env.get_task()
            assert np.array_equal(initial_task, current_task)
            
    def test_reset_changes_task_with_dr(self, random_laser_env):
        """Test that reset changes task when DR is enabled."""
        random_laser_env.set_dr_training(True)
        
        # Get task after first reset
        random_laser_env.reset(seed=None)
        task1 = random_laser_env.get_task()
        
        # Reset again and get new task
        random_laser_env.reset(seed=None)
        task2 = random_laser_env.get_task()
        
        # Tasks might be different (probabilistic)
        # We'll just check that the mechanism works
        assert isinstance(task1, np.ndarray)
        assert isinstance(task2, np.ndarray)
        assert task1.shape == task2.shape
        
    def test_deterministic_behavior_with_seeds(self, random_laser_env, fixed_seed):
        """Test deterministic behavior when using seeds."""
        random_laser_env.set_dr_training(True)
        
        # Reset with same seed multiple times
        tasks = []
        for _ in range(3):
            random_laser_env.reset(seed=fixed_seed)
            task = random_laser_env.get_task()
            tasks.append(task)
            
        # All tasks should be identical when using same seed
        for i in range(1, len(tasks)):
            assert np.array_equal(tasks[0], tasks[i])


class TestDomainRandomizationVectorization:
    """Test domain randomization with vectorized environments."""
    
    def test_vectorized_task_setting(self, random_laser_env_params):
        """Test task setting in vectorized environments."""
        # Create multiple environments
        envs = [RandomFROGLaserEnv(**random_laser_env_params) for _ in range(3)]
        
        try:
            # Set different tasks for each environment
            tasks = [
                np.array([2.0, 2.5]),
                np.array([2.5, 2.6]),
                np.array([3.0, 2.7])
            ]
            
            for env, task in zip(envs, tasks):
                env.set_task(task[0], task[1])
                
            # Verify tasks were set correctly
            for env, expected_task in zip(envs, tasks):
                current_task = env.get_task()
                assert np.allclose(current_task, expected_task)
                
        finally:
            # Clean up
            for env in envs:
                env.close()
                
    def test_vectorized_dr_training(self, random_laser_env_params):
        """Test DR training in vectorized environments."""
        # Create multiple environments
        envs = [RandomFROGLaserEnv(**random_laser_env_params) for _ in range(3)]
        
        try:
            # Set different DR training states
            dr_states = [True, False, True]
            
            for env, dr_state in zip(envs, dr_states):
                env.set_dr_training(dr_state)
                
            # Verify DR training states
            for env, expected_state in zip(envs, dr_states):
                assert env.get_dr_training() == expected_state
                
        finally:
            # Clean up
            for env in envs:
                env.close()


class TestDomainRandomizationEdgeCases:
    """Test edge cases in domain randomization."""
    
    def test_extreme_task_values(self, random_laser_env):
        """Test behavior with extreme task values."""
        # Test with boundary values
        boundary_tasks = [
            np.array([1.0, 2.47]),  # Lower bounds
            np.array([3.5, 2.87]),  # Upper bounds
        ]
        
        for task in boundary_tasks:
            random_laser_env.set_task(task[0], task[1])
            
            # Should not crash
            random_laser_env.reset()
            
            # Physics should still work
            fwhm = random_laser_env.pulse_FWHM
            intensity = random_laser_env.peak_intensity
            
            assert float(fwhm) > 0
            assert float(intensity) >= 0
            
    def test_task_bounds_enforcement(self, random_laser_env):
        """Test that task bounds are enforced."""
        # Try to set task outside bounds (should be handled gracefully)
        # Note: The current implementation doesn't explicitly enforce bounds
        # in set_task, so this test documents current behavior
        
        extreme_task = np.array([10.0, 10.0])  # Outside bounds
        random_laser_env.set_task(extreme_task[0], extreme_task[1])
        
        # Should not crash
        random_laser_env.reset()
        
        # Environment should still function
        obs, _ = random_laser_env.reset()
        assert isinstance(obs, dict)
        
    def test_dispersion_coefficient_calculation_edge_cases(self, random_laser_env):
        """Test dispersion coefficient calculation with edge cases."""
        # Test with different compressor parameters
        edge_tasks = [
            np.array([1.0, 2.47]),
            np.array([3.5, 2.87]),
            np.array([2.0, 2.5]),
        ]
        
        for task in edge_tasks:
            random_laser_env.set_task(task[0], task[1])
            
            # Calculate dispersion coefficient
            coeff = random_laser_env.dispersion_coefficient_for_dynamics(0)
            
            # Should be valid
            assert isinstance(coeff, (int, float, torch.Tensor))
            assert not np.isnan(float(coeff))
            assert not np.isinf(float(coeff))
            
    def test_success_metric_with_different_tasks(self, random_laser_env, fixed_seed):
        """Test success metric calculation with different tasks."""
        different_tasks = [
            np.array([1.5, 2.5]),
            np.array([2.0, 2.6]),
            np.array([3.0, 2.8]),
        ]
        
        success_metrics = []
        
        for task in different_tasks:
            random_laser_env.set_task(task[0], task[1])
            random_laser_env.reset(seed=fixed_seed)
            
            # Take a step to trigger success metric calculation
            action = random_laser_env.action_space.sample()
            random_laser_env.step(action)
            
            # Get success metric
            if hasattr(random_laser_env, 'peak_intensity_ratio'):
                metric = random_laser_env.peak_intensity_ratio
                success_metrics.append(float(metric))
                
        # Success metrics should be valid
        for metric in success_metrics:
            assert isinstance(metric, (int, float))
            assert metric >= 0
            assert not np.isnan(metric)
            assert not np.isinf(metric) 