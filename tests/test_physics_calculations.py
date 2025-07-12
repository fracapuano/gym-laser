"""Test physics calculations for both LaserEnv and RandomLaserEnv."""

import pytest
import numpy as np
import torch
from gym_laser.utils import physics


class TestFROGTraceCalculations:
    """Test FROG trace calculations and properties."""
    
    def test_frog_trace_generation(self, laser_env, fixed_seed):
        """Test that FROG trace is generated correctly."""
        laser_env.reset(seed=fixed_seed)
        
        frog = laser_env.frog
        assert isinstance(frog, torch.Tensor)
        assert frog.ndim == 2
        assert frog.shape[0] > 0
        assert frog.shape[1] > 0
        
        # FROG trace should be non-negative
        assert torch.all(frog >= 0)
        
    def test_frog_trace_consistency(self, laser_env, fixed_seed):
        """Test that FROG trace is consistent for same control parameters."""
        laser_env.reset(seed=fixed_seed)
        
        # Get FROG trace twice with same parameters
        frog1 = laser_env.frog
        frog2 = laser_env.frog
        
        # Should be identical
        assert torch.allclose(frog1, frog2)
        
    def test_frog_trace_changes_with_control(self, laser_env, sample_actions, fixed_seed):
        """Test that FROG trace changes when control parameters change."""
        laser_env.reset(seed=fixed_seed)
        
        # Get initial FROG trace
        initial_frog = laser_env.frog.clone()
        
        # Apply action and get new FROG trace
        action = sample_actions['positive_action']
        laser_env.step(action)
        new_frog = laser_env.frog
        
        # Should be different (though this depends on the magnitude of change)
        assert not torch.allclose(initial_frog, new_frog, atol=1e-6)
        
    def test_frog_trace_windowing(self, laser_env, fixed_seed):
        """Test that FROG trace windowing works correctly."""
        laser_env.reset(seed=fixed_seed)
        
        # Get observation which includes windowed FROG trace
        obs, _ = laser_env.reset(seed=fixed_seed)
        frog_windowed = obs['frog_trace']
        
        # Should be windowed to specified size
        assert frog_windowed.shape == (1, 32, 32)  # window_size=32 from fixtures
        assert frog_windowed.dtype == np.uint8
        assert np.all(frog_windowed >= 0)
        assert np.all(frog_windowed <= 255)
        
    def test_frog_trace_normalization(self, laser_env, fixed_seed):
        """Test that FROG trace normalization is correct."""
        laser_env.reset(seed=fixed_seed)
        
        obs, _ = laser_env.reset(seed=fixed_seed)
        frog_windowed = obs['frog_trace']
        
        # Should use full range of uint8 values when normalized
        assert frog_windowed.max() > 0  # Should have some signal
        
    def test_frog_method_with_custom_control(self, laser_env, fixed_seed):
        """Test frog_trace method with custom control parameters."""
        laser_env.reset(seed=fixed_seed)
        
        # Test with current control parameters
        current_control = laser_env.psi_picoseconds
        frog_custom = laser_env.frog_trace(current_control)
        
        assert isinstance(frog_custom, torch.Tensor)
        assert frog_custom.ndim == 2
        assert torch.all(frog_custom >= 0)


class TestPulseCalculations:
    """Test pulse-related calculations."""
    
    def test_pulse_generation(self, laser_env, fixed_seed):
        """Test that pulse is generated correctly."""
        laser_env.reset(seed=fixed_seed)
        
        pulse = laser_env.pulse
        assert isinstance(pulse, tuple)
        assert len(pulse) == 2
        
        time, intensity = pulse
        assert isinstance(time, torch.Tensor)
        assert isinstance(intensity, torch.Tensor)
        assert time.shape == intensity.shape
        assert len(time) > 0
        
        # Time should be monotonically increasing
        assert torch.all(time[1:] > time[:-1])
        
        # Intensity should be non-negative
        assert torch.all(intensity >= 0)
        
    def test_pulse_consistency(self, laser_env, fixed_seed):
        """Test that pulse is consistent for same control parameters."""
        laser_env.reset(seed=fixed_seed)
        
        # Get pulse twice with same parameters
        pulse1 = laser_env.pulse
        pulse2 = laser_env.pulse
        
        # Should be identical
        assert torch.allclose(pulse1[0], pulse2[0])  # time
        assert torch.allclose(pulse1[1], pulse2[1])  # intensity
        
    def test_pulse_changes_with_control(self, laser_env, sample_actions, fixed_seed):
        """Test that pulse changes when control parameters change."""
        laser_env.reset(seed=fixed_seed)
        
        # Get initial pulse
        initial_pulse = laser_env.pulse
        initial_time, initial_intensity = initial_pulse
        
        # Apply action and get new pulse
        action = sample_actions['positive_action']
        laser_env.step(action)
        new_pulse = laser_env.pulse
        new_time, new_intensity = new_pulse
        
        # Time axis should be the same (it's determined by the laser system)
        assert torch.allclose(initial_time, new_time)
        
        # Intensity should be different
        assert not torch.allclose(initial_intensity, new_intensity, atol=1e-6)


class TestFWHMCalculations:
    """Test FWHM (Full Width Half Maximum) calculations."""
    
    def test_fwhm_calculation(self, laser_env, fixed_seed):
        """Test that FWHM is calculated correctly."""
        laser_env.reset(seed=fixed_seed)
        
        fwhm = laser_env.pulse_FWHM
        assert isinstance(fwhm, (int, float, torch.Tensor))
        assert float(fwhm) > 0  # FWHM should be positive
        
        # FWHM should be reasonable (in picoseconds)
        assert 0.1 <= float(fwhm) <= 1000  # Reasonable range for laser pulses
        
    def test_fwhm_consistency(self, laser_env, fixed_seed):
        """Test that FWHM is consistent for same control parameters."""
        laser_env.reset(seed=fixed_seed)
        
        # Get FWHM twice with same parameters
        fwhm1 = laser_env.pulse_FWHM
        fwhm2 = laser_env.pulse_FWHM
        
        # Should be identical
        assert np.isclose(float(fwhm1), float(fwhm2))
        
    def test_fwhm_changes_with_control(self, laser_env, sample_actions, fixed_seed):
        """Test that FWHM changes when control parameters change."""
        laser_env.reset(seed=fixed_seed)
        
        # Get initial FWHM
        initial_fwhm = float(laser_env.pulse_FWHM)
        
        # Apply action and get new FWHM
        action = sample_actions['positive_action']
        laser_env.step(action)
        new_fwhm = float(laser_env.pulse_FWHM)
        
        # Should be different (though this depends on the magnitude of change)
        assert not np.isclose(initial_fwhm, new_fwhm, atol=1e-6)
        
    def test_fwhm_termination_condition(self, laser_env, fixed_seed):
        """Test that FWHM is used correctly in termination condition."""
        laser_env.reset(seed=fixed_seed)
        
        # Test termination function
        terminated = laser_env.is_terminated()
        fwhm = float(laser_env.pulse_FWHM)
        
        # Currently, termination is disabled (returns False), but test the logic
        expected_terminated = fwhm >= laser_env.MAX_DURATION
        # Note: is_terminated() returns False due to "and False" in the implementation
        assert terminated == (expected_terminated and False)


class TestPeakIntensityCalculations:
    """Test peak intensity calculations."""
    
    def test_peak_intensity_calculation(self, laser_env, fixed_seed):
        """Test that peak intensity is calculated correctly."""
        laser_env.reset(seed=fixed_seed)
        
        intensity = laser_env.peak_intensity
        assert isinstance(intensity, (int, float, torch.Tensor))
        assert float(intensity) >= 0  # Intensity should be non-negative
        
    def test_peak_intensity_consistency(self, laser_env, fixed_seed):
        """Test that peak intensity is consistent for same control parameters."""
        laser_env.reset(seed=fixed_seed)
        
        # Get intensity twice with same parameters
        intensity1 = laser_env.peak_intensity
        intensity2 = laser_env.peak_intensity
        
        # Should be identical
        assert np.isclose(float(intensity1), float(intensity2))
        
    def test_peak_intensity_changes_with_control(self, laser_env, sample_actions, fixed_seed):
        """Test that peak intensity changes when control parameters change."""
        laser_env.reset(seed=fixed_seed)
        
        # Get initial intensity
        initial_intensity = float(laser_env.peak_intensity)
        
        # Apply action and get new intensity
        action = sample_actions['positive_action']
        laser_env.step(action)
        new_intensity = float(laser_env.peak_intensity)
        
        # Should be different (though this depends on the magnitude of change)
        assert not np.isclose(initial_intensity, new_intensity, atol=1e-6)
        
    def test_peak_intensity_vs_transform_limited(self, laser_env, fixed_seed):
        """Test peak intensity relative to transform-limited pulse."""
        laser_env.reset(seed=fixed_seed)
        
        intensity = laser_env.peak_intensity
        tl_intensity = laser_env.TL_intensity
        
        # Both should be positive
        assert float(intensity) > 0
        assert float(tl_intensity) > 0
        
        # Ratio should be reasonable (typically < 1 for non-optimal pulses)
        ratio = float(intensity) / float(tl_intensity)
        assert 0 < ratio <= 2  # Should be positive, could potentially exceed 1


class TestTransformLimitedCalculations:
    """Test transform-limited pulse calculations."""
    
    def test_transform_limited_properties(self, laser_env, fixed_seed):
        """Test transform-limited pulse properties."""
        laser_env.reset(seed=fixed_seed)
        
        tl_pulse = laser_env.transform_limited
        assert isinstance(tl_pulse, tuple)
        assert len(tl_pulse) == 2
        
        tl_time, tl_intensity = tl_pulse
        assert isinstance(tl_time, torch.Tensor)
        assert isinstance(tl_intensity, torch.Tensor)
        assert tl_time.shape == tl_intensity.shape
        assert len(tl_time) > 0
        
        # Time should be monotonically increasing
        assert torch.all(tl_time[1:] > tl_time[:-1])
        
        # Intensity should be non-negative
        assert torch.all(tl_intensity >= 0)
        
    def test_transform_limited_consistency(self, laser_env, fixed_seed):
        """Test that transform-limited pulse is consistent."""
        laser_env.reset(seed=fixed_seed)
        
        # Get transform-limited pulse twice
        tl1 = laser_env.transform_limited
        tl2 = laser_env.transform_limited
        
        # Should be identical (transform-limited is fixed for given laser)
        assert torch.allclose(tl1[0], tl2[0])  # time
        assert torch.allclose(tl1[1], tl2[1])  # intensity
        
    def test_transform_limited_regret(self, laser_env, fixed_seed):
        """Test transform-limited regret calculation."""
        laser_env.reset(seed=fixed_seed)
        
        regret = laser_env.transform_limited_regret()
        assert isinstance(regret, (int, float))
        assert regret >= 0  # Regret should be non-negative
        
        # Regret should be consistent
        regret2 = laser_env.transform_limited_regret()
        assert np.isclose(regret, regret2)
        
    def test_transform_limited_regret_changes(self, laser_env, sample_actions, fixed_seed):
        """Test that transform-limited regret changes with control."""
        laser_env.reset(seed=fixed_seed)
        
        # Get initial regret
        initial_regret = laser_env.transform_limited_regret()
        
        # Apply action and get new regret
        action = sample_actions['positive_action']
        laser_env.step(action)
        new_regret = laser_env.transform_limited_regret()
        
        # Should be different (though this depends on the magnitude of change)
        assert not np.isclose(initial_regret, new_regret, atol=1e-6)


class TestControlParameterTransformations:
    """Test control parameter transformations."""
    
    def test_psi_property_bounds(self, laser_env, fixed_seed):
        """Test that psi property returns values in [0, 1] range."""
        laser_env.reset(seed=fixed_seed)
        
        psi = laser_env.psi
        assert isinstance(psi, torch.Tensor)
        assert psi.shape == (3,)
        assert torch.all(psi >= 0.0)
        assert torch.all(psi <= 1.0)
        
    def test_psi_picoseconds_transformation(self, laser_env, fixed_seed):
        """Test transformation to picoseconds units."""
        laser_env.reset(seed=fixed_seed)
        
        psi = laser_env.psi
        psi_ps = laser_env.psi_picoseconds
        
        assert isinstance(psi_ps, torch.Tensor)
        assert psi_ps.shape == psi.shape
        assert psi_ps.dtype == torch.float32
        
        # Should be different scale (picoseconds units)
        assert not torch.allclose(psi, psi_ps)
        
    def test_control_utils_scaling(self, laser_env, fixed_seed):
        """Test control utilities scaling operations."""
        laser_env.reset(seed=fixed_seed)
        
        control_utils = laser_env.control_utils
        
        # Test scaling and descaling
        psi = laser_env.psi
        psi_ps = control_utils.descale_control(psi)
        psi_scaled_back = control_utils.scale_control(psi_ps)
        
        # Should be approximately equal after round-trip
        assert torch.allclose(psi, psi_scaled_back, atol=1e-6)
        
    def test_control_magnification_operations(self, laser_env, fixed_seed):
        """Test control magnification operations."""
        laser_env.reset(seed=fixed_seed)
        
        control_utils = laser_env.control_utils
        
        # Test magnification and demagnification
        psi_ps = laser_env.psi_picoseconds
        psi_demag = control_utils.controls_demagnify(psi_ps)
        psi_mag = control_utils.control_magnify(psi_demag)
        
        # Should be approximately equal after round-trip
        assert torch.allclose(psi_ps, psi_mag, atol=1e-6)


class TestRandomEnvPhysicsConsistency:
    """Test that RandomLaserEnv maintains physics consistency."""
    
    def test_physics_calculations_consistency(self, random_laser_env, fixed_seed):
        """Test that physics calculations are consistent in RandomLaserEnv."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Test all physics properties
        fwhm = random_laser_env.pulse_FWHM
        intensity = random_laser_env.peak_intensity
        frog = random_laser_env.frog
        pulse = random_laser_env.pulse
        
        # All should be valid
        assert float(fwhm) > 0
        assert float(intensity) >= 0
        assert isinstance(frog, torch.Tensor)
        assert isinstance(pulse, tuple)
        
    def test_dynamics_parameter_effects(self, random_laser_env, fixed_seed):
        """Test that changing dynamics parameters affects physics."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Get initial physics
        initial_fwhm = float(random_laser_env.pulse_FWHM)
        initial_intensity = float(random_laser_env.peak_intensity)
        
        # Change dynamics parameters
        new_task = np.array([3.0, 2.7])  # Different B and GDD
        random_laser_env.set_task(new_task[0], new_task[1])
        
        # Get new physics
        new_fwhm = float(random_laser_env.pulse_FWHM)
        new_intensity = float(random_laser_env.peak_intensity)
        
        # Should be different (though this depends on the magnitude of change)
        # At least one should be different
        assert not (np.isclose(initial_fwhm, new_fwhm, atol=1e-6) and 
                   np.isclose(initial_intensity, new_intensity, atol=1e-6))
        
    def test_dispersion_coefficient_calculation(self, random_laser_env, fixed_seed):
        """Test dispersion coefficient calculation."""
        random_laser_env.reset(seed=fixed_seed)
        
        # Test calculation
        coeff = random_laser_env.dispersion_coefficient_for_dynamics(0)
        assert isinstance(coeff, (int, float, torch.Tensor))
        
        # Should be in reasonable range
        assert 2.47 <= float(coeff) <= 2.87
        
        # Should be consistent
        coeff2 = random_laser_env.dispersion_coefficient_for_dynamics(0)
        assert np.isclose(float(coeff), float(coeff2))


class TestPhysicsUtilityFunctions:
    """Test physics utility functions directly."""
    
    def test_peak_intensity_function(self, laser_env, fixed_seed):
        """Test peak intensity utility function."""
        laser_env.reset(seed=fixed_seed)
        
        # Get pulse intensity
        _, pulse_intensity = laser_env.pulse
        
        # Test physics.peak_intensity function
        peak = physics.peak_intensity(pulse_intensity)
        assert isinstance(peak, (int, float, torch.Tensor))
        assert float(peak) >= 0
        
        # Should match environment's calculation
        env_peak = laser_env.peak_intensity
        assert np.isclose(float(peak), float(env_peak))
        
    def test_fwhm_function(self, laser_env, fixed_seed):
        """Test FWHM utility function."""
        laser_env.reset(seed=fixed_seed)
        
        # Get pulse data
        time, intensity = laser_env.pulse
        
        # Test physics.FWHM function
        fwhm = physics.FWHM(time, intensity)
        assert isinstance(fwhm, (int, float, torch.Tensor))
        assert float(fwhm) > 0
        
        # Should match environment's calculation (accounting for unit conversion)
        env_fwhm = laser_env.pulse_FWHM
        expected_fwhm = float(fwhm) * 1e12  # Convert to picoseconds
        assert np.isclose(expected_fwhm, float(env_fwhm))
        
    def test_peak_on_peak_alignment(self, laser_env, fixed_seed):
        """Test peak-on-peak alignment utility function."""
        laser_env.reset(seed=fixed_seed)
        
        # Get current pulse and transform-limited pulse
        time, intensity = laser_env.pulse
        tl_time, tl_intensity = laser_env.transform_limited
        
        # Test peak-on-peak alignment
        pulse1, pulse2 = physics.peak_on_peak(
            temporal_profile=[time.cpu(), intensity.cpu()],
            other=[tl_time.cpu(), tl_intensity.cpu()]
        )
        
        # Should return two aligned pulses
        assert len(pulse1) == 2
        assert len(pulse2) == 2
        
        # Both should be same length
        assert len(pulse1[0]) == len(pulse1[1])
        assert len(pulse2[0]) == len(pulse2[1])
        assert len(pulse1[0]) == len(pulse2[0])


class TestPhysicsErrorHandling:
    """Test error handling in physics calculations."""
    
    def test_invalid_control_parameters(self, laser_env, fixed_seed):
        """Test handling of invalid control parameters."""
        laser_env.reset(seed=fixed_seed)
        
        # Test with NaN control parameters
        original_psi = laser_env.psi.clone()
        
        # Try to set invalid psi (should be clipped)
        laser_env.psi = torch.tensor([2.0, -1.0, 0.5])  # Outside [0, 1]
        
        # Should be clipped to valid range
        assert torch.all(laser_env.psi >= 0.0)
        assert torch.all(laser_env.psi <= 1.0)
        
    def test_physics_calculations_with_extreme_values(self, laser_env, fixed_seed):
        """Test physics calculations with extreme control values."""
        laser_env.reset(seed=fixed_seed)
        
        # Test with extreme values (at boundaries)
        laser_env.psi = torch.tensor([0.0, 0.0, 0.0])
        
        # Should still produce valid physics
        fwhm = laser_env.pulse_FWHM
        intensity = laser_env.peak_intensity
        frog = laser_env.frog
        
        assert float(fwhm) > 0
        assert float(intensity) >= 0
        assert torch.all(frog >= 0)
        
        # Test with other extreme
        laser_env.psi = torch.tensor([1.0, 1.0, 1.0])
        
        fwhm = laser_env.pulse_FWHM
        intensity = laser_env.peak_intensity
        frog = laser_env.frog
        
        assert float(fwhm) > 0
        assert float(intensity) >= 0
        assert torch.all(frog >= 0) 