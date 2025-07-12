# gym-laser

[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`gym-laser` is a physics-informed simulated environment for laser pulse optimization using []`gymnasium`](LINK_TO_GYMNASIUM).

## Features

- **Physics-informed environments**: Based on real laser physics simulations <!-- Add link to ELIopt notebook describing the simulation process -->
- **Two environment variants**:
  - `FROGLaserEnv`: Environment for standard RL training
  - `RandomFROGLaserEnv`: Environment enabling domain randomization environment for robust training

## Installation

We recommend installing this environment within a [Conda environment](LINK_TO_DOWNLOAD_MINICONDA). Then, simply run:
```bash
conda create -n rlaser python=3.11 -y
conda activate rlaser

pip install gym-laser
```

## Quick Start

### Basic Usage

```python
import numpy as np
from gym_laser.env_utils import EnvParametrization
from gym_laser.LaserEnv import FROGLaserEnv

# Create environment with default parameters
params = EnvParametrization().get_parametrization_dict()
env = FROGLaserEnv(**params)

# Reset environment
obs, info = env.reset()

# Take a random action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Access physics properties
print(f"Pulse FWHM: {env.pulse_FWHM:.2f} ps")
print(f"Peak intensity: {env.peak_intensity:.2e}")

env.close()
```

### Domain Randomization

```python
from gym_laser.RandomLaserEnv import RandomFROGLaserEnv

# Create environment with domain randomization
env = RandomFROGLaserEnv(**params)

# Enable domain randomization
env.set_dr_training(True)

# Environment will sample different dynamics at each reset
for episode in range(5):
    obs, info = env.reset()
    print(f"Episode {episode} task: {env.get_task()}")
    
env.close()
```

### Training with Stable Baselines3

```python
from stable_baselines3 import PPO
from gym_laser.LaserEnv import FROGLaserEnv
from gym_laser.env_utils import EnvParametrization

# Create environment
params = EnvParametrization().get_parametrization_dict()
env = FROGLaserEnv(**params)

# Train agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test trained agent
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Environment Details

### Observation Space

The environments provide dict observations containing:

- **`frog_trace`**: FROG trace as uint8 image (shape: `(1, window_size, window_size)`)
- **`psi`**: Current control parameters in [0, 1] range (shape: `(3,)`)
- **`action`**: Last applied action (shape: `(3,)`)

Additional for `RandomFROGLaserEnv`:
- **`B_integral`**: Current B-integral value (shape: `(1,)`)
- **`compressor_GDD`**: Current compressor GDD parameter (shape: `(1,)`)

### Action Space

3-dimensional continuous action space representing changes to:
- GDD (Group Delay Dispersion)
- TOD (Third Order Dispersion)
- FOD (Fourth Order Dispersion)

Actions are in the range `[-1, 1]` and are applied as deltas to current control parameters.

### Reward Function

The reward function combines multiple components:
- **Intensity component**: Encourages higher pulse intensity
- **Duration component**: Penalizes longer pulse durations
- **Alive bonus**: Reward for not terminating episode

## Testing

This project includes a comprehensive test suite covering:

- Environment interface compliance with Gymnasium
- Physics calculations accuracy
- Domain randomization functionality
- Environment semantics and behavior

### Running Tests

```bash
# Run all tests
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --integration   # Integration tests only
python run_tests.py --physics       # Physics tests only
python run_tests.py --fast          # Fast tests only

# Run with coverage
python run_tests.py --coverage

# Run linting and formatting checks
python run_tests.py --lint --format
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Test fixtures
├── test_env_interface.py         # Gymnasium interface tests
├── test_env_semantics.py         # Environment behavior tests
├── test_physics_calculations.py  # Physics accuracy tests
└── test_domain_randomization.py  # Domain randomization tests
```

## Development

### Setting up development environment

```bash
git clone https://github.com/fracapuano/gym-laser.git
cd gym-laser
pip install -e ".[dev]"
```

### Code Quality

The project uses several tools to maintain code quality:

- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security checking

### CI/CD Pipeline

The project includes a comprehensive GitHub Actions CI/CD pipeline that:

- Tests on Python 3.8, 3.9, 3.10, and 3.11
- Runs all test categories
- Checks code quality and security
- Generates coverage reports
- Tests installation and basic functionality
- Includes performance benchmarks

## Physics Background

The environments simulate ultrashort laser pulse propagation and control using:

- **FROG (Frequency-Resolved Optical Gating)**: For pulse characterization
- **Dispersion control**: Via GDD, TOD, and FOD parameters
- **Non-linear effects**: Modeled through B-integral
- **Transform-limited pulses**: As optimization targets

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{gym_laser,
  title={Gym-Laser: A Reinforcement Learning Environment for Laser Pulse Optimization},
  author={Francesco Capuano},
  year={2024},
  url={https://github.com/fracapuano/gym-laser}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Make sure to:

1. Add tests for new functionality
2. Update documentation as needed
3. Follow the existing code style
4. Ensure all CI checks pass

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/fracapuano/gym-laser/issues) on GitHub.
