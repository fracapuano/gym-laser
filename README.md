# gym-laser

[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`gym-laser` is a physics-informed simulated environment for laser pulse optimization using [`gymnasium`](https://gymnasium.farama.org/). Check out [our demo](https://huggingface.co/spaces/fracapuano/RLaser) to train and upload your own policy for pulse shaping.

![](https://huggingface.co/datasets/fracapuano/rlaser-assets/resolve/main/assets/gym-laser-render.gif)

## Features

- **Physics-informed environments**: Based on real-world laser physics simulations, as described in [this guide](https://github.com/fracapuano/ELIopt/blob/main/notebooks/SemiPhysicalModel/SemiPhysicalModel_v2.ipynb).
- **Two environment variants**:
  - `FROGLaserEnv`: Environment for standard RL training
  - `RandomFROGLaserEnv`: Environment enabling domain randomization environment for robust training

## Installation

We recommend installing this environment within a [Conda environment](https://repo.anaconda.com/miniconda/). Then, simply run:
```bash
conda create -n gymlaser python=3.11 -y
conda activate gymlaser

pip install gym-laser
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
import gym_laser  # triggers environment registration

render = True
env = gym.make("LaserEnv", render_mode="human" if render else "rgb_array")

# Test trained agent
obs, info = env.reset()
for _ in range(20):  # max timesteps in one episode
    action, _states = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if render:
        env.render()
    
    if terminated or truncated:
        obs, info = env.reset()
```

### Training with Stable Baselines3

Install `stable-baselines3` via `pip` to train a RL policy directly on this environment.

```python
from stable_baselines3 import PPO
import gymnasium as gym
import gym_laser  # triggers environment registration

env = gym.make("LaserEnv")

# Train agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Environment Details

### Observation Space

The environments provide dict observations containing:

- **`frog_trace`**: FROG trace as uint8 image (shape: `(1, window_size, window_size)`)
- **`psi`**: Current control parameters in [0, 1] range (shape: `(3,)`)
- **`action`**: Last applied action (shape: `(3,)`)

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

## Citation

If you use this environment in your research, please cite:

```bibtex
@article{capuano2025shaping,
  title={Shaping Laser Pulses with Reinforcement Learning},
  author={Capuano, Francesco and Peceli, Davorin and Tiboni, Gabriele},
  journal={arXiv preprint arXiv:2503.00499},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Check out the [TODO.md](TODO.md) file!

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/fracapuano/gym-laser/issues) on GitHub.

