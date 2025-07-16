import matplotlib
matplotlib.use('Agg')

import gradio as gr
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# This is needed to register the custom environment
import gym_laser


def make_env_fn():
    """Helper function to create a single environment instance."""
    return gym.make("LaserEnv", render_mode="rgb_array")


def initialize_environment():
    """Initializes the environment on app load."""
    try:
        env = DummyVecEnv([make_env_fn])
        env = VecFrameStack(env, n_stack=5)
        obs = env.reset()
        state = { "env": env, "obs": obs, "model": None, "step_num": 0 }
        return state, "Environment ready. Running with random policy."
    except Exception as e:
        return None, f"Error: {e}"


def load_model(state, model_path):
    """Loads a model into the existing environment state."""
    if model_path is None or state is None:
        return state, "Upload failed or environment not ready."
    try:
        state["model"] = SAC.load(model_path.name)
        state["obs"] = state["env"].reset() # Reset for the new policy
        state["step_num"] = 0
        return state, "Model loaded. Running simulation."
    except Exception as e:
        return state, f"Error loading model: {e}"


def run_continuous_simulation(state, b_integral):
    """Runs the simulation continuously, yielding frames."""
    if not state or "env" not in state:
        yield state, None, "Environment not ready."
        return

    env = state["env"]
    obs = state["obs"]
    model = state.get("model")
    step_num = state.get("step_num", 0)
    
    # Run for a large number of steps to simulate "always-on"
    for i in range(100000):  # Large number for continuous simulation
        # Apply the current B-integral value
        env.envs[0].unwrapped.laser.B = float(b_integral)

        if model:
            action, _ = model.predict(obs, deterministic=True)
            status = f"Running model... Step {step_num} (B={b_integral:.1f})"
        else:
            action = env.action_space.sample().reshape(1, -1)
            status = f"Running random policy... Step {step_num} (B={b_integral:.1f})"
            
        obs, _, done, _ = env.step(action)
        frame = env.render()
        
        if done[0]:
            obs = env.reset()
            step_num = 0
        else:
            step_num += 1

        state["obs"] = obs
        state["step_num"] = step_num
        
        yield state, frame, status
        # time.sleep(0.05)  # Small delay for smooth animation


with gr.Blocks() as demo:
    gr.Markdown("# DRL for Laser Pulse Shaping")
    gr.Markdown(
        "A random policy simulation runs automatically with live updates. "
        "Upload your own SAC model to see it take over. Adjust B-integral to see live effects."
    )

    sim_state = gr.State()

    with gr.Row():
        with gr.Column():
            model_uploader = gr.UploadButton(
                "Upload Model (.zip)",
                file_types=['.zip'],
                elem_id="model-upload",
            )
        with gr.Column():
            b_slider = gr.Slider(
                minimum=0,
                maximum=10,
                step=0.5,
                value=2.0,
                label="B-integral",
                info="Adjust nonlinearity live during simulation.",
            )

    with gr.Row():
        with gr.Column():
            image_display = gr.Image(label="Environment Render", interactive=False, height=480)
            status_box = gr.Textbox(label="Status", interactive=False)

    # On page load, initialize and start the continuous simulation
    continuous_event = demo.load(
        fn=initialize_environment,
        inputs=None,
        outputs=[sim_state, status_box]
    ).then(
        fn=run_continuous_simulation,
        inputs=[sim_state, b_slider],
        outputs=[sim_state, image_display, status_box]
    )

    # When a model is uploaded, cancel the current simulation and start a new one
    model_upload_event = model_uploader.upload(
        fn=load_model,
        inputs=[sim_state, model_uploader],
        outputs=[sim_state, status_box],
        cancels=[continuous_event]
    ).then(
        fn=run_continuous_simulation,
        inputs=[sim_state, b_slider],
        outputs=[sim_state, image_display, status_box]
    )

    # When B-integral slider changes, restart the simulation with the new value
    b_slider.change(
        fn=run_continuous_simulation,
        inputs=[sim_state, b_slider],
        outputs=[sim_state, image_display, status_box],
        cancels=[continuous_event, model_upload_event]
    )

demo.launch()