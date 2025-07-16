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
        state = { 
            "env": env, 
            "obs": obs, 
            "model": None, 
            "step_num": 0,
            "current_b_integral": 2.0  # Store current B-integral in state
        }
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


def update_b_integral(state, b_integral):
    """Updates the B-integral value in the state without restarting simulation."""
    if state is not None:
        state["current_b_integral"] = b_integral
    return state


def run_continuous_simulation(state):
    """Runs the simulation continuously, using the current B-integral from state."""
    if not state or "env" not in state:
        yield state, None, "Environment not ready."
        return

    env = state["env"]
    obs = state["obs"]
    step_num = state.get("step_num", 0)
    
    # Run for a large number of steps to simulate "always-on"
    for i in range(100000):  # Large number for continuous simulation
        model = state.get("model")
        current_b = state.get("current_b_integral", 2.0)
        
        # Apply the current B-integral value from state
        env.envs[0].unwrapped.laser.B = float(current_b)

        if model:
            action, _ = model.predict(obs, deterministic=True)
            status = f"Running model / Step {step_num} (B={current_b:.1f})"
        else:
            action = env.action_space.sample().reshape(1, -1)
            status = f"Running random policy / Step {step_num} (B={current_b:.1f})"
            
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
    init_event = demo.load(
        fn=initialize_environment,
        inputs=None,
        outputs=[sim_state, status_box]
    )
    
    continuous_event = init_event.then(
        fn=run_continuous_simulation,
        inputs=[sim_state],
        outputs=[sim_state, image_display, status_box]
    )

    # When a model is uploaded, restart simulation (this needs to restart to reset for new policy)
    model_upload_event = model_uploader.upload(
        fn=load_model,
        inputs=[sim_state, model_uploader],
        outputs=[sim_state, status_box],
        cancels=[continuous_event]
    ).then(
        fn=run_continuous_simulation,
        inputs=[sim_state],
        outputs=[sim_state, image_display, status_box]
    )

    # When B-integral slider changes, just update the value in state (no restart needed)
    b_slider.change(
        fn=update_b_integral,
        inputs=[sim_state, b_slider],
        outputs=[sim_state]
    )

demo.launch()