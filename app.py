import gradio as gr
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import time

# This is needed to register the custom environment
import gym_laser

MAX_STEPS = 200

def make_env_fn():
    """Helper function to create a single environment instance."""
    return gym.make("LaserEnv", render_mode="rgb_array")


def load_model_and_init(model_path, b_integral):
    """
    Loads the uploaded model, creates and resets the environment.
    """
    if model_path is None:
        return None, None, "Upload a model to begin."

    try:
        env = DummyVecEnv([make_env_fn])
        env = VecFrameStack(env, n_stack=5)
        model = SAC.load(model_path.name)
        
        env.envs[0].unwrapped.laser.B = float(b_integral)
        obs = env.reset()
        
        initial_frame = env.render()
        
        state = { "env": env, "model": model, "obs": obs }
        
        return state, initial_frame, "Model loaded. Adjust B-integral or run simulation."

    except Exception as e:
        return None, None, f"Error: {e}"

def update_b_and_render(state, b_integral):
    """
    Updates the B-integral from the slider and renders the current state.
    """
    if not state or not state.get("env"):
        return state, None, "Please upload a model first."
        
    env = state["env"]
    env.envs[0].unwrapped.laser.B = float(b_integral)
    
    frame = env.render()
    
    return state, frame, f"B-integral set to {b_integral:.2f}"

def run_random_policy(state, b_integral):
    env = state["env"]
    model = state["model"]
    obs = state["obs"]

    # Set B-integral at the start of the simulation
    env.envs[0].unwrapped.laser.B = float(b_integral)

    for i in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        frame = env.render()
        
        yield frame, f"Running... Step {i+1}/{MAX_STEPS}"
        time.sleep(0.05)
        
        if done[0]:  # Episode finished (e.g., pulse duration limit)
            break
            
    state["obs"] = env.reset() if done[0] else obs
    yield env.render(), f"B-integral set to {b_integral:.2f}"

def run_simulation_loop(state, b_integral):
    """
    Runs a simulation loop for MAX_STEPS, yielding intermediate frames.
    """
    if not state or not state.get("env"):
        yield None, "Please upload a model first."
        return

    env = state["env"]
    model = state["model"]
    obs = state["obs"]

    # Set B-integral at the start of the simulation
    env.envs[0].unwrapped.laser.B = float(b_integral)

    for i in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frame = env.render()
        
        yield frame, f"Running... Step {i+1}/{MAX_STEPS}"
        time.sleep(0.05)
        
        if done[0]:  # Episode finished (e.g., pulse duration limit)
            break
            
    state["obs"] = env.reset() if done[0] else obs
    yield env.render(), "Simulation finished."

with gr.Blocks() as demo:
    title = ""
    copy = """
    """
    gr.Markdown(title)
    gr.Markdown(copy)

    sim_state = gr.State(None)


    model_uploader = gr.UploadButton(
        "Upload Model (.zip)",
        file_types=['.zip'],
        elem_id="model-upload",
    )

    b_slider = gr.Slider(
        minimum=0,
        maximum=10,
        step=0.5,
        value=2.0,
        label="B-integral",
        info="Tweak system's nonlinearity",
    )

    with gr.Row():
        with gr.Column():
            image_display = gr.Image(label="Environment Render", interactive=False, height=360)
            with gr.Row():
                run_button = gr.Button("Run Simulation")
                status_box = gr.Textbox(label="Status", interactive=False, scale=4)


    # Event handlers
    model_uploader.upload(
        fn=load_model_and_init,
        inputs=[model_uploader, b_slider],
        outputs=[sim_state, image_display, status_box]
    )

    b_slider.release(
        fn=update_b_and_render,
        inputs=[sim_state, b_slider],
        outputs=[sim_state, image_display, status_box]
    )
    
    run_button.click(
        fn=run_simulation_loop,
        inputs=[sim_state, b_slider],
        outputs=[image_display, status_box]
    )

demo.launch()