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

def initialize_environment(b_integral):
    """Initializes the environment on app load."""
    try:
        env = DummyVecEnv([make_env_fn])
        env = VecFrameStack(env, n_stack=5)
        env.envs[0].unwrapped.laser.B = float(b_integral)
        obs = env.reset()
        initial_frame = env.render()
        state = { "env": env, "obs": obs, "model": None }
        return state, initial_frame, "Environment ready. Run with a random policy or upload a model."
    
    except Exception as e:
        return None, None, f"Error: {e}"

def load_model(state, model_path):
    """Loads a model into the existing environment state."""
    if model_path is None:
        return state, "Upload failed."
    if not state or "env" not in state:
        return state, "Environment not initialized. Please refresh."
    try:
        state["model"] = SAC.load(model_path.name)
        state["obs"] = state["env"].reset() # Reset for the new policy
        initial_frame = state["env"].render()
        return state, initial_frame, "Model loaded. Ready to run simulation."
    except Exception as e:
        return state, None, f"Error loading model: {e}"

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

def run_policy_loop(state, b_integral):
    """
    Dispatcher that runs the appropriate simulation loop 
    based on whether a model is loaded.
    """
    if not state or "env" not in state:
        yield None, "Environment not ready. Please refresh."
        return
    
    if state.get("model"):
        # Model is loaded, run the intelligent agent
        yield from run_simulation_loop(state, b_integral)
    else:
        # No model, run a random agent
        yield from run_random_policy(state, b_integral)

def run_random_policy(state, b_integral):
    env = state["env"]
    obs = state["obs"]

    env.envs[0].unwrapped.laser.B = float(b_integral)

    for i in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action.reshape(1, -1))
        frame = env.render()
        
        yield frame, f"Running random policy... Step {i+1}/{MAX_STEPS}"
        time.sleep(0.05)
            
    state["obs"] = env.reset() if done[0] else obs
    yield env.render(), "Random policy run finished."

def run_simulation_loop(state, b_integral):
    """
    Runs a simulation loop for MAX_STEPS, yielding intermediate frames.
    """
    env = state["env"]
    model = state["model"]
    obs = state["obs"]

    env.envs[0].unwrapped.laser.B = float(b_integral)

    for i in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frame = env.render()
        
        yield frame, f"Running model... Step {i+1}/{MAX_STEPS}"
        time.sleep(0.05)
        
        if done[0]:
            break
            
    state["obs"] = env.reset() if done[0] else obs
    yield env.render(), "Simulation finished."

with gr.Blocks() as demo:
    gr.Markdown("# DRL for Laser Pulse Shaping")
    gr.Markdown(
        "Run a simulation with a random agent, or upload your own SAC model to see it in action. "
        "Adjust the B-integral slider to see its effect on the pulse."
    )

    sim_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            model_uploader = gr.UploadButton(
                "Upload Model (.zip)",
                file_types=['.zip'],
                elem_id="model-upload",
            )

        with gr.Column():
            run_button = gr.Button("Run Simulation")

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
                status_box = gr.Textbox(label="Status", interactive=False, scale=4)

    # Event Handlers
    demo.load(
        fn=initialize_environment,
        inputs=[b_slider],
        outputs=[sim_state, image_display, status_box]
    )

    model_uploader.upload(
        fn=load_model,
        inputs=[sim_state, model_uploader],
        outputs=[sim_state, image_display, status_box]
    )

    b_slider.release(
        fn=update_b_and_render,
        inputs=[sim_state, b_slider],
        outputs=[sim_state, image_display, status_box]
    )
    
    run_button.click(
        fn=run_policy_loop,
        inputs=[sim_state, b_slider],
        outputs=[image_display, status_box]
    )

demo.launch()