import matplotlib
matplotlib.use('Agg')

import gradio as gr
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import os

from huggingface_hub import hf_hub_download

# This is needed to register the custom environment
import gym_laser

# Pre-trained model configurations (TODO: add models by hosting them on huggingface)
PRETRAINED_MODELS = {
    "Random Policy": None,
    "Upload Custom Model": "upload",
    "SAC-UDR(1.5,2.5)": "sac-udr-narrow", 
    "SAC-UDR(1.0,9.0)": "sac-udr-wide-extra",
}

MAX_STEPS = 100_000  # large number for continuous simulation

def get_model_path(model_id):
    """Get the path to a pre-trained model."""
    return f"pretrained-policies/{model_id}.zip"


def load_pretrained_model(model_id):
    """Load a pre-trained model."""
    model = hf_hub_download(
        repo_id=f"fracapuano/{model_id}", filename=f"{model_id}.zip"
    )
    return SAC.load(model)


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
            "current_b_integral": 2.0,  # Store current B-integral in state
            "model_filename": "Random Policy"  # Default model name
        }
        return state
    except Exception as e:
        return None, f"Error: {e}"


def load_selected_model(state, model_selection, uploaded_file):
    """Loads a model based on selection (pre-trained or uploaded)."""
    if state is None:
        return state, gr.update()
    
    try:
        if model_selection == "Random Policy":
            state["model"] = None
            state["model_filename"] = "Random Policy"
            state["obs"] = state["env"].reset()
            state["step_num"] = 0
            return state, gr.update()
        
        elif model_selection == "Upload Custom Model":
            if uploaded_file is None:
                return state, "Please upload a model file.", gr.update()
            
            model_filename = uploaded_file.name.split('/')[-1]
            state["model"] = SAC.load(uploaded_file.name)
            state["model_filename"] = model_filename
            state["obs"] = state["env"].reset()
            state["step_num"] = 0
            return state, gr.update()
        
        else:
            model_id = PRETRAINED_MODELS[model_selection]
            model = load_pretrained_model(model_id)
            
            state["model"] = model
            state["model_filename"] = model_selection
            state["obs"] = state["env"].reset()
            state["step_num"] = 0
            return state, gr.update()
            
    except Exception as e:
        return state, f"Error loading model: {e}", gr.update()

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
    for i in range(MAX_STEPS):
        model = state.get("model")
        model_filename = state.get("model_filename", "Random Policy")
        current_b = state.get("current_b_integral", 2.0)
        
        # Apply the current B-integral value from state
        env.envs[0].unwrapped.laser.B = float(current_b)

        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample().reshape(1, -1)
            
        obs, _, done, _ = env.step(action)
        frame = env.render()
        
        if done[0]:
            obs = env.reset()
            step_num = 0
        else:
            step_num += 1

        state["obs"] = obs
        state["step_num"] = step_num
        
        yield state, frame


with gr.Blocks(css="body {zoom: 90%}") as demo:
    gr.Markdown("# Shaping Laser Pulses with Reinforcement Learning")
    
    with gr.Tab("Demo"):
        sim_state = gr.State()

        with gr.Row():
            b_slider = gr.Slider(
                minimum=0,
                maximum=10,
                step=0.5,
                value=2.0,
                label="B-integral",
                info="Adjust nonlinearity live during simulation.",
            )

        with gr.Row():
            image_display = gr.Image(label="Environment Render", interactive=False, height=360)
        
        with gr.Row():
            with gr.Column():
                model_selector = gr.Dropdown(
                    choices=list(PRETRAINED_MODELS.keys()),
                    value="Random Policy",
                    label="Model Selection",
                    info="Choose a pre-trained model or upload your own"
                )

        with gr.Row():
            with gr.Column(scale=1):
                model_uploader = gr.UploadButton(
                    "Upload Model (.zip)",
                    file_types=['.zip'],
                    elem_id="model-upload",
                    visible=False  # Initially hidden
                )

        # Show/hide upload button based on selection
        def update_upload_visibility(selection):
            return gr.update(visible=(selection == "Upload Custom Model"))
        
        model_selector.change(
            fn=update_upload_visibility,
            inputs=[model_selector],
            outputs=[model_uploader]
        )

        # On page load, initialize and start the continuous simulation
        init_event = demo.load(
            fn=initialize_environment,
            inputs=None,
            outputs=[sim_state]
        )
        
        continuous_event = init_event.then(
            fn=run_continuous_simulation,
            inputs=[sim_state],
            outputs=[sim_state, image_display]
        )

        # When model selection changes, load the selected model
        model_change_event = model_selector.change(
            fn=load_selected_model,
            inputs=[sim_state, model_selector, model_uploader],
            outputs=[sim_state, model_uploader],
            cancels=[continuous_event]
        ).then(
            fn=run_continuous_simulation,
            inputs=[sim_state],
            outputs=[sim_state, image_display]
        )

        # When a custom model is uploaded, load it
        model_upload_event = model_uploader.upload(
            fn=load_selected_model,
            inputs=[sim_state, model_selector, model_uploader],
            outputs=[sim_state, model_uploader],
            cancels=[continuous_event]
        ).then(
            fn=run_continuous_simulation,
            inputs=[sim_state],
            outputs=[sim_state, image_display]
        )

        # When B-integral slider changes, just update the value in state (no restart needed)
        b_slider.change(
            fn=update_b_integral,
            inputs=[sim_state, b_slider],
            outputs=[sim_state]
        )
    
    with gr.Tab("About"):
        with open("copy.md", "r") as f:
            gr.Markdown(f.read())

demo.launch()