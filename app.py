# aqq.qy
# -------------------------------------
# Streamlit Frontend for Innovation Lab
# Agentic AI Supply Chain Visualization
# -------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from innovation_lab import load_retrained_model, generate_for_agent

st.set_page_config(page_title="Innovation Lab AI", layout="wide")
st.title("🧠 Innovation Lab — Agentic AI Flight & Path Predictor")

# Load model
@st.cache_resource
def get_model():
    return load_retrained_model("best_transformer_wind.pt")

model = get_model()

# Input controls
st.sidebar.header("Simulation Controls")
grid_size = st.sidebar.slider("Grid Size", 8, 20, 12)
wind_intensity = st.sidebar.slider("Wind Intensity", 0.1, 1.0, 0.5)
agent_pos = st.sidebar.number_input("Agent Token Position", 0, grid_size*grid_size-1, 10)
goal_pos = st.sidebar.number_input("Goal Token Position", 0, grid_size*grid_size-1, 50)

if st.button("🚀 Run AI Simulation"):
    st.info("Running AI inference with retrained model...")

    # Dummy grid + wind data
    grid = np.zeros((grid_size, grid_size))
    wind = np.random.uniform(-wind_intensity, wind_intensity, size=(grid_size, grid_size, 2))
    hist = [agent_pos-3, agent_pos-2, agent_pos-1] if agent_pos > 3 else [1, 2, 3]
    others = [goal_pos]
    wind_flat = np.concatenate([wind[:,:,0].flatten(), wind[:,:,1].flatten(),
                                np.zeros_like(grid.flatten())]).astype(np.float32)

    preds = generate_for_agent(model, grid, wind_flat, hist, others)
    st.success(f"Predicted Path Tokens: {preds}")

    # Visualization
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(grid, cmap='gray_r', alpha=0.6)
    ax.scatter(*divmod(agent_pos, grid_size)[::-1], c='blue', label='Start')
    ax.scatter(*divmod(goal_pos, grid_size)[::-1], c='red', label='Goal')

    for p in preds:
        px, py = divmod(p, grid_size)
        ax.scatter(py, px, c='orange', s=50)

    ax.legend()
    ax.set_title("Predicted Agent Path")
    st.pyplot(fig)

st.markdown("---")
st.caption("Developed by Pratyush • IIT Kharagpur 🧩")
