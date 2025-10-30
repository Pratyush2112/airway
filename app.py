# ===========================
# app.py
# Streamlit Frontend for Transformer-based AI Flight Simulator
# ===========================

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from innovation_lab import gen_episode, run_episode_horizon, evaluate_horizon_model, AutoregrTransformerV2

# ---------- Setup ----------
st.set_page_config(page_title="AI Flight Simulator", layout="wide")
st.title("✈️ AI-Powered Multi-Agent Flight Path Simulator 🌪️")
st.markdown(
    "Simulate **optimal aircraft trajectories** using a Transformer model trained on dynamic wind and turbulence conditions."
)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Simulation Controls")

grid_size = st.sidebar.slider("Grid Size", 8, 20, 12)
num_agents = st.sidebar.slider("Number of Agents", 1, 5, 3)
obstacle_prob = st.sidebar.slider("Obstacle Probability", 0.05, 0.3, 0.12)
wind_intensity = st.sidebar.slider("Wind Intensity", 0.1, 1.0, 0.6)
turb_prob = st.sidebar.slider("Turbulence Probability", 0.01, 0.1, 0.05)
episode_len = st.sidebar.slider("Episode Length", 10, 50, 40)
run_button = st.sidebar.button("🚀 Run Simulation")

st.sidebar.markdown("---")
evaluate_button = st.sidebar.button("📊 Evaluate Model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    try:
        model = AutoregrTransformerV2().to(device)
        model.load_state_dict(torch.load("best_transformer_v2.pt", map_location=device))
        model.eval()
        return model
    except Exception:
        st.warning("⚠️ No trained model found. Using random initialization.")
        return AutoregrTransformerV2().to(device)

model = load_model()

# ---------- Simulation ----------
if run_button:
    with st.spinner("Generating environment and running simulation..."):
        ep = None
        while ep is None:
            ep = gen_episode(grid_size)
        grid, starts, goals, trajs_truth = ep

        trajs_pred, success, collision = run_episode_horizon(model, grid, starts, goals)
        succ_rate = sum(success) / num_agents * 100

    # ---------- Visualization ----------
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(grid, cmap='gray_r')
    colors = ['tab:blue','tab:orange','tab:purple','tab:green','tab:red']

    for ag in range(num_agents):
        gt = trajs_truth[ag][:len(trajs_pred[ag])]
        px, py = zip(*gt)
        ax.plot(py, px, '--', color=colors[ag], alpha=0.5, label=f'Agent{ag} GT')
        pr = trajs_pred[ag]
        qx, qy = zip(*pr)
        ax.plot(qy, qx, '-', color=colors[ag], label=f'Agent{ag} Pred')
        ax.scatter(starts[ag][1], starts[ag][0], marker='D', color=colors[ag])
        ax.scatter(goals[ag][1], goals[ag][0], marker='X', color=colors[ag])

    ax.set_title(f"Simulation Results (Collision={collision}, Success={succ_rate:.1f}%)")
    ax.legend()
    plt.gca().invert_yaxis()
    st.pyplot(fig)

# ---------- Evaluation ----------
if evaluate_button:
    with st.spinner("Evaluating model performance on random episodes..."):
        succ_rate, collision_rate, avg_time = evaluate_horizon_model(model, n_episodes=30)
        st.success("✅ Model Evaluation Complete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Success Rate", f"{succ_rate:.1f}%")
        col2.metric("Collision Rate", f"{collision_rate:.1f}%")
        col3.metric("Avg Time-to-Goal", f"{avg_time:.1f} steps")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "Developed by **Pratyush Singh** — IIT Kharagpur 🧠 | Powered by Transformer Models & Streamlit"
)
