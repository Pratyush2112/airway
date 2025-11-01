# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from innovation_lab import (GRID_SIZE, VOCAB, NUM_AGENTS, K, H, DEVICE,
                            gen_episode_dynamic, gen_wind_field, gen_turbulence_mask,
                            token_to_xy, xy_to_token, generate_h_for_agent_wind,
                            load_retrained_model, AutoregrTransformerWind, fallback_next_step, token_to_xy as tok2xy)

st.set_page_config(layout="wide", page_title="Airway — Dynamic Simulation")

st.title("Airway — Dynamic multi-agent sim (wind, turbulence, obstacles)")

# Sidebar controls
st.sidebar.header("Environment controls")
obstacle_prob = st.sidebar.slider("Obstacle probability", 0.0, 0.4, 0.12, 0.01)
turb_prob = st.sidebar.slider("Turbulence probability (per cell)", 0.0, 0.3, 0.06, 0.01)
wind_intensity = st.sidebar.slider("Wind intensity", 0.0, 2.0, 0.6, 0.05)
timesteps = st.sidebar.slider("Episode timesteps", 10, 80, 40, 1)
num_agents = st.sidebar.slider("Num agents", 1, 4, NUM_AGENTS, 1)
seed = st.sidebar.number_input("Random seed (0 = random)", min_value=0, step=1, value=0)

st.sidebar.markdown("---")
st.sidebar.header("Simulation / Model")
use_model = st.sidebar.checkbox("Use Transformer model if available", value=False)
model_path = st.sidebar.text_input("Model path (relative)", value="best_transformer_wind.pt")
run_eval = st.sidebar.button("Generate & Run episode")

# manual controls
st.sidebar.markdown("---")
st.sidebar.header("Manual obstacle editing")
if "manual_grid" not in st.session_state:
    st.session_state.manual_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
place_mode = st.sidebar.selectbox("Place mode", ["None", "Toggle cell", "Randomize obstacles", "Clear obstacles"])
if st.sidebar.button("Apply manual change"):
    if place_mode == "Randomize obstacles":
        st.session_state.manual_grid = (np.random.rand(GRID_SIZE, GRID_SIZE) < obstacle_prob).astype(int)
    elif place_mode == "Clear obstacles":
        st.session_state.manual_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

# Attempt to load model if requested
model = None
if use_model:
    if os.path.exists(model_path):
        try:
            model = load_retrained_model(model_path)
            st.sidebar.success(f"Model loaded from {model_path}")
        except Exception as e:
            st.sidebar.error(f"Failed loading model: {e}")
            model = None
    else:
        st.sidebar.warning("Model file not found; fallback planner will be used.")
        model = None

# Run the episode generation and simulation
if run_eval:
    chosen_seed = None if seed == 0 else int(seed)
    ep = gen_episode_dynamic(grid_size=GRID_SIZE, num_agents=num_agents,
                             obstacle_prob=obstacle_prob, timesteps=timesteps,
                             wind_intensity=wind_intensity, turb_prob=turb_prob, seed=chosen_seed)
    if ep is None:
        st.error("Failed to generate an episode — try lowering obstacle probability or reducing agents.")
    else:
        base_grid, wind_seq, turb_seq, starts, goals, truth_trajs = ep

        # If manual grid provided, overlay it (manual overrides base)
        if st.session_state.manual_grid.sum() > 0:
            base_grid = st.session_state.manual_grid.copy()

        # Prepare wind_flat per timestep as model expects (concat u,v,turb)
        # Run an H-step predicted rolling simulation
        cur = list(starts)
        pred_trajs = [[s] for s in starts]
        for t in range(timesteps - 1):
            wind_t = wind_seq[t]
            wind_flat = np.concatenate([wind_t[:, :, 0].flatten(), wind_t[:, :, 1].flatten(), turb_seq[t].flatten()]).astype(np.float32)
            proposals = []
            for ag in range(num_agents):
                hist = pred_trajs[ag]
                # build K-history tokens (oldest->latest); pad if needed
                hist_tokens = [xy_to_token(hist[max(0, len(hist) - K + k)]) for k in range(K)]
                other_pos = [xy_to_token(cur[o]) for o in range(len(cur)) if o != ag]
                if model is not None:
                    try:
                        preds = generate_h_for_agent_wind(model, base_grid.flatten(), wind_flat, hist_tokens, other_pos)
                        proposed = tok2xy(preds[0])
                    except Exception as e:
                        st.sidebar.warning(f"Model inference failed, using fallback for agent {ag}: {e}")
                        proposed = fallback_next_step(base_grid, cur[ag], goals[ag], turb_seq[t])
                else:
                    proposed = fallback_next_step(base_grid, cur[ag], goals[ag], turb_seq[t])
                # ensure neighbor or remain
                if proposed not in list(neighbors(cur[ag], base_grid)) and proposed != cur[ag]:
                    proposed = cur[ag]
                proposals.append(proposed)
            # collision resolution (simple)
            counts = {}
            for p in proposals: counts[p] = counts.get(p, 0) + 1
            new_positions = []
            for ag, p in enumerate(proposals):
                if counts[p] > 1:
                    new_positions.append(cur[ag])
                else:
                    new_positions.append(p)
            cur = new_positions
            for ag in range(num_agents):
                pred_trajs[ag].append(cur[ag])

        # Visualization
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(base_grid, cmap="gray_r", alpha=0.6)
        # show turbulence at last timestep as red transparent patches
        turb_show = turb_seq[min(timesteps - 1, 0)]
        for (i, j), val in np.ndenumerate(turb_show):
            if val:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='red', alpha=0.18)
                ax.add_patch(rect)
        # draw wind arrows sampled on a coarse grid
        wind_vis = wind_seq[min(0, timesteps - 1)]
        step = max(1, GRID_SIZE // 8)
        for i in range(0, GRID_SIZE, step):
            for j in range(0, GRID_SIZE, step):
                u, v = wind_vis[i, j]
                ax.arrow(j, i, v * 0.6, u * 0.6, head_width=0.15, head_length=0.15, fc='k', ec='k', alpha=0.7)

        colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:green']
        for ag in range(num_agents):
            gt = truth_trajs[ag]
            pr = pred_trajs[ag]
            gx, gy = zip(*gt)
            px, py = zip(*pr)
            ax.plot(gy, gx, '--', color=colors[ag % len(colors)], alpha=0.5, label=f'Agent{ag} GT')
            ax.plot(py, px, '-', color=colors[ag % len(colors)], label=f'Agent{ag} Pred')
            ax.scatter(starts[ag][1], starts[ag][0], marker='D', color=colors[ag % len(colors)], s=50)
            ax.scatter(goals[ag][1], goals[ag][0], marker='X', color=colors[ag % len(colors)], s=50)
        ax.set_title("Episode (green=goals, D=starts). Red=turbu cells. Arrows=wind")
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(GRID_SIZE - 0.5, -0.5)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.grid(color='lightgray', linestyle=':', linewidth=0.4)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        # show key stats
        succ = [pred_trajs[i][-1] == goals[i] for i in range(num_agents)]
        st.write(f"Success per agent: {succ}")
        collisions = any(len(set([pred_trajs[a][t] for a in range(num_agents)])) != num_agents for t in range(len(pred_trajs[0])))
        st.write(f"Collision occurred: {collisions}")

# Small helper for manual toggle of cells (web UI)
def neighbors(cell, grid):
    x, y = cell
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
            yield (nx, ny)
