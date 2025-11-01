# innovation_lab.py
# Core utilities, episode generation, model wrapper and inference helpers.

import numpy as np
import random
from heapq import heappush, heappop
import torch
import torch.nn as nn

# ---------- Config ----------
GRID_SIZE = 12
VOCAB = GRID_SIZE * GRID_SIZE
NUM_AGENTS = 3
K = 3      # history length
H = 5      # prediction horizon
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Small utils ----------
def xy_to_token(xy, grid_size=GRID_SIZE):
    x, y = xy
    return int(x * grid_size + y)

def token_to_xy(tok, grid_size=GRID_SIZE):
    return (tok // grid_size, tok % grid_size)

def one_hot_token(tok, vocab=VOCAB):
    vec = np.zeros(vocab, dtype=np.float32)
    vec[int(tok)] = 1.0
    return vec

# ---------- A* pathfinding ----------
def neighbors(cell, grid):
    x, y = cell
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
            yield (nx, ny)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_with_avoid(grid, start, goal, avoid_set):
    """Return shortest path from start to goal avoiding coordinates in avoid_set; returns list of coords or None."""
    open_set = []
    heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()
    while open_set:
        _, cost, current, path = heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        for nb in neighbors(current, grid):
            if nb in avoid_set:
                continue
            new_cost = cost + 1
            heappush(open_set, (new_cost + heuristic(nb, goal), new_cost, nb, path + [nb]))
    return None

# ---------- Wind & turbulence helpers ----------
def gen_wind_field(grid_size=GRID_SIZE, intensity=0.6):
    """Generate wind field shape (grid, grid, 2) with values in [-intensity, intensity]."""
    return np.random.uniform(-intensity, intensity, size=(grid_size, grid_size, 2)).astype(np.float32)

def gen_turbulence_mask(grid_size=GRID_SIZE, prob=0.06):
    """Return binary mask (grid, grid) where 1 indicates turbulence cell."""
    return (np.random.rand(grid_size, grid_size) < prob).astype(np.int32)

# ---------- Episode generator ----------
def gen_episode_dynamic(grid_size=GRID_SIZE, num_agents=NUM_AGENTS,
                        obstacle_prob=0.12, timesteps=40,
                        wind_intensity=0.6, turb_prob=0.06, seed=None):
    """
    Create:
      base_grid: 0-free, 1-obstacle  (shape grid_size x grid_size)
      wind_seq: [T, G, G, 2]
      turb_seq: [T, G, G]
      starts, goals: lists of coords
      trajs: ground-truth trajectories computed by dynamic replanning (list of agents x timesteps)
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    base_grid = (np.random.rand(grid_size, grid_size) < obstacle_prob).astype(np.int32)
    free_cells = [(i, j) for i in range(grid_size) for j in range(grid_size) if base_grid[i, j] == 0]
    if len(free_cells) < num_agents * 2:
        return None
    choices = random.sample(free_cells, num_agents * 2)
    starts = [choices[i] for i in range(num_agents)]
    goals  = [choices[i + num_agents] for i in range(num_agents)]

    wind_seq = np.stack([gen_wind_field(grid_size, intensity=wind_intensity) for _ in range(timesteps)], axis=0)
    turb_seq = np.stack([gen_turbulence_mask(grid_size, prob=turb_prob) for _ in range(timesteps)], axis=0)

    planned_paths = []
    used_cells = set()
    for i in range(num_agents):
        cur = starts[i]
        path = [cur]
        for t in range(timesteps - 1):
            grid_t = np.clip(base_grid + turb_seq[t], 0, 1)
            subpath = astar_with_avoid(grid_t, cur, goals[i], used_cells.copy())
            if subpath is None or len(subpath) < 2:
                next_pos = cur
            else:
                next_pos = subpath[1]
            path.append(next_pos)
            cur = next_pos
            if cur == goals[i]:
                for _ in range(t + 1, timesteps):
                    path.append(cur)
                break
        if len(path) < timesteps:
            path.extend([path[-1]] * (timesteps - len(path)))
        planned_paths.append(path)
        used_cells.update(planned_paths[-1])
    return base_grid, wind_seq, turb_seq, starts, goals, planned_paths

# ---------- Dataset builder convenience (kept minimal) ----------
def build_sequence_dataset_dynamic(n_samples=2000, grid_size=GRID_SIZE, num_agents=NUM_AGENTS,
                                   obstacle_prob=0.12, timesteps=40, wind_intensity=0.6, turb_prob=0.06):
    """
    Build a small dataset for training/evaluation. Returns numpy arrays.
    """
    Xg, Xw, Xh, Xo, Yseq = [], [], [], [], []
    created = 0
    attempts = 0
    max_attempts = n_samples * 10
    while created < n_samples and attempts < max_attempts:
        attempts += 1
        ep = gen_episode_dynamic(grid_size=grid_size, num_agents=num_agents, obstacle_prob=obstacle_prob,
                                 timesteps=timesteps, wind_intensity=wind_intensity, turb_prob=turb_prob)
        if ep is None:
            continue
        base_grid, wind_seq, turb_seq, starts, goals, trajs = ep
        for t in range(K, timesteps - H - 1):
            for ag in range(num_agents):
                wind_t = wind_seq[t]
                wind_flat = np.concatenate([wind_t[:, :, 0].flatten(), wind_t[:, :, 1].flatten(), turb_seq[t].flatten()]).astype(np.float32)
                g_flat = base_grid.flatten().astype(np.float32)
                hist_tokens = [xy_to_token(trajs[ag][t - K + 1 + k], grid_size=grid_size) for k in range(K)]
                hist_oh = np.concatenate([one_hot_token(tok) for tok in hist_tokens])
                others = []
                for other in range(num_agents):
                    if other == ag: continue
                    others.append(one_hot_token(xy_to_token(trajs[other][t], grid_size=grid_size)))
                others_concat = np.concatenate(others) if len(others) > 0 else np.zeros(VOCAB * (num_agents - 1), dtype=np.float32)
                target_seq = [xy_to_token(trajs[ag][t + 1 + h], grid_size=grid_size) for h in range(H)]
                Xg.append(g_flat); Xw.append(wind_flat); Xh.append(hist_oh); Xo.append(others_concat); Yseq.append(target_seq)
                created += 1
                if created >= n_samples:
                    break
            if created >= n_samples:
                break
    if created == 0:
        raise RuntimeError("Dataset build failed; try different obstacle probability")
    return (np.stack(Xg), np.stack(Xw), np.stack(Xh), np.stack(Xo), np.array(Yseq, dtype=np.int64))

# ---------- Model (same as in your file with safety) ----------
class AutoregrTransformerWind(nn.Module):
    def __init__(self, vocab=VOCAB, hist_dim=VOCAB * K, others_dim=VOCAB * (NUM_AGENTS - 1),
                 grid_dim=VOCAB, wind_dim=GRID_SIZE * GRID_SIZE * 3,
                 d_model=192, nhead=6, num_layers=3):
        super().__init__()
        self.expected_dims = {"grid": grid_dim, "wind": wind_dim, "hist": hist_dim, "others": others_dim}
        self.grid_fc = nn.Linear(grid_dim, d_model)
        self.wind_fc = nn.Linear(wind_dim, d_model)
        self.hist_fc = nn.Linear(hist_dim, d_model)
        self.others_fc = nn.Linear(others_dim, d_model) if others_dim > 0 else nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=384, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(60, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=384, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(d_model, vocab)

    def encode_src(self, grid, wind, hist, others):
        B = grid.shape[0]
        grid = grid.view(B, -1).float()
        wind = wind.view(B, -1).float()
        hist = hist.view(B, -1).float()
        others = others.view(B, -1).float()
        # checks
        if grid.shape[1] != self.expected_dims["grid"]:
            raise RuntimeError(f"grid flattened dim {grid.shape[1]} != expected {self.expected_dims['grid']}")
        if wind.shape[1] != self.expected_dims["wind"]:
            raise RuntimeError(f"wind flattened dim {wind.shape[1]} != expected {self.expected_dims['wind']}")
        if hist.shape[1] != self.expected_dims["hist"]:
            raise RuntimeError(f"hist flattened dim {hist.shape[1]} != expected {self.expected_dims['hist']}")
        if others.shape[1] != self.expected_dims["others"]:
            raise RuntimeError(f"others flattened dim {others.shape[1]} != expected {self.expected_dims['others']}")
        g = self.grid_fc(grid).unsqueeze(0)
        w = self.wind_fc(wind).unsqueeze(0)
        h = self.hist_fc(hist).unsqueeze(0)
        o = self.others_fc(others).unsqueeze(0)
        src = torch.cat([g, w, h, o], dim=0)
        return self.encoder(src)

    def forward(self, grid, wind, hist, others, tgt_tokens=None, teacher_forcing_prob=0.0):
        B = grid.shape[0]
        memory = self.encode_src(grid, wind, hist, others)
        last_tok_idx = torch.argmax(hist.reshape(B, K, -1)[:, -1, :], dim=-1).to(grid.device)
        dec_input_idxs = last_tok_idx.unsqueeze(1)
        generated_logits = []
        for step in range(H):
            seq_len = dec_input_idxs.shape[1]
            positions = torch.arange(seq_len, device=grid.device).unsqueeze(0).repeat(B, 1)
            tok_emb = self.token_emb(dec_input_idxs) + self.pos_emb(positions)
            tgt = tok_emb.permute(1, 0, 2)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(grid.device)
            dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.out_fc(dec_out[-1, :, :])
            generated_logits.append(logits.unsqueeze(1))
            next_tok = torch.argmax(logits, dim=-1)
            dec_input_idxs = torch.cat([dec_input_idxs, next_tok.unsqueeze(1)], dim=1)
        return torch.cat(generated_logits, dim=1)

def load_retrained_model(weights_path="best_transformer_wind.pt"):
    model = AutoregrTransformerWind().to(DEVICE)
    try:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"Loaded retrained weights from {weights_path}")
    except Exception as e:
        print(f"No weights loaded ({e}); continuing with fresh model.")
    return model

# ---------- Inference wrapper that ensures fixed-length inputs ----------
def build_fixed_length_hist(hist_tokens, vocab=VOCAB, K=K):
    vec = np.zeros(vocab * K, dtype=np.float32)
    used = hist_tokens[-K:] if len(hist_tokens) > K else hist_tokens
    start_idx = (K - len(used)) * vocab
    for i, t in enumerate(used):
        vec[start_idx + i * vocab: start_idx + (i + 1) * vocab] = one_hot_token(t, vocab=vocab)
    return vec

def build_fixed_length_others(other_positions, vocab=VOCAB, n_agents=NUM_AGENTS):
    slots = n_agents - 1
    vec = np.zeros(vocab * slots, dtype=np.float32)
    used = other_positions[:slots] if len(other_positions) > 0 else []
    for i, t in enumerate(used):
        vec[i * vocab:(i + 1) * vocab] = one_hot_token(t, vocab=vocab)
    return vec

def generate_h_for_agent_wind(model, grid, wind_flat, hist_tokens, other_positions, device=DEVICE, H=H):
    model.eval()
    grid_np = np.array(grid, dtype=np.float32).reshape(-1)
    wind_np = np.array(wind_flat, dtype=np.float32).reshape(-1)
    if grid_np.size != VOCAB:
        raise RuntimeError(f"grid flattened size {grid_np.size} != {VOCAB}")
    if wind_np.size != GRID_SIZE * GRID_SIZE * 3:
        raise RuntimeError(f"wind flat size {wind_np.size} != {GRID_SIZE * GRID_SIZE * 3}")
    hist_vec = build_fixed_length_hist(hist_tokens, vocab=VOCAB, K=K)
    other_vec = build_fixed_length_others(other_positions, vocab=VOCAB, n_agents=NUM_AGENTS)
    g_t = torch.tensor(grid_np, dtype=torch.float32, device=device).unsqueeze(0)
    w_t = torch.tensor(wind_np, dtype=torch.float32, device=device).unsqueeze(0)
    h_t = torch.tensor(hist_vec, dtype=torch.float32, device=device).unsqueeze(0)
    o_t = torch.tensor(other_vec, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits_seq = model(g_t, w_t, h_t, o_t)
        preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().astype(int)[0].tolist()
    return preds

# ---------- Fallback single-step planner when model unavailable ----------
def fallback_next_step(base_grid, cur_pos, goal, turb_mask=None):
    # Plan one-step using A* on grid union turb_mask
    grid_use = base_grid.copy()
    if turb_mask is not None:
        grid_use = np.clip(grid_use + turb_mask, 0, 1)
    path = astar_with_avoid(grid_use, cur_pos, goal, set())
    if path is None or len(path) < 2:
        return cur_pos
    return path[1]

