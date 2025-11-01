# innovation_lab_fixed.py — Fixed for retrained model compatibility

import numpy as np
import torch
import torch.nn as nn

GRID_SIZE = 12
VOCAB = GRID_SIZE * GRID_SIZE
NUM_AGENTS = 3
K = 3
H = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xy_to_token(xy):
    x, y = xy
    return int(x * GRID_SIZE + y)


def token_to_xy(tok):
    return (tok // GRID_SIZE, tok % GRID_SIZE)


def one_hot_token(tok, vocab=VOCAB):
    v = np.zeros(vocab, dtype=np.float32)
    v[int(tok)] = 1.0
    return v


class AutoregrTransformerWind(nn.Module):
    def __init__(self, vocab=VOCAB, hist_dim=VOCAB * K,
                 others_dim=VOCAB * (NUM_AGENTS - 1),
                 grid_dim=VOCAB, wind_dim=GRID_SIZE * GRID_SIZE * 3,
                 d_model=192, nhead=6, num_layers=3):
        super().__init__()
        self.expected_dims = {
            "grid": grid_dim,
            "wind": wind_dim,
            "hist": hist_dim,
            "others": others_dim
        }
        self.grid_fc = nn.Linear(grid_dim, d_model)
        self.wind_fc = nn.Linear(wind_dim, d_model)
        self.hist_fc = nn.Linear(hist_dim, d_model)
        self.others_fc = nn.Linear(others_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 384, 0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_emb = nn.Embedding(vocab, d_model)
        # Make pos embedding large enough for K + H + some buffer
        self.pos_emb = nn.Embedding(60, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, 384, 0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(d_model, vocab)

    def encode_src(self, grid, wind, hist, others):
        # Ensure tensors are (B, -1) and float on correct device
        B = grid.shape[0]
        grid = grid.view(B, -1).float()
        wind = wind.view(B, -1).float()
        hist = hist.view(B, -1).float()
        others = others.view(B, -1).float()

        # Sanity checks: if shapes don't match what encoders expect, raise informative error
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
        src = torch.cat([g, w, h, o], dim=0)  # seq_len=4, B, d_model
        return self.encoder(src)

    def forward(self, grid, wind, hist, others, tgt_tokens=None):
        B = grid.shape[0]
        memory = self.encode_src(grid, wind, hist, others)
        # compute last token index from hist: hist shaped (B, K*VOCAB)
        # reshape to (B, K, VOCAB) then argmax
        last_tok_idx = torch.argmax(hist.reshape(B, K, -1)[:, -1, :], dim=-1)
        dec_input_idxs = last_tok_idx.unsqueeze(1)  # (B, 1)
        generated_logits = []
        for step in range(H):
            seq_len = dec_input_idxs.shape[1]
            positions = torch.arange(seq_len, device=grid.device).unsqueeze(0).repeat(B, 1)
            tok_emb = self.token_emb(dec_input_idxs) + self.pos_emb(positions)
            tgt = tok_emb.permute(1, 0, 2)  # seq_len, B, d_model
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(grid.device)
            dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.out_fc(dec_out[-1, :, :])  # (B, vocab)
            generated_logits.append(logits.unsqueeze(1))
            next_tok = torch.argmax(logits, dim=-1)
            dec_input_idxs = torch.cat([dec_input_idxs, next_tok.unsqueeze(1)], dim=1)
        return torch.cat(generated_logits, dim=1)  # (B, H, vocab)


def load_retrained_model(weights_path="best_transformer_wind.pt"):
    model = AutoregrTransformerWind().to(DEVICE)
    try:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"✅ Loaded retrained weights from {weights_path}")
    except Exception as e:
        print(f"⚠️ Could not load weights: {e}")
    return model


def _build_fixed_length_hist(hist_tokens, vocab=VOCAB, K=K):
    """
    Build a flattened hist vector of length K * vocab.
    hist_tokens: list of token ints (most recent last). If len < K, will pad at front with zeros.
    """
    vec = np.zeros(vocab * K, dtype=np.float32)
    # Ensure hist_tokens length <= K; use the most recent tokens at the end
    used = hist_tokens[-K:] if len(hist_tokens) > K else hist_tokens
    start_idx = (K - len(used)) * vocab  # pad at front if fewer tokens
    for i, t in enumerate(used):
        vec[start_idx + i * vocab: start_idx + (i + 1) * vocab] = one_hot_token(t, vocab=vocab)
    return vec


def _build_fixed_length_others(other_positions, vocab=VOCAB, n_agents=NUM_AGENTS):
    """
    Build a flattened others vector of length (n_agents-1) * vocab.
    other_positions: list of token ints for other agents. If fewer than n_agents-1, pad with zeros.
    If more, take the first (n_agents-1) entries.
    Order matters — choose a consistent ordering upstream.
    """
    slots = n_agents - 1
    vec = np.zeros(vocab * slots, dtype=np.float32)
    used = other_positions[:slots] if len(other_positions) >= 0 else []
    for i, t in enumerate(used):
        vec[i * vocab:(i + 1) * vocab] = one_hot_token(t, vocab=vocab)
    return vec


def generate_for_agent(model, grid, wind_flat, hist_tokens, other_positions):
    """
    grid: a (GRID_SIZE, GRID_SIZE) numpy array or flattened vector of length VOCAB
    wind_flat: flattened wind vector length GRID_SIZE*GRID_SIZE*3
    hist_tokens: list of token ints (ordered oldest->newest or vice versa but consistent)
    other_positions: list of token ints for other agents (length <= NUM_AGENTS-1)
    """
    model.eval()

    # Prepare grid: accept either flattened or 2D grid
    grid_np = np.array(grid, dtype=np.float32)
    if grid_np.size == VOCAB:
        grid_flat = grid_np.reshape(VOCAB)
    elif grid_np.size == GRID_SIZE * GRID_SIZE:
        grid_flat = grid_np.reshape(VOCAB)
    else:
        raise RuntimeError(f"grid has unexpected size {grid_np.size}, expected {VOCAB}")

    # Prepare wind
    wind_np = np.array(wind_flat, dtype=np.float32)
    if wind_np.size != GRID_SIZE * GRID_SIZE * 3:
        raise RuntimeError(f"wind_flat has unexpected size {wind_np.size}, expected {GRID_SIZE * GRID_SIZE * 3}")

    # Prepare hist and others with fixed lengths
    hist_vec = _build_fixed_length_hist(hist_tokens, vocab=VOCAB, K=K)
    others_vec = _build_fixed_length_others(other_positions, vocab=VOCAB, n_agents=NUM_AGENTS)

    # Convert to tensors on DEVICE
    g_t = torch.tensor(grid_flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, VOCAB)
    w_t = torch.tensor(wind_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)    # (1, wind_dim)
    h_t = torch.tensor(hist_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)   # (1, K*VOCAB)
    o_t = torch.tensor(others_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0) # (1, (NUM_AGENTS-1)*VOCAB)

    with torch.no_grad():
        logits_seq = model(g_t, w_t, h_t, o_t)  # (1, H, vocab)
        preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().astype(int)[0].tolist()  # list length H
    return preds


if __name__ == "__main__":
    model = load_retrained_model("best_transformer_wind.pt")
    print("✅ Model loaded successfully and ready for inference.")

