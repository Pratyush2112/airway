# innovation_lab.py
# ---------------------------------------------------
# Transformer-based Multi-Agent Flight Path Simulator
# with Dynamic Wind + Turbulence + Retrained Weights
# ---------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from heapq import heappush, heappop
import matplotlib.pyplot as plt

# ----------------- Config -----------------
GRID_SIZE = 12
VOCAB = GRID_SIZE * GRID_SIZE
NUM_AGENTS = 3
K = 3
H = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Utils -----------------
def xy_to_token(xy):
    x, y = xy
    return int(x * GRID_SIZE + y)

def token_to_xy(tok):
    return (tok // GRID_SIZE, tok % GRID_SIZE)

def one_hot_token(tok, vocab=VOCAB):
    v = np.zeros(vocab, dtype=np.float32)
    v[tok] = 1.0
    return v

def neighbors(cell, grid):
    x, y = cell
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
            yield (nx, ny)

# ----------------- Model -----------------
class AutoregrTransformerWind(nn.Module):
    def __init__(self, vocab=VOCAB, hist_dim=VOCAB*K,
                 others_dim=VOCAB*(NUM_AGENTS-1),
                 grid_dim=VOCAB, wind_dim=GRID_SIZE*GRID_SIZE*3,
                 d_model=192, nhead=6, num_layers=3):
        super().__init__()
        self.grid_fc = nn.Linear(grid_dim, d_model)
        self.wind_fc = nn.Linear(wind_dim, d_model)
        self.hist_fc = nn.Linear(hist_dim, d_model)
        self.others_fc = nn.Linear(others_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 384, 0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(60, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, 384, 0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(d_model, vocab)

    def encode_src(self, grid, wind, hist, others):
        g = self.grid_fc(grid).unsqueeze(0)
        w = self.wind_fc(wind).unsqueeze(0)
        h = self.hist_fc(hist).unsqueeze(0)
        o = self.others_fc(others).unsqueeze(0)
        src = torch.cat([g, w, h, o], dim=0)
        return self.encoder(src)

    def forward(self, grid, wind, hist, others, tgt_tokens=None):
        B = grid.shape[0]
        memory = self.encode_src(grid, wind, hist, others)
        last_tok_idx = torch.argmax(hist.reshape(B, K, VOCAB)[:, -1, :], dim=-1)
        dec_input_idxs = last_tok_idx.unsqueeze(1)
        generated_logits = []
        for step in range(H):
            seq_len = dec_input_idxs.shape[1]
            positions = torch.arange(seq_len, device=grid.device).unsqueeze(0).repeat(B,1)
            tok_emb = self.token_emb(dec_input_idxs) + self.pos_emb(positions)
            tgt = tok_emb.permute(1,0,2)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(grid.device)
            dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.out_fc(dec_out[-1,:,:])
            generated_logits.append(logits.unsqueeze(1))
            next_tok = torch.argmax(logits, dim=-1)
            dec_input_idxs = torch.cat([dec_input_idxs, next_tok.unsqueeze(1)], dim=1)
        return torch.cat(generated_logits, dim=1)

# ----------------- Load Retrained Weights -----------------
def load_retrained_model(weights_path="best_transformer_wind.pt"):
    model = AutoregrTransformerWind().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded retrained model from {weights_path}")
    return model

# ----------------- Inference -----------------
def generate_for_agent(model, grid, wind_flat, hist_tokens, other_positions):
    model.eval()
    g_t = torch.tensor(grid.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    w_t = torch.tensor(wind_flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    h_t = torch.tensor(np.concatenate([one_hot_token(t) for t in hist_tokens]),
                       dtype=torch.float32, device=DEVICE).unsqueeze(0)
    o_t = torch.tensor(np.concatenate([one_hot_token(t) for t in other_positions])
                       if len(other_positions)>0 else np.zeros(VOCAB*(NUM_AGENTS-1)),
                       dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits_seq = model(g_t, w_t, h_t, o_t)
        preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().astype(int)[0].tolist()
    return preds

# ----------------- Simple Visual Test -----------------
if __name__ == "__main__":
    model = load_retrained_model("best_transformer_wind.pt")
    print("✅ Model ready for Streamlit frontend integration.")
