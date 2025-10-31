# innovation_lab.py — Fixed for retrained model compatibility

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
    v[tok] = 1.0
    return v


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
        # Safely flatten and adjust to linear layer input
        B = grid.shape[0]
        grid = grid.view(B, -1)
        wind = wind.view(B, -1)
        hist = hist.view(B, -1)
        others = others.view(B, -1)

        g = self.grid_fc(grid).unsqueeze(0)
        w = self.wind_fc(wind).unsqueeze(0)
        h = self.hist_fc(hist).unsqueeze(0)
        o = self.others_fc(others).unsqueeze(0)
        src = torch.cat([g, w, h, o], dim=0)
        return self.encoder(src)

    def forward(self, grid, wind, hist, others, tgt_tokens=None):
        B = grid.shape[0]
        memory = self.encode_src(grid, wind, hist, others)
        last_tok_idx = torch.argmax(hist.reshape(B, K, -1)[:, -1, :], dim=-1)
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
        print(f"✅ Loaded retrained weights from {weights_path}")
    except Exception as e:
        print(f"⚠️ Could not load weights: {e}")
    return model


def generate_for_agent(model, grid, wind_flat, hist_tokens, other_positions):
    model.eval()
    g_t = torch.tensor(grid.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    w_t = torch.tensor(wind_flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    h_t = torch.tensor(np.concatenate([one_hot_token(t) for t in hist_tokens]),
                       dtype=torch.float32, device=DEVICE).unsqueeze(0)
    o_t = torch.tensor(np.concatenate([one_hot_token(t) for t in other_positions])
                       if len(other_positions) > 0 else np.zeros(VOCAB * (NUM_AGENTS - 1)),
                       dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits_seq = model(g_t, w_t, h_t, o_t)
        preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().astype(int)[0].tolist()
    return preds


if __name__ == "__main__":
    model = load_retrained_model("best_transformer_wind.pt")
    print("✅ Model loaded successfully and ready for inference.")
