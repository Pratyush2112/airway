import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import warnings

# ==============================
#  GLOBAL CONFIG
# ==============================
GRID_SIZE = 12
VOCAB = GRID_SIZE * GRID_SIZE
K = 3               # history length
H = 3               # prediction horizon
NUM_AGENTS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
#  UTILS
# ==============================
def one_hot_token(t, vocab=VOCAB):
    x = np.zeros(vocab, dtype=np.float32)
    if 0 <= t < vocab:
        x[t] = 1.0
    return x


# ==============================
#  MODEL ARCHITECTURE
# ==============================
class AutoregrTransformerV2(nn.Module):
    def __init__(self, vocab_size=VOCAB, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.grid_proj = nn.Linear(GRID_SIZE * GRID_SIZE, d_model)
        self.wind_proj = nn.Linear(GRID_SIZE * GRID_SIZE * 3, d_model)
        self.hist_proj = nn.Linear(vocab_size * K, d_model)
        self.other_proj = nn.Linear(vocab_size * (NUM_AGENTS - 1), d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, grid, wind, hist, others, tgt_tokens=None):
        # grid, wind, hist, others => [B, ...]
        g = self.grid_proj(grid)
        w = self.wind_proj(wind)
        h = self.hist_proj(hist)
        o = self.other_proj(others)

        x = torch.stack([g, w, h, o], dim=1)  # [B, 4, d_model]
        enc = self.encoder(x)                 # [B, 4, d_model]

        out = self.fc_out(enc[:, -1, :])      # [B, vocab]
        out = out.unsqueeze(1).repeat(1, H, 1)  # predict H steps
        return out


# ==============================
#  FAST MODEL LOAD / EXPORT
# ==============================
def load_pretrained(model, path_state_dict=None, path_torchscript=None, device=device,
                    use_half_on_cuda=True, try_torchscript_first=True, use_torch_compile=False):
    """Load model or TorchScript with safety & speed"""
    if try_torchscript_first and path_torchscript and os.path.exists(path_torchscript):
        try:
            scripted = torch.jit.load(path_torchscript, map_location=device)
            scripted.eval()
            if device.type == "cuda":
                scripted.to(device)
            print(f"✅ Loaded TorchScript from {path_torchscript}")
            return scripted
        except Exception as e:
            warnings.warn(f"TorchScript load failed: {e}, falling back to weights.")

    if path_state_dict and os.path.exists(path_state_dict):
        sd = torch.load(path_state_dict, map_location=device)
        model.load_state_dict(sd)
        model.eval().to(device)

        if device.type == "cuda" and use_half_on_cuda:
            for name, module in model.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.float()
            model.half()
            print("✅ Half precision (FP16) enabled on CUDA")

        if use_torch_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("✅ Model compiled with torch.compile()")
            except Exception as e:
                warnings.warn(f"torch.compile failed: {e}")

        print(f"✅ Loaded state_dict from {path_state_dict}")
        return model

    warnings.warn("⚠️ No model file found; returning empty model.")
    return model.to(device)


def export_torchscript(model, example_inputs, save_path="best_transformer_ts.pt", device=device):
    """Export model to TorchScript"""
    model.eval().to(device)
    try:
        traced = torch.jit.trace(model.cpu(), example_inputs, strict=False)
        traced.save(save_path)
        print(f"✅ TorchScript saved to {save_path}")
        return save_path
    except Exception as e:
        warnings.warn(f"Trace failed: {e}")
        try:
            scripted = torch.jit.script(model.cpu())
            scripted.save(save_path)
            print(f"✅ Scripted TorchScript saved to {save_path}")
            return save_path
        except Exception as e2:
            warnings.warn(f"Script also failed: {e2}")
            return None


# ==============================
#  FAST INFERENCE HELPERS
# ==============================
def generate_h_for_agent_fast(model, grid, wind_flat, hist_tokens, other_positions,
                              device=device, H=H, vocab=VOCAB):
    model_device = device
    g_t = torch.tensor(grid.flatten(), dtype=torch.float32, device=model_device).unsqueeze(0)
    w_t = torch.tensor(wind_flat, dtype=torch.float32, device=model_device).unsqueeze(0)
    h_t = torch.tensor(np.concatenate([one_hot_token(t) for t in hist_tokens]),
                       dtype=torch.float32, device=model_device).unsqueeze(0)
    if len(other_positions) > 0:
        others_oh = np.concatenate([one_hot_token(t) for t in other_positions]).astype(np.float32)
    else:
        others_oh = np.zeros(vocab * (NUM_AGENTS - 1), dtype=np.float32)
    o_t = torch.tensor(others_oh, dtype=torch.float32, device=model_device).unsqueeze(0)

    if next(model.parameters()).dtype == torch.float16:
        g_t = g_t.half(); w_t = w_t.half(); h_t = h_t.half(); o_t = o_t.half()

    with torch.inference_mode():
        logits_seq = model(g_t, w_t, h_t, o_t, tgt_tokens=None)
        preds = torch.argmax(logits_seq, dim=-1).cpu().numpy().astype(int)[0].tolist()
    return preds


def try_load_best_weights(model, filename="best_transformer_v2.pt", ts_filename="best_transformer_v2_ts.pt"):
    if os.path.exists(ts_filename):
        try:
            return load_pretrained(model, path_torchscript=ts_filename)
        except Exception:
            pass
    if os.path.exists(filename):
        return load_pretrained(model, path_state_dict=filename)
    return model
