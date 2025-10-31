import streamlit as st
import torch
from innovation_lab import (
    AutoregrTransformerV2,
    load_pretrained,
    export_torchscript,
    generate_h_for_agent_fast,
    device,
)
import numpy as np
import os

st.set_page_config(page_title="Agentic ERP AI", layout="wide")

st.title("🤖 Agentic ERP Supply Chain Dashboard")

# Sidebar: model loading options
st.sidebar.header("⚙️ Model Options")
weight_path = st.sidebar.text_input("Weights (.pt)", "best_transformer_v2.pt")
ts_path = st.sidebar.text_input("TorchScript (.pt)", "best_transformer_v2_ts.pt")
prefer_ts = st.sidebar.checkbox("Prefer TorchScript if available", True)
use_half = st.sidebar.checkbox("Use FP16 on CUDA", True)
use_compile = st.sidebar.checkbox("Use torch.compile()", False)

@st.cache_resource
def load_model(weight_path, ts_path, prefer_ts, use_half, use_compile):
    model = AutoregrTransformerV2().to(device)
    loaded = load_pretrained(
        model,
        path_state_dict=weight_path,
        path_torchscript=ts_path,
        device=device,
        use_half_on_cuda=use_half,
        try_torchscript_first=prefer_ts,
        use_torch_compile=use_compile,
    )
    return loaded

model = load_model(weight_path, ts_path, prefer_ts, use_half, use_compile)

if st.sidebar.button("Export TorchScript (for deployment)"):
    dummy_grid = torch.zeros((1, 144))
    dummy_wind = torch.zeros((1, 144 * 3))
    dummy_hist = torch.zeros((1, 144 * 3))
    dummy_others = torch.zeros((1, 144 * 2))
    export_torchscript(model, (dummy_grid, dummy_wind, dummy_hist, dummy_others), save_path=ts_path)
    st.sidebar.success("✅ TorchScript exported successfully!")

# -----------------
#  Input section
# -----------------
st.subheader("📦 Simulation Inputs")

grid = np.random.rand(12, 12)
wind = np.random.rand(12, 12, 3)
hist = [5, 20, 30]
others = [10, 50]

if st.button("Run Agentic AI Inference"):
    preds = generate_h_for_agent_fast(model, grid, wind.flatten(), hist, others)
    st.success(f"✅ Predicted next sequence: {preds}")
    st.write("These represent the next predicted positions or tokens for supply chain flow.")

st.info("💡 Tip: You can replace the weights file with your retrained version for better performance.")
