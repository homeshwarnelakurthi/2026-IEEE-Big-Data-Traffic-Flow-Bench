# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Inference-only replay of the Task 2 spatio-temporal CNN (task2_nn.py) from a saved task2_cnn.pt.

The network is a stack of dilated 1-D residual convolutions over the corridor's links: the input channels are the
12-step history, profile and static link features; the head emits six horizon logits per link. Candidate W uses the
average of three seeds (30, 31, 32) for onset windows at T+30 only.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
import task2_nn as nnm

H = nnm.H


def build_net(C, width=128):
    import torch.nn as nn, torch.nn.functional as F

    class Block(nn.Module):
        def __init__(s, c, dil): super().__init__(); s.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil); s.c2 = nn.Conv1d(c, c, 1); s.n = nn.GroupNorm(8, c)
        def forward(s, x): return x + s.c2(F.gelu(s.n(s.c1(x))))

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Conv1d(C, width, 1); s.blocks = nn.Sequential(*[Block(width, d) for d in (1, 2, 4, 8, 16, 32, 1, 2, 4, 8)]); s.head = nn.Conv1d(width, H, 1)
        def forward(s, x): return s.head(s.blocks(F.gelu(s.inp(x))))
    return Net()


def window_inputs(splits=("validation", "private"), panels=None):
    panels = panels or nnm.v.Q2
    return [r for s in splits for r in nnm.window_inputs(s, panels)]


def dump(weights, W, device="cpu"):
    """weights: path to task2_cnn.pt; W: list from window_inputs(). Returns the window_probs layout."""
    import torch
    net = build_net(W[0]["X"].shape[0]); net.load_state_dict(torch.load(weights, map_location=device)); net.to(device).eval(); rows = []
    with torch.no_grad():
        for r in W:
            p = torch.sigmoid(net(torch.tensor(r["X"][None].astype(np.float32), device=device)))[0].cpu().numpy()[:, :r["M"]]
            for h in range(H):
                rows.append(pd.DataFrame({"split": r["split"], "panel": r["panel"], "window_id": r["window_id"], "condition": r["condition"], "h": h + 1,
                                          "link_id": r["links"], "p_general": p[h], "p_general_w": p[h],
                                          "p_onsetT30": p[h] if (r["condition"] == "queue_onset" and h == H - 1) else np.nan}))
    return pd.concat(rows, ignore_index=True)
