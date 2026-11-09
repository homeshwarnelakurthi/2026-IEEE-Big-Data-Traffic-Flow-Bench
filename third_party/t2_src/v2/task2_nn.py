# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 2 spatio-temporal CNN. The corridor is a 1-D sequence of links; the 13-step history (T-60..T), leave-one-day-out
profile slices and static link attributes are stacked as channels; dilated residual Conv1d blocks along space give each
link a wide spatial view; the head emits 6 horizon logits per link. Onset windows are trained and predicted at T+30 only
(organizer-confirmed selection rule). Training windows: organizer-rule picks + random picks (as v19), history blanking.
Outputs window_probs.parquet in the same format as the LightGBM versions (p_general = p_general_w = CNN prob)."""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import task2_v10 as v, task2_v11 as v11
from common import DATA, T, network, day_files, date_of, weekday_of, slots_of, day_grid

HV, H, MMAX = 13, 6, 352

def static_feats(net):
    M = len(net); pos = np.arange(M) / max(M - 1, 1)
    return np.stack([net.free_speed_kmh.to_numpy() / 120, net.lanes.to_numpy() / 6, np.log1p(net.length_km.to_numpy()),
                     net.ramp_on.to_numpy(), net.ramp_off.to_numpy(), pos, net.v_cut.to_numpy() / net.free_speed_kmh.to_numpy()], 0).astype(np.float32)

def make_input(Hv, Hq, Ho, net, prof, t0, wd, onset, st):
    """Channels x links. Hv/Hq/Ho: (M,13) ending at T."""
    vf = net.free_speed_kmh.to_numpy()[:, None]; cap = net.capacity_vph.to_numpy()[:, None]
    ok = np.isfinite(Hv) & np.isfinite(Hq)
    vr = np.where(ok, Hv / vf, 0); qn = np.where(ok, Hq / cap, 0); oc = np.where(np.isfinite(Ho), Ho, 0)
    qf = prof["qfrac"][wd]; pv = prof["v"][wd] / vf
    hs = np.clip(np.arange(t0 - HV + 1, t0 + 1), 0, T - 1); fs = np.clip(np.arange(t0 + 1, t0 + H + 1), 0, T - 1)
    tt = 2 * np.pi * t0 / T; M = Hv.shape[0]
    glob = np.stack([np.full(M, np.sin(tt)), np.full(M, np.cos(tt)), np.full(M, wd / 6), np.full(M, float(wd >= 5)), np.full(M, float(onset))], 0)
    X = np.concatenate([vr.T, qn.T, oc.T, ok.T.astype(float), qf[:, hs].T, qf[:, fs].T, pv[:, np.r_[hs[-3:], fs]].T, st, glob], 0)
    return X.astype(np.float16)

def build_training(panels, per_day, rng, blank=0.15):
    """Returns dict of arrays: X (N,C,MMAX) f16, Y (N,6,MMAX) f16, E (N,6,MMAX) bool, cond, part, panel_idx, M."""
    out = {k: [] for k in ("X", "Y", "E", "cond", "part", "M")}
    for p in panels:
        net = network(p); li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); vc = net.v_cut.to_numpy()[:, None]
        st_stats, days = v11.profile_stats(p, net); stf = static_feats(net)
        excl = set(pd.read_csv(DATA / "task2" / p / "train" / "window_index.csv").date.astype(str)); dates = sorted(days); hold = set(d for d in dates[-24:] if d not in excl)
        for f in day_files(p, "train", masked=False):
            d = date_of(f)
            if d in excl or d not in days: continue
            prof = v11.profile_from(st_stats, days[d]); g = day_grid(f, li, M, with_occ=True); wd = weekday_of(d)
            V = g["v"]; obs = np.isfinite(V); Lraw = (V <= vc) & obs; L = (v.t2.denoise(V) <= vc) & obs; anyq = Lraw.any(0)
            cands = np.arange(HV - 1, T - H)
            hist_any = np.array([anyq[t - (HV - 1):t].any() for t in cands]); fut_any = np.array([anyq[t + 1:t + H + 1].any() for t in cands])
            early = np.array([anyq[t:t + H].any() for t in cands]); last = np.array([anyq[t + H] for t in cands])
            groups = {0: cands[anyq[cands - 1] & fut_any], 2: cands[~hist_any & ~early & last]}   # ongoing, onset-at-T+30
            k = per_day if d not in hold else 1
            for cond, cc in groups.items():
                if len(cc) == 0: continue
                org = []
                for t0 in np.sort(cc):
                    if not org or t0 - org[-1] >= 72: org.append(int(t0))
                rnd = []
                for t0 in rng.permutation(cc):
                    if all(abs(t0 - q) >= 36 for q in rnd + org): rnd.append(int(t0))
                    if len(rnd) >= max(1, k - 1): break
                for t0 in org[:k] + rnd:
                    sl = slice(t0 - HV + 1, t0 + 1); Hv = V[:, sl].copy(); Hq = g["q"][:, sl].copy(); Ho = g["occ"][:, sl].copy()
                    dr = rng.random(M) < rng.choice([0.2, 0.3, 0.5]); Hv[dr, -1] = np.nan; Hq[dr, -1] = np.nan; Ho[dr, -1] = np.nan
                    b = rng.random(M) < blank; Hv[b] = np.nan; Hq[b] = np.nan; Ho[b] = np.nan
                    X = make_input(Hv, Hq, Ho, net, prof, t0, wd, cond == 2, stf)
                    y = L[:, t0 + 1:t0 + H + 1].T; e = (obs & g["elig"])[:, t0 + 1:t0 + H + 1].T.copy()
                    if cond == 2: e[:H - 1] = False                      # onset: score T+30 only
                    Xp = np.zeros((X.shape[0], MMAX), np.float16); Xp[:, :M] = X
                    Yp = np.zeros((H, MMAX), np.float16); Yp[:, :M] = y; Ep = np.zeros((H, MMAX), bool); Ep[:, :M] = e
                    out["X"].append(Xp); out["Y"].append(Yp); out["E"].append(Ep); out["cond"].append(cond); out["part"].append(int(d in hold)); out["M"].append(M)
        print(f"{p}: {len(out['X'])} windows", flush=True)
    return {k: np.stack(a) if k in ("X", "Y", "E") else np.array(a) for k, a in out.items()}

def window_inputs(split, panels):
    """Inputs for every released window (train windows use LOO profiles)."""
    rows = []
    for p in panels:
        net = network(p); li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); st_stats, days = v11.profile_stats(p, net); full = v11.profile_from(st_stats); stf = static_feats(net)
        wi = pd.read_csv(DATA / "task2" / p / split / "window_index.csv"); hist = pd.read_parquet(DATA / "task2" / p / split / "window_history.parquet")
        mf = {date_of(f): f for f in day_files(p, split, masked=True)}
        for w in wi.itertuples(index=False):
            t0 = int(slots_of(pd.Series([w.forecast_origin]))[0]); onset = w.condition == "queue_onset"
            prof = v11.profile_from(st_stats, days[w.date]) if (split == "train" and w.date in days) else full
            Hv, Hq, Ho = v.load_window(hist, w, li, M, day_grid(mf[w.date], li, M, with_occ=True) if w.date in mf else None, t0)
            X = make_input(Hv, Hq, Ho, net, prof, t0, weekday_of(w.date), onset, stf); Xp = np.zeros((X.shape[0], MMAX), np.float16); Xp[:, :M] = X
            rows.append(dict(split=split, panel=p, window_id=w.window_id, condition=w.condition, X=Xp, M=M, links=net.link_id.to_numpy()))
    return rows

def train_and_predict(D, W, out, epochs, device, seed, width=128):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(seed); np.random.seed(seed)
    C = D["X"].shape[1]
    class Block(nn.Module):
        def __init__(s, c, dil): super().__init__(); s.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil); s.c2 = nn.Conv1d(c, c, 1); s.n = nn.GroupNorm(8, c)
        def forward(s, x): return x + s.c2(F.gelu(s.n(s.c1(x))))
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Conv1d(C, width, 1); s.blocks = nn.Sequential(*[Block(width, d) for d in (1, 2, 4, 8, 16, 32, 1, 2, 4, 8)]); s.head = nn.Conv1d(width, H, 1)
        def forward(s, x): return s.head(s.blocks(F.gelu(s.inp(x))))
    net = Net().to(device); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    tr = np.nonzero(D["part"] == 0)[0]; va = np.nonzero(D["part"] == 1)[0]; bs = 32; steps = epochs * (len(tr) // bs + 1)
    assert len(tr) > 0 and len(va) > 0, f"need train and validation windows (train {len(tr)}, valid {len(va)})"
    print(f"train windows {len(tr)}, valid windows {len(va)}", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    def batch(ix):
        x = torch.tensor(D["X"][ix].astype(np.float32), device=device); y = torch.tensor(D["Y"][ix].astype(np.float32), device=device); e = torch.tensor(D["E"][ix], device=device)
        return x, y, e
    def loss_fn(logit, y, e):
        bce = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * e).sum() / e.sum().clamp(min=1)
        p = torch.sigmoid(logit) * e; t = y * e; inter = (p * t).sum((1, 2)); union = (p + t - p * t).sum((1, 2))
        siou = 1 - ((inter + 1) / (union + 1)).mean()
        return bce + 0.5 * siou
    best = (1e9, None)
    for ep in range(epochs):
        net.train(); rng = np.random.default_rng(seed + ep); tl = []
        for i in range(0, len(tr), bs):
            x, y, e = batch(rng.permutation(tr)[i:i + bs] if i == 0 else tr[rng.integers(0, len(tr), bs)])
            loss = loss_fn(net(x), y, e); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl.append(loss.item())
        net.eval(); vl = []
        with torch.no_grad():
            for i in range(0, len(va), 64): x, y, e = batch(va[i:i + 64]); vl.append(loss_fn(net(x), y, e).item())
        v_loss = float(np.mean(vl)) if vl else float(np.mean(tl)); print(f"epoch {ep}: train {np.mean(tl):.4f} valid {v_loss:.4f}", flush=True)
        if v_loss < best[0]: best = (v_loss, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); torch.save(best[1], f"{out}/task2_cnn.pt"); net.eval(); rows = []
    with torch.no_grad():
        for r in W:
            x = torch.tensor(r["X"][None].astype(np.float32), device=device); p = torch.sigmoid(net(x))[0].cpu().numpy()[:, :r["M"]]   # (6, M)
            for h in range(H):
                rows.append(pd.DataFrame({"split": r["split"], "panel": r["panel"], "window_id": r["window_id"], "condition": r["condition"], "h": h + 1, "link_id": r["links"],
                                          "p_general": p[h], "p_general_w": p[h], "p_onsetT30": p[h] if (r["condition"] == "queue_onset" and h == H - 1) else np.nan}))
    pd.concat(rows, ignore_index=True).to_parquet(f"{out}/window_probs.parquet", index=False); return best[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out_nn"); ap.add_argument("--panels", nargs="*", default=v.Q2)
    ap.add_argument("--per-day", type=int, default=3); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--max-days", type=int, default=0, help="smoke-test limit on days per panel (0 = all)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(a.seed)
    if a.max_days:
        orig = v11.profile_stats
        def limited(p, net, *args, **kw):
            st, days = orig(p, net, *args, **kw); keep = sorted(days)[:: max(1, len(days) // a.max_days)]; return st, {d: days[d] for d in keep}
        v11.profile_stats = limited
    D = build_training(a.panels, a.per_day, rng); print(f"training tensors {D['X'].shape} ({time.time()-t0:.0f}s)", flush=True)
    W = sum((window_inputs(s, a.panels) for s in ("train", "validation", "private")), []); print(f"window inputs {len(W)} ({time.time()-t0:.0f}s)", flush=True)
    try:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        print("torch not available: data path verified, skipping training"); return
    print("device", device, flush=True); bl = train_and_predict(D, W, a.out, a.epochs, device, a.seed)
    json.dump({"best_valid_loss": bl, "runtime_s": time.time() - t0, "channels": int(D["X"].shape[1]), "n_windows": int(len(D["X"])), "epochs": a.epochs, "seed": a.seed}, open(f"{a.out}/run_info.json", "w"))
    print("done", f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
