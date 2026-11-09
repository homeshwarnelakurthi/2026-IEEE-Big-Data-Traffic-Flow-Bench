# Source: yeashusemwal, "TrafficFlowBench Trees + U-Net + CNN" (Kaggle notebook, 2026),
# kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87. Redistributed for reproduction with credit; see NOTICE.
"""Task 1 U-Net v2 = task1_unet.py + density-consistency loss (per-lane error of q/v at target cells, weight --dens-w),
for Task 3 (LWR depends on density N = (q/v)*L). Task 1 U-Net inpainting over (links x 288) day grids.
Training uses the UNMASKED train days with fresh random masks every epoch (Task 1 masks are per-cell random draws at
20/30/50%) plus simulated 90-minute Task 2 gaps, so data is effectively unlimited. Loss is metric-shaped:
0.54*(dv_kmh/25)^2 + 0.46*(dq_per_lane/600)^2 on target cells. Profiles are leave-one-day-out for train days.
Outputs: task1_{validation,private}.csv (template rows) and holdout.npz (real masked held-out days) for local scoring."""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import task2_v11 as v11
from common import DATA, T, panels, network, day_files, date_of, regime_of, weekday_of, slots_of, day_grid
MMAX = 352; RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}

def static(net):
    return dict(vf=net.free_speed_kmh.to_numpy(np.float32), cap=net.capacity_vph.to_numpy(np.float32), lanes=net.lanes.to_numpy(np.float32), L=net.length_km.to_numpy(np.float32))

def make_x(V, Q, O, obs, prof, wd, rate, st):
    """(13, M, T) float32 inputs; V/Q/O NaN where missing."""
    M = V.shape[0]; vf = st["vf"][:, None]; cap = st["cap"][:, None]; tt = 2 * np.pi * np.arange(T) / T
    x = np.zeros((13, M, T), np.float32)
    x[0] = np.where(obs, V / vf, 0); x[1] = np.where(obs, Q / cap, 0); x[2] = np.where(obs & np.isfinite(O), O, 0); x[3] = obs
    x[4] = prof["v"][wd] / vf; x[5] = prof["q"][wd] / cap
    x[6] = vf / 120; x[7] = st["lanes"][:, None] / 6; x[8] = np.log1p(st["L"])[:, None]
    x[9] = np.sin(tt)[None]; x[10] = np.cos(tt)[None]; x[11] = rate; x[12] = wd / 6
    return x

def load_panel(p, holdout_n=20):
    net = network(p); li = {l: i for i, l in enumerate(net.link_id)}; M = len(net); st = static(net)
    stats, days = v11.profile_stats(p, net)
    unm = {date_of(f): f for f in day_files(p, "train", masked=False)}; msk = {date_of(f): (f, regime_of(f)) for f in day_files(p, "train", masked=True)}
    md = sorted(msk); hold = set(sorted(md[-holdout_n:])[::4])
    train = []
    for d, f in unm.items():
        if d in hold: continue
        g = day_grid(f, li, M, with_occ=True); wd = weekday_of(d); lp = v11.profile_from(stats, days[d])   # LOO profile cached once per day
        train.append(dict(d=d, wd=wd, V=g["v"].astype(np.float16), Q=g["q"].astype(np.float16), O=g["occ"].astype(np.float16), el=g["elig"] & np.isfinite(g["v"]) & np.isfinite(g["q"]),
                          prof={"v": {wd: lp["v"][wd]}, "q": {wd: lp["q"][wd]}}))
    hold_items = []
    for d in sorted(hold):
        f, r = msk[d]; g = day_grid(f, li, M, with_occ=True); gt = day_grid(unm[d], li, M)
        obs = np.isfinite(g["v"]) & np.isfinite(g["q"]); tgt = ~obs & np.isfinite(gt["v"]) & np.isfinite(gt["q"]) & gt["elig"]
        hold_items.append(dict(d=d, wd=weekday_of(d), r=r, V=g["v"], Q=g["q"], O=g["occ"], obs=obs, tgt=tgt, tv=gt["v"], tq=gt["q"]))
    return dict(p=p, net=net, li=li, M=M, st=st, stats=stats, days=days, full=v11.profile_from(stats), train=train, hold=hold_items)

def synth(item, P, rng):
    """Random Task-1-style mask + optional 90-min gaps on an unmasked train day."""
    V = item["V"].astype(np.float32); Q = item["Q"].astype(np.float32); O = item["O"].astype(np.float32); el = item["el"]
    rate = float(rng.choice([0.2, 0.3, 0.5])); m = el & (rng.random(el.shape) < rate)
    gap = np.zeros(T, bool)
    if rng.random() < 0.6:
        for s in rng.integers(20, 250, rng.integers(1, 3)): gap[s:s + 18] = True
    obs = np.isfinite(V) & np.isfinite(Q) & ~m & ~gap[None]
    tgt = el & ~obs
    prof = item["prof"]
    return make_x(V, Q, O, obs, prof, item["wd"], rate, P["st"]), V, Q, tgt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out_unet"); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--panels", nargs="*", default=None); ap.add_argument("--bs", type=int, default=6); ap.add_argument("--width", type=int, default=32); ap.add_argument("--dens-w", type=float, default=0.05)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(a.seed)
    Ps = [load_panel(p) for p in (a.panels or panels())]; print(f"loaded {sum(len(P['train']) for P in Ps)} train days, {sum(len(P['hold']) for P in Ps)} holdout days ({time.time()-t0:.0f}s)", flush=True)
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
    class UNet(nn.Module):
        def __init__(s, w=a.width):
            super().__init__(); s.e1 = blk(13, w); s.e2 = blk(w, 2 * w); s.e3 = blk(2 * w, 4 * w); s.b = blk(4 * w, 8 * w)
            s.u3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2); s.d3 = blk(8 * w, 4 * w); s.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2); s.d2 = blk(4 * w, 2 * w)
            s.u1 = nn.ConvTranspose2d(2 * w, w, 2, 2); s.d1 = blk(2 * w, w); s.o = nn.Conv2d(w, 2, 1)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); b = s.b(F.max_pool2d(e3, 2))
            d3 = s.d3(torch.cat([s.u3(b), e3], 1)); d2 = s.d2(torch.cat([s.u2(d3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
            base = torch.stack([x[:, 4], x[:, 5]], 1)          # predict a residual on top of the profile
            return base + s.o(d1)
    net = UNet().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    items = [(pi, it) for pi, P in enumerate(Ps) for it in P["train"]]; steps_per = len(items) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * steps_per)
    def to_batch(samples):
        X = np.zeros((len(samples), 13, MMAX, T), np.float32); Y = np.zeros((len(samples), 2, MMAX, T), np.float32); W = np.zeros((len(samples), 2, MMAX, T), np.float32)
        for k, (P, x, V, Q, tgt) in enumerate(samples):
            M = P["M"]; X[k, :, :M] = x; st = P["st"]
            Y[k, 0, :M] = np.where(tgt, V / st["vf"][:, None], 0); Y[k, 1, :M] = np.where(tgt, Q / st["cap"][:, None], 0)
            W[k, 0, :M] = np.where(tgt, 0.54 * (st["vf"][:, None] / 25) ** 2, 0); W[k, 1, :M] = np.where(tgt, 0.46 * (st["cap"][:, None] / st["lanes"][:, None] / 600) ** 2, 0)
        S = np.zeros((len(samples), 4, MMAX, T), np.float32)   # vf, cap, lanes, target mask for the density term
        for k, (P, x, V, Q, tgt) in enumerate(samples):
            M = P["M"]; st = P["st"]; S[k, 0, :M] = st["vf"][:, None]; S[k, 1, :M] = st["cap"][:, None]; S[k, 2, :M] = st["lanes"][:, None]; S[k, 3, :M] = tgt
        return [torch.tensor(z, device=dev) for z in (X, Y, W, S)]
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    def hold_eval():
        net.eval(); se_v = se_q = n = 0.0
        with torch.no_grad():
            for P in Ps:
                for h in P["hold"]:
                    prof = v11.profile_from(P["stats"], P["days"][h["d"]]); x = make_x(h["V"], h["Q"], h["O"], h["obs"], prof, h["wd"], RATE[h["r"]], P["st"])
                    X = np.zeros((1, 13, MMAX, T), np.float32); X[0, :, :P["M"]] = x; out = net(torch.tensor(X, device=dev))[0].float().cpu().numpy()[:, :P["M"]]
                    pv = out[0] * P["st"]["vf"][:, None]; pq = out[1] * P["st"]["cap"][:, None]; t = h["tgt"]
                    se_v += float(((pv - h["tv"])[t] ** 2).sum()); se_q += float((((pq - h["tq"]) / P["st"]["lanes"][:, None])[t] ** 2).sum()); n += float(t.sum()); h["pv"], h["pq"] = pv, pq
        rv, rq = (se_v / n) ** 0.5, (se_q / n) ** 0.5; return rv, rq, 0.54 * (1 - rv / 25) + 0.46 * (1 - rq / 600)
    best = (-1, None)
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(len(items)); losses = []
        for s in range(steps_per):
            samples = []
            for j in order[s * a.bs:(s + 1) * a.bs]:
                pi, it = items[j]; x, V, Q, tgt = synth(it, Ps[pi], rng); samples.append((Ps[pi], x, V, Q, tgt))
            X, Y, W, S = to_batch(samples)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                out = net(X)
            out = out.float(); loss = ((out - Y) ** 2 * W).sum() / (W > 0).sum().clamp(min=1)
            m = S[:, 3] > 0
            if a.dens_w > 0 and m.any():
                vp = (out[:, 0] * S[:, 0]).clamp(min=5.0); qp = out[:, 1] * S[:, 1]; vt = (Y[:, 0] * S[:, 0]).clamp(min=5.0); qt = Y[:, 1] * S[:, 1]
                dk = (qp / vp - qt / vt) / S[:, 2].clamp(min=1)          # per-lane density error (veh/km/lane)
                loss = loss + a.dens_w * (dk[m] ** 2).mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
        rv, rq, ss = hold_eval(); print(f"epoch {ep}: loss {np.mean(losses):.5f} | holdout RMSE speed {rv:.3f} flow/lane {rq:.2f} S_state {ss:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ss > best[0]: best = (ss, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); torch.save(best[1], f"{a.out}/task1_unet.pt"); hold_eval()
    np.savez_compressed(f"{a.out}/holdout.npz", **{f"{P['p']}|{h['d']}|{k}": h[k] for P in Ps for h in P["hold"] for k in ("pv", "pq", "tgt")})
    net.eval()
    for split in ("validation", "private"):
        outs = []
        for P in Ps:
            tpl = pd.read_csv(DATA / "task1" / P["p"] / split / "sample_submission_state.csv", dtype=str)
            d_arr = tpl.timestamp.str[:10].to_numpy(); slot = slots_of(tpl.timestamp); lidx = tpl.link_id.map(P["li"]).to_numpy(dtype=int)
            pv = np.zeros(len(tpl), np.float32); pq = np.zeros(len(tpl), np.float32)
            for f in day_files(P["p"], split, masked=True):
                d = date_of(f); rows = np.nonzero(d_arr == d)[0]
                if len(rows) == 0: continue
                g = day_grid(f, P["li"], P["M"], with_occ=True); obs = np.isfinite(g["v"]) & np.isfinite(g["q"])
                x = make_x(g["v"], g["q"], g["occ"], obs, P["full"], weekday_of(d), RATE[regime_of(f)], P["st"]); X = np.zeros((1, 13, MMAX, T), np.float32); X[0, :, :P["M"]] = x
                with torch.no_grad(): out = net(torch.tensor(X, device=dev))[0].float().cpu().numpy()[:, :P["M"]]
                pv[rows] = (out[0] * P["st"]["vf"][:, None])[lidx[rows], slot[rows]]; pq[rows] = (out[1] * P["st"]["cap"][:, None])[lidx[rows], slot[rows]]
            vf = P["st"]["vf"][lidx]; tpl["speed_kmh"] = np.clip(pv, 5.0, vf * 1.08); tpl["flow_vph"] = np.clip(pq, 100.0, None); outs.append(tpl)
        pd.concat(outs, ignore_index=True).to_csv(f"{a.out}/task1_{split}.csv", index=False); print(f"[{split}] written", flush=True)
    json.dump({"best_holdout_S_state": best[0], "runtime_s": time.time() - t0, "epochs": a.epochs, "seed": a.seed, "width": a.width, "dens_w": a.dens_w}, open(f"{a.out}/run_info.json", "w")); print("done", flush=True)

if __name__ == "__main__":
    main()
