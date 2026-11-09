# Source: anubisyy, "CatBoost-LGBM-NNLS pipeline" (Kaggle notebook, 2026), kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517.
# Redistributed for reproduction with credit; see NOTICE. Changes by Homeshwar Nelakurthi: state_inference.py replaced by the
# notebook's tfb_source version (fixed 0.5/0.5 blend, no state_settings.json).
"""LightGBM residual reconstruction with temporal and spatial context."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import RELEASE, OUT, THREADS
from base_panel import Panel as BasePanel

TRAIN_MONTHS = ("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01")
VALID_MONTH = "2031-02"
SEED = 20261001
RATE = np.array([.20, .30, .50], dtype=np.float32)


class Panel(BasePanel):
    """Add milepost order from static, public training metadata."""

    def __init__(self, name, panel_no):
        super().__init__(name, panel_no)
        sample = sorted((RELEASE / "corridors" / name / "train" / "mainline_states" /
                         "year_month=2030-06").glob("*.parquet"))[0]
        location = pd.read_parquet(sample, columns=["link_id", "milepost"]).drop_duplicates("link_id")
        location["link_id"] = location.link_id.astype(str)
        milepost = location.set_index("link_id").reindex(self.links).milepost.to_numpy(dtype=np.float32)
        if not np.isfinite(milepost).all():
            raise ValueError(f"Missing mileposts for {name}")
        self.milepost = milepost
        self.sorted_links = np.argsort(milepost)
        self.link_rank = np.empty(len(self.links), dtype=np.int32)
        self.link_rank[self.sorted_links] = np.arange(len(self.links))


def all_panels():
    config = json.loads((RELEASE / "config" / "corridors.json").read_text())
    return [x["corridor_id"] for x in config["panels"]]


def month_frame(panel, month, split="train"):
    directory = RELEASE / "corridors" / panel / split / "mainline_states"
    if split != "train":
        directory = RELEASE / "corridors" / panel / split / "mainline_states_masked"
        paths = sorted(directory.glob("**/*.parquet"))
    else:
        paths = sorted(directory.glob(f"year_month={month}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No state data: {panel} {split} {month}")
    cols = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
    frame = pd.concat((pd.read_parquet(path, columns=cols) for path in paths), ignore_index=True)
    frame["link_id"] = frame.link_id.astype(str)
    frame["station_id"] = frame.station_id.astype(str)
    if frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError(f"Duplicate link/time for {panel} {month}")
    return frame


def matrices(frame, panel):
    index = pd.Index(sorted(frame.timestamp.unique()))
    cols = panel.links
    speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)
    flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)
    eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=cols).fillna(0).to_numpy(dtype=bool)
    return index, speed, flow, eligible


def simulated_mask(panel_name, times, speed, flow, eligible):
    dates = pd.Index([text[:10] for text in times])
    result = np.zeros(speed.shape, dtype=bool)
    regimes = np.empty(speed.shape[0], dtype=np.uint8)
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel_name}|{date}".encode(), digest_size=8).digest(), "big")
        regime = seed % 3
        rng = np.random.default_rng(seed)
        regimes[rows] = regime
        result[rows] = (rng.random((len(rows), speed.shape[1])) < RATE[regime]) & eligible[rows] \
                       & np.isfinite(speed[rows]) & np.isfinite(flow[rows])
    return result, regimes


def sampled_targets(mask, times, per_day):
    dates = pd.Index([text[:10] for text in times])
    row_out, col_out = [], []
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        coords = np.argwhere(mask[rows])
        if not len(coords):
            continue
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|sample|{date}|{mask.shape[1]}".encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        chosen = coords[rng.choice(len(coords), size=min(per_day, len(coords)), replace=False)]
        row_out.append(rows[chosen[:, 0]])
        col_out.append(chosen[:, 1])
    return np.concatenate(row_out), np.concatenate(col_out)


def graph_mean(graph, values):
    observed = np.isfinite(values).astype(np.float32)
    count = (graph @ observed.T).T
    mean = (graph @ np.nan_to_num(values, nan=0.0).T).T / np.maximum(count, 1)
    mean = np.where(count > 0, mean, np.nan)
    return mean, count


def compute_features(panel, panel_no, times, observed_speed, observed_flow, rows, cols, regime):
    n_times, n_links = observed_speed.shape
    temporal_speed = pd.DataFrame(observed_speed).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_flow = pd.DataFrame(observed_flow).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_speed = np.where(np.isfinite(temporal_speed), temporal_speed, panel.vf[None, :])
    temporal_flow = np.where(np.isfinite(temporal_flow), temporal_flow, .6 * panel.cap[None, :])

    graph = (panel.incoming + panel.outgoing).tocsr()
    graph.data[:] = 1.0
    spatial_flow, spatial_count = graph_mean(graph, observed_flow)
    flow_base = np.where(spatial_count > 0, .5 * temporal_flow + .5 * spatial_flow, temporal_flow)
    up_speed, up_count = graph_mean(panel.incoming, observed_speed)
    down_speed, down_count = graph_mean(panel.outgoing, observed_speed)
    up_flow, _ = graph_mean(panel.incoming, observed_flow)
    down_flow, _ = graph_mean(panel.outgoing, observed_flow)

    valid = np.isfinite(observed_speed)
    positions = np.arange(n_times, dtype=np.int32)[:, None]
    prev_index = np.maximum.accumulate(np.where(valid, positions, -1), axis=0)
    next_index = np.minimum.accumulate(np.where(valid, positions, n_times)[::-1], axis=0)[::-1]
    p = prev_index[rows, cols]
    n = next_index[rows, cols]
    prev_v = observed_speed[np.maximum(p, 0), cols].copy()
    next_v = observed_speed[np.minimum(n, n_times - 1), cols].copy()
    prev_q = observed_flow[np.maximum(p, 0), cols].copy()
    next_q = observed_flow[np.minimum(n, n_times - 1), cols].copy()
    prev_v[p < 0] = np.nan
    prev_q[p < 0] = np.nan
    next_v[n >= n_times] = np.nan
    next_q[n >= n_times] = np.nan
    prev_gap = np.where(p >= 0, rows - p, 300).astype(np.float32)
    next_gap = np.where(n < n_times, n - rows, 300).astype(np.float32)

    stamp = pd.to_datetime(times, utc=True)
    tod = stamp.hour.to_numpy(dtype=np.float32) + stamp.minute.to_numpy(dtype=np.float32) / 60
    dow = stamp.dayofweek.to_numpy(dtype=np.float32)
    lanes = np.maximum(panel.lanes[cols], 1)
    base_speed = temporal_speed[rows, cols]
    base_flow = flow_base[rows, cols] / lanes
    temporal_flow_lane = temporal_flow[rows, cols] / lanes
    spatial_flow_lane = spatial_flow[rows, cols] / lanes
    up_q = up_flow[rows, cols] / lanes
    down_q = down_flow[rows, cols] / lanes
    regime_vector = regime if len(regime) == len(rows) else regime[rows]
    features = [
        np.full(len(rows), panel_no), cols, regime_vector,
        tod[rows], np.sin(2*np.pi*tod[rows]/24), np.cos(2*np.pi*tod[rows]/24), dow[rows],
        panel.vf[cols], panel.cap[cols] / lanes, lanes, panel.length[cols],
        base_speed, base_flow, temporal_flow_lane, spatial_flow_lane,
        base_speed / panel.vf[cols], base_flow * lanes / panel.cap[cols],
        prev_v, next_v, prev_q / lanes, next_q / lanes, prev_gap, next_gap,
        next_v - prev_v, (next_q - prev_q) / lanes,
        up_speed[rows, cols], down_speed[rows, cols], up_q, down_q,
        up_count[rows, cols], down_count[rows, cols], spatial_count[rows, cols],
        up_speed[rows, cols] - down_speed[rows, cols], up_q - down_q,
    ]
    for offset in (-2, -1, 1, 2):
        ranks = panel.link_rank[cols] + offset
        valid_neighbor = (ranks >= 0) & (ranks < n_links)
        neighbor = panel.sorted_links[np.clip(ranks, 0, n_links-1)]
        neighbor_lanes = np.maximum(panel.lanes[neighbor], 1)
        same_time_v = observed_speed[rows, neighbor]
        same_time_q = observed_flow[rows, neighbor] / neighbor_lanes
        filled_v = temporal_speed[rows, neighbor]
        filled_q = temporal_flow[rows, neighbor] / neighbor_lanes
        features.extend((
            np.where(valid_neighbor, np.abs(panel.milepost[neighbor]-panel.milepost[cols]), np.nan),
            np.where(valid_neighbor, same_time_v, np.nan),
            np.where(valid_neighbor, same_time_q, np.nan),
            np.where(valid_neighbor, filled_v, np.nan),
            np.where(valid_neighbor, filled_q, np.nan),
        ))
    X = np.column_stack(features).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, base_speed, base_flow


def collect():
    train_X, train_v, train_q = [], [], []
    valid_X, valid_v, valid_q = [], [], []
    val_base_v, val_base_q, val_panel, val_regime = [], [], [], []
    for panel_no, name in enumerate(all_panels()):
        panel = Panel(name, panel_no)
        for month in (*TRAIN_MONTHS, VALID_MONTH):
            frame = month_frame(name, month)
            times, speed, flow, eligible = matrices(frame, panel)
            mask, regime = simulated_mask(name, times, speed, flow, eligible)
            observed_speed = np.where(mask, np.nan, speed)
            observed_flow = np.where(mask, np.nan, flow)
            is_val = month == VALID_MONTH
            rows, cols = sampled_targets(mask, times, 1500 if is_val else 1000)
            X, base_v, base_q = compute_features(panel, panel_no, times, observed_speed, observed_flow,
                                                  rows, cols, regime)
            truth_v = speed[rows, cols]
            truth_q = flow[rows, cols] / np.maximum(panel.lanes[cols], 1)
            if is_val:
                valid_X.append(X)
                valid_v.append(truth_v - base_v)
                valid_q.append(truth_q - base_q)
                val_base_v.append(base_v)
                val_base_q.append(base_q)
                val_panel.append(np.full(len(rows), panel_no, dtype=np.uint8))
                val_regime.append(regime[rows])
            else:
                train_X.append(X)
                train_v.append(truth_v - base_v)
                train_q.append(truth_q - base_q)
            print(name, month, len(rows), "baseline speed RMSE", round(float(np.sqrt(np.mean((truth_v-base_v)**2))), 3), flush=True)
    train = (np.vstack(train_X), np.concatenate(train_v), np.concatenate(train_q))
    valid = (np.vstack(valid_X), np.concatenate(valid_v), np.concatenate(valid_q),
             np.concatenate(val_base_v), np.concatenate(val_base_q),
             np.concatenate(val_panel), np.concatenate(val_regime))
    print("train/valid shapes", train[0].shape, valid[0].shape, flush=True)
    return train, valid


def state_score(v_residual, q_residual, panel_codes, regime_codes):
    panel_scores = []
    detail = []
    for panel_no, name in enumerate(all_panels()):
        scores = []
        for regime in range(3):
            mask = (panel_codes == panel_no) & (regime_codes == regime)
            if not mask.any():
                continue
            rv = float(np.sqrt(np.mean(v_residual[mask]**2)))
            rq = float(np.sqrt(np.mean(q_residual[mask]**2)))
            score = .54*max(0, 1-rv/25) + .46*max(0, 1-rq/600)
            scores.append(score)
            detail.append((name, regime, rv, rq, score))
        panel_scores.append(float(np.mean(scores)))
    return float(np.mean(panel_scores)), detail


def train():
    (X, yv, yq), (Xval, yv_val, yq_val, base_v, base_q, panel_code, regime_code) = collect()
    params = {"objective":"regression", "metric":"rmse", "learning_rate":.05,
              "num_leaves":31, "max_depth":9, "min_data_in_leaf":120,
              "feature_fraction":.85, "bagging_fraction":.85, "bagging_freq":1,
              "lambda_l2":5.0, "max_bin":127, "num_threads":THREADS,
              "verbosity":-1, "seed":SEED}
    train_set = lgb.Dataset(X, label=yv)
    valid_set = lgb.Dataset(Xval, label=yv_val, reference=train_set)
    speed_model = lgb.train(params, train_set, num_boost_round=600, valid_sets=[valid_set],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    speed_model.save_model(str(OUT / "state_speed_lgb.txt"))
    del train_set, valid_set
    train_set = lgb.Dataset(X, label=yq)
    valid_set = lgb.Dataset(Xval, label=yq_val, reference=train_set)
    flow_model = lgb.train(params, train_set, num_boost_round=600, valid_sets=[valid_set],
                           callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    flow_model.save_model(str(OUT / "state_flow_lgb.txt"))
    pred_v = speed_model.predict(Xval, num_threads=THREADS)
    pred_q = flow_model.predict(Xval, num_threads=THREADS)
    trials = []
    for av in (0, .25, .5, .75, 1.0):
        for aq in (0, .25, .5, .75, 1.0):
            score, _ = state_score(yv_val-av*pred_v, yq_val-aq*pred_q, panel_code, regime_code)
            trials.append((score, av, aq))
    trials.sort(reverse=True)
    baseline, baseline_detail = state_score(yv_val, yq_val, panel_code, regime_code)
    best_score, av, aq = trials[0]
    print("baseline score", baseline, "best", trials[:10], flush=True)
    for name, regime, rv, rq, score in baseline_detail:
        print("baseline detail", name, regime, rv, rq, score, flush=True)
    settings = {"speed_iteration":speed_model.best_iteration, "flow_iteration":flow_model.best_iteration,
                "speed_scale":av, "flow_scale":aq, "baseline_proxy_score":baseline,
                "best_proxy_score":best_score, "train_rows":len(X), "valid_rows":len(Xval)}
    (OUT / "state_model.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(settings, flush=True)
    return speed_model, flow_model, settings


def predict(speed_model, flow_model, settings):
    outfile = OUT / "state.csv"
    first = True
    for split in ("validation", "private"):
        for panel_no, name in enumerate(all_panels()):
            panel = Panel(name, panel_no)
            frame = month_frame(name, "", split=split)
            times, speed, flow, eligible = matrices(frame, panel)
            target = pd.read_csv(RELEASE / "task1" / name / split / "sample_submission_state.csv",
                                 dtype={"station_id":str, "link_id":str})
            row = times.get_indexer(target.timestamp)
            col = pd.Index(panel.links).get_indexer(target.link_id)
            if (row<0).any() or (col<0).any():
                raise ValueError(f"Unknown target: {name} {split}")
            rates = target.mask_regime.map({"R1":0, "R2":1, "R3":2}).to_numpy(dtype=np.uint8)
            X, base_v, base_q = compute_features(panel, panel_no, times, speed, flow, row, col, rates)
            v = np.clip(base_v + settings["speed_scale"]*speed_model.predict(X, num_threads=THREADS), 1, 180)
            q = np.clip((base_q + settings["flow_scale"]*flow_model.predict(X, num_threads=THREADS))
                        * panel.lanes[col], 0, 150000)
            target["speed_kmh"] = v
            target["flow_vph"] = q
            if not np.isfinite(target[["speed_kmh", "flow_vph"]].to_numpy()).all():
                raise ValueError(f"Nonfinite prediction: {name} {split}")
            target[["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]].to_csv(
                outfile, mode="w" if first else "a", header=first, index=False)
            first = False
            print("predicted", name, split, len(target), flush=True)


