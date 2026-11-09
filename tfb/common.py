"""Shared paths, panel metadata and the dense (day, slot, link) array cache."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]  # defaults: <repo>/data/kaggle_public and <repo>/cache
RELEASE = Path(os.environ.get("TFB_DATA_ROOT", _REPO / "data" / "kaggle_public"))
CACHE = Path(os.environ.get("TFB_CACHE", _REPO / "cache"))
SLOTS = 288
SPLITS = ("train", "validation", "private")
TASK2_EXCLUDED = {"D12_I405_N", "D12_I405_S"}


def panels() -> list[str]:
    cfg = json.loads((RELEASE / "config" / "corridors.json").read_text())
    return [p["corridor_id"] for p in cfg["panels"]]


def family(panel: str) -> str:
    return panel.rsplit("_", 1)[0]


@lru_cache(maxsize=None)
def meta(panel: str) -> pd.DataFrame:
    """One row per sensored link, sorted by milepost (the scorer's order), with FD parameters."""
    net = RELEASE / "corridors" / panel / "network"
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
    first = sorted((RELEASE / "corridors" / panel / "train" / "mainline_states").glob("*/*.parquet"))[0]
    loc = pd.read_parquet(first, columns=["link_id", "station_id", "milepost"]).drop_duplicates("link_id")
    loc["link_id"] = loc.link_id.astype(str)
    m = loc.merge(fd, on="link_id", how="left").sort_values(["milepost", "link_id"]).reset_index(drop=True)
    m["v_cut"] = 0.6 * m.free_speed_kmh
    return m


def day_files(panel: str, split: str, masked: bool) -> list[Path]:
    sub = "mainline_states_masked" if masked else "mainline_states"
    return sorted((RELEASE / "corridors" / panel / split / sub).glob("*/*.parquet"), key=lambda p: p.name)


def build_cache(panel: str, split: str, masked: bool) -> Path:
    """Dense arrays: speed, flow, occupancy (float32, NaN = unavailable), eligible (bool),
    pct (uint8), plus dates and per-day mask regime (masked views only)."""
    tag = "masked" if masked else "full"
    out = CACHE / panel / f"{split}_{tag}.npz"
    if out.exists():
        return out
    m = meta(panel)
    col = {l: i for i, l in enumerate(m.link_id)}
    files = day_files(panel, split, masked)
    D, L = len(files), len(m)
    arr = {k: np.full((D, SLOTS, L), np.nan, np.float32) for k in ("speed", "flow", "occ")}
    elig = np.zeros((D, SLOTS, L), bool)
    pct = np.zeros((D, SLOTS, L), np.uint8)
    dates, regimes = [], []
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=["date", "timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy",
                                        "pct_observed", "is_score_eligible"])
        x["link_id"] = x.link_id.astype(str)
        ts = x.timestamp.astype(str)
        slot = (ts.str[11:13].astype(int) * 60 + ts.str[14:16].astype(int)) // 5
        j = x.link_id.map(col).to_numpy()
        ok = ~pd.isna(j)
        s, j = slot.to_numpy()[ok], j[ok].astype(int)
        arr["speed"][d, s, j] = x.speed_kmh.to_numpy()[ok]
        arr["flow"][d, s, j] = x.flow_vph.to_numpy()[ok]
        arr["occ"][d, s, j] = x.occupancy.to_numpy()[ok]
        elig[d, s, j] = x.is_score_eligible.to_numpy()[ok].astype(bool)
        pct[d, s, j] = x.pct_observed.to_numpy()[ok]
        dates.append(str(x.date.iloc[0]))
        regimes.append(f.parent.name.split("=")[-1] if masked else "")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arr, eligible=elig, pct=pct, dates=np.array(dates), regimes=np.array(regimes))
    return out


def load(panel: str, split: str, masked: bool) -> dict:
    z = np.load(build_cache(panel, split, masked))
    return {k: z[k] for k in z.files}


def slot_of(ts: str) -> int:
    return (int(ts[11:13]) * 60 + int(ts[14:16])) // 5
