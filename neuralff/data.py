"""Parquet -> tensors for the neural fake factor.

One row per loose gen-fake muon in the measurement region, already trigger-
stitched by `fakeflavor/skim.py`. This module only selects columns, encodes the
categorical ones, standardizes the numerical ones and splits.

The split is group-aware on (sample, era, run, lumi, evt), copied from
`fakeflavor/train.py::split_by_event` for the same reason: MC run number is 1 and
event numbers restart per sample, so 12 QCD pT bins x 4 eras collide constantly
on (run, lumi, evt) alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np

from src.utils import get_logger

log = get_logger("neuralff.data")

# Carried through for the comparison, never fed to the model. mu_conept and
# mu_abseta are the axes of the reference map; isTight is the label; weight is
# the analyzer weight for the muon's stitched trigger path.
META_COLUMNS = ["run", "lumi", "evt", "sample", "era",
                "mu_conept", "mu_abseta", "isTight", "weight"]


@dataclass
class Encoding:
    """Everything needed to turn raw columns into model inputs, twice.

    Fitted on the training split only and stored with the model, so that
    `compare.py` and any later consumer reproduce the exact same inputs.
    """
    numerical: list[str]
    categorical: list[str]
    mean: np.ndarray                       # per numerical feature
    std: np.ndarray
    lo: np.ndarray = None                  # winsorization bounds, see fit_encoding
    hi: np.ndarray = None
    cat_levels: dict[str, list] = field(default_factory=dict)   # value -> index

    @property
    def cat_sizes(self) -> list[int]:
        return [len(self.cat_levels[c]) for c in self.categorical]

    def to_json(self) -> dict:
        return {
            "numerical": self.numerical,
            "categorical": self.categorical,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "lo": self.lo.tolist(), "hi": self.hi.tolist(),
            # JSON keys must be strings; values may be ints or strings.
            "cat_levels": {k: [str(v) for v in vs]
                           for k, vs in self.cat_levels.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> "Encoding":
        n = len(d["numerical"])
        return cls(numerical=d["numerical"], categorical=d["categorical"],
                   mean=np.asarray(d["mean"], dtype=np.float64),
                   std=np.asarray(d["std"], dtype=np.float64),
                   lo=np.asarray(d.get("lo", [-np.inf] * n), dtype=np.float64),
                   hi=np.asarray(d.get("hi", [np.inf] * n), dtype=np.float64),
                   cat_levels={k: list(v) for k, v in d["cat_levels"].items()})


def feature_spec(cfg: dict, name: str | None = None) -> tuple[str, dict]:
    """Resolve a named feature set from the config.

    Returns (name, {'numerical': [...], 'categorical': [...]}). The name is
    stored with the model so `compare.py` never has to guess which arm it is
    looking at.
    """
    name = name or cfg.get("default_set")
    sets = cfg["feature_sets"]
    if name not in sets:
        raise KeyError(f"unknown feature set {name!r}; have {sorted(sets)}")
    return name, sets[name]


def load(parquet: Path, cfg: dict, feature_set: str | None = None,
         extra: list[str] | None = None) -> dict:
    """Read the parquet and return raw columns (no encoding yet).

    `extra` pulls additional columns that are not model inputs — observables the
    closure plots bin in, for instance. They ride along untouched.
    """
    _, fcfg = feature_spec(cfg, feature_set)
    cols = list(fcfg["numerical"]) + list(fcfg["categorical"]) + META_COLUMNS
    cols += list(extra or [])
    arr = ak.from_parquet(parquet, columns=sorted(set(cols)))

    out = {}
    for c in sorted(set(cols)):
        if c == "era":                       # keep as string, encoded below
            out[c] = np.asarray(ak.to_numpy(arr[c])).astype(str)
        elif c == "sample":
            out[c] = np.asarray(ak.to_numpy(arr[c])).astype(str)
        else:
            out[c] = ak.to_numpy(arr[c])

    n = len(out["isTight"])
    log.info("Loaded %s: %d rows", parquet, n)
    if n == 0:
        raise ValueError(f"{parquet} has no rows")
    return out


def fit_encoding(cols: dict, cfg: dict, train_mask: np.ndarray,
                 feature_set: str | None = None) -> Encoding:
    """Standardization stats and categorical levels, from the training rows only."""
    _, fcfg = feature_spec(cfg, feature_set)
    num, cat = list(fcfg["numerical"]), list(fcfg["categorical"])

    X = np.column_stack([np.asarray(cols[f], dtype=np.float64) for f in num])
    Xt = X[train_mask]

    # Winsorize before standardizing. Removing the MET/MT cut from the ntuple
    # exposed pathological PUPPI MET values that the cut used to hide: in one
    # QCD pT bin, 0.011% of rows have MET > 1 TeV and the maximum is 5.9e6 GeV.
    # Left alone, a single such row drags the mean by ~13 GeV and inflates the
    # std to ~9000, which collapses every physical MET onto z ~ 0 and destroys
    # the feature. Clipping bounds the influence without dropping the rows, so
    # they still carry their weight in the loss and in the closure sum.
    q = float(cfg.get("features", {}).get("clip_quantile", 0.9999))
    lo = np.quantile(Xt, 1.0 - q, axis=0)
    hi = np.quantile(Xt, q, axis=0)
    for j, f in enumerate(num):
        n_clip = int(((X[:, j] < lo[j]) | (X[:, j] > hi[j])).sum())
        if n_clip:
            log.info("  %-12s clipped to [%.4g, %.4g]  (%d rows, %.4f%%, "
                     "raw max %.4g)", f, lo[j], hi[j], n_clip,
                     100 * n_clip / len(X), X[:, j].max())
    Xt = np.clip(Xt, lo, hi)

    mean = Xt.mean(axis=0)
    # A constant column would divide by zero; leave it alone instead.
    std = Xt.std(axis=0)
    std[std <= 0] = 1.0

    levels: dict[str, list] = {}
    njet_max = int(cfg.get("features", {}).get("njet_max", 6))
    for c in cat:
        v = cols[c]
        if np.issubdtype(np.asarray(v).dtype, np.number):
            # Integer counts: clip the sparse tail into the last category so the
            # embedding table does not carry near-empty rows.
            levels[c] = list(range(njet_max + 1))
        else:
            levels[c] = sorted(set(v[train_mask].tolist()))
    return Encoding(numerical=num, categorical=cat, mean=mean, std=std,
                    lo=lo, hi=hi, cat_levels=levels)


def encode(cols: dict, enc: Encoding) -> tuple[np.ndarray, np.ndarray]:
    """Raw columns -> (standardized numerical, categorical index) arrays."""
    Xn = np.column_stack([np.asarray(cols[f], dtype=np.float64)
                          for f in enc.numerical])
    Xn = (np.clip(Xn, enc.lo, enc.hi) - enc.mean) / enc.std

    idx = []
    for c in enc.categorical:
        lv = enc.cat_levels[c]
        v = np.asarray(cols[c])
        if np.issubdtype(v.dtype, np.number):
            # levels are 0..njet_max; everything above goes in the last one.
            i = np.clip(v.astype(np.int64), 0, len(lv) - 1)
        else:
            lookup = {str(x): k for k, x in enumerate(lv)}
            # An era absent from training (cannot happen in the current setup,
            # but a silent 0 would be a wrong answer, not a crash) -> last index.
            i = np.array([lookup.get(str(x), len(lv) - 1) for x in v],
                         dtype=np.int64)
        idx.append(i)
    Xc = np.column_stack(idx) if idx else np.zeros((len(Xn), 0), dtype=np.int64)
    return Xn.astype(np.float32), Xc.astype(np.int64)


def split_by_event(cols: dict, frac: float, seed: int,
                   sel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware split: returns (keep, held-out) boolean masks over `sel` rows.

    `sel` is an index array into the full table; the returned masks are over the
    same rows in the same order.
    """
    take = (lambda a: a) if sel is None else (lambda a: a[sel])
    _, sample_code = np.unique(take(cols["sample"]), return_inverse=True)
    _, era_code = np.unique(take(cols["era"]), return_inverse=True)
    key = np.stack([sample_code, era_code, take(cols["run"]),
                    take(cols["lumi"]), take(cols["evt"])], axis=1)
    _, group = np.unique(key, axis=0, return_inverse=True)

    rng = np.random.default_rng(seed)
    n_groups = int(group.max()) + 1
    perm = rng.permutation(n_groups)
    held = np.zeros(n_groups, dtype=bool)
    held[perm[:int(round(frac * n_groups))]] = True
    is_held = held[group]
    return ~is_held, is_held


def make_splits(cols: dict, cfg: dict) -> dict[str, np.ndarray]:
    """50 / 25 / 25 train / val / test index arrays (the paper's split)."""
    tcfg = cfg["train"]
    seed = int(tcfg["seed"])
    n = len(cols["isTight"])
    all_idx = np.arange(n)

    keep, test = split_by_event(cols, float(tcfg["test_size"]), seed)
    idx_te = all_idx[test]
    idx_tv = all_idx[keep]
    tr, va = split_by_event(cols, float(tcfg["val_size"]), seed + 1, sel=idx_tv)
    idx_tr, idx_va = idx_tv[tr], idx_tv[va]

    log.info("Split: train=%d (%.0f%%) val=%d (%.0f%%) test=%d (%.0f%%)",
             len(idx_tr), 100 * len(idx_tr) / n, len(idx_va),
             100 * len(idx_va) / n, len(idx_te), 100 * len(idx_te) / n)
    return {"train": idx_tr, "val": idx_va, "test": idx_te}
