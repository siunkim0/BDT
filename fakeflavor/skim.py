"""Phase 1: turn the SKNanoAnalyzer `mu` ntuples into training parquet.

Reads /gv0/.../SKNanoOutput/fake/MakeTree/{era}_bdtflav/*.root, applies the
measurement-region selection and the Mu8/Mu17 trigger stitch, and writes one
parquet per (sample, era) plus a combined file.

Run from the BDT project root:

    python -m fakeflavor.skim --config fakeflavor/config/samples.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .features import CLASSES, FEATURE_SETS
from src.utils import get_logger, load_yaml

log = get_logger("fakeflavor.skim")

TREE = "mu"

# Mu8 covers ptcorr [10, 30), Mu17 covers [30, inf). Same stitch point as
# Tools/fakerate_2d.py — each muon is therefore assigned to exactly one path
# and appears exactly once in the output.
STITCH_PTCORR = 30.0

# fake::fillTreeRow writes these when the muon has no matched jet.
FLOAT_SENTINEL = -999.0
INT_SENTINEL = -99


def read_tree(path: Path) -> ak.Array:
    with uproot.open(path) as f:
        if TREE not in f:
            log.warning("%s has no tree '%s' — skipping", path, TREE)
            return None
        return f[TREE].arrays()


def stitch(arr: ak.Array) -> ak.Array:
    """Assign each muon to exactly one trigger path and build `weight`.

    Below the stitch point the FR is measured with Mu8, above it with Mu17.
    A muon that fails the path it is assigned to is dropped — it is not
    'moved' to the other path, because the other path's efficiency does not
    cover that ptcorr range.
    """
    low = ak.to_numpy(arr["mu_conept"]) < STITCH_PTCORR
    pass_mu8 = ak.to_numpy(arr["pass_mu8"])
    pass_mu17 = ak.to_numpy(arr["pass_mu17"])

    keep = np.where(low, pass_mu8, pass_mu17)
    weight = np.where(low, ak.to_numpy(arr["w_mu8"]), ak.to_numpy(arr["w_mu17"]))
    path = np.where(low, 0, 1).astype(np.int32)   # 0 = Mu8, 1 = Mu17

    arr = arr[keep]
    arr["weight"] = weight[keep]
    arr["path"] = path[keep]
    return arr


def derive(arr: ak.Array) -> ak.Array:
    """Add derived columns and convert jet sentinels to NaN.

    xgboost treats NaN as 'missing' and learns a default direction per split,
    which is the right behaviour for muons with no matched jet (~2-3%). Leaving
    -999 in place would instead put them at the far end of every jet axis and
    let the trees carve out a spurious 'flavor' region for them.
    """
    arr["jet_abseta"] = np.abs(ak.to_numpy(arr["jet_eta"]))

    for field in arr.fields:
        if not field.startswith("jet_"):
            continue
        v = ak.to_numpy(arr[field])
        if v.dtype == bool:
            continue
        v = v.astype(np.float32)
        v[(v == FLOAT_SENTINEL) | (v == INT_SENTINEL)] = np.nan
        arr[field] = v
    return arr


def skim_file(path: Path, era: str, sample: str, is_data: bool) -> ak.Array | None:
    arr = read_tree(path)
    if arr is None or len(arr) == 0:
        return None
    n_raw = len(arr)

    # Central measurement region. The ntuple was written with the loosest
    # away-jet cut (30 GeV) so that Central (40) and both away-jet systematics
    # are offline cuts on the same rows.
    arr = arr[ak.to_numpy(arr["ev_hasAway40"])]
    n_central = len(arr)

    arr = stitch(arr)
    n_stitch = len(arr)

    if not is_data:
        # Fakes only: a prompt muon's 'source jet' is not a fake source.
        arr = arr[~ak.to_numpy(arr["isPrompt"])]
        # Keep the four trainable classes; drops pileup (4) and unmatched (-1).
        flav = ak.to_numpy(arr["flavor"])
        arr = arr[np.isin(flav, list(CLASSES))]
    n_final = len(arr)

    arr = derive(arr)
    # str() is not redundant: an unquoted 2017 in the YAML arrives as an int,
    # which would make this column int64 for some eras and string for others —
    # ak.concatenate then builds a union type that parquet refuses to write.
    arr["era"] = ak.Array([str(era)] * n_final)
    arr["sample"] = ak.Array([str(sample)] * n_final)

    log.info("  %-34s raw=%8d  away40=%8d  stitch=%8d  final=%8d",
             path.name, n_raw, n_central, n_stitch, n_final)
    return arr if n_final else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="fakeflavor/config/samples.yaml")
    p.add_argument("--out", default="fakeflavor/data/ntuples")
    p.add_argument("--data", action="store_true",
                   help="also skim DoubleMuon (for data/MC input plots only)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["ntuple_root"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = dict(cfg["mc"])
    if args.data:
        groups.update(cfg["data"])
    data_names = set(cfg["data"])

    all_mc = []
    for sample, spec in groups.items():
        is_data = sample in data_names
        parts = []
        for era in cfg["eras"]:
            era_dir = root / f"{era}_{cfg['tag']}"
            files = sorted(era_dir.glob(spec["glob"]))
            if not files:
                log.warning("No files for %s in %s", sample, era_dir)
                continue
            log.info("%s / %s: %d file(s)", sample, era, len(files))
            for f in files:
                a = skim_file(f, era, sample, is_data)
                if a is not None:
                    parts.append(a)
        if not parts:
            continue
        merged = ak.concatenate(parts)
        fp = out_dir / f"{sample}.parquet"
        ak.to_parquet(merged, fp)
        log.info("Wrote %s: %d rows", fp, len(merged))
        if not is_data:
            all_mc.append(merged)
            report(merged, sample)

    if all_mc:
        combined = ak.concatenate(all_mc)
        fp = out_dir / "train.parquet"
        ak.to_parquet(combined, fp)
        log.info("Wrote %s: %d rows", fp, len(combined))
        report(combined, "ALL MC")
        check_features(combined)


def report(arr: ak.Array, name: str) -> None:
    """Per-class row counts and effective sample sizes.

    N_eff = (Σw)² / Σw² — the count that matters for a weighted fit. QCD pt-bin
    weights span many orders of magnitude, so raw row counts overstate the
    statistics badly.
    """
    flav = ak.to_numpy(arr["flavor"])
    w = ak.to_numpy(arr["weight"]).astype(np.float64)
    log.info("--- %s: %d rows", name, len(arr))
    for code, label in CLASSES.items():
        m = flav == code
        if not m.any():
            log.info("    %-4s %8d rows   (none)", label, 0)
            continue
        sw, sw2 = w[m].sum(), (w[m] ** 2).sum()
        log.info("    %-4s %8d rows   sum(w)=%.4g   N_eff=%.0f",
                 label, int(m.sum()), sw, sw * sw / max(sw2, 1e-30))


def check_features(arr: ak.Array) -> None:
    """Warn about features that are constant or entirely missing.

    A constant column is dead weight and usually means a branch was never
    filled for this era/sample combination (e.g. jet_qgl and jet_puIdDisc are
    Run 2 only — fine here, but the same check will catch it if Run 3 is added).
    """
    fields = set(arr.fields)
    for feat in FEATURE_SETS["full"]:
        if feat not in fields:
            log.error("Feature %s is MISSING from the skim", feat)
            continue
        v = ak.to_numpy(arr[feat]).astype(np.float64)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            log.warning("Feature %s is all-NaN", feat)
        elif np.all(finite == finite[0]):
            log.warning("Feature %s is CONSTANT = %g", feat, finite[0])


if __name__ == "__main__":
    main()
