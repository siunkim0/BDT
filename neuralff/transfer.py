"""QCD -> TTLJ closure test: neural fake rate against the original binned map.

The main comparison (`compare.py`) asks whether the network reproduces the map on
the sample the map was measured on. That is a consistency check. This asks the
question that decides whether any of it is useful:

    measure the fake rate on QCD, apply it to ttbar, and see which estimator
    predicts ttbar's observed number of tight muons.

Two estimators, both measured on QCD, evaluated on the same TTLJ rows:

    flat    fakerate_muon_QCD_MuEnriched(conePt, |eta|) from fr/*.json, era-matched
            -- literally the deployed object, not a proxy: Gate 1 showed our rows
            reproduce it to four decimals
    neural  sigmoid(q(x)) from nff_qcd_paper5.pt

and two forms of every prediction, both from `fakeflavor.closure.forms`:

    rate    sum_{all loose}           w f
    matrix  sum_{loose and NOT tight}  w F,   F = exp(q)

Both are unbiased when f is the true conditional, so `matrix` is not a second
physics test — it is the *conditioning* test, and it is what bkg.cc computes.
The previous direct-FR attempt scored 1.0007 in rate and 0.9384 in matrix; quoting
rate alone would have hidden that.

    python -m neuralff.transfer --source qcd --target TTLJ
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils import get_logger, load_yaml
from fakeflavor.closure import (FR_CLAMP_HI, FR_CLAMP_LO, forms, logit,
                                matrix_calibration)
from . import data as D
from .compare import cell_conept, official_fr
from .fakefactor import FakeFactor

log = get_logger("neuralff.transfer")

# Observables the closure plots bin in. `ev_awayjet_pt` is not a model input, so
# it rides along via data.load(extra=...).
PLOT_VARS = ["mu_conept", "mu_abseta", "ev_met", "ev_mt", "ev_njet40",
             "ev_awayjet_pt"]
EXTRA_COLUMNS = ["ev_awayjet_pt"]


def flat_fake_rate(cols: dict, cfg: dict, source_sample: str) -> np.ndarray:
    """Look up the source sample's official binned map for every target row.

    Era-matched: each row is evaluated with its own era's map, exactly as the
    binned FR would be applied in an analysis.

    conePt above the last edge has no bin, because the reference map drops the
    overflow (established in the main study). bkg.cc clamps into the last bin, so
    that is what is done here; the caller is told how much weight that touches.
    """
    ptb = np.asarray(cfg["binning"]["ptcorr"], dtype=np.float64)
    etab = np.asarray(cfg["binning"]["abseta"], dtype=np.float64)
    n = len(cols["isTight"])
    f = np.full(n, np.nan)

    # Clip, i.e. bkg.cc's convention, NOT compare.py's `drop`: here the map is
    # being *applied*, not compared against.
    cp = cell_conept(cols, np.arange(n), ptb, overflow="clip")
    ae = np.asarray(cols["mu_abseta"], dtype=np.float64)
    era = np.asarray(cols["era"]).astype(str)

    ix = np.clip(np.searchsorted(ptb, cp, side="right") - 1, 0, len(ptb) - 2)
    iy = np.clip(np.searchsorted(etab, ae, side="right") - 1, 0, len(etab) - 2)

    for e in np.unique(era):
        m = era == e
        table, opt, oeta = official_fr(cfg, source_sample, e)
        if not (np.allclose(opt, ptb) and np.allclose(oeta, etab)):
            raise ValueError(f"official binning for {e} differs from the config")
        f[m] = table[ix[m], iy[m]]
    if not np.isfinite(f).all():
        raise ValueError("flat map lookup left NaNs")
    return f


def summarize(w, f, is_tight, q=None, truth=None) -> dict:
    d = forms(w, f, is_tight, q=q)
    if truth:
        d["rate_ratio"] = d["rate"] / truth
        d["matrix_ratio"] = d["matrix"] / truth
        d["matrix_clamped_ratio"] = d["matrix_clamped"] / truth
    return d


def differential(w, f, is_tight, q, truth_mask, cols, cfg):
    """The same ratios split by conePt bin, |eta| bin and era.

    One inclusive number can hide compensating errors; this is where a transfer
    failure actually shows up.
    """
    ptb = np.asarray(cfg["binning"]["ptcorr"], dtype=np.float64)
    etab = np.asarray(cfg["binning"]["abseta"], dtype=np.float64)
    cp = np.asarray(cols["mu_conept"], dtype=np.float64)
    ae = np.asarray(cols["mu_abseta"], dtype=np.float64)
    era = np.asarray(cols["era"]).astype(str)

    out = {"ptcorr": [], "abseta": [], "era": []}
    for key, edges, v in (("ptcorr", ptb, np.minimum(cp, ptb[-1] - 0.1)),
                          ("abseta", etab, ae)):
        for i in range(len(edges) - 1):
            m = (v >= edges[i]) & (v < edges[i + 1])
            if not m.any():
                continue
            t = float((w[m] * truth_mask[m]).sum())
            if t <= 0:
                continue
            d = summarize(w[m], f[m], is_tight[m], q=q[m], truth=t)
            d["edges"] = [float(edges[i]), float(edges[i + 1])]
            d["truth"] = t
            out[key].append(d)
    for e in sorted(set(era.tolist())):
        m = era == e
        t = float((w[m] * truth_mask[m]).sum())
        if t <= 0:
            continue
        d = summarize(w[m], f[m], is_tight[m], q=q[m], truth=t)
        d["era"] = e
        d["truth"] = t
        out["era"].append(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="neuralff/config/nff.yaml")
    p.add_argument("--source", default="qcd",
                   help="where the fake rate was measured (model/map stem)")
    p.add_argument("--source-sample", default="QCD_MuEnriched",
                   help="key into compare.official_name for the binned map")
    p.add_argument("--target", default="TTLJ",
                   help="sample the fake rate is applied to")
    p.add_argument("--ntuples", default=None,
                   help="defaults to fakeflavor/data/ntuples_nff/<target>.parquet")
    p.add_argument("--arm", default="paper5")
    p.add_argument("--models", default="neuralff/data/models")
    p.add_argument("--out", default="neuralff/plots/transfer")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ntuples = Path(args.ntuples or
                   f"fakeflavor/data/ntuples_nff/{args.target}.parquet")

    model = Path(args.models) / f"nff_{args.source}_{args.arm}.pt"
    ff = FakeFactor(model)
    cols = D.load(ntuples, cfg, ff.feature_set, extra=EXTRA_COLUMNS)

    w = np.asarray(cols["weight"], dtype=np.float64)
    is_tight = np.asarray(cols["isTight"]).astype(bool)
    truth = float((w * is_tight).sum())
    log.info("Target %s: %d rows, truth N(tight) = %.1f (sum w)",
             args.target, len(w), truth)

    # How much of the target sits above the map's last edge.
    ptb = np.asarray(cfg["binning"]["ptcorr"], dtype=np.float64)
    over = np.asarray(cols["mu_conept"], dtype=np.float64) >= ptb[-1]
    log.info("conePt >= %g: %d rows (%.3f%% of weight) — clamped into the last "
             "bin for `flat`, handled natively by `neural`",
             ptb[-1], int(over.sum()), 100 * w[over].sum() / w.sum())

    results = {"source": args.source, "target": args.target, "arm": args.arm,
               "model": str(model), "ntuples": str(ntuples),
               "n_rows": int(len(w)), "truth": truth,
               "overflow_weight_frac": float(w[over].sum() / w.sum()),
               "estimators": {}, "differential": {}}

    # ---- flat: the original binned map, measured on the source sample --------
    f_flat = flat_fake_rate(cols, cfg, args.source_sample)
    q_flat = logit(f_flat)
    results["estimators"]["flat"] = summarize(w, f_flat, is_tight, q=q_flat,
                                              truth=truth)
    results["differential"]["flat"] = differential(w, f_flat, is_tight, q_flat,
                                                   is_tight, cols, cfg)

    # ---- neural: the same question, unbinned --------------------------------
    # The exact logit, not f: errors in q are multiplicative in F, which is what
    # the matrix sum is linear in.
    q_nn = ff.logit({k: v for k, v in cols.items()}).astype(np.float64)
    f_nn = 1.0 / (1.0 + np.exp(-q_nn))
    results["estimators"]["neural"] = summarize(w, f_nn, is_tight, q=q_nn,
                                                truth=truth)
    results["differential"]["neural"] = differential(w, f_nn, is_tight, q_nn,
                                                     is_tight, cols, cfg)

    results["matrix_calibration_neural"] = [
        dict(zip(["q_lo", "q_hi", "sumw", "f_meas", "f_pred", "F_meas", "F_pred"],
                 map(float, r)))
        for r in matrix_calibration(w, q_nn, is_tight)]

    # ---- report --------------------------------------------------------------
    print(f"\n=== {args.source} -> {args.target}   truth N(tight) = {truth:.1f} ===")
    print(f"{'estimator':>10} {'rate':>12} {'rate/truth':>11} {'matrix':>12} "
          f"{'matrix/truth':>13} {'clamped/truth':>14} {'n_clamp':>8} {'top1%':>7}")
    for name in ("flat", "neural"):
        d = results["estimators"][name]
        print(f"{name:>10} {d['rate']:12.1f} {d['rate_ratio']:11.4f} "
              f"{d['matrix']:12.1f} {d['matrix_ratio']:13.4f} "
              f"{d['matrix_clamped_ratio']:14.4f} {d['n_clamp']:8d} "
              f"{100*d['top1pct']:6.2f}%")

    for key, lab in (("ptcorr", "conePt"), ("abseta", "|eta|"), ("era", "era")):
        print(f"\n-- differential rate/truth and matrix/truth by {lab} --")
        rows = results["differential"]["flat"][key]
        print(f"{'bin':>16} {'flat rate':>10} {'nn rate':>9} "
              f"{'flat matrix':>12} {'nn matrix':>10}")
        for i, rf in enumerate(rows):
            rn = results["differential"]["neural"][key][i]
            tag = rf.get("era") or f"[{rf['edges'][0]:g},{rf['edges'][1]:g}]"
            print(f"{tag:>16} {rf['rate_ratio']:10.4f} {rn['rate_ratio']:9.4f} "
                  f"{rf['matrix_ratio']:12.4f} {rn['matrix_ratio']:10.4f}")

    (out / "transfer.json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out / "transfer.json")


if __name__ == "__main__":
    main()
