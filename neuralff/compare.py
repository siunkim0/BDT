"""Compare the neural fake factor with the binned FR map, per era.

    python -m neuralff.compare --model neuralff/data/models/nff_qcd.pt \
        --ntuples fakeflavor/data/ntuples_nff/QCD_MuEnriched.parquet \
        --sample QCD_MuEnriched --out neuralff/plots/qcd

Three curves per (era, |eta|) slice:

  binned    Sum w(tight) / Sum w(loose) over ALL rows, in the standard
            (conePt, |eta|) cells. This is the measurement the map encodes.
  official  the same quantity from fr/fakerate_muon_*.json, i.e. the existing
            result, read straight out of the correctionlib file.
  ML        the model's f(x), averaged over the loose population of the cell:
            f_ML(b) = Sum_{i in b} w_i f(x_i) / Sum_{i in b} w_i,
            evaluated on the TEST split only. The band is the weighted standard
            deviation of f(x_i) inside the cell (the paper's Fig. 10/11
            definition), i.e. how much structure the map is averaging over —
            not a statistical error.

If f is the true conditional then Sum_b w f = Sum_b w 1[tight] identically, so
the ML points and the binned points estimate the same number and any difference
is model error plus the (small) difference between the test split and the whole.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_logger, load_yaml
from . import data as D
from .fakefactor import FakeFactor

log = get_logger("neuralff.compare")

ETA_COLORS = ["tab:blue", "tab:orange", "tab:red"]


# ---------------------------------------------------------------- binned maps
def cell_conept(cols: dict, idx: np.ndarray, ptb: np.ndarray,
                overflow: str = "drop") -> np.ndarray:
    """conePt used for cell assignment, with the overflow convention applied.

    'drop' leaves values above the last edge alone, so the `< ptb[-1]` test
    excludes them from every cell — what fr/*.json does. 'clip' folds them into
    the last bin — what fake.cc's histograms do. See binning.overflow in the
    config for the measurement that settles which is which.
    """
    cp = np.asarray(cols["mu_conept"], dtype=np.float64)[idx]
    return np.minimum(cp, ptb[-1] - 0.1) if overflow == "clip" else cp


def binned_fr(cols: dict, idx: np.ndarray, ptb: np.ndarray, etab: np.ndarray,
              overflow: str = "drop"):
    """Sum w(tight)/Sum w(loose) per cell, with binomial-style weighted errors."""
    cp = cell_conept(cols, idx, ptb, overflow)
    ae = np.asarray(cols["mu_abseta"], dtype=np.float64)[idx]
    w = np.asarray(cols["weight"], dtype=np.float64)[idx]
    t = np.asarray(cols["isTight"])[idx].astype(bool)

    nx, ny = len(ptb) - 1, len(etab) - 1
    fr = np.full((nx, ny), np.nan)
    err = np.full((nx, ny), np.nan)
    for ix in range(nx):
        mx = (cp >= ptb[ix]) & (cp < ptb[ix + 1])
        for iy in range(ny):
            m = mx & (ae >= etab[iy]) & (ae < etab[iy + 1])
            L, T = w[m].sum(), w[m & t].sum()
            if L <= 0:
                continue
            p = T / L
            # Weighted binomial: Var(p) = sum w^2 p(1-p) / (sum w)^2.
            fr[ix, iy] = p
            err[ix, iy] = np.sqrt(max(p * (1 - p), 0.) * (w[m] ** 2).sum()) / L
    return fr, err


def official_fr(cfg: dict, sample: str, era: str):
    """The existing binned result from fr/fakerate_muon_*.json."""
    ccfg = cfg["compare"]
    # str() on both sides: an unquoted 2017 in the YAML arrives as an int, so a
    # lookup by the era string misses. compare.py catches KeyError and carries on
    # without the overlay, so this failed *silently* for 2017 and 2018 until
    # transfer.py hit it as a hard error (2026-08-14).
    era_tag = {str(k): str(v) for k, v in ccfg["era_tag"].items()}
    tag = era_tag[str(era)]
    path = ccfg["official_json"].format(tag=tag)
    name = ccfg["official_name"][sample]
    with open(path) as f:
        corrections = json.load(f)["corrections"]
    corr = next(c for c in corrections if c["name"] == name)
    eta_edges, pt_edges = corr["data"]["edges"]
    content = np.asarray(corr["data"]["content"], dtype=np.float64)
    n_pt = len(pt_edges) - 1
    # correctionlib stores absEta-major with ptCorr inner; we want [ipt, ieta].
    return content.reshape(len(eta_edges) - 1, n_pt).T, np.asarray(pt_edges), \
        np.asarray(eta_edges)


def ml_projection(f: np.ndarray, cols: dict, idx: np.ndarray,
                  ptb: np.ndarray, etab: np.ndarray, overflow: str = "drop"):
    """Weighted mean and weighted std of f(x) inside each cell."""
    cp = cell_conept(cols, idx, ptb, overflow)
    ae = np.asarray(cols["mu_abseta"], dtype=np.float64)[idx]
    w = np.asarray(cols["weight"], dtype=np.float64)[idx]

    nx, ny = len(ptb) - 1, len(etab) - 1
    mean = np.full((nx, ny), np.nan)
    spread = np.full((nx, ny), np.nan)
    for ix in range(nx):
        mx = (cp >= ptb[ix]) & (cp < ptb[ix + 1])
        for iy in range(ny):
            m = mx & (ae >= etab[iy]) & (ae < etab[iy + 1])
            sw = w[m].sum()
            if sw <= 0:
                continue
            mu = (w[m] * f[m]).sum() / sw
            var = (w[m] * (f[m] - mu) ** 2).sum() / sw
            mean[ix, iy] = mu
            spread[ix, iy] = np.sqrt(max(var, 0.))
    return mean, spread


# -------------------------------------------------------------------- plotting
def plot_vs_ptcorr(ptb, etab, binned, berr, ml, mlsd, official, title, path):
    fig, axes = plt.subplots(1, len(etab) - 1, figsize=(4.6 * (len(etab) - 1), 4.2),
                             sharey=True)
    centers = 0.5 * (ptb[:-1] + ptb[1:])
    xerr = 0.5 * (ptb[1:] - ptb[:-1])
    for iy, ax in enumerate(np.atleast_1d(axes)):
        col = ETA_COLORS[iy % len(ETA_COLORS)]
        # The spread is drawn ONCE, as the band. Putting it on the markers as
        # well (it was, until 2026-08-12) reads as an error bar and buries the
        # binned points, which are the thing being compared against.
        ax.fill_between(centers, ml[:, iy] - mlsd[:, iy], ml[:, iy] + mlsd[:, iy],
                        color=col, alpha=0.18, lw=0, zorder=1,
                        label=r"ML: $\pm 1$ s.d. of $f$ within the cell")
        ax.errorbar(centers, ml[:, iy], xerr=xerr, fmt="s", ms=5, lw=1.1,
                    color=col, zorder=2,
                    label=r"ML: weighted mean of $f$ in cell")
        if official is not None:
            ax.step(np.append(ptb[:-1], ptb[-1]),
                    np.append(official[:, iy], official[-1, iy]),
                    where="post", color="tab:green", ls="--", lw=1.3, zorder=3,
                    label="official fr/*.json")
        # Last, so it sits on top of everything it is being compared with.
        ax.errorbar(centers, binned[:, iy], yerr=berr[:, iy], xerr=xerr,
                    fmt="o", color="k", ms=4.5, lw=1.2, zorder=4,
                    label="binned FR (this sample)")
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_T^{corr}$ [GeV]")
        ax.set_title(rf"${etab[iy]:g} < |\eta| < {etab[iy+1]:g}$", fontsize=10)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("fake rate $f$")
    np.atleast_1d(axes)[0].set_ylim(0, None)
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path)


def plot_vs_eta(ptb, etab, binned, berr, ml, mlsd, official, title, path):
    n = len(ptb) - 1
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    centers = 0.5 * (etab[:-1] + etab[1:])
    xerr = 0.5 * (etab[1:] - etab[:-1])
    for ix in range(n):
        ax = axes[ix]
        # Same convention as the conePt view: the band is the within-cell
        # spread of f, the marker is its weighted mean.
        ax.fill_between(centers, ml[ix] - mlsd[ix], ml[ix] + mlsd[ix],
                        color="tab:blue", alpha=0.18, lw=0, zorder=1,
                        label=r"ML: $\pm 1$ s.d. in cell")
        ax.errorbar(centers, ml[ix], xerr=xerr, fmt="^", color="tab:blue",
                    ms=5, lw=1.1, zorder=2, label="ML: mean")
        if official is not None:
            ax.errorbar(centers, official[ix], xerr=xerr, fmt="s", mfc="none",
                        color="tab:green", ms=6, lw=1, zorder=3, label="official")
        ax.errorbar(centers, binned[ix], yerr=berr[ix], xerr=xerr, fmt="o",
                    color="k", ms=4.5, lw=1.2, zorder=4, label="binned")
        ax.set_title(rf"$p_T^{{corr}} \in [{ptb[ix]:g}, {ptb[ix+1]:g}]$", fontsize=9)
        ax.set_xlabel(r"$|\eta|$")
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("fake rate $f$")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path)


def plot_maps(ptb, etab, binned, ml, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    ratio = ml / np.where(binned > 0, binned, np.nan)
    for ax, arr, name, kw in [
            (axes[0], binned, "binned", dict(vmin=0, vmax=np.nanmax(binned))),
            (axes[1], ml, "ML (test)", dict(vmin=0, vmax=np.nanmax(binned))),
            (axes[2], ratio, "ML / binned", dict(vmin=0.8, vmax=1.2, cmap="RdBu_r"))]:
        im = ax.pcolormesh(ptb, etab, arr.T, shading="flat", **kw)
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_T^{corr}$ [GeV]")
        ax.set_ylabel(r"$|\eta|$")
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax)
        for ix in range(len(ptb) - 1):
            for iy in range(len(etab) - 1):
                v = arr[ix, iy]
                if np.isfinite(v):
                    ax.text(np.sqrt(ptb[ix] * ptb[ix + 1]),
                            0.5 * (etab[iy] + etab[iy + 1]), f"{v:.3f}",
                            ha="center", va="center", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path)


def plot_ff_surface(ff, cols, idx, title, path):
    """Paper Fig. 7/8: F(x) over (mu_pt, ev_met) and (mu_pt, |eta|) grids.

    Every non-scanned feature is fixed at the weighted median of the sample, so
    the panel shows the model's own structure rather than a population average.
    """
    from .fakefactor import grid_columns
    num = ff.enc.numerical
    cat = ff.enc.categorical
    base = {}
    for c in num:
        base[c] = float(np.median(np.asarray(cols[c], dtype=np.float64)[idx]))
    for c in cat:
        v = np.asarray(cols[c])[idx]
        vals, counts = np.unique(v, return_counts=True)
        base[c] = vals[np.argmax(counts)]

    pt = np.linspace(10, 100, 90)
    scans = []
    if "ev_met" in num:
        scans.append(("ev_met", np.linspace(0, 60, 60), r"$E_T^{miss}$ [GeV]"))
    if "mu_abseta" in num:
        scans.append(("mu_abseta", np.linspace(0, 2.4, 60), r"$|\eta|$"))
    if not scans:
        return

    fig, axes = plt.subplots(1, len(scans), figsize=(6.0 * len(scans), 4.2))
    for ax, (name, yv, ylab) in zip(np.atleast_1d(axes), scans):
        PT, YV = np.meshgrid(pt, yv, indexing="ij")
        g = grid_columns(base, mu_pt=PT.ravel(), **{name: YV.ravel()})
        f = ff.fake_rate(g).reshape(PT.shape)
        im = ax.pcolormesh(pt, yv, f.T, shading="auto", cmap="viridis")
        ax.set_xlabel(r"$p_T^{\mu}$ [GeV]")
        ax.set_ylabel(ylab)
        fig.colorbar(im, ax=ax, label="fake rate $f$")
    fixed = ", ".join(f"{k}={base[k]:.3g}" if isinstance(base[k], float)
                      else f"{k}={base[k]}" for k in base
                      if k not in ("mu_pt",) + tuple(s[0] for s in scans))
    fig.suptitle(f"{title}\nfixed: {fixed}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path)


# ------------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="neuralff/config/nff.yaml")
    p.add_argument("--model", default=None,
                   help="omit to run the binned-vs-official check alone — the "
                        "gate that says the parquet reproduces fr/*.json before "
                        "anything is trained on it")
    p.add_argument("--ntuples", required=True)
    p.add_argument("--sample", required=True,
                   help="key into compare.official_name (QCD_MuEnriched | TTLJ)")
    p.add_argument("--out", default="neuralff/plots/compare")
    p.add_argument("--feature-set", default=None,
                   help="only needed without --model; otherwise the bundle's "
                        "own feature set is used")
    p.add_argument("--no-official", action="store_true",
                   help="skip the fr/*.json overlay (e.g. when the ntuple region "
                        "does not match the one the official map was measured in)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    ptb = np.asarray(cfg["binning"]["ptcorr"], dtype=np.float64)
    etab = np.asarray(cfg["binning"]["abseta"], dtype=np.float64)
    overflow = cfg["binning"].get("overflow", "drop")
    log.info("conePt overflow convention: %s", overflow)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # The model's own feature set decides which columns are read, so a bundle
    # trained on one arm can never be evaluated through the other's columns.
    ff = FakeFactor(args.model) if args.model else None
    fset = args.feature_set or (ff.feature_set if ff else None)
    log.info("Feature set: %s", fset or cfg.get("default_set"))
    cols = D.load(Path(args.ntuples), cfg, fset)
    n_rows = len(cols["isTight"])
    if ff is not None:
        idx_test = ff.splits["test"]
        if idx_test.max() >= n_rows:
            raise ValueError("model splits do not match this ntuple — the "
                             "parquet changed since training")
        log.info("Evaluating the model on %d test rows", len(idx_test))
        f_all = ff.fake_rate({k: v for k, v in cols.items()})
    else:
        # Binned-vs-official only. Every row is its own 'test' set so the
        # per-cell bookkeeping below stays one code path.
        idx_test, f_all = np.arange(n_rows), None
        log.info("No --model: binned vs official only")

    eras = sorted(set(cols["era"].tolist()))
    report = {"model": args.model, "ntuples": args.ntuples,
              "sample": args.sample, "eras": {}}

    test_mask = np.zeros(len(cols["isTight"]), dtype=bool)
    test_mask[idx_test] = True

    for era in eras + ["ALL"]:
        sel = np.ones(len(cols["isTight"]), dtype=bool) if era == "ALL" \
            else (cols["era"] == era)
        idx_all = np.flatnonzero(sel)
        idx_te = np.flatnonzero(sel & test_mask)

        binned, berr = binned_fr(cols, idx_all, ptb, etab, overflow)
        b_te, _ = binned_fr(cols, idx_te, ptb, etab, overflow)
        if ff is None:
            ml = mlsd = np.full_like(binned, np.nan)
        else:
            ml, mlsd = ml_projection(f_all[idx_te], cols, idx_te, ptb, etab,
                                     overflow)

        official = None
        if not args.no_official and era != "ALL":
            try:
                official, opt, oeta = official_fr(cfg, args.sample, era)
                if not (np.allclose(opt, ptb) and np.allclose(oeta, etab)):
                    log.warning("%s: official binning differs from the config; "
                                "not overlaying", era)
                    official = None
            except (KeyError, StopIteration, FileNotFoundError) as e:
                log.warning("No official map for %s / %s (%s)",
                            args.sample, era, e)

        # closure on the test split: sum(w f) must equal sum(w) over tight muons
        w = np.asarray(cols["weight"], dtype=np.float64)[idx_te]
        t = np.asarray(cols["isTight"])[idx_te].astype(float)
        ae = np.asarray(cols["mu_abseta"], dtype=np.float64)[idx_te]
        closure, closure_eta = float("nan"), []
        if ff is not None:
            closure = float((w * f_all[idx_te]).sum() / (w * t).sum())
            for iy in range(len(etab) - 1):
                m = (ae >= etab[iy]) & (ae < etab[iy + 1])
                closure_eta.append(float((w[m] * f_all[idx_te][m]).sum()
                                         / max((w[m] * t[m]).sum(), 1e-30)))

        tag = f"{args.sample}_{era}"
        title = f"{args.sample}, {era} — neural fake factor vs binned FR"
        if ff is not None:
            plot_vs_ptcorr(ptb, etab, binned, berr, ml, mlsd, official, title,
                           outdir / f"FR_vs_ptcorr_{tag}.png")
            plot_vs_eta(ptb, etab, binned, berr, ml, mlsd, official, title,
                        outdir / f"FR_vs_eta_{tag}.png")
            plot_maps(ptb, etab, binned, ml, title, outdir / f"FR_map_{tag}.png")
            if era == "ALL":
                plot_ff_surface(ff, cols, idx_te,
                                f"{args.sample} — learned fake rate $f(x)$",
                                outdir / f"FF_surface_{tag}.png")

        rows = []
        for iy in range(len(etab) - 1):
            for ix in range(len(ptb) - 1):
                rows.append({
                    "ptcorr": [ptb[ix], ptb[ix + 1]],
                    "abseta": [etab[iy], etab[iy + 1]],
                    "binned": _f(binned[ix, iy]), "binned_err": _f(berr[ix, iy]),
                    "binned_test": _f(b_te[ix, iy]),
                    "ml": _f(ml[ix, iy]), "ml_spread": _f(mlsd[ix, iy]),
                    "official": (_f(official[ix, iy]) if official is not None
                                 else None),
                })
        report["eras"][era] = {"closure": closure, "closure_by_eta": closure_eta,
                              "cells": rows}
        log.info("%-12s closure sum(w f)/sum(w tight) = %.4f  (per |eta|: %s)",
                 era, closure, " ".join(f"{c:.4f}" for c in closure_eta))
        print_table(era, ptb, etab, binned, berr, ml, mlsd, official)

    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", outdir / "report.json")


def _f(x):
    return None if not np.isfinite(x) else float(x)


def print_table(era, ptb, etab, binned, berr, ml, mlsd, official):
    print(f"\n=== {era} ===")
    head = f"{'|eta|':>12} {'ptcorr':>12} {'binned':>8} {'stat':>7} " \
           f"{'ML':>8} {'spread':>7} {'ML-bin':>8} {'rel%':>7} {'pull':>6}"
    if official is not None:
        head += f" {'official':>9} {'bin/off':>8}"
    print(head)
    for iy in range(len(etab) - 1):
        for ix in range(len(ptb) - 1):
            b, e, m = binned[ix, iy], berr[ix, iy], ml[ix, iy]
            if not np.isfinite(b):
                continue
            d = m - b
            line = (f"[{etab[iy]:.1f},{etab[iy+1]:.1f}]".rjust(12) +
                    f" [{ptb[ix]:5.0f},{ptb[ix+1]:5.0f}]".rjust(13) +
                    f"{b:9.4f}{e:8.4f}{m:9.4f}{mlsd[ix,iy]:8.4f}"
                    f"{d:+9.4f}{100*d/b:+8.1f}{(d/e if e else np.nan):+7.1f}")
            if official is not None:
                o = official[ix, iy]
                line += f"{o:10.4f}{b/o if o else np.nan:9.3f}"
            print(line)


if __name__ == "__main__":
    main()
