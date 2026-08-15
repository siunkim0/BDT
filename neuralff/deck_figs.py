"""Figures for the neural-fake-factor decks.

Everything the deck shows beyond `compare.py`'s own plots. Same discipline as
`fakeflavor/plots/flavor_v3/deck_figs.py`: one script builds every figure, the
numbers come from the model meta.json and the compare report at build time, and
nothing is typed into a slide by hand.

    python -m neuralff.deck_figs --sample QCD_MuEnriched --out neuralff/plots/deck/qcd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_logger, load_yaml
from . import data as D
from .fakefactor import FakeFactor

log = get_logger("neuralff.deck_figs")

SERIF = "Times New Roman"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [SERIF, "Nimbus Roman", "Liberation Serif",
                                     "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

INK, GREY, ACC = "#111111", "#555555", "#1a5276"
C_TIGHT, C_LOOSE = "#1a5276", "#a93226"
ARMS = ("paper5", "paper5_rawpt")
ARM_LABEL = {"paper5": "conePt arm", "paper5_rawpt": "raw-pT arm"}

# Axis labels, so no slide has to spell out a column name.
LABEL = {
    "mu_conept": r"$p_T^{corr}$ [GeV]", "mu_pt": r"$p_T^{\mu}$ [GeV]",
    "mu_abseta": r"$|\eta^{\mu}|$", "ev_met": r"$E_T^{miss}$ [GeV]",
    "ev_mt": r"$M_T$ [GeV]", "ev_njet40": r"$N_{jet}$ ($p_T > 40$ GeV)",
    "era": "era",
}


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", path)


def meta(models: Path, stem: str) -> dict:
    return json.loads((models / f"{stem}.meta.json").read_text())


def report(plots: Path, arm: str, sample_tag: str) -> dict:
    return json.loads((plots / f"{sample_tag}_{arm}" / "report.json").read_text())


# --------------------------------------------------------------------- tables
def _table(ax, rows, header, widths, title=None):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, color=INK, pad=14)
    t = ax.table(cellText=rows, colLabels=header, colWidths=widths,
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.55)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(ACC)


def fig_arms(models: Path, plots: Path, sample: str, tag: str, out: Path):
    """The two arms side by side: AUC, closure, and agreement with the map."""
    rows = []
    for arm in ARMS:
        M = meta(models, f"nff_{tag}_{arm}")
        R = report(plots, arm, tag)["eras"]["ALL"]
        cells = [c for c in R["cells"] if c["binned"] and c["ml"]]
        rel = np.array([100 * (c["ml"] - c["binned"]) / c["binned"] for c in cells])
        rows.append([ARM_LABEL[arm],
                     str(len(M["features"]["numerical"]) +
                         len(M["features"]["categorical"])),
                     f"{M['test_auc']:.4f}", f"{R['closure']:.4f}",
                     f"{np.median(np.abs(rel)):.2f}%",
                     f"{np.max(np.abs(rel)):.1f}%",
                     f"{int(np.sum(np.abs(rel) < 5))} / {len(rel)}"])
    fig, ax = plt.subplots(figsize=(11.4, 2.9))
    _table(ax, rows,
           ["arm", "inputs", "test AUC", "closure", "median |ML-binned|",
            "worst", "cells within 5%"],
           [0.17, 0.09, 0.12, 0.12, 0.20, 0.12, 0.18])
    save(fig, out / "A_arms.png")


def fig_training_table(models: Path, tag: str, out: Path):
    rows = []
    for arm in ARMS:
        M = meta(models, f"nff_{tag}_{arm}")
        t = M["train_config"]
        rows.append([ARM_LABEL[arm], f"{M['n_rows']:,}",
                     f"{M['n_parameters']:,}", str(M["best_epoch"]),
                     f"{M['val_loss']:.5f}", f"{M['test_loss']:.5f}",
                     f"{t['lr']:g} / {t['batch_size']}"])
    fig, ax = plt.subplots(figsize=(11.4, 2.9))
    _table(ax, rows,
           ["arm", "loose muons", "parameters", "best epoch", "val loss",
            "test loss", "lr / batch"],
           [0.17, 0.15, 0.14, 0.12, 0.13, 0.13, 0.16])
    save(fig, out / "T_training_table.png")


def fig_history(models: Path, tag: str, arm: str, out: Path):
    M = meta(models, f"nff_{tag}_{arm}")
    h = M["history"]
    ep = [r["epoch"] for r in h]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
    a1.plot(ep, [r["train_loss"] for r in h], color=C_TIGHT, lw=1.4, label="train")
    a1.plot(ep, [r["val_loss"] for r in h], color=C_LOOSE, lw=1.4, label="validation")
    a1.axvline(M["best_epoch"], color=GREY, ls=":", lw=1.2,
               label=f"best epoch {M['best_epoch']}")
    a1.set_ylabel("weighted binary cross-entropy")
    a1.legend(frameon=False)
    a1.grid(alpha=0.3)
    a2.plot(ep, [r["lr"] for r in h], color=GREY, lw=1.4)
    a2.set_yscale("log")
    a2.set_ylabel("learning rate")
    a2.set_xlabel("epoch")
    a2.grid(alpha=0.3)
    save(fig, out / f"H_history_{arm}.png")


# ------------------------------------------------------------ model behaviour
def fig_roc_scores(ff: FakeFactor, cols: dict, idx: np.ndarray, arm: str,
                   out: Path):
    """ROC and the f(x) distribution split by tight / loose-not-tight."""
    from sklearn.metrics import roc_curve, roc_auc_score
    f = ff.fake_rate({k: v for k, v in cols.items()})[idx]
    y = np.asarray(cols["isTight"])[idx].astype(int)
    w = np.abs(np.asarray(cols["weight"], dtype=np.float64)[idx])

    fpr, tpr, _ = roc_curve(y, f, sample_weight=w)
    auc = roc_auc_score(y, f, sample_weight=w)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot(fpr, tpr, color=ACC, lw=1.6, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    ax.set_xlabel("loose-not-tight efficiency")
    ax.set_ylabel("tight efficiency")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, out / f"R_roc_{arm}.png")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    b = np.linspace(0, 1, 60)
    for m, c, lab in [(y == 1, C_TIGHT, "tight"),
                      (y == 0, C_LOOSE, "loose, not tight")]:
        ax.hist(f[m], bins=b, weights=w[m] / w[m].sum(), histtype="step",
                color=c, lw=1.6, label=lab)
    ax.set_xlabel("predicted fake rate $f(x)$")
    ax.set_ylabel("fraction of weight / bin")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    save(fig, out / f"S_scores_{arm}.png")


def fig_calibration(ff: FakeFactor, cols: dict, idx: np.ndarray, arm: str,
                    out: Path, nbins: int = 12):
    """Measured tight fraction vs predicted f, in weighted quantile bins.

    This is the analogue of a confusion matrix for a model whose output is used
    as a rate rather than a decision: what matters is not whether the ranking is
    right but whether the number is right, everywhere on its range.
    """
    f = ff.fake_rate({k: v for k, v in cols.items()})[idx]
    y = np.asarray(cols["isTight"])[idx].astype(float)
    w = np.asarray(cols["weight"], dtype=np.float64)[idx]

    o = np.argsort(f)
    f, y, w = f[o], y[o], w[o]
    edges = np.searchsorted(np.cumsum(w), np.linspace(0, w.sum(), nbins + 1)[1:-1])
    pred, meas, err = [], [], []
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(f)]):
        if hi <= lo:
            continue
        ww = w[lo:hi]
        p = (ww * f[lo:hi]).sum() / ww.sum()
        m = (ww * y[lo:hi]).sum() / ww.sum()
        pred.append(p)
        meas.append(m)
        err.append(np.sqrt(max(m * (1 - m), 0) * (ww ** 2).sum()) / ww.sum())

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 6.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
    lim = [0, max(max(pred), max(meas)) * 1.08]
    a1.plot(lim, lim, color=GREY, ls="--", lw=1)
    a1.errorbar(pred, meas, yerr=err, fmt="o", color=ACC, ms=5, lw=1.2)
    a1.set_ylabel("measured tight fraction")
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.grid(alpha=0.3)
    a2.axhline(1, color=GREY, ls="--", lw=1)
    a2.errorbar(pred, np.array(meas) / np.array(pred),
                yerr=np.array(err) / np.array(pred), fmt="o", color=ACC, ms=4, lw=1)
    a2.set_ylim(0.8, 1.2)
    a2.set_ylabel("meas / pred")
    a2.set_xlabel("predicted fake rate $f(x)$  (weighted quantile bins)")
    a2.grid(alpha=0.3)
    save(fig, out / f"K_calibration_{arm}.png")


def fig_importance(ff: FakeFactor, cols: dict, idx: np.ndarray, arm: str,
                   out: Path, seed: int = 42):
    """Permutation importance — a network has no gain, so shuffle instead.

    Each input is shuffled in turn and the rise in the weighted cross-entropy is
    recorded. This is the honest analogue of the BDT decks' gain ranking: it
    says what the fitted model actually leans on.
    """
    rng = np.random.default_rng(seed)
    base_cols = {k: np.asarray(v)[idx] for k, v in cols.items()}
    y = base_cols["isTight"].astype(np.float64)
    w = np.abs(base_cols["weight"].astype(np.float64))

    def logloss(c):
        f = np.clip(ff.fake_rate(c), 1e-9, 1 - 1e-9)
        return float((-w * (y * np.log(f) + (1 - y) * np.log(1 - f))).sum() / w.sum())

    base = logloss(base_cols)
    feats = list(ff.enc.numerical) + list(ff.enc.categorical)
    imp = []
    for f_ in feats:
        c = dict(base_cols)
        c[f_] = rng.permutation(c[f_])
        imp.append(logloss(c) - base)
    o = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.barh([LABEL.get(feats[i], feats[i]) for i in o], [imp[i] for i in o],
            color=ACC, height=0.6)
    ax.set_xlabel("rise in weighted cross-entropy when the input is shuffled")
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate([imp[i] for i in o]):
        ax.text(v, i, f"  {v:+.4f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(imp) * 1.25)
    save(fig, out / f"I_importance_{arm}.png")


# ------------------------------------------------------------------- appendix
def fig_feature_table(ff: FakeFactor, out: Path):
    rows = []
    for f_ in ff.enc.numerical:
        rows.append([f_, "numerical", LABEL.get(f_, ""), "standardized, winsorized"])
    for f_ in ff.enc.categorical:
        n = len(ff.enc.cat_levels[f_])
        rows.append([f_, "categorical", LABEL.get(f_, ""),
                     f"embedded, {n} levels"])
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    _table(ax, rows, ["feature", "kind", "meaning", "preprocessing"],
           [0.22, 0.15, 0.28, 0.35])
    save(fig, out / "D_features.png")


def fig_inputs(ff: FakeFactor, cols: dict, idx: np.ndarray, out: Path):
    """One figure per input: tight vs loose-not-tight, weighted and normalized."""
    y = np.asarray(cols["isTight"])[idx].astype(bool)
    w = np.abs(np.asarray(cols["weight"], dtype=np.float64)[idx])
    made = []
    for f_ in list(ff.enc.numerical) + list(ff.enc.categorical):
        v = np.asarray(cols[f_])[idx]
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        if f_ in ff.enc.categorical and not np.issubdtype(v.dtype, np.number):
            levels = sorted(set(v.tolist()))
            x = np.arange(len(levels))
            for m, c, lab in [(y, C_TIGHT, "tight"), (~y, C_LOOSE, "loose, not tight")]:
                h = np.array([w[m & (v == l)].sum() for l in levels], dtype=float)
                ax.step(x, h / h.sum(), where="mid", color=c, lw=1.6, label=lab)
            ax.set_xticks(x); ax.set_xticklabels(levels)
        else:
            vv = v.astype(float)
            hi = np.quantile(vv, 0.995)
            lo = max(vv.min(), 0.0) if f_ != "mu_abseta" else 0.0
            b = np.linspace(lo, hi if hi > lo else lo + 1, 60)
            for m, c, lab in [(y, C_TIGHT, "tight"), (~y, C_LOOSE, "loose, not tight")]:
                ax.hist(np.clip(vv[m], b[0], b[-1]), bins=b,
                        weights=w[m] / w[m].sum(), histtype="step",
                        color=c, lw=1.6, label=lab)
        ax.set_xlabel(LABEL.get(f_, f_))
        ax.set_ylabel("fraction of weight / bin")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        save(fig, out / f"X_input_{f_}.png")
        made.append(f_)
    return made


# ---------------------------------------------------------------- transfer
# Binning for the closure distributions. Chosen so every bin has weight in both
# samples; the last bin of each is an overflow so the histograms integrate to the
# totals in the transfer table (verification item 4).
CLOSURE_BINS = {
    "mu_conept": np.array([10, 12, 14, 17, 20, 25, 30, 40, 50, 70, 100, 200.]),
    "mu_abseta": np.linspace(0, 2.4, 13),
    "ev_met": np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 130, 170, 400.]),
    "ev_mt": np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 130, 170, 400.]),
    "ev_njet40": np.arange(0.5, 7.5),
    "ev_awayjet_pt": np.array([40, 50, 60, 70, 85, 100, 120, 150, 200, 300, 600.]),
}
CLOSURE_LABEL = {
    "mu_conept": r"$p_T^{corr}(\mu^{fake})$ [GeV]",
    "mu_abseta": r"$|\eta(\mu^{fake})|$",
    "ev_met": r"$E_T^{miss}$ [GeV]", "ev_mt": r"$M_T$ [GeV]",
    "ev_njet40": r"$N_{jet}$ ($p_T > 40$ GeV)",
    "ev_awayjet_pt": r"$p_T$(away jet) [GeV]",
}


def _chi2(obs, exp, eobs, eexp):
    """chi2/ndf and p-value over the bins with any expectation."""
    from scipy.stats import chi2 as chi2dist
    m = (exp > 0) | (obs > 0)
    var = eobs[m] ** 2 + eexp[m] ** 2
    ok = var > 0
    ndf = int(ok.sum())
    if ndf == 0:
        return float("nan"), float("nan"), 0
    c2 = float((((obs[m] - exp[m]) ** 2)[ok] / var[ok]).sum())
    return c2 / ndf, float(chi2dist.sf(c2, ndf)), ndf


def closure_hists(x, w, F, is_tight, edges):
    """Observed = tight muons; Expected = matrix-method sum over fail-side muons.

    Identical definitions to `transfer.py`, binned — so summing either histogram
    reproduces the corresponding number in the transfer table exactly. Values are
    clipped into the end bins so nothing is silently dropped.
    """
    xc = np.clip(x, edges[0], edges[-1] - 1e-9)
    t, f_ = is_tight, ~is_tight
    obs, _ = np.histogram(xc[t], bins=edges, weights=w[t])
    eobs = np.sqrt(np.histogram(xc[t], bins=edges, weights=w[t] ** 2)[0])
    exp, _ = np.histogram(xc[f_], bins=edges, weights=w[f_] * F[f_])
    eexp = np.sqrt(np.histogram(xc[f_], bins=edges,
                                weights=(w[f_] * F[f_]) ** 2)[0])
    return obs, eobs, exp, eexp


def fig_closure(cols, w, F, is_tight, title, path,
                lumi="Run 2, 138 fb$^{-1}$ (13 TeV)", ratio_ylim=None):
    """AN-25-154 Fig. 50 anatomy: Expected filled, Observed points, Obs/Exp pad.

    One panel per observable in a 2x3 grid, each with rate and shape chi2/ndf.
    'Rate' keeps the normalization; 'Shape' rescales Expected to Observed first,
    which separates 'the total is wrong' from 'the distribution is wrong'.
    """
    from matplotlib.gridspec import GridSpec
    variables = list(CLOSURE_BINS)
    fig = plt.figure(figsize=(15.0, 9.0))
    # Row 2 is an empty spacer: without it the upper row's x-axis label collides
    # with the lower row's CMS header.
    gs = GridSpec(5, 3, figure=fig, height_ratios=[3, 1.1, 0.75, 3, 1.1],
                  hspace=0.06, wspace=0.26)
    for k, var in enumerate(variables):
        r, c = divmod(k, 3)
        row0 = 0 if r == 0 else 3
        a1 = fig.add_subplot(gs[row0, c])
        a2 = fig.add_subplot(gs[row0 + 1, c], sharex=a1)
        edges = CLOSURE_BINS[var]
        obs, eobs, exp, eexp = closure_hists(
            np.asarray(cols[var], dtype=np.float64), w, F, is_tight, edges)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        wid = np.diff(edges)

        a1.bar(ctr, exp, width=wid, color="#5b8fd0", edgecolor="#2b5c9e",
               linewidth=0.6, label="Expected", zorder=1)
        a1.bar(ctr, 2 * eexp, width=wid, bottom=exp - eexp, color="none",
               hatch="///", edgecolor="#444444", linewidth=0.0,
               label="error, stat", zorder=2)
        a1.errorbar(ctr, obs, yerr=eobs, fmt="o", color="k", ms=3.6, lw=1.0,
                    label="Observed", zorder=3)
        a1.set_ylabel("Weighted muons / bin", fontsize=9)
        a1.tick_params(labelbottom=False, labelsize=8)
        a1.set_ylim(0, max(obs.max(), exp.max()) * 1.45)

        r_c2, r_p, _ = _chi2(obs, exp, eobs, eexp)
        s = obs.sum() / exp.sum() if exp.sum() > 0 else 1.0
        s_c2, s_p, _ = _chi2(obs, exp * s, eobs, eexp * s)
        a1.text(0.04, 0.94, f"Rate: $\\chi^2$/ndf = {r_c2:.2f} (p = {r_p:.2f})\n"
                            f"Shape: $\\chi^2$/ndf = {s_c2:.2f} (p = {s_p:.2f})",
                transform=a1.transAxes, va="top", fontsize=7.5, color=INK)
        if k == 0:
            a1.legend(fontsize=7.5, frameon=False, loc="upper right")
        a1.text(0.0, 1.02, "CMS", transform=a1.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
        a1.text(1.0, 1.02, lumi, transform=a1.transAxes, fontsize=7,
                va="bottom", ha="right")

        ratio = np.divide(obs, exp, out=np.full_like(obs, np.nan), where=exp > 0)
        rerr = np.divide(eobs, exp, out=np.full_like(obs, np.nan), where=exp > 0)
        band = np.divide(eexp, exp, out=np.full_like(obs, np.nan), where=exp > 0)
        a2.axhline(1.0, color="#2b5c9e", lw=1.0)
        a2.bar(ctr, 2 * band, width=wid, bottom=1 - band, color="none",
               hatch="///", edgecolor="#444444", linewidth=0.0)
        a2.errorbar(ctr, ratio, yerr=rerr, fmt="o", color="k", ms=3.2, lw=0.9)
        # The AN uses [0, 2] because its errors are data-sized. Ours are MC-sized
        # and tiny, so that range would hide every deviation. `ratio_ylim` is
        # passed in and shared by both estimators — auto-scaling each one made
        # the two figures silently incomparable side by side.
        if ratio_ylim is not None:
            a2.set_ylim(*ratio_ylim)
        else:
            fin = ratio[np.isfinite(ratio)]
            half = max(0.12, 1.25 * np.max(np.abs(fin - 1.0))) if fin.size else 0.4
            a2.set_ylim(1 - half, 1 + half)
        a2.set_ylabel("Obs / Exp", fontsize=8)
        a2.set_xlabel(CLOSURE_LABEL[var], fontsize=9)
        a2.tick_params(labelsize=8)
        a2.grid(alpha=0.25)
    fig.suptitle(title, fontsize=12, y=0.985)
    fig.subplots_adjust(top=0.92)
    save(fig, path)


def fig_closure_overlay(cols, w, F_by_est, is_tight, title, path):
    """Both estimators' Obs/Exp on one axis per observable — the direct read."""
    variables = list(CLOSURE_BINS)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0))
    colors = {"flat": "#a93226", "neural": "#1a5276"}
    for k, var in enumerate(variables):
        ax = axes.ravel()[k]
        edges = CLOSURE_BINS[var]
        ctr = 0.5 * (edges[:-1] + edges[1:])
        for name, F in F_by_est.items():
            obs, eobs, exp, eexp = closure_hists(
                np.asarray(cols[var], dtype=np.float64), w, F, is_tight, edges)
            ratio = np.divide(obs, exp, out=np.full_like(obs, np.nan), where=exp > 0)
            rerr = np.divide(eobs, exp, out=np.full_like(obs, np.nan), where=exp > 0)
            ax.errorbar(ctr, ratio, yerr=rerr, fmt="o-", ms=3.4, lw=1.0,
                        color=colors.get(name, "k"), label=name)
        ax.axhline(1.0, color="k", lw=0.9, ls="--")
        ax.set_xlabel(CLOSURE_LABEL[var], fontsize=9)
        ax.set_ylabel("Observed / Expected", fontsize=9)
        ax.set_ylim(0.4, 1.3)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=12)
    save(fig, path)


def fig_transfer_table(res: dict, out: Path):
    rows = []
    for name in ("flat", "neural"):
        d = res["estimators"][name]
        rows.append([{"flat": "flat map (original)",
                      "neural": "neural FR"}[name],
                     f"{d['rate']:.0f}", f"{d['rate_ratio']:.4f}",
                     f"{d['matrix']:.0f}", f"{d['matrix_ratio']:.4f}",
                     f"{d['n_clamp']:d}", f"{100*d['top1pct']:.1f}%"])
    fig, ax = plt.subplots(figsize=(11.6, 2.6))
    _table(ax, rows, ["estimator", "rate", "rate / truth", "matrix",
                      "matrix / truth", "n_clamp", "top 1%"],
           [0.22, 0.13, 0.14, 0.13, 0.15, 0.11, 0.12])
    save(fig, out / "A_transfer.png")


def fig_transfer_diff(res: dict, out: Path):
    """Closure ratio vs conePt / |eta| / era, both estimators, both forms."""
    keys = [("ptcorr", r"$p_T^{corr}$ bin"), ("abseta", r"$|\eta|$ bin"),
            ("era", "era")]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    for ax, (key, lab) in zip(axes, keys):
        rows_f = res["differential"]["flat"][key]
        rows_n = res["differential"]["neural"][key]
        x = np.arange(len(rows_f))
        tags = [r.get("era") or f"{r['edges'][0]:g}-{r['edges'][1]:g}"
                for r in rows_f]
        for rows, col, name in ((rows_f, "#a93226", "flat"),
                                (rows_n, "#1a5276", "neural")):
            ax.plot(x, [r["rate_ratio"] for r in rows], "o-", color=col, ms=5,
                    lw=1.2, label=f"{name}, rate")
            ax.plot(x, [r["matrix_ratio"] for r in rows], "s--", color=col,
                    ms=5, lw=1.2, mfc="none", label=f"{name}, matrix")
        ax.axhline(1.0, color="k", lw=1.0, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(lab)
        ax.set_ylabel("predicted / truth")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("QCD-measured fake rate applied to ttbar, differentially",
                 fontsize=12)
    save(fig, out / "B_transfer_diff.png")


def fig_transfer_cells(cols, w, F_by_est, is_tight, cfg, out: Path):
    """Predicted/truth per (conePt, |eta|) cell, one panel per estimator.

    The 1D views can hide a cancellation between cells; this is the joint map.
    Matrix form, i.e. the quantity bkg.cc would actually compute.
    """
    ptb = np.asarray(cfg["binning"]["ptcorr"], dtype=np.float64)
    etab = np.asarray(cfg["binning"]["abseta"], dtype=np.float64)
    cp = np.minimum(np.asarray(cols["mu_conept"], dtype=np.float64), ptb[-1] - 0.1)
    ae = np.asarray(cols["mu_abseta"], dtype=np.float64)
    fail = ~is_tight

    fig, axes = plt.subplots(1, len(F_by_est), figsize=(6.4 * len(F_by_est), 3.9))
    for ax, (name, F) in zip(np.atleast_1d(axes), F_by_est.items()):
        arr = np.full((len(ptb) - 1, len(etab) - 1), np.nan)
        for ix in range(len(ptb) - 1):
            mx = (cp >= ptb[ix]) & (cp < ptb[ix + 1])
            for iy in range(len(etab) - 1):
                m = mx & (ae >= etab[iy]) & (ae < etab[iy + 1])
                truth = (w[m] * is_tight[m]).sum()
                if truth <= 0:
                    continue
                arr[ix, iy] = (w[m & fail] * F[m & fail]).sum() / truth
        im = ax.pcolormesh(ptb, etab, arr.T, shading="flat", cmap="RdBu_r",
                           vmin=0.6, vmax=1.4)
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_T^{corr}$ [GeV]")
        ax.set_ylabel(r"$|\eta|$")
        ax.set_title(f"{name}: predicted / truth (matrix form)", fontsize=10)
        fig.colorbar(im, ax=ax)
        for ix in range(len(ptb) - 1):
            for iy in range(len(etab) - 1):
                if np.isfinite(arr[ix, iy]):
                    ax.text(np.sqrt(ptb[ix] * ptb[ix + 1]),
                            0.5 * (etab[iy] + etab[iy + 1]),
                            f"{arr[ix, iy]:.2f}", ha="center", va="center",
                            fontsize=8)
    save(fig, out / "C_transfer_cells.png")


def fig_matrix_calib(res: dict, out: Path):
    rows = res["matrix_calibration_neural"]
    fp = [r["F_pred"] for r in rows]
    fm = [r["F_meas"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    lim = [0, max(max(fp), max(fm)) * 1.1]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1)
    ax.plot(fp, fm, "o", color=ACC, ms=6)
    ax.set_xlabel(r"predicted fake factor $F = e^{q}$ (weighted quantile bins)")
    ax.set_ylabel(r"measured $F = n/(d-n)$")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(alpha=0.3)
    save(fig, out / "K_matrix_calib.png")


# ------------------------------------------------- why the transfer fails
C_QCD, C_TT, C_NN, C_FL = "#a93226", "#111111", "#1a5276", "#7f8c8d"
FLAV_NAME = {0: "b", 1: "c", 2: "uds", 3: "g", 4: "pileup", -1: "unmatched"}


def _wfr(w, t, m):
    """weighted fake rate in a mask, and its weighted stat error."""
    if w[m].sum() <= 0:
        return np.nan, np.nan
    p = (w[m] * t[m]).sum() / w[m].sum()
    e = np.sqrt(max(p * (1 - p), 0) * (w[m] ** 2).sum()) / w[m].sum()
    return p, e


def fig_trend(Q, T, out: Path, cone_min=50.0):
    """The headline diagnostic: FR vs N_jet and vs MET, QCD against ttbar.

    In QCD the fake rate rises with jet activity and MET — there high MET IS
    jet mismeasurement. In ttbar it does not, because MET is the W neutrino.
    The network, fitted on QCD, follows the QCD curve into a sample where the
    truth goes the other way.
    """
    # M_T first: it is the network's most load-bearing input by a factor six
    # (permutation importance +1.885 against +0.302 for conePt), and it is also
    # the input whose MEANING differs most between the two processes.
    panels = [("ev_mt", np.array([0, 30, 60, 90, 1000.]), r"$M_T$ [GeV]"),
              ("ev_njet40", np.array([0.5, 1.5, 2.5, 3.5, 8.5]),
               r"$N_{jet}$ ($p_T > 40$ GeV)"),
              ("ev_met", np.array([0, 40, 80, 1000.]), r"$E_T^{miss}$ [GeV]")]
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.8))
    for ax, (var, edges, xlab) in zip(axes, panels):
        ctr = 0.5 * (edges[:-1] + np.minimum(edges[1:], edges[:-1] * 2 + 40))
        xs = np.arange(len(edges) - 1)
        for src, w, t, col, lab, mk in (
                (Q, Q["w"], Q["t"], C_QCD, "QCD truth", "o"),
                (T, T["w"], T["t"], C_TT, r"$t\bar{t}$ truth", "s")):
            v = src[var]
            m0 = src["cone"] >= cone_min
            y, e = zip(*[_wfr(w, t, m0 & (v >= lo) & (v < hi))
                         for lo, hi in zip(edges[:-1], edges[1:])])
            ax.errorbar(xs, y, yerr=e, fmt=mk + "-", color=col, ms=6, lw=1.6,
                        label=lab)
        # what each estimator predicts for the ttbar rows
        m0 = T["cone"] >= cone_min
        for pred, col, lab, mk in ((T["f_nn"], C_NN, "neural predicts", "^"),
                                   (T["f_flat"], C_FL, "map predicts", "v")):
            y = [np.average(pred[m0 & (T[var] >= lo) & (T[var] < hi)],
                            weights=T["w"][m0 & (T[var] >= lo) & (T[var] < hi)])
                 for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(xs, y, mk + "--", color=col, ms=6, lw=1.4, label=lab)
        ax.set_xticks(xs)
        if var == "ev_njet40":      # integer counts, so label them as integers
            ax.set_xticklabels([f"{int(lo + 0.5)}" if hi < 8
                                else rf"$\geq {int(lo + 0.5)}$"
                                for lo, hi in zip(edges[:-1], edges[1:])])
        else:
            ax.set_xticklabels([f"{lo:g}-{hi:g}" if hi < 900 else f"> {lo:g}"
                                for lo, hi in zip(edges[:-1], edges[1:])])
        ax.set_xlabel(xlab)
        ax.set_ylabel("fake rate")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(rf"Fake rate vs jet activity and MET, conePt > {cone_min:g} GeV"
                 "\n"
                 r"QCD rises, $t\bar{t}$ does not — and the network follows QCD",
                 fontsize=12)
    save(fig, out / "M_trend.png")


def fig_trend_both(Q, T, out: Path, cone_min=50.0):
    """truth / binned / neural, in BOTH samples, against M_T, N_jet and MET.

    Top row is QCD, where the map was measured and the network trained: all
    three curves should lie on top of each other, and they do. Bottom row is
    ttbar, where the same two estimators are applied unchanged. Everything that
    opens up in the bottom row is transfer error — the estimators themselves are
    demonstrably fine in the top row.
    """
    panels = [("ev_mt", np.array([0, 30, 60, 90, 1000.]), r"$M_T$ [GeV]"),
              ("ev_njet40", np.array([0.5, 1.5, 2.5, 3.5, 8.5]),
               r"$N_{jet}$ ($p_T > 40$ GeV)"),
              ("ev_met", np.array([0, 40, 80, 1000.]), r"$E_T^{miss}$ [GeV]")]
    rows = [(Q, "QCD  (measured / trained here)"),
            (T, r"$t\bar{t}$  (the same estimators, applied)")]

    fig, axes = plt.subplots(2, 3, figsize=(16.6, 8.6), sharey="row")
    for r, (src, rowlab) in enumerate(rows):
        m0 = src["cone"] >= cone_min
        for c, (var, edges, xlab) in enumerate(panels):
            ax = axes[r, c]
            xs = np.arange(len(edges) - 1)
            masks = [m0 & (src[var] >= lo) & (src[var] < hi)
                     for lo, hi in zip(edges[:-1], edges[1:])]
            y, e = zip(*[_wfr(src["w"], src["t"], m) for m in masks])
            ax.errorbar(xs, y, yerr=e, fmt="o-", color=C_TT, ms=7, lw=2,
                        label="truth", zorder=3)
            for key, col, lab, mk in ((("f_flat"), C_FL, "binned map", "v"),
                                      (("f_nn"), C_NN, "neural", "^")):
                yy = [np.average(src[key][m], weights=src["w"][m])
                      if src["w"][m].sum() > 0 else np.nan for m in masks]
                ax.plot(xs, yy, mk + "--", color=col, ms=7, lw=1.8, label=lab)
            ax.set_xticks(xs)
            if var == "ev_njet40":
                ax.set_xticklabels([f"{int(lo + 0.5)}" if hi < 8
                                    else rf"$\geq {int(lo + 0.5)}$"
                                    for lo, hi in zip(edges[:-1], edges[1:])])
            else:
                ax.set_xticklabels([f"{lo:g}-{hi:g}" if hi < 900 else f"> {lo:g}"
                                    for lo, hi in zip(edges[:-1], edges[1:])])
            ax.set_xlabel(xlab)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(f"{rowlab}\n\nfake rate", fontsize=10)
    axes[0, 0].legend(fontsize=10, frameon=False)
    fig.suptitle(rf"conePt > {cone_min:g} GeV — the two estimators agree with "
                 r"truth in QCD and split in $t\bar{t}$", fontsize=13)
    save(fig, out / "M_trend_both.png")


def fig_population(Q, T, out: Path):
    """Where each sample lives. The network is fitted on the red distribution
    and asked about the black one."""
    panels = [("ev_mt", np.array([0, 10, 20, 30, 40, 60, 80, 100, 130, 170,
                                  400.]), r"$M_T$ [GeV]  (rank 1 input)", False),
              ("cone", np.array([10, 12, 14, 17, 20, 25, 30, 40, 50, 70, 100,
                                 200.]), r"$p_T^{corr}$ [GeV]", True),
              ("ev_met", np.array([0, 20, 40, 60, 80, 120, 200, 400.]),
               r"$E_T^{miss}$ [GeV]", True),
              ("ev_njet40", np.arange(0.5, 8.5), r"$N_{jet}$ ($p_T>40$)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.4))
    for ax, (var, edges, xlab, logx) in zip(axes, panels):
        for src, col, lab in ((Q, C_QCD, "QCD"), (T, C_TT, r"$t\bar{t}$")):
            v = np.clip(src[var], edges[0], edges[-1] - 1e-9)
            h, _ = np.histogram(v, bins=edges, weights=src["w"])
            ax.step(edges, np.r_[h[0], h] / h.sum(), where="pre", color=col,
                    lw=1.8, label=lab)
        ax.set_xlabel(xlab)
        ax.set_ylabel("fraction of weight")
        if logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=10, frameon=False)
    fig.suptitle("Population overlap — the network is fitted on QCD and applied "
                 r"to $t\bar{t}$", fontsize=12)
    save(fig, out / "M_population.png")


def fig_cone_regions(Q, T, out: Path):
    """predicted/truth by conePt region, with how much weight each sample puts
    there — the failure tracks the regions QCD barely covers."""
    regions = [(10, 20), (20, 50), (50, 100), (100, 1e9)]
    labs = ["10-20", "20-50", "50-100", "> 100"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.6))
    x = np.arange(len(regions))
    qf, tf, rf, rn = [], [], [], []
    for lo, hi in regions:
        mq = (Q["cone"] >= lo) & (Q["cone"] < hi)
        mt = (T["cone"] >= lo) & (T["cone"] < hi)
        qf.append(Q["w"][mq].sum() / Q["w"].sum())
        tf.append(T["w"][mt].sum() / T["w"].sum())
        truth = (T["w"][mt] * T["t"][mt]).sum() / T["w"][mt].sum()
        rf.append(np.average(T["f_flat"][mt], weights=T["w"][mt]) / truth)
        rn.append(np.average(T["f_nn"][mt], weights=T["w"][mt]) / truth)

    a1.bar(x - 0.2, qf, 0.4, color=C_QCD, label="QCD")
    a1.bar(x + 0.2, tf, 0.4, color=C_TT, label=r"$t\bar{t}$")
    a1.set_yscale("log")
    a1.set_ylabel("fraction of the sample's weight")
    a1.set_title("where each sample lives", fontsize=10)
    a1.legend(fontsize=9, frameon=False)
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_xlabel(r"$p_T^{corr}$ [GeV]")
        ax.grid(alpha=0.3, axis="y")
    a2.axhline(1.0, color="k", ls="--", lw=1)
    a2.bar(x - 0.2, rf, 0.4, color=C_FL, label="map")
    a2.bar(x + 0.2, rn, 0.4, color=C_NN, label="neural")
    a2.set_ylabel("predicted / truth")
    a2.set_title(r"accuracy on $t\bar{t}$", fontsize=10)
    a2.legend(fontsize=9, frameon=False)
    for xi, v in zip(x, rn):
        a2.text(xi + 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("The neural miss grows exactly where QCD has no statistics",
                 fontsize=12)
    save(fig, out / "M_cone_regions.png")


def fig_flavour(Q, T, out: Path, cone_min=50.0):
    """Composition differs a lot, per-flavour FR barely — so the map's own error
    is composition, and the extra variables are NOT acting as a flavour proxy."""
    codes = [0, 1, 2, 3]
    x = np.arange(len(codes))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.6))
    for src, off, col, lab in ((Q, -0.2, C_QCD, "QCD"),
                               (T, 0.2, C_TT, r"$t\bar{t}$")):
        m0 = src["cone"] >= cone_min
        fr = [src["w"][m0 & (src["flav"] == c)].sum() / src["w"][m0].sum()
              for c in codes]
        a1.bar(x + off, fr, 0.4, color=col, label=lab)
        vals, errs = zip(*[_wfr(src["w"], src["t"], m0 & (src["flav"] == c))
                           for c in codes])
        a2.bar(x + off, vals, 0.4, yerr=errs, color=col, label=lab)
    a1.set_ylabel("fraction of weight")
    a1.set_title("source-jet flavour composition", fontsize=10)
    a2.set_ylabel("fake rate")
    a2.set_title("fake rate per flavour", fontsize=10)
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels([FLAV_NAME[c] for c in codes])
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle(rf"conePt > {cone_min:g} GeV: composition differs, per-flavour "
                 "fake rate does not", fontsize=12)
    save(fig, out / "M_flavour.png")


def build_transfer(cfg, args, out: Path):
    from .transfer import flat_fake_rate, EXTRA_COLUMNS
    res = json.loads((Path(args.transfer_json)).read_text())
    ff = FakeFactor(Path(args.models) / f"nff_{args.source}_{args.arm}.pt")
    cols = D.load(Path(args.ntuples), cfg, ff.feature_set, extra=EXTRA_COLUMNS)
    w = np.asarray(cols["weight"], dtype=np.float64)
    is_tight = np.asarray(cols["isTight"]).astype(bool)

    f_flat = flat_fake_rate(cols, cfg, args.source_sample)
    F_flat = f_flat / (1.0 - f_flat)
    q_nn = ff.logit({k: v for k, v in cols.items()}).astype(np.float64)
    F_nn = np.exp(q_nn)

    lumi = "Run 2, 138 fb$^{-1}$ (13 TeV)"
    # One range for both figures, from the worse of the two, so the two slides
    # can be read against each other.
    rlim = (0.40, 1.40)
    fig_closure(cols, w, F_flat, is_tight,
                "MC closure — ttbar observed vs matrix-method prediction using the "
                "QCD-measured binned map", out / "D_closure_flat.png", lumi, rlim)
    fig_closure(cols, w, F_nn, is_tight,
                "MC closure — ttbar observed vs matrix-method prediction using the "
                "QCD-trained neural fake rate", out / "D_closure_neural.png",
                lumi, rlim)
    fig_closure_overlay(cols, w, {"flat": F_flat, "neural": F_nn}, is_tight,
                        "Observed / Expected, both estimators",
                        out / "D_closure_ratio_overlay.png")
    fig_transfer_table(res, out)
    fig_transfer_diff(res, out)
    fig_transfer_cells(cols, w, {"flat": F_flat, "neural": F_nn}, is_tight,
                       cfg, out)
    fig_matrix_calib(res, out)

    # --- the "why" figures: they need the SOURCE sample too ------------------
    import awkward as ak
    f_nn = 1.0 / (1.0 + np.exp(-q_nn))
    T = {"w": w, "t": is_tight.astype(float),
         "cone": np.asarray(cols["mu_conept"], dtype=np.float64),
         "ev_njet40": np.asarray(cols["ev_njet40"], dtype=np.float64),
         "ev_met": np.asarray(cols["ev_met"], dtype=np.float64),
         "ev_mt": np.asarray(cols["ev_mt"], dtype=np.float64),
         "f_nn": f_nn, "f_flat": f_flat,
         "flav": ak.to_numpy(ak.from_parquet(args.ntuples,
                                             columns=["flavor"])["flavor"])}

    # The source sample, with the SAME two estimators evaluated on it. That is
    # what makes the comparison honest: in QCD all three curves should agree
    # (the map is measured there and the network trained there), so any split in
    # ttbar is transfer, not a broken estimator.
    qcols = D.load(Path(args.source_ntuples), cfg, ff.feature_set)
    q_flat = flat_fake_rate(qcols, cfg, args.source_sample)
    Q = {"w": np.asarray(qcols["weight"], dtype=np.float64),
         "t": np.asarray(qcols["isTight"]).astype(float),
         "cone": np.asarray(qcols["mu_conept"], dtype=np.float64),
         "ev_njet40": np.asarray(qcols["ev_njet40"], dtype=np.float64),
         "ev_met": np.asarray(qcols["ev_met"], dtype=np.float64),
         "ev_mt": np.asarray(qcols["ev_mt"], dtype=np.float64),
         "f_flat": q_flat,
         "f_nn": ff.fake_rate({k: v for k, v in qcols.items()}),
         "flav": ak.to_numpy(ak.from_parquet(args.source_ntuples,
                                             columns=["flavor"])["flavor"])}
    del qcols

    fig_trend(Q, T, out)
    fig_trend_both(Q, T, out)
    fig_population(Q, T, out)
    fig_cone_regions(Q, T, out)
    fig_flavour(Q, T, out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="neuralff/config/nff.yaml")
    p.add_argument("--sample", required=False)
    p.add_argument("--tag", required=False, help="qcd | ttlj — model/plot stem")
    p.add_argument("--ntuples", required=True)
    p.add_argument("--models", default="neuralff/data/models")
    p.add_argument("--plots", default="neuralff/plots")
    p.add_argument("--out", required=True)
    p.add_argument("--transfer", action="store_true",
                   help="build the QCD->TTLJ transfer figures instead")
    p.add_argument("--transfer-json", default="neuralff/plots/transfer/transfer.json")
    p.add_argument("--source", default="qcd")
    p.add_argument("--source-sample", default="QCD_MuEnriched")
    p.add_argument("--source-ntuples",
                   default="fakeflavor/data/ntuples_nff/QCD_MuEnriched.parquet",
                   help="the sample the fake rate was measured on — needed by "
                        "the diagnostic figures, which compare the two")
    p.add_argument("--arm", default="paper5")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.out)
    models, plots = Path(args.models), Path(args.plots)

    if args.transfer:
        build_transfer(cfg, args, out)
        return

    fig_arms(models, plots, args.sample, args.tag, out)
    fig_training_table(models, args.tag, out)
    for arm in ARMS:
        fig_history(models, args.tag, arm, out)

    for arm in ARMS:
        ff = FakeFactor(models / f"nff_{args.tag}_{arm}.pt")
        cols = D.load(Path(args.ntuples), cfg, ff.feature_set)
        idx = ff.splits["test"]
        fig_roc_scores(ff, cols, idx, arm, out)
        fig_calibration(ff, cols, idx, arm, out)
        fig_importance(ff, cols, idx, arm, out)
        if arm == "paper5":
            fig_feature_table(ff, out)
            fig_inputs(ff, cols, idx, out)
        del cols


if __name__ == "__main__":
    main()
