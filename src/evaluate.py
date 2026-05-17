"""Phase 4: evaluate the trained BDT.

Produces:
  - plots/roc.png                   : ROC curve with AUC (weighted + unweighted)
  - plots/feature_importance.png    : gain-based feature importance
  - plots/overtraining.png          : train vs test BDT score distribution
                                       (Kolmogorov-Smirnov test)
  - plots/m4l_vs_score.png          : m4l vs BDT score 2D + profile,
                                       diagnostic for decorrelation
  - reports/phase4_summary.md       : short summary of the results

Run from project root:

    python -m src.evaluate --model data/models/bdt_v1.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from .features import FEATURES
from .train import load_dataset
from .utils import PLOT_DIR, PROJECT_ROOT, SEED, get_logger, load_yaml

log = get_logger("evaluate")


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray,
             out_path: Path) -> tuple[float, float]:
    """ROC curve with weighted (training metric) and unweighted AUC."""
    fpr_w, tpr_w, _ = roc_curve(y_true, y_score, sample_weight=weights)
    auc_w = roc_auc_score(y_true, y_score, sample_weight=weights)
    fpr_u, tpr_u, _ = roc_curve(y_true, y_score)
    auc_u = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_w, tpr_w, label=f"weighted, AUC = {auc_w:.4f}")
    ax.plot(fpr_u, tpr_u, label=f"unweighted, AUC = {auc_u:.4f}",
            linestyle="--")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    ax.set_xlabel("background efficiency (FPR)")
    ax.set_ylabel("signal efficiency (TPR)")
    ax.set_title("ROC — held-out test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved ROC to %s (AUC w=%.4f, u=%.4f)", out_path, auc_w, auc_u)
    return auc_w, auc_u


def plot_feature_importance(model: xgb.XGBClassifier, feature_names: list[str],
                            out_path: Path) -> pd.Series:
    """Gain-based importance from the booster, sorted high → low."""
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    # Booster keys are like "f0", "f1", ... — map back to feature names.
    importances = {
        feature_names[int(k[1:])]: v for k, v in raw.items()
    }
    series = pd.Series(importances).reindex(feature_names, fill_value=0.0)
    ordered = series.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 0.3 * len(feature_names) + 1.5))
    ax.barh(ordered.index, ordered.values)
    ax.set_xlabel("gain")
    ax.set_title("BDT feature importance (gain)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved feature importance to %s", out_path)
    return ordered.sort_values(ascending=False)


def plot_overtraining(model: xgb.XGBClassifier,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      w_train: np.ndarray, w_test: np.ndarray,
                      out_path: Path) -> dict[str, float]:
    """Stepped score histograms for sig/bkg in train and test, plus KS tests.

    Histograms are weight-normalized (area=1) so train and test shapes are
    directly comparable. KS is unweighted on the predicted scores — scipy's
    ks_2samp doesn't take weights, but the stat is still a useful diagnostic.
    """
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    bins = np.linspace(0.0, 1.0, 41)

    splits = {
        "sig train": (p_train[y_train == 1], w_train[y_train == 1]),
        "sig test":  (p_test[y_test == 1],  w_test[y_test == 1]),
        "bkg train": (p_train[y_train == 0], w_train[y_train == 0]),
        "bkg test":  (p_test[y_test == 0],  w_test[y_test == 0]),
    }
    styles = {
        "sig train": dict(color="C0", linestyle="-",  histtype="step", lw=1.5),
        "sig test":  dict(color="C0", linestyle="--", histtype="step", lw=1.5),
        "bkg train": dict(color="C3", linestyle="-",  histtype="step", lw=1.5),
        "bkg test":  dict(color="C3", linestyle="--", histtype="step", lw=1.5),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (vals, wts) in splits.items():
        ax.hist(vals, bins=bins, weights=wts, density=True,
                label=label, **styles[label])

    ks_sig = ks_2samp(splits["sig train"][0], splits["sig test"][0])
    ks_bkg = ks_2samp(splits["bkg train"][0], splits["bkg test"][0])

    ax.set_xlabel("BDT score")
    ax.set_ylabel("normalized density")
    ax.set_title(
        f"Overtraining check — KS sig p={ks_sig.pvalue:.3g}, "
        f"bkg p={ks_bkg.pvalue:.3g}"
    )
    ax.set_yscale("log")
    ax.legend(loc="upper center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved overtraining plot to %s (KS sig p=%.3g, bkg p=%.3g)",
             out_path, ks_sig.pvalue, ks_bkg.pvalue)
    return {
        "ks_sig_stat": float(ks_sig.statistic),
        "ks_sig_p":    float(ks_sig.pvalue),
        "ks_bkg_stat": float(ks_bkg.statistic),
        "ks_bkg_p":    float(ks_bkg.pvalue),
    }


def plot_m4l_vs_score(model: xgb.XGBClassifier,
                      X_test: np.ndarray, y_test: np.ndarray,
                      m4l_test: np.ndarray, w_test: np.ndarray,
                      out_path: Path,
                      m4l_range: tuple[float, float] = (70.0, 250.0),
                      n_m4l_bins: int = 36,
                      n_score_bins: int = 30) -> float:
    """2D m4l vs BDT-score for signal and background, plus a profile.

    The profile (mean BDT score per m4l bin, background only) is the key
    decorrelation check: if the line is flat, the score does not sculpt the
    background m4l shape. Returns Pearson correlation of (m4l, score) on bkg.
    """
    p_test = model.predict_proba(X_test)[:, 1]
    sig = y_test == 1
    bkg = y_test == 0

    m4l_bins = np.linspace(m4l_range[0], m4l_range[1], n_m4l_bins + 1)
    score_bins = np.linspace(0.0, 1.0, n_score_bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [1, 1, 1]})

    # Per-class 2D histograms — column-normalized so each m4l slice sums to 1.
    # Reveals score *shape* dependence on m4l rather than the underlying yield.
    for ax, mask, title in (
        (axes[0], sig, "signal"),
        (axes[1], bkg, "background"),
    ):
        h, xe, ye = np.histogram2d(
            m4l_test[mask], p_test[mask],
            bins=[m4l_bins, score_bins],
            weights=w_test[mask],
        )
        col_sum = h.sum(axis=1, keepdims=True)
        h_norm = np.divide(h, col_sum, out=np.zeros_like(h),
                           where=col_sum > 0)
        im = ax.pcolormesh(xe, ye, h_norm.T, cmap="viridis", shading="auto")
        ax.set_xlabel("m4l [GeV]")
        ax.set_ylabel("BDT score")
        ax.set_title(f"{title} — column-normalized")
        fig.colorbar(im, ax=ax, label="P(score | m4l)")

    # Profile: mean BDT score in each m4l bin, separately for sig and bkg.
    ax = axes[2]
    centers = 0.5 * (m4l_bins[:-1] + m4l_bins[1:])
    for mask, label, color in ((sig, "signal", "C0"), (bkg, "background", "C3")):
        means = np.zeros(n_m4l_bins)
        stds = np.zeros(n_m4l_bins)
        counts = np.zeros(n_m4l_bins)
        idx = np.digitize(m4l_test[mask], m4l_bins) - 1
        ps = p_test[mask]
        ws = w_test[mask]
        for b in range(n_m4l_bins):
            sel = idx == b
            counts[b] = sel.sum()
            if counts[b] > 1:
                ww = ws[sel]
                means[b] = np.average(ps[sel], weights=ww)
                var = np.average((ps[sel] - means[b]) ** 2, weights=ww)
                stds[b] = np.sqrt(var) / np.sqrt(max(counts[b], 1))
        valid = counts > 0
        ax.errorbar(centers[valid], means[valid], yerr=stds[valid],
                    fmt="o-", label=label, color=color, lw=1.2, ms=3)
    ax.set_xlabel("m4l [GeV]")
    ax.set_ylabel("⟨BDT score⟩")
    ax.set_title("Profile (mean ± SEM)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Pearson correlation on background (the relevant one for sculpting).
    if bkg.sum() > 1:
        corr_bkg = float(np.corrcoef(m4l_test[bkg], p_test[bkg])[0, 1])
    else:
        corr_bkg = float("nan")
    fig.suptitle(f"m4l vs BDT score — bkg Pearson r = {corr_bkg:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved m4l-vs-score plot to %s (bkg Pearson r=%.3f)",
             out_path, corr_bkg)
    return corr_bkg


def write_summary(out_path: Path, *, auc_weighted: float, auc_unweighted: float,
                  importance: pd.Series, ks: dict[str, float],
                  best_iter: int | None, n_train: int, n_val: int, n_test: int,
                  feature_names: list[str], m4l_corr_bkg: float) -> None:
    """Short markdown summary of Phase 4 results."""
    top5 = importance.head(5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 — Evaluation summary",
        "",
        "## Held-out test performance",
        f"- Weighted AUC (training metric, class-balanced):  **{auc_weighted:.4f}**",
        f"- Unweighted AUC (event-counting):                 **{auc_unweighted:.4f}**",
        f"- Best boosting iteration (early stopping):        {best_iter}",
        f"- Split sizes: train={n_train}, val={n_val}, test={n_test}",
        "",
        "## Top features (gain)",
    ]
    for name, gain in top5.items():
        lines.append(f"- `{name}`: {gain:.3g}")
    lines += [
        "",
        "## Overtraining check (KS, train vs test scores)",
        f"- signal:     stat={ks['ks_sig_stat']:.3g}, p={ks['ks_sig_p']:.3g}",
        f"- background: stat={ks['ks_bkg_stat']:.3g}, p={ks['ks_bkg_p']:.3g}",
        "",
        "## m4l decorrelation",
        f"- Background Pearson r(m4l, score) = **{m4l_corr_bkg:.3f}** (lower → better)",
        "- See `plots/m4l_vs_score.png` for the column-normalized 2D and the",
        "  per-class profile of ⟨score⟩ vs m4l.",
        "",
        "## Caveats",
        "- `m4l`, `mZ1`, `mZ2` are excluded from training inputs to keep the",
        "  score decorrelated from the Higgs mass peak. The pT/m4l ratios still",
        "  reference m4l as a normalization — residual correlation, if any, is",
        "  visible in `plots/m4l_vs_score.png`.",
        "- DY M-50 and TTto2L2Nu surviving the 4-muon skim are tiny",
        "  (~10² events each). Per-sample equal weighting (see train.py)",
        "  protects the loss from being dominated by xsec, but the BDT cannot",
        "  learn rich shapes for these reducible backgrounds.",
        "- PU reweighting and lepton SFs are not applied.",
        "- The `Muon` PD data has not been included; this is signal vs. MC",
        "  background only.",
        "",
        "## Plots",
        "- `plots/roc.png`",
        "- `plots/feature_importance.png`",
        "- `plots/overtraining.png`",
        "- `plots/m4l_vs_score.png`",
        "- `plots/training_history.png` (from Phase 3)",
        "",
        f"_Features evaluated_ ({len(feature_names)}): "
        + ", ".join(f"`{f}`" for f in feature_names),
        "",
    ]
    out_path.write_text("\n".join(lines))
    log.info("Wrote summary to %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/bdt_v1.json")
    p.add_argument("--ntuples", default="data/ntuples/")
    p.add_argument("--samples", default="config/samples.yaml")
    p.add_argument("--config", default="config/selection.yaml")
    p.add_argument("--tag", default=None,
                   help="Plot/summary suffix; defaults to v<N> from --model.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    samples_cfg = load_yaml(args.samples)
    ntuple_dir = Path(args.ntuples)

    df = load_dataset(ntuple_dir, samples_cfg, samples_cfg["lumi_fb"])

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    # Two weights are tracked through the split:
    #   w_train = train_weight (the weight the BDT *was* trained against);
    #            used here only for the overtraining KS check, which compares
    #            the same training-time score distribution.
    #   w_eval  = xsec_weight  (the physics weight); used for the ROC, the
    #            m4l-vs-score 2D, and the profile, so the numbers reflect
    #            real expected yields rather than the planed/balanced
    #            training distribution.
    w_train = df["train_weight"].to_numpy(dtype=np.float32)
    # |xsec_weight| — sklearn's roc_auc_score and weighted histograms reject
    # negative weights (amcatnlo NLO cancellations). Sign info is only
    # meaningful for cross-section yields, not for classification metrics.
    w_eval = np.abs(df["xsec_weight"].to_numpy(dtype=np.float32))
    m4l = df["m4l"].to_numpy(dtype=np.float32)

    seed = cfg["train"].get("seed", SEED)

    # Reproduce the exact two-step split from train.py, also carrying m4l +
    # xsec_weight so the diagnostics use the same test events as the AUC.
    (X_tv, X_test, y_tv, y_test,
     wtr_tv, wtr_test, we_tv, we_test, m4l_tv, m4l_test) = train_test_split(
        X, y, w_train, w_eval, m4l,
        test_size=cfg["train"]["test_size"],
        stratify=y, random_state=seed,
    )
    (X_tr, X_val, y_tr, y_val,
     wtr_tr, wtr_val, we_tr, we_val, m4l_tr, m4l_val) = train_test_split(
        X_tv, y_tv, wtr_tv, we_tv, m4l_tv,
        test_size=cfg["train"]["val_size"],
        stratify=y_tv, random_state=seed,
    )
    log.info("Split sizes: train=%d val=%d test=%d",
             len(y_tr), len(y_val), len(y_test))

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    log.info("Loaded model from %s", args.model)
    best_iter = getattr(model, "best_iteration", None)

    # Plot/summary tag — derive "v3" from "bdt_v3_planed.json" by default.
    if args.tag:
        tag = args.tag
    else:
        import re as _re
        m = _re.match(r"bdt_(v\d+)", Path(args.model).stem)
        tag = m.group(1) if m else Path(args.model).stem
    log.info("Output tag: %s", tag)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    p_test = model.predict_proba(X_test)[:, 1]
    # Use xsec_weight for ROC so the curve corresponds to physical event
    # yields (not the training-time class-balanced distribution). For
    # planed models, using planed_weight here would give an inflated AUC.
    auc_w, auc_u = plot_roc(y_test, p_test, we_test,
                            PLOT_DIR / f"roc_{tag}.png")

    importance = plot_feature_importance(
        model, FEATURES, PLOT_DIR / f"feature_importance_{tag}.png",
    )

    ks = plot_overtraining(
        model, X_tr, y_tr, X_test, y_test, wtr_tr, wtr_test,
        PLOT_DIR / f"overtraining_{tag}.png",
    )

    corr_bkg = plot_m4l_vs_score(
        model, X_test, y_test, m4l_test, we_test,
        PLOT_DIR / f"m4l_vs_score_{tag}.png",
    )

    write_summary(
        PROJECT_ROOT / "reports" / f"phase4_summary_{tag}.md",
        auc_weighted=auc_w, auc_unweighted=auc_u,
        importance=importance, ks=ks, best_iter=best_iter,
        n_train=len(y_tr), n_val=len(y_val), n_test=len(y_test),
        feature_names=FEATURES,
        m4l_corr_bkg=corr_bkg,
    )


if __name__ == "__main__":
    main()
