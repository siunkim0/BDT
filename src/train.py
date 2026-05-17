"""Phase 3: train an xgboost BDT on the engineered features.

Reads parquet files from data/ntuples/ (must have FEATURES columns from
features.py), splits into train/val/test, trains, saves model + plots.

Run from project root:

    python -m src.train --config config/selection.yaml --out data/models/bdt_v1.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FEATURES
from .utils import PLOT_DIR, SEED, detect_gpu, get_logger, load_yaml

log = get_logger("train")


def load_dataset(ntuple_dir: Path, samples_cfg: dict, lumi_fb: float) -> pd.DataFrame:
    """Concatenate all MC samples into one DataFrame with class labels and weights.

    Two weight columns are produced:
      - `xsec_weight`  : signed lumi × xsec × genWeight/Σgw — for yield/evaluation.
      - `train_weight` : per-sample-equal, then class-balanced — for BDT training.

    Why not weight train_weight by xsec? With this skim, DY M-50 keeps only
    ~10² 4μ events but has σ≈6225 pb, giving each event a 10⁷× larger weight
    than a qqZZ event. The BDT then optimizes for DY while val is dominated
    by qqZZ → AUC collapses to 0.5. Equal per-sample weighting avoids this.
    """
    frames = []
    for category in ("signal", "background"):
        for name, cfg in samples_cfg.get(category, {}).items():
            fp = ntuple_dir / f"{name}_features.parquet"
            if not fp.exists():
                log.warning("Missing %s, skipping", fp)
                continue
            arr = ak.from_parquet(fp)
            df = ak.to_dataframe(arr) if hasattr(ak, "to_dataframe") else \
                 pd.DataFrame({k: ak.to_numpy(arr[k]) for k in arr.fields})
            df["label"] = cfg["label"]
            df["sample"] = name
            # Signed yield weight (NLO cancellations preserved). Normalize by
            # sum(genWeight) — i.e. signed effective sample size — so the sum
            # of xsec_weight over the sample equals lumi×xsec.
            sum_gw_signed = float(np.sum(df["genWeight"]))
            df["xsec_weight"] = (
                lumi_fb * 1000.0 * cfg["xsec"]                # pb → fb
                * df["genWeight"]
                / max(abs(sum_gw_signed), 1.0)
            )
            # Absolute training weight. xgboost requires sample_weight ≥ 0,
            # so for training we drop the sign and normalize by sum|gw|.
            sum_gw_abs = float(np.sum(np.abs(df["genWeight"])))
            df["abs_weight"] = (
                lumi_fb * 1000.0 * cfg["xsec"]
                * np.abs(df["genWeight"])
                / max(sum_gw_abs, 1.0)
            )
            frames.append(df)
            log.info("Loaded %s: %d events, sum xsec_weight = %.2f, sum abs_weight = %.2f",
                     name, len(df), df["xsec_weight"].sum(), df["abs_weight"].sum())
    full = pd.concat(frames, ignore_index=True)

    # Per-sample equal contribution within class.
    # Each sample's training weights sum to 1; then divide by # samples in
    # that class so the class total is 1. Result: every sample contributes
    # the same to the loss regardless of its yield, and signal vs background
    # are balanced.
    full["train_weight"] = 0.0
    for label_value in (0, 1):
        cls = full.label == label_value
        names = full.loc[cls, "sample"].unique()
        n = max(len(names), 1)
        for name in names:
            sel = cls & (full["sample"] == name)
            sample_sum = full.loc[sel, "abs_weight"].sum()
            full.loc[sel, "train_weight"] = (
                full.loc[sel, "abs_weight"] / max(sample_sum, 1e-12) / n
            )
    # Rescale so mean per-event weight = 1.
    # The class-balance step above leaves total weight ≈ 2 across N events,
    # i.e. ~1e-6 per event. With weights that small, hyperparameters like
    # min_child_weight (default 5) become unreachable and the BDT can't
    # build any splits. Normalizing to mean=1 makes YAML hyperparameters
    # behave as if weights were uniform.
    full["train_weight"] *= len(full) / full["train_weight"].sum()
    log.info("Class balance: sig=%d events, bkg=%d events",
             int((full.label == 1).sum()), int((full.label == 0).sum()))
    return full


def plot_planing_check(df: pd.DataFrame, out_path: Path,
                       m4l_col: str = "m4l",
                       n_bins: int = 50,
                       m4l_range: tuple[float, float] = (70.0, 250.0)) -> None:
    """Sanity plot: m4l histograms, signal/bkg × original/planed weights.

    The right column should look flat (within statistical noise) where the
    sample is populated. Empty bins (e.g. signal outside ~[100, 145]) just
    stay at zero — they were never going to be flattened.
    """
    edges = np.linspace(m4l_range[0], m4l_range[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    panels = [
        (axes[0, 0], df["label"] == 1, "xsec_weight",   "signal — original",     "C0"),
        (axes[0, 1], df["label"] == 1, "planed_weight", "signal — planed",       "C0"),
        (axes[1, 0], df["label"] == 0, "xsec_weight",   "background — original", "C3"),
        (axes[1, 1], df["label"] == 0, "planed_weight", "background — planed",   "C3"),
    ]
    for ax, mask, wcol, title, color in panels:
        h, _ = np.histogram(df.loc[mask, m4l_col],
                            bins=edges,
                            weights=df.loc[mask, wcol])
        ax.step(centers, h, where="mid", color=color, lw=1.4)
        ax.fill_between(centers, 0, h, step="mid", color=color, alpha=0.25)
        # CV across populated bins (>0) — flat means CV ≈ 0.
        populated = h > 0
        if populated.sum() > 1:
            cv = h[populated].std() / max(abs(h[populated].mean()), 1e-12)
            ax.set_title(f"{title}   (CV[populated]={cv:.2f})")
        else:
            ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Σ weight / bin")
    for ax in axes[1, :]:
        ax.set_xlabel("m4l [GeV]")

    fig.suptitle(
        f"Planing check — {n_bins} bins in [{m4l_range[0]}, {m4l_range[1]}] GeV\n"
        f"right column should be flat where the sample is populated"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved planing check to %s", out_path)


def apply_planing(df: pd.DataFrame,
                  m4l_col: str = "m4l",
                  n_bins: int = 50,
                  m4l_range: tuple[float, float] = (70.0, 250.0)) -> pd.DataFrame:
    """Per-class m4l planing: reweight signal and background separately so
    each class has a flat m4l distribution at training time.

    With both classes flat in m4l, m4l carries zero discriminating power and
    the BDT must rely on m4l-independent kinematics. xsec_weight is left
    untouched (still the right weight for evaluation/yields); the new
    `planed_weight` column is the per-class-normalized training weight.

    Events outside [m4l_range] keep their xsec_weight before normalization
    (np.digitize -> clip pushes them to the edge bins). For the planed
    sample this is fine since events outside the planing window are rare in
    the training selection (m4l_min=70 already cuts the low side).
    """
    df = df.copy()
    df["planed_weight"] = 0.0

    edges = np.linspace(m4l_range[0], m4l_range[1], n_bins + 1)

    for label in (0, 1):
        mask = (df["label"] == label).to_numpy()
        if not mask.any():
            continue
        m4l_vals = df.loc[mask, m4l_col].to_numpy()
        # xgboost requires sample_weight >= 0. amcatnlo samples (qqZZ, DY)
        # have negative genWeights, so xsec_weight can be negative; we use
        # |xsec_weight| both as the per-event weight and as the density we
        # invert. The NLO sign cancellation is preserved for evaluation
        # via the untouched xsec_weight column.
        abs_w = np.abs(df.loc[mask, "xsec_weight"].to_numpy())

        hist, _ = np.histogram(m4l_vals, bins=edges, weights=abs_w)
        bin_idx = np.clip(np.digitize(m4l_vals, edges) - 1, 0, n_bins - 1)
        local_density = hist[bin_idx]

        # 1/density gives a flat m4l shape; floor avoids div-by-zero in
        # bins where the class has no events.
        planed = abs_w / np.maximum(local_density, 1e-12)
        # Per-class normalization: each class sums to 1, so signal and
        # background contribute equally to the loss.
        total = planed.sum()
        if total > 0:
            planed = planed / total

        df.loc[mask, "planed_weight"] = planed

    sig_sum = float(df.loc[df.label == 1, "planed_weight"].sum())
    bkg_sum = float(df.loc[df.label == 0, "planed_weight"].sum())
    log.info("Planing applied. Signal weight sum: %.4f, Bkg weight sum: %.4f",
             sig_sum, bkg_sum)
    # xgboost wants mean-≈-1 weights so YAML hyperparameters (min_child_weight
    # etc.) behave as if uniform. Match the train_weight rescale convention.
    df["planed_weight"] *= len(df) / max(df["planed_weight"].sum(), 1e-12)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="config/samples.yaml")
    p.add_argument("--config", default="config/selection.yaml")
    p.add_argument("--ntuples", default="data/ntuples/")
    p.add_argument("--out", default="data/models/bdt_v1.json")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    samples_cfg = load_yaml(args.samples)
    ntuple_dir = Path(args.ntuples)

    df = load_dataset(ntuple_dir, samples_cfg, samples_cfg["lumi_fb"])

    planing_cfg = cfg["train"].get("planing", {}) or {}
    if planing_cfg.get("enabled", False):
        m4l_col   = planing_cfg.get("m4l_col", "m4l")
        n_bins    = int(planing_cfg.get("n_bins", 50))
        m4l_range = tuple(planing_cfg.get("m4l_range", [70.0, 250.0]))
        df = apply_planing(df, m4l_col=m4l_col, n_bins=n_bins, m4l_range=m4l_range)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_planing_check(df, PLOT_DIR / "planing_check.png",
                           m4l_col=m4l_col, n_bins=n_bins, m4l_range=m4l_range)
        weight_col = "planed_weight"
        log.info("Using planed_weight for training (m4l-decorrelated).")
    else:
        weight_col = "train_weight"
        log.info("Using train_weight for training (no planing).")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    w = df[weight_col].to_numpy(dtype=np.float32)

    seed = cfg["train"].get("seed", SEED)

    X_tv, X_test, y_tv, y_test, w_tv, w_test = train_test_split(
        X, y, w,
        test_size=cfg["train"]["test_size"],
        stratify=y, random_state=seed,
    )
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_tv, y_tv, w_tv,
        test_size=cfg["train"]["val_size"],
        stratify=y_tv, random_state=seed,
    )
    log.info("Split sizes: train=%d val=%d test=%d", len(y_tr), len(y_val), len(y_test))

    use_gpu = cfg["train"].get("use_gpu", True) and detect_gpu()
    log.info("Using device: %s", "cuda" if use_gpu else "cpu")

    bdt_cfg = dict(cfg["bdt"])
    early_stopping = bdt_cfg.pop("early_stopping_rounds", 30)
    eval_metric = bdt_cfg.pop("eval_metric", "auc")

    model = xgb.XGBClassifier(
        **bdt_cfg,
        objective="binary:logistic",
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping,
        device="cuda" if use_gpu else "cpu",
        tree_method="hist",
        random_state=seed,
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        sample_weight_eval_set=[w_tr, w_val],
        verbose=False,
    )

    best_iter = getattr(model, "best_iteration", None)
    log.info("Best iteration: %s / %d", best_iter, bdt_cfg["n_estimators"])

    # Held-out test AUC (weighted with class-balanced weights → matches training metric)
    p_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, p_test, sample_weight=w_test)
    log.info("Test AUC (weighted): %.4f", auc_test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out_path)
    log.info("Saved model to %s", out_path)

    # Tag plots by major model version (bdt_v3_planed -> v3) so v1/v2/v3
    # outputs don't overwrite each other.
    import re as _re
    m = _re.match(r"bdt_(v\d+)", out_path.stem)
    model_tag = m.group(1) if m else out_path.stem

    # Training history plot — train + val curves of the eval_metric.
    history = model.evals_result()
    train_curve = history["validation_0"][eval_metric]
    val_curve = history["validation_1"][eval_metric]
    iters = np.arange(1, len(train_curve) + 1)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"training_history_{model_tag}.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, train_curve, label="train")
    ax.plot(iters, val_curve, label="val")
    if best_iter is not None:
        ax.axvline(best_iter + 1, color="grey", ls="--", lw=1,
                   label=f"best iter ({best_iter + 1})")
    ax.set_xlabel("boosting round")
    ax.set_ylabel(eval_metric)
    ax.set_title(f"BDT training history — test {eval_metric}={auc_test:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    log.info("Saved training history to %s", plot_path)

    # Tree structure — first tree is usually the most informative single split.
    # Change tree_idx to inspect other rounds (e.g. best_iter, last tree).
    tree_idx = 100
    booster = model.get_booster()
    booster.feature_names = list(FEATURES)
    fig, ax = plt.subplots(figsize=(16, 8))
    xgb.plot_tree(booster, num_trees=tree_idx, ax=ax)
    ax.set_title(
        f"Tree {tree_idx} (of {bdt_cfg['n_estimators']}) — "
        f"depth={bdt_cfg['max_depth']}, lr={bdt_cfg['learning_rate']}"
    )
    fig.tight_layout()
    tree_path = PLOT_DIR / f"tree_structure_{model_tag}_tree{tree_idx}.png"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tree_path, dpi=300)
    plt.close(fig)
    log.info("Saved tree %d to %s", tree_idx, tree_path)


if __name__ == "__main__":
    main()
