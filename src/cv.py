"""Phase 3 add-on: 5-fold cross-validation for the BDT hyperparameters in
config/selection.yaml.

The held-out test set (same split as train.py, seed=42) is never touched
during CV — it stays as the honest final number. Each fold uses 4/5 of the
remaining pool to fit and 1/5 for early-stopping validation.

Reported:
  - Per-fold validation AUC and best_iteration
  - Mean ± std AUC across folds
  - Test AUC of a final model refit on the full pool with
    n_estimators = round(mean(best_iteration)) + 1

Run from project root:

    python -m src.cv --config config/selection.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features import FEATURES
from .train import apply_planing, load_dataset
from .utils import SEED, detect_gpu, get_logger, load_yaml

log = get_logger("cv")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="config/samples.yaml")
    p.add_argument("--config", default="config/selection.yaml")
    p.add_argument("--ntuples", default="data/ntuples/")
    p.add_argument("--n_splits", type=int, default=5)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    samples_cfg = load_yaml(args.samples)
    df = load_dataset(Path(args.ntuples), samples_cfg, samples_cfg["lumi_fb"])

    planing_cfg = cfg["train"].get("planing", {}) or {}
    if planing_cfg.get("enabled", False):
        df = apply_planing(
            df,
            m4l_col=planing_cfg.get("m4l_col", "m4l"),
            n_bins=int(planing_cfg.get("n_bins", 50)),
            m4l_range=tuple(planing_cfg.get("m4l_range", [70.0, 250.0])),
        )
        weight_col = "planed_weight"
        log.info("CV: using planed_weight (planing enabled).")
    else:
        weight_col = "train_weight"
        log.info("CV: using train_weight (no planing).")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    w = df[weight_col].to_numpy(dtype=np.float32)

    seed = cfg["train"].get("seed", SEED)

    X_pool, X_test, y_pool, y_test, w_pool, w_test = train_test_split(
        X, y, w,
        test_size=cfg["train"]["test_size"],
        stratify=y, random_state=seed,
    )

    use_gpu = cfg["train"].get("use_gpu", True) and detect_gpu()
    log.info("Using device: %s", "cuda" if use_gpu else "cpu")

    bdt_cfg = dict(cfg["bdt"])
    early_stopping = bdt_cfg.pop("early_stopping_rounds", 30)
    eval_metric = bdt_cfg.pop("eval_metric", "auc")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
    fold_aucs: list[float] = []
    best_iters: list[int] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_pool, y_pool), start=1):
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
            X_pool[tr_idx], y_pool[tr_idx],
            sample_weight=w_pool[tr_idx],
            eval_set=[(X_pool[val_idx], y_pool[val_idx])],
            sample_weight_eval_set=[w_pool[val_idx]],
            verbose=False,
        )
        p_val = model.predict_proba(X_pool[val_idx])[:, 1]
        auc = roc_auc_score(y_pool[val_idx], p_val, sample_weight=w_pool[val_idx])
        bi = int(getattr(model, "best_iteration", 0))
        fold_aucs.append(auc)
        best_iters.append(bi)
        log.info("Fold %d/%d  val AUC=%.4f  best_iter=%d",
                 fold, args.n_splits, auc, bi)

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs, ddof=1))
    # +1 because xgboost best_iteration is 0-indexed.
    mean_iter = int(round(float(np.mean(best_iters)))) + 1
    log.info("CV summary: AUC = %.4f ± %.4f over %d folds  (mean n_trees = %d)",
             mean_auc, std_auc, args.n_splits, mean_iter)

    final_cfg = dict(bdt_cfg)
    final_cfg["n_estimators"] = mean_iter
    final = xgb.XGBClassifier(
        **final_cfg,
        objective="binary:logistic",
        eval_metric=eval_metric,
        device="cuda" if use_gpu else "cpu",
        tree_method="hist",
        random_state=seed,
    )
    final.fit(X_pool, y_pool, sample_weight=w_pool, verbose=False)
    p_test = final.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, p_test, sample_weight=w_test)
    log.info("Held-out test AUC (refit on full pool): %.4f", auc_test)


if __name__ == "__main__":
    main()
