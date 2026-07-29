"""5-fold cross-validation for the flavor BDT, mirroring src/cv.py.

The held-out test set (same event-level split as train.py, same seed) is never
touched during CV — it stays as the honest final number. Each fold fits on 4/5
of the remaining pool and early-stops on 1/5.

    python -m fakeflavor.cv --config fakeflavor/config/train.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import CLASSES, FEATURE_SETS, N_CLASSES
from .train import build_weights, detect_gpu, load, split_by_event
from src.utils import SEED, get_logger, load_yaml

log = get_logger("fakeflavor.cv")


def mean_ovr_auc(y, proba, w) -> tuple[float, dict[str, float]]:
    per = {}
    for c, name in CLASSES.items():
        if (y == c).sum() == 0:
            continue
        per[name] = float(roc_auc_score((y == c).astype(int), proba[:, c],
                                        sample_weight=w))
    return float(np.mean(list(per.values()))), per


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="fakeflavor/config/train.yaml")
    p.add_argument("--ntuples", default="fakeflavor/data/ntuples/train.parquet")
    p.add_argument("--feature-set", default=None)
    p.add_argument("--n_splits", type=int, default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    fs_name = args.feature_set or tcfg.get("feature_set", "full")
    features = FEATURE_SETS[fs_name]
    seed = int(tcfg.get("seed", SEED))
    n_splits = args.n_splits or int(cfg.get("cv", {}).get("n_splits", 5))

    X, y, w_phys, meta = load(Path(args.ntuples), features)
    w = build_weights(y, w_phys, tcfg)

    # Same pool/test split as train.py so the two are directly comparable.
    pool_mask, _ = split_by_event(meta, y, float(tcfg["test_size"]), seed)
    pool = np.flatnonzero(pool_mask)
    X_pool, y_pool, w_pool = X[pool], y[pool], w[pool]
    log.info("CV pool: %d rows (%d held out)", len(pool), len(y) - len(pool))

    use_gpu = tcfg.get("use_gpu", True) and detect_gpu()
    log.info("Device: %s", "cuda" if use_gpu else "cpu")

    bdt_cfg = dict(cfg["bdt"])
    early = bdt_cfg.pop("early_stopping_rounds", 40)
    metric = bdt_cfg.pop("eval_metric", "mlogloss")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs, best_iters = [], []
    for fold, (tr, va) in enumerate(skf.split(X_pool, y_pool), start=1):
        model = xgb.XGBClassifier(
            **bdt_cfg,
            objective="multi:softprob",
            num_class=N_CLASSES,
            eval_metric=metric,
            early_stopping_rounds=early,
            device="cuda" if use_gpu else "cpu",
            tree_method="hist",
            random_state=seed,
        )
        model.fit(X_pool[tr], y_pool[tr], sample_weight=w_pool[tr],
                  eval_set=[(X_pool[va], y_pool[va])],
                  sample_weight_eval_set=[w_pool[va]], verbose=False)
        auc, per = mean_ovr_auc(y_pool[va], model.predict_proba(X_pool[va]),
                                w_pool[va])
        bi = int(model.best_iteration)
        fold_aucs.append(auc)
        best_iters.append(bi)
        log.info("Fold %d/%d  mean OvR AUC=%.4f  best_iter=%d  %s",
                 fold, n_splits, auc, bi,
                 {k: round(v, 4) for k, v in per.items()})

    log.info("CV summary: mean OvR AUC = %.4f ± %.4f over %d folds "
             "(mean n_trees = %d)",
             float(np.mean(fold_aucs)), float(np.std(fold_aucs, ddof=1)),
             n_splits, int(round(float(np.mean(best_iters)))) + 1)


if __name__ == "__main__":
    main()
