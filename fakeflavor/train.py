"""Train either flavor BDT of the two routes.

`--target flavor` (default) is the 4-class (b / c / uds / g) source-jet flavor
classifier of Phase 1:

    python -m fakeflavor.train --config fakeflavor/config/train.yaml \
        --ntuples fakeflavor/data/ntuples/train.parquet \
        --out fakeflavor/data/models/flavor_full.json

`--target tight` is the direct route: a binary model of P(tight | loose, x)
whose calibrated output is used *as* the fake rate, replacing the FR(conePt,
|eta|) lookup rather than reweighting it.

    python -m fakeflavor.train --target tight \
        --config fakeflavor/config/direct.yaml \
        --ntuples fakeflavor/data/ntuples/QCD_MuEnriched.parquet \
        --feature-set direct --out fakeflavor/data/models/direct.json

Three differences matter for the direct target and are enforced below rather
than left to the config:
  * **no class balancing** — the output has to be a calibrated probability, and
    rescaling the tight/loose classes to equal weight would shift it away from
    the measured FR by construction;
  * **no weight clipping** — for the same reason. The tail is affordable here:
    the top 0.1% of QCD rows by weight carry only 1.4% of the total;
  * training runs on **QCD alone**, not train.parquet. TTLJ is weightless next
    to QCD (sum w 6.7e3 vs 7.8e8) so it contributes ~12 effective events, but
    it is also the transfer/closure target, so it is excluded explicitly
    instead of being harmlessly diluted.
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
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from .features import CLASSES, FEATURE_SETS, N_CLASSES
from src.utils import SEED, get_logger, load_yaml

log = get_logger("fakeflavor.train")

# --target -> (label column, class-code map). The direct target's "classes" are
# just the two arms of the tight WP; they exist so the weighting and reporting
# code below stays shared between the two routes.
TARGET_COLUMN = {"flavor": "flavor", "tight": "isTight"}
TARGET_CLASSES = {"flavor": CLASSES, "tight": {0: "loose", 1: "tight"}}


def detect_gpu() -> bool:
    """Return True only if xgboost can really train on a GPU.

    Not `src/utils.py::detect_gpu`. That one trains a one-row DMatrix with
    device="cuda" inside a try/except, but xgboost does not *raise* when CUDA
    is missing — it emits a UserWarning and silently falls back to CPU — so it
    returns True on every machine. Verified on tamsa2, which has no GPU.
    Checking `build_info()["USE_CUDA"]` is no good either: the PyPI wheel is
    always built with CUDA. So run the probe and read the warnings.
    """
    try:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dm = xgb.DMatrix(np.array([[1.0]]), label=np.array([0]))
            xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 1},
                      dm, num_boost_round=1)
        msgs = " ".join(str(w.message) for w in caught)
        return not ("No visible GPU" in msgs or "not compiled with CUDA" in msgs)
    except Exception:
        return False


def load(parquet: Path, features: list[str], target: str = "flavor"):
    arr = ak.from_parquet(parquet)
    X = np.column_stack([
        ak.to_numpy(arr[f]).astype(np.float32) for f in features
    ])
    y = ak.to_numpy(arr[TARGET_COLUMN[target]]).astype(np.int64)
    w = ak.to_numpy(arr["weight"]).astype(np.float64)
    meta = {k: ak.to_numpy(arr[k]) for k in ("run", "lumi", "evt", "mu_conept",
                                             "mu_abseta", "isTight",
                                             "sample", "era")}
    log.info("Loaded %s: %d rows x %d features", parquet, len(y), len(features))
    return X, y, w, meta


def build_weights(y: np.ndarray, w_phys: np.ndarray, cfg: dict,
                  classes: dict | None = None) -> np.ndarray:
    """Physical weight -> training weight (clipped, optionally class-balanced)."""
    classes = CLASSES if classes is None else classes
    w = w_phys.astype(np.float64).copy()
    if not cfg.get("physical_weight", True):
        w = np.ones_like(w)

    w = np.abs(w)
    q = cfg.get("weight_clip_quantile")
    if q:
        for c in classes:
            m = y == c
            if m.sum() > 10:
                cap = np.quantile(w[m], q)
                n_clipped = int((w[m] > cap).sum())
                w[m] = np.minimum(w[m], cap)
                log.info("  class %-4s clipped %d/%d rows at w=%.4g",
                         classes[c], n_clipped, int(m.sum()), cap)

    if cfg.get("class_balance", True):
        for c in classes:
            m = y == c
            s = w[m].sum()
            if s > 0:
                w[m] /= s
    # xgboost hyperparameters (min_child_weight) assume ~unit weights.
    w *= len(w) / max(w.sum(), 1e-30)
    return w


def split_by_event(meta: dict, y: np.ndarray, test_size: float, seed: int):
    """Group-aware split on (sample, era, run, lumi, evt) so no event straddles it.

    `sample` and `era` are part of the key, not decoration: MC run number is 1
    and event numbers restart per sample, so 12 QCD pt bins plus TTLJ across
    four eras collide constantly on (run, lumi, evt) alone. Grouping on that
    would silently tie unrelated events from different samples into one group
    and correlate the split with the sample.
    """
    _, sample_code = np.unique(meta["sample"], return_inverse=True)
    _, era_code = np.unique(meta["era"], return_inverse=True)
    key = np.stack([sample_code, era_code,
                    meta["run"], meta["lumi"], meta["evt"]], axis=1)
    _, group = np.unique(key, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    n_groups = group.max() + 1
    perm = rng.permutation(n_groups)
    n_test = int(round(test_size * n_groups))
    test_groups = np.zeros(n_groups, dtype=bool)
    test_groups[perm[:n_test]] = True
    is_test = test_groups[group]
    return ~is_test, is_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="fakeflavor/config/train.yaml")
    p.add_argument("--ntuples", default="fakeflavor/data/ntuples/train.parquet")
    p.add_argument("--out", default="fakeflavor/data/models/flavor_full.json")
    p.add_argument("--feature-set", default=None,
                   help="override train.feature_set in the config")
    p.add_argument("--target", default="flavor", choices=sorted(TARGET_COLUMN),
                   help="'flavor' = 4-class source-jet flavor; "
                        "'tight' = direct P(tight|loose,x) used as the FR")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    classes = TARGET_CLASSES[args.target]
    n_classes = len(classes)
    binary = args.target == "tight"
    fs_name = args.feature_set or tcfg.get("feature_set", "full")
    features = FEATURE_SETS[fs_name]
    seed = int(tcfg.get("seed", SEED))
    log.info("Target '%s' (%d classes), feature set '%s' (%d features)",
             args.target, n_classes, fs_name, len(features))

    if binary:
        # Not negotiable via config: both would decalibrate the output, which
        # for this target *is* the fake rate. See the module docstring.
        for k in ("class_balance", "weight_clip_quantile"):
            if tcfg.get(k):
                log.warning("target=tight: forcing %s off (was %r) — the output "
                            "must stay a calibrated probability", k, tcfg[k])
            tcfg[k] = False if k == "class_balance" else None

    X, y, w_phys, meta = load(Path(args.ntuples), features, args.target)
    for c, name in classes.items():
        m = y == c
        log.info("  class %-6s %8d rows  sum w = %.4g", name, int(m.sum()),
                 float(np.abs(w_phys[m]).sum()))
    if binary:
        wa = np.abs(w_phys)
        log.info("  measured inclusive FR = %.4f (weighted), %.4f (unweighted)",
                 float(wa[y == 1].sum() / wa.sum()), float((y == 1).mean()))

    w = build_weights(y, w_phys, tcfg, classes)

    if tcfg.get("split_by_event", True):
        tv, te = split_by_event(meta, y, float(tcfg["test_size"]), seed)
        tr, va = split_by_event({k: v[tv] for k, v in meta.items()}, y[tv],
                                float(tcfg["val_size"]), seed + 1)
        idx_tv = np.flatnonzero(tv)
        idx_tr, idx_va = idx_tv[tr], idx_tv[va]
        idx_te = np.flatnonzero(te)
    else:
        from sklearn.model_selection import train_test_split
        i = np.arange(len(y))
        i_tv, idx_te = train_test_split(i, test_size=float(tcfg["test_size"]),
                                        stratify=y, random_state=seed)
        idx_tr, idx_va = train_test_split(i_tv, test_size=float(tcfg["val_size"]),
                                          stratify=y[i_tv], random_state=seed)
    log.info("Split: train=%d val=%d test=%d", len(idx_tr), len(idx_va), len(idx_te))

    use_gpu = tcfg.get("use_gpu", True) and detect_gpu()
    log.info("Device: %s", "cuda" if use_gpu else "cpu")

    bdt_cfg = dict(cfg["bdt"])
    early = bdt_cfg.pop("early_stopping_rounds", 40)
    metric = bdt_cfg.pop("eval_metric", "mlogloss")

    model = xgb.XGBClassifier(
        **bdt_cfg,
        **({"objective": "binary:logistic"} if binary else
           {"objective": "multi:softprob", "num_class": N_CLASSES}),
        eval_metric=metric,
        early_stopping_rounds=early,
        device="cuda" if use_gpu else "cpu",
        tree_method="hist",
        random_state=seed,
    )
    model.fit(
        X[idx_tr], y[idx_tr], sample_weight=w[idx_tr],
        eval_set=[(X[idx_tr], y[idx_tr]), (X[idx_va], y[idx_va])],
        sample_weight_eval_set=[w[idx_tr], w[idx_va]],
        verbose=False,
    )
    log.info("Best iteration: %s / %d", model.best_iteration, bdt_cfg["n_estimators"])

    proba = model.predict_proba(X[idx_te])
    aucs = {}
    for c, name in classes.items():
        if binary and c == 0:
            continue    # one number, not two mirror-image ones
        aucs[name] = float(roc_auc_score((y[idx_te] == c).astype(int), proba[:, c],
                                         sample_weight=w[idx_te]))
        log.info("Test one-vs-rest AUC  %-6s = %.4f", name, aucs[name])

    if binary:
        # The two things that decide whether this is usable as a fake rate.
        #
        # (1) Sum rule. sum_i w_i p(x_i) must reproduce sum_{i tight} w_i on the
        #     held-out rows. This is the same-process closure: it can only fail
        #     through miscalibration, so a deviation here invalidates any
        #     cross-process closure number before it is measured.
        # (2) Reliability. Predicted probability vs the weighted tight fraction
        #     in bins of the prediction. This is the direct route's analogue of
        #     the flavor route's leakage diagnostic -- except that here it is a
        #     proper scoring rule, so it is expected to hold, and the point of
        #     measuring it is to have the in-sample reference against which the
        #     transfer sample can be judged.
        wt, yt, pt = w[idx_te], y[idx_te], proba[:, 1]
        pred, obs = float((wt * pt).sum()), float((wt * (yt == 1)).sum())
        log.info("Test sum rule: predicted %.1f vs observed %.1f  -> %.4f",
                 pred, obs, pred / obs)
        log.info("Reliability (test):  %8s %8s %10s %10s",
                 "p_lo", "p_hi", "<p>", "FR_obs")
        qs = np.quantile(pt, np.linspace(0, 1, 11))
        qs[0], qs[-1] = 0.0, 1.0
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (pt >= lo) & (pt < hi) if hi < 1.0 else (pt >= lo)
            if not m.any():
                continue
            log.info("                     %8.4f %8.4f %10.4f %10.4f",
                     lo, hi, float(np.average(pt[m], weights=wt[m])),
                     float((wt[m] * (yt[m] == 1)).sum() / wt[m].sum()))
        aucs["_sum_rule"] = pred / obs

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    booster = model.get_booster()
    booster.feature_names = list(features)
    model.save_model(out)
    # Feature order is load-bearing for any downstream C++/ONNX consumer, so
    # persist it next to the model rather than relying on features.py matching.
    out.with_suffix(".meta.json").write_text(json.dumps({
        "target": args.target,
        "ntuples": str(args.ntuples),
        "feature_set": fs_name,
        "features": list(features),
        "classes": {str(k): v for k, v in classes.items()},
        "best_iteration": int(model.best_iteration),
        "test_auc_ovr": aucs,
        "train_config": tcfg,
    }, indent=2))
    log.info("Saved model to %s (+ .meta.json)", out)

    # Save the test-set indices so evaluate.py scores exactly the held-out rows.
    np.savez_compressed(out.with_suffix(".split.npz"),
                        test=idx_te, train=idx_tr, val=idx_va)

    hist = model.evals_result()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["validation_0"][metric], label="train")
    ax.plot(hist["validation_1"][metric], label="val")
    ax.axvline(model.best_iteration, color="grey", ls="--", lw=1, label="best")
    ax.set_xlabel("boosting round")
    ax.set_ylabel(metric)
    auc_vals = [v for k, v in aucs.items() if not k.startswith("_")]
    ax.set_title(f"{args.target} BDT ({fs_name}) — "
                 f"mean OvR AUC {np.mean(auc_vals):.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_dir = Path("fakeflavor/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = fs_name if args.target == "flavor" else f"{args.target}_{fs_name}"
    fig.savefig(plot_dir / f"training_history_{tag}.png", dpi=300)
    plt.close(fig)
    log.info("Saved training history to %s",
             plot_dir / f"training_history_{tag}.png")


if __name__ == "__main__":
    main()
