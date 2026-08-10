"""Train either flavor BDT of the two routes.

`--target flavor` (default) is the 4-class (b / c / uds / g) source-jet flavor
classifier of Phase 1:

    python -m fakeflavor.train --config fakeflavor/config/train.yaml \
        --ntuples fakeflavor/data/ntuples/train.parquet \
        --out fakeflavor/data/models/flavor_full.json

`--target genflavor` is the same classifier trained on the gen-jet label. It is
the better of the two (+0.0054 mean OvR AUC, almost all of it in the g class)
and should be preferred unless you specifically need the reco-jet definition:

    python -m fakeflavor.train --target genflavor \
        --config fakeflavor/config/train.yaml \
        --ntuples fakeflavor/data/ntuples_sv/train.parquet \
        --out fakeflavor/data/models/flavor_genlabel.json

`--target tight` is the direct route: a binary model of P(tight | loose, x)
whose calibrated output is used *as* the fake rate, replacing the FR(conePt,
|eta|) lookup rather than reweighting it.

    python -m fakeflavor.train --target tight \
        --config fakeflavor/config/direct.yaml \
        --ntuples fakeflavor/data/ntuples/QCD_MuEnriched.parquet \
        --feature-set direct --out fakeflavor/data/models/direct.json

`--target origin` is the process test: a binary QCD-vs-ttbar model trained at
FIXED true flavor, i.e. "is a b-jet fake in ttbar a different object from a
b-jet fake in QCD?". Per-flavor FR was measured NOT to be sample-universal
(RESULTS.md §7), which is what killed the flavor-weighted route; this quantifies
the difference and says which variables carry it. Run it once per flavor:

    for FL in b c uds g; do
      python -m fakeflavor.train --target origin --flavor $FL \
        --config fakeflavor/config/origin.yaml \
        --ntuples fakeflavor/data/ntuples_sv/train.parquet \
        --feature-set local --out fakeflavor/data/models/origin_${FL}_local.json
    done

Read the `local` arm (no ev_*, so the model cannot just count jets) against the
`direct_map` arm (conePt, |eta| only), which is the part the existing flat FR
map already absorbs. Unlike `tight`, this target *wants* class balance: QCD
outweighs TTLJ by 1.1e5 in physical weight.

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

from .features import CLASSES, FEATURE_SETS
from src.utils import SEED, get_logger, load_yaml

log = get_logger("fakeflavor.train")

# --target -> (label column, class-code map). The direct target's "classes" are
# just the two arms of the tight WP; they exist so the weighting and reporting
# code below stays shared between the two routes.
#
# `genflavor` is the same 4-class problem read off the *gen* jet instead of the
# reco jet, and it is the more accurate target of the two. The reco label is
# noisy in exactly one place: matching the muon to a reco jet and reading its
# partonFlavour makes reco-`g` only ~53% true gluon, because 22.2% of reco-g
# rows have no gen jet at all and another ~24% are really uds. b and c are
# unaffected (gen-unmatched fractions 0.14% and 0.09%). Measured on the full
# Run 2 sample, switching the label is worth +0.0054 mean OvR AUC — five times
# the sv_* block — and costs only the 1.38% of rows with no gen jet, which are
# dropped below rather than kept as a fifth class (tried: that scores worse).
#
# `origin` is the odd one out: its label is not an integer column but the
# `sample` string the skim stamps on every row, so `load()` maps it. Combined
# with --flavor it answers "at fixed true flavor, is a QCD fake distinguishable
# from a ttbar fake?" — see FEATURES_LOCAL in features.py for why that matters.
TARGET_COLUMN = {"flavor": "flavor", "genflavor": "genflavor",
                 "tight": "isTight", "origin": "sample"}
TARGET_CLASSES = {"flavor": CLASSES, "genflavor": CLASSES,
                  "tight": {0: "loose", 1: "tight"},
                  "origin": {0: "QCD", 1: "TT"}}

# sample string -> origin class code. Anything not listed raises rather than
# being silently bucketed: a third process would otherwise be folded into QCD
# and quietly change what the model means.
ORIGIN_CODE = {"QCD_MuEnriched": 0, "TTLJ": 1}

# --flavor slice: reco `flavor` codes, by name. "all" keeps every row.
FLAVOR_SLICE = {v: k for k, v in CLASSES.items()}


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


def build_labels(arr, target: str) -> np.ndarray:
    """Integer class label for `target`, read off an already-loaded array.

    Shared with evaluate.py so the two cannot drift: a mismatch here would be
    invisible (the AUC would simply be scored against the wrong truth).
    """
    raw = ak.to_numpy(arr[TARGET_COLUMN[target]])
    if target != "origin":
        return raw.astype(np.int64)
    # `sample` is a string column, so the usual .astype(int64) would crash.
    unknown = sorted(set(np.unique(raw)) - set(ORIGIN_CODE))
    if unknown:
        raise ValueError(
            f"sample value(s) {unknown} have no origin code; known: "
            f"{sorted(ORIGIN_CODE)}. Add them to ORIGIN_CODE deliberately — "
            "do not let a third process fall into one of the two classes.")
    y = np.zeros(len(raw), dtype=np.int64)
    for name, code in ORIGIN_CODE.items():
        y[raw == name] = code
    return y


def load(parquet: Path, features: list[str], target: str = "flavor"):
    arr = ak.from_parquet(parquet)
    X = np.column_stack([
        ak.to_numpy(arr[f]).astype(np.float32) for f in features
    ])
    y = build_labels(arr, target)
    w = ak.to_numpy(arr["weight"]).astype(np.float64)
    meta = {k: ak.to_numpy(arr[k]) for k in ("run", "lumi", "evt", "mu_conept",
                                             "mu_abseta", "isTight",
                                             "sample", "era", "flavor")}
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


class ReduceLROnPlateau(xgb.callback.TrainingCallback):
    """Halve the learning rate when the validation metric stops improving.

    Ports the schedule from ref/2511.06972v2-2.pdf §3.2 ("if the validation
    loss does not decrease for five consecutive epochs, the learning rate is
    reduced by a factor of 0.5"), with one deliberate deviation: the paper
    counts *epochs*, i.e. full passes over the data, while a boosting round
    adds a single tree. Copying patience=5 literally would collapse the
    learning rate inside the first hundred trees, so the default is 20 rounds
    and lives in the config.

    Reads `validation_1` because main()'s eval_set is [(train), (val)] — index
    1 is the validation fold. Lower-is-better is assumed, which holds for
    mlogloss and logloss, the only metrics this project uses.
    """

    def __init__(self, metric: str, factor: float = 0.5, patience: int = 20,
                 min_lr: float = 0.005, lr0: float = 0.05,
                 eval_key: str = "validation_1") -> None:
        self.metric, self.factor, self.patience = metric, factor, patience
        self.min_lr, self.lr, self.eval_key = min_lr, lr0, eval_key
        self.best, self.wait, self.n_drops = float("inf"), 0, 0

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        hist = evals_log.get(self.eval_key, {}).get(self.metric)
        if not hist:
            return False
        cur = hist[-1]
        if cur < self.best - 1e-6:
            self.best, self.wait = cur, 0
            return False
        self.wait += 1
        if self.wait >= self.patience and self.lr > self.min_lr:
            new = max(self.lr * self.factor, self.min_lr)
            log.info("  round %d: val %s plateaued at %.5f -> learning rate "
                     "%.4g to %.4g", epoch, self.metric, cur, self.lr, new)
            self.lr, self.wait = new, 0
            self.n_drops += 1
            model.set_param("learning_rate", self.lr)
        return False    # never stop training; that is early stopping's job


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
                   help="'flavor' = 4-class source-jet flavor off the reco jet; "
                        "'genflavor' = same but off the gen jet (more accurate, "
                        "see TARGET_COLUMN); "
                        "'tight' = direct P(tight|loose,x) used as the FR; "
                        "'origin' = binary QCD vs ttbar, normally with --flavor")
    p.add_argument("--plot-dir", default="fakeflavor/plots",
                   help="parent directory for this run's per-model plot "
                        "folders. Point it at a run directory (e.g. "
                        "fakeflavor/plots/<run>/models) so a 20-model scan "
                        "adds one folder to plots/, not twenty.")
    p.add_argument("--flavor", default="all",
                   choices=["all"] + sorted(FLAVOR_SLICE),
                   help="keep only rows of this reco source-jet flavor before "
                        "training. Meant for --target origin, where the "
                        "question is posed at FIXED true flavor; it is a no-op "
                        "sanity knob for the other targets.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    classes = TARGET_CLASSES[args.target]
    n_classes = len(classes)
    # A property of the target's class count, not of its name: `tight` and
    # `origin` are both two-class problems and both need binary:logistic.
    binary = n_classes == 2
    fs_name = args.feature_set or tcfg.get("feature_set", "full")
    features = FEATURE_SETS[fs_name]
    seed = int(tcfg.get("seed", SEED))
    log.info("Target '%s' (%d classes), feature set '%s' (%d features)",
             args.target, n_classes, fs_name, len(features))

    if args.target == "tight":
        # Gated on the target, NOT on `binary`. The direct model's output *is*
        # the fake rate, so balancing or clipping decalibrates the answer. The
        # origin model is equally binary and wants the opposite: QCD outweighs
        # TTLJ by 1.1e5 in physical weight, so without class balance it would
        # see ~12 effective ttbar events. See the module docstring.
        for k in ("class_balance", "weight_clip_quantile"):
            if tcfg.get(k):
                log.warning("target=tight: forcing %s off (was %r) — the output "
                            "must stay a calibrated probability", k, tcfg[k])
            tcfg[k] = False if k == "class_balance" else None

    X, y, w_phys, meta = load(Path(args.ntuples), features, args.target)

    if args.target == "genflavor":
        # The skim guarantees `flavor` in 0..3 but says nothing about the gen
        # label, so drop the rows with no gen jet (-1) here. See TARGET_COLUMN.
        keep = np.isin(y, list(classes))
        log.info("genflavor: keeping %d/%d rows (%.2f%% have no gen jet)",
                 int(keep.sum()), len(y), 100.0 * (1.0 - keep.mean()))
        X, y, w_phys = X[keep], y[keep], w_phys[keep]
        meta = {k: v[keep] for k, v in meta.items()}

    if args.flavor != "all":
        # Applied before weighting and before the split, so class balance and
        # the weight clip are computed within the slice, which is what makes
        # the per-flavor origin AUCs comparable to each other.
        keep = meta["flavor"] == FLAVOR_SLICE[args.flavor]
        log.info("flavor slice '%s': keeping %d/%d rows",
                 args.flavor, int(keep.sum()), len(y))
        if keep.sum() == 0:
            raise ValueError(f"no rows with flavor == {args.flavor}")
        X, y, w_phys = X[keep], y[keep], w_phys[keep]
        meta = {k: v[keep] for k, v in meta.items()}

    for c, name in classes.items():
        m = y == c
        log.info("  class %-6s %8d rows  sum w = %.4g", name, int(m.sum()),
                 float(np.abs(w_phys[m]).sum()))
    if args.target == "tight":
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

    # Optional, and absent from direct.yaml on purpose: without the key the
    # callback list is empty and training behaves exactly as before.
    plateau_cfg = dict(bdt_cfg.pop("lr_plateau", {}) or {})
    callbacks = []
    if plateau_cfg.pop("enabled", False):
        sched = ReduceLROnPlateau(metric=metric,
                                  lr0=float(bdt_cfg.get("learning_rate", 0.05)),
                                  **{k: float(v) if k != "patience" else int(v)
                                     for k, v in plateau_cfg.items()})
        callbacks.append(sched)
        log.info("LR-on-plateau: factor %.2f, patience %d rounds, floor %.4g",
                 sched.factor, sched.patience, sched.min_lr)
    else:
        sched = None

    model = xgb.XGBClassifier(
        **bdt_cfg,
        callbacks=callbacks,
        **({"objective": "binary:logistic"} if binary else
           {"objective": "multi:softprob", "num_class": n_classes}),
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
    if sched is not None:
        log.info("LR-on-plateau fired %d time(s); final learning rate %.4g",
                 sched.n_drops, sched.lr)

    proba = model.predict_proba(X[idx_te])
    aucs = {}
    for c, name in classes.items():
        if binary and c == 0:
            continue    # one number, not two mirror-image ones
        aucs[name] = float(roc_auc_score((y[idx_te] == c).astype(int), proba[:, c],
                                         sample_weight=w[idx_te]))
        log.info("Test one-vs-rest AUC  %-6s = %.4f", name, aucs[name])

    if args.target == "tight":
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
        "flavor_slice": args.flavor,
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
    plot_dir = Path(args.plot_dir) / out.stem
    plot_dir.mkdir(parents=True, exist_ok=True)
    # The model file stem, not the feature set: two models sharing a feature set
    # (QCD vs ttbar `full`, or the five flavor slices of one origin arm) used to
    # silently overwrite each other's history PNG. evaluate.py names its plots
    # the same way for the same reason.
    tag = out.stem
    fig.savefig(plot_dir / "training_history.png", dpi=300)
    plt.close(fig)
    log.info("Saved training history to %s",
             plot_dir / "training_history.png")


if __name__ == "__main__":
    main()
