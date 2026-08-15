"""Phase 3: evaluate the flavor BDT.

Produces, on the held-out test set:
  1. one-vs-rest ROC / AUC per class, inclusive
  2. row-normalized confusion matrix
  3. gain feature importance
  4. per-true-class score distributions
  5. **AUC inside each (conePt, |eta|) FR cell** — the headline number. A
     classifier that only rediscovers the existing 2D FR binning adds nothing
     to a flavor-weighted fake rate.
  6. **leakage diagnostic** — FR vs predicted-flavor probability within each
     *true* flavor. Must be flat; see the docstring of `leakage_check`.
     Flavor targets only.

    python -m fakeflavor.evaluate --model fakeflavor/data/models/flavor_full.json

Target-aware since 2026-08-09: the label column, the class list and any row
filter are read from the model's own `.meta.json`, so the same command scores a
4-class flavor model, a binary `tight` model and a per-flavor `origin` model.
Metas written before that date carry no `target` key and default to `flavor`,
which is what they all were. Plot filenames key on the **model file stem**, not
the feature set — two models sharing a feature set used to overwrite each other.

Cross-process transfer (train QCD, test TTLJ) is not done here: pass a different
`--ntuples` with `--all-rows`, or use `closure.py`, which is built for it.
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .features import CLASSES, N_CLASSES
from .train import FLAVOR_SLICE, build_labels, build_weights, cell_mask
from src.utils import get_logger, load_yaml

log = get_logger("fakeflavor.evaluate")

PLOT_DIR = Path("fakeflavor/plots")


def load_model(model_path: Path):
    """Load a saved model, restoring base_score explicitly.

    xgboost 2.1.4 does not round-trip `base_score` through the sklearn wrapper:
    the file stores the fitted value, but `XGBClassifier().load_model()` leaves
    the booster configured at the constructor default 0.5 and reports no error.
    For a *classifier* that is usually invisible, since a constant offset to the
    margin does not change the ranking and therefore not the AUC. Here it is
    fatal: the output of the direct model *is* the fake rate, so an offset of
    logit(0.258) - logit(0.5) = -1.06 in the margin inflates every prediction.
    Measured on QCD: sum_i w_i p_i / sum_{tight} w_i = 1.53 unpatched, 1.0029
    patched, the latter being exactly the number train.py recorded in-memory.

    Re-applying the value from the JSON is the fix; passing base_score= to the
    constructor is not (load_model overwrites it).
    """
    meta = json.loads(model_path.with_suffix(".meta.json").read_text())
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    saved = json.loads(Path(model_path).read_text())
    base = saved["learner"]["learner_model_param"].get("base_score")
    if base is not None:
        vals = [float(v) for v in base.strip("[]").split(",")]
        if len(vals) == 1:
            model.get_booster().set_param("base_score", vals[0])
        else:
            # multi:softprob stores one intercept per class and set_param takes
            # only a scalar. Harmless in practice for the flavor models (values
            # are O(1e-3) against class priors of O(0.1)) and irrelevant to the
            # prior-corrected outputs, but say so rather than pass silently.
            log.warning("%s has a vector base_score %s that cannot be restored "
                        "after load; per-class margins are offset by that much",
                        model_path.name, vals)
    return model, meta


def prior_correct(proba: np.ndarray, train_prior: np.ndarray,
                  true_prior: np.ndarray) -> np.ndarray:
    """Undo class balancing.

    Training with class_balance=true fits P(flavor|x) under a uniform prior.
    To get the probability under the physical mixture, multiply by the ratio
    of priors and renormalize:  P_true ∝ P_bal x (pi_true / pi_train).
    """
    adj = proba * (true_prior / np.maximum(train_prior, 1e-12))[None, :]
    return adj / np.maximum(adj.sum(axis=1, keepdims=True), 1e-12)


def cell_auc(y, proba, w, conept, abseta, ptbins, etabins,
             classes: dict | None = None) -> None:
    """One-vs-rest AUC computed separately inside every FR cell.

    This is the question the whole study rests on: the flat FR map already
    knows (conePt, |eta|), so any discrimination the BDT provides *inside* a
    cell is genuinely new information. Cells with fewer than 50 rows of a class
    are reported as '-'.

    Equally meaningful for the origin target, where it reads as: is QCD still
    separable from ttbar *within* one cell of the map the fake rate is
    currently binned in?
    """
    classes = CLASSES if classes is None else classes
    # For a binary target the two columns are mirror images (AUC_0 = 1 - AUC_1),
    # so report only the positive class — same convention as the ROC block.
    cols = [c for c in classes if not (len(classes) == 2 and c == 0)]
    log.info("--- one-vs-rest AUC inside each FR cell (conePt x |eta|)")
    header = "  ".join(f"{classes[c]:>5s}" for c in cols)
    log.info("  %-18s %-12s  %s   n", "conePt", "|eta|", header)
    for i in range(len(ptbins) - 1):
        for j in range(len(etabins) - 1):
            m = ((conept >= ptbins[i]) & (conept < ptbins[i + 1])
                 & (abseta >= etabins[j]) & (abseta < etabins[j + 1]))
            if m.sum() < 100:
                continue
            cells = []
            for c in cols:
                pos = (y[m] == c).sum()
                if pos < 50 or pos == m.sum():
                    cells.append("    -")
                    continue
                a = roc_auc_score((y[m] == c).astype(int), proba[m, c],
                                  sample_weight=w[m])
                cells.append(f"{a:5.3f}")
            log.info("  [%5.0f,%5.0f)      [%.1f,%.1f)   %s   %d",
                     ptbins[i], ptbins[i + 1], etabins[j], etabins[j + 1],
                     "  ".join(cells), int(m.sum()))


def leakage_check(y, proba, w, is_tight, out_dir: Path) -> None:
    """FR vs predicted-flavor probability, within each TRUE flavor.

    Why this matters: `mu_miniiso`, `mu_sip3d`, `mu_tkreliso` and `mu_conept`
    are the variables that *define* the tight WP, i.e. the FR numerator
    (conePt literally contains miniiso). Using them to predict flavor is
    legitimate, but a flavor-weighted FR built on the prediction is only valid
    if, at fixed true flavor, the prediction carries no information about
    tightness. So bin by the predicted probability of the true class and plot
    FR = N(tight)/N(loose) in each bin: it must be flat.

    If it is not flat, retrain with the WP-defining variables removed and
    compare — the difference is the size of the bias.
    """
    log.info("--- leakage diagnostic: FR vs predicted p(true class), per true flavor")
    edges = np.linspace(0.0, 1.0, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(4 * N_CLASSES, 3.5), sharey=True)
    for c, name in CLASSES.items():
        ax = axes[c]
        m = y == c
        if m.sum() < 100:
            ax.set_title(f"{name} (too few rows)")
            continue
        p = proba[m, c]
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        fr, err = [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            den = w[m][sel].sum()
            num = w[m][sel & is_tight[m]].sum()
            n_eff = den ** 2 / max((w[m][sel] ** 2).sum(), 1e-30)
            # A bin with a handful of effective entries, or one that happens to
            # be pure loose / pure tight, gives a binomial error of ~0 and would
            # otherwise contribute an unbounded term to the chi2 below.
            if den <= 0 or n_eff < 50:
                fr.append(np.nan)
                err.append(np.nan)
                continue
            r = num / den
            fr.append(r)
            err.append(max(np.sqrt(max(r * (1 - r), 0) / n_eff), 0.5 / n_eff))
        fr, err = np.array(fr), np.array(err)
        flat = w[m][is_tight[m]].sum() / max(w[m].sum(), 1e-30)
        ok = np.isfinite(fr)
        if ok.sum() > 1:
            chi2 = np.nansum(((fr[ok] - flat) / np.maximum(err[ok], 1e-9)) ** 2)
            log.info("    %-4s inclusive FR=%.4f   chi2/ndf vs flat = %.1f/%d",
                     name, flat, chi2, int(ok.sum() - 1))
        ax.errorbar(centers, fr, yerr=err, fmt="o", color=f"C{c}")
        ax.axhline(flat, color="grey", ls="--", lw=1)
        ax.set_title(f"true {name}")
        ax.set_xlabel(f"predicted p({name})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("FR = N(tight)/N(loose)")
    fig.suptitle("Leakage diagnostic — flat means the prediction is independent "
                 "of tightness at fixed true flavor")
    fig.tight_layout()
    fig.savefig(out_dir / "leakage.png", dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fakeflavor/data/models/flavor_full.json")
    p.add_argument("--ntuples", default="fakeflavor/data/ntuples/train.parquet")
    p.add_argument("--config", default="fakeflavor/config/train.yaml")
    p.add_argument("--plot-dir", default=str(PLOT_DIR),
                   help="parent directory for this model's plot folder; see "
                        "the same flag on train.py")
    p.add_argument("--prior-correct", action="store_true",
                   help="rescale probabilities back to the physical class mixture")
    p.add_argument("--all-rows", action="store_true",
                   help="evaluate on every row instead of the held-out test set")
    args = p.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    model_path = Path(args.model)
    model, meta = load_model(model_path)
    features = meta["features"]
    # One directory per model, named by the model file stem, with short file
    # names inside. Keyed on meta["feature_set"] before 2026-08-09, which meant
    # models sharing a feature set (QCD vs ttbar `full`, or the flavor slices of
    # one origin arm) silently overwrote each other's plots — and 21 models
    # writing 6 PNGs each into one flat directory was unreadable regardless.
    # train.py puts its history PNG in the same directory.
    tag = model_path.stem
    out_dir = Path(args.plot_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Target-aware. Metas written before 2026-08-09 carry no "target" key and
    # are all 4-class reco-flavor models, which is the default.
    target = meta.get("target", "flavor")
    classes = {int(k): v for k, v in meta["classes"].items()}
    n_classes = len(classes)
    flavor_slice = meta.get("flavor_slice", "all")
    cell = meta.get("cell")
    log.info("Model %s: target '%s', %d classes %s, flavor slice '%s', cell %s",
             model_path.name, target, n_classes, list(classes.values()),
             flavor_slice, cell or "-")

    arr = ak.from_parquet(args.ntuples)
    X = np.column_stack([ak.to_numpy(arr[f]).astype(np.float32) for f in features])
    y = build_labels(arr, target)
    w = np.abs(ak.to_numpy(arr["weight"]).astype(np.float64))
    conept = ak.to_numpy(arr["mu_conept"])
    abseta = ak.to_numpy(arr["mu_abseta"])
    is_tight = ak.to_numpy(arr["isTight"]).astype(bool)

    # The row filters train.py applied before splitting, replayed in the same
    # order — otherwise the saved test indices point at the wrong rows.
    if target == "genflavor":
        keep = np.isin(y, list(classes))
        X, y, w, conept, abseta, is_tight = (a[keep] for a in
                                             (X, y, w, conept, abseta, is_tight))
    if flavor_slice != "all":
        keep = ak.to_numpy(arr["flavor"]) == FLAVOR_SLICE[flavor_slice]
        if target == "genflavor":
            raise NotImplementedError("genflavor + flavor slice not supported")
        log.info("Replaying flavor slice '%s': %d rows", flavor_slice,
                 int(keep.sum()))
        X, y, w, conept, abseta, is_tight = (a[keep] for a in
                                             (X, y, w, conept, abseta, is_tight))
    if cell:
        # After the flavor slice, because that is the order train.py applies
        # them in. Built from the already-filtered conept/abseta locals, not
        # from `arr`, for the same reason. `cell_mask` is imported from train.py
        # rather than reimplemented so the two boundary conventions cannot
        # drift apart by a row.
        keep = cell_mask(conept, abseta, cell)
        log.info("Replaying cell '%s': %d rows", cell, int(keep.sum()))
        X, y, w, conept, abseta, is_tight = (a[keep] for a in
                                             (X, y, w, conept, abseta, is_tight))

    if not args.all_rows:
        split = np.load(model_path.with_suffix(".split.npz"))
        idx = split["test"]
        log.info("Evaluating on the held-out test set: %d rows", len(idx))
        X, y, w, conept, abseta, is_tight = (a[idx] for a in
                                             (X, y, w, conept, abseta, is_tight))

    proba = model.predict_proba(X)

    if args.prior_correct:
        train_prior = np.full(n_classes, 1.0 / n_classes)   # class_balance=true
        true_prior = np.array([w[y == c].sum() for c in classes])
        true_prior /= true_prior.sum()
        proba = prior_correct(proba, train_prior, true_prior)
        log.info("Prior-corrected to physical mixture: %s",
                 {classes[c]: round(float(true_prior[c]), 4) for c in classes})

    # 1. inclusive ROC.
    #
    # Two weightings, and they differ by a lot — quote the right one. `phys` is
    # the analyzer weight, i.e. AUC in the population the FR is actually applied
    # to, where b dominates. `bal` is the class-balanced training weight and is
    # the number train.py prints; it is the one to compare across models,
    # because it does not move when the class mixture does. Curves are drawn
    # with `phys`.
    w_bal = build_weights(y, w, cfg["train"], classes)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c, name in classes.items():
        if n_classes == 2 and c == 0:
            continue        # one number, not two mirror-image ones
        yb = (y == c).astype(int)
        auc = roc_auc_score(yb, proba[:, c], sample_weight=w)
        auc_bal = roc_auc_score(yb, proba[:, c], sample_weight=w_bal)
        fpr, tpr, _ = roc_curve(yb, proba[:, c], sample_weight=w)
        ax.plot(fpr, tpr, label=f"{name} vs rest (AUC {auc:.3f})", color=f"C{c}")
        log.info("Inclusive one-vs-rest AUC  %-4s = %.4f (phys)  %.4f (class-balanced)",
                 name, auc, auc_bal)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{target} BDT ({tag}) — held-out test set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=300)
    plt.close(fig)

    # 2. confusion matrix (row-normalized: P(predicted | true))
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=list(classes), sample_weight=w)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-30)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    names = [classes[c] for c in classes]
    ax.set_xticks(range(n_classes), names)
    ax.set_yticks(range(n_classes), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_title(f"confusion ({tag}), row-normalized")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png", dpi=300)
    plt.close(fig)
    log.info("Confusion (P(pred|true)):\n%s", np.round(cm, 3))

    # 3. gain importance
    booster = model.get_booster()
    booster.feature_names = list(features)
    gain = booster.get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:25]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1])
    ax.set_xlabel("gain")
    ax.set_title(f"top 25 features by gain ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / "importance.png", dpi=300)
    plt.close(fig)
    log.info("Top 10 by gain: %s", [(k, round(v, 1)) for k, v in top[:10]])

    # 4. per-true-class score distributions
    # squeeze=False so a 2-class model still gets an indexable axes array.
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3.5),
                             squeeze=False)
    axes = axes[0]
    for c, name in classes.items():
        ax = axes[c]
        for c2, name2 in classes.items():
            m = y == c2
            ax.hist(proba[m, c], bins=40, range=(0, 1), weights=w[m],
                    histtype="step", density=True, label=f"true {name2}",
                    color=f"C{c2}")
        ax.set_yscale("log")
        ax.set_xlabel(f"p({name})")
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"score distributions ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / "scores.png", dpi=300)
    plt.close(fig)

    # 5. the headline: does it add anything inside an FR cell?
    ecfg = cfg["evaluate"]
    cell_auc(y, proba, w, conept, abseta,
             np.array(ecfg["ptcorr_bins"]), np.array(ecfg["abseta_bins"]),
             classes)

    # 6. leakage. Defined against a TRUE FLAVOR — it asks whether the predicted
    # flavor probability carries tightness information at fixed flavor — so it
    # is meaningless for the origin and tight targets and is skipped there.
    if target in ("flavor", "genflavor"):
        leakage_check(y, proba, w, is_tight, out_dir)

    log.info("Plots written to %s", out_dir)


if __name__ == "__main__":
    main()
