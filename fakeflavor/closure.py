"""Physics closure test for candidate fake rates (TODO.md Step E item 3).

The question: does any more finely parametrized fake rate predict the observed
number of tight muons in a *different* process better than the flat
FR(ptCorr, |eta|) map already used by bkg.cc?

Every method reduces to one per-muon number f_i, and predictions of N(tight) are
compared against the observed count in TTLJ with all FR maps measured on QCD
MuEnriched (the "FR is sample-universal" assumption from the v4 flavor study is
exactly what is being tested):

    truth   f_i = FR_{f_i}(ptcorr_i, |eta|_i)         -- MC-truth flavor, the best
                                                        the flavor route could do
    bdt     f_i = Sum_f p_f(x_i) * FR_f(...)          -- deployable flavor route
    flat    f_i = FR_incl(ptcorr_i, |eta|_i)          -- what bkg.cc does today
    direct  f_i = P(tight | loose, x_i)               -- --direct-model, the direct
                                                        route (target=tight)

`truth` is the ceiling of the flavor route: if it does not beat `flat`, that route
is pointless regardless of how good the classifier is.

**Two forms of every prediction are reported, and they are not equivalent.**

    rate    sum_over_all_loose  w_i f_i
    matrix  sum_over_loose_and_NOT_tight  w_i F_i ,   F = f/(1-f)

`rate` tests whether the parametrization transports between processes. `matrix`
is what bkg.cc actually computes (Analyzers/src/bkg.cc:237-252, f clamped to
[0.001, 0.950]): the FR is evaluated only on muons that *failed* tight. F is the
*fake factor*, and in the fake-factor literature (arXiv:2511.06972) it is the
primary object -- the density ratio p^T(x)/p^L(x), estimated directly by a
classifier separating tight from loose-not-tight, F = exp(q) with q the logit.
Our direct model is exactly that classifier, so f and F are two readings of one
fit; `matrix` is reported from the exact margin q wherever one exists. Both are
unbiased when f is the true conditional -- the identity is
sum_i w_i (1-f_i) * f_i/(1-f_i) = sum_i w_i f_i -- so `matrix` is not a second
physics test, it is an *efficiency* test. A model whose inputs nearly determine
tightness returns f ~ 0 on every fail-side muon and its `matrix` prediction
collapses to zero while its `rate` prediction still looks perfect. That is the
failure mode the feature-set rule in features.py exists to prevent, and quoting
`rate` alone would hide it completely. `n_clamp` counts fail-side muons pushed to
the 0.95 ceiling; a nonzero count means the same illness.

Population. Everything is restricted to the rows in the parquet, i.e. non-prompt
loose muons with flavor in {b,c,uds,g} -- pileup and unmatched are excluded from
both the maps and the observed count, so the comparison is self-consistent and
the pileup/unmatched handling question is deferred (see TODO.md Step E note).
For the same reason `flat` is built as Sum_f tight_f / Sum_f loose_f over those
four flavors rather than from FR2D_QCD.root, whose denominator is all loose
muons: mixing populations would confound the method comparison with a
selection difference.

Prior correction. The model is trained class-balanced, so its raw outputs are
posteriors under a uniform prior and do not sum to a calibrated expectation.
Sum_f p_f * FR_f is only an estimate of E[tight|x] if p_f is calibrated in the
population it is applied to, so three variants are reported:
  uniform  raw class-balanced output (no correction) -- diagnostic only
  qcd      corrected to the QCD training mixture -- the honest deployable choice,
           since the target composition is not known without MC truth
  ttlj     corrected to TTLJ's own truth mixture -- cheating, quoted as the
           optimistic bound on what calibration could buy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from src.utils import get_logger, load_yaml
from .evaluate import PLOT_DIR, load_model, prior_correct
from .features import CLASSES, N_CLASSES
from .skim import STITCH_PTCORR

log = get_logger(__name__)

FR_ROOT = Path("/data6/Users/snuintern2/fake")
# u/d/s are separate directories in the analyzer output but one BDT class.
FLAVOR_MEMBERS = {"b": ["b"], "c": ["c"], "uds": ["u", "d", "s"], "g": ["g"]}
PATHS = {0: "Mu8", 1: "Mu17"}
HIST = "MeasReg/Central/{wp}/recoflav/{flav}/fake_ptcorr_abseta"


def load_fr_maps(eras: list[str], sample: str = "QCD", suffix: str = "flavor"):
    """Sum loose/tight counts over eras, then divide -> FR per (path, flavor).

    recoflav (not genflav) because the parquet's `flavor` column comes from
    fake::FlavorCode, which matches the jet by NanoAOD Muon_jetIdx. Mixing the
    two schemes would compare different labels.

    Returns (fr, err, edges) where fr[path][flavor] is a 2D array and `incl` is
    the flavor-summed inclusive map over the same four flavors.
    """
    num = {p: {} for p in PATHS}
    den = {p: {} for p in PATHS}
    edges = None

    for era in eras:
        path_root = FR_ROOT / f"{era}_{suffix}" / f"{sample}.root"
        if not path_root.exists():
            raise FileNotFoundError(f"missing per-flavor FR input: {path_root}")
        with uproot.open(path_root) as f:
            for pcode, pname in PATHS.items():
                for flav, members in FLAVOR_MEMBERS.items():
                    for wp, acc in (("tight", num), ("loose", den)):
                        for member in members:
                            key = f"{pname}/" + HIST.format(wp=wp, flav=member)
                            h = f[key]
                            v = h.values()
                            if edges is None:
                                edges = (h.axis(0).edges(), h.axis(1).edges())
                            acc[pcode][flav] = acc[pcode].get(flav, 0.0) + v
                            # Sumw2 -> variance, summed the same way.
                            e2key = (pcode, flav, wp)
                            acc[pcode].setdefault("_var", {})
                            acc[pcode]["_var"][e2key] = (
                                acc[pcode]["_var"].get(e2key, 0.0) + h.errors() ** 2)

    fr, err = {}, {}
    for pcode in PATHS:
        fr[pcode], err[pcode] = {}, {}
        tot_n = sum(num[pcode][f] for f in FLAVOR_MEMBERS)
        tot_d = sum(den[pcode][f] for f in FLAVOR_MEMBERS)
        for flav in list(FLAVOR_MEMBERS) + ["incl"]:
            if flav == "incl":
                n, d = tot_n, tot_d
                vn = sum(num[pcode]["_var"][(pcode, f, "tight")]
                         for f in FLAVOR_MEMBERS)
                vd = sum(den[pcode]["_var"][(pcode, f, "loose")]
                         for f in FLAVOR_MEMBERS)
            else:
                n, d = num[pcode][flav], den[pcode][flav]
                vn = num[pcode]["_var"][(pcode, flav, "tight")]
                vd = den[pcode]["_var"][(pcode, flav, "loose")]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(d > 0, n / np.maximum(d, 1e-30), 0.0)
                # var(n/d) with n subset of d: binomial-like, but the weighted
                # counts are not integers, so propagate the stored Sumw2 terms
                # treating numerator and denominator errors as independent
                # (conservative -- they are positively correlated).
                v = np.where(d > 0,
                             (vn + r ** 2 * vd) / np.maximum(d ** 2, 1e-30), 0.0)
            fr[pcode][flav] = r
            err[pcode][flav] = np.sqrt(np.maximum(v, 0.0))
    return fr, err, edges


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index with the analyzer's overflow convention (last bin absorbs it)."""
    clipped = np.clip(values, edges[0], edges[-1] - 1e-6)
    return np.clip(np.searchsorted(edges, clipped, side="right") - 1,
                   0, len(edges) - 2)


def lookup(fr_by_path, flavname, ix, iy, path):
    """Per-muon FR, picking the map of the muon's own trigger path."""
    out = np.empty(len(ix))
    for pcode in PATHS:
        m = path == pcode
        if m.any():
            out[m] = fr_by_path[pcode][flavname][ix[m], iy[m]]
    return out


def map_stat_err(coef, fr_err_by_path, flavname, ix, iy, path, nx, ny):
    """FR-map statistical error on sum_i coef_i * FR(bin_i).

    `coef` is d(prediction)/d(FR) per muon: just w_i for the rate form, and
    w_i / (1 - f_i)^2 on fail-side muons for the matrix form.

    The map error is fully correlated among muons landing in the same bin, so
    the coefficients must be summed per bin *before* squaring -- treating them
    as independent would understate this badly for the thin light-flavor bins.
    """
    var = 0.0
    for pcode in PATHS:
        m = path == pcode
        if not m.any():
            continue
        flat = ix[m] * ny + iy[m]
        wsum = np.bincount(flat, weights=coef[m], minlength=nx * ny)
        e = fr_err_by_path[pcode][flavname].reshape(-1)
        var += float(((wsum * e) ** 2).sum())
    return np.sqrt(var)


# bkg.cc::GetFakeRate clamps the looked-up FR before forming e = f/(1-f).
FR_CLAMP_LO, FR_CLAMP_HI = 0.001, 0.950


def logit(f):
    """log(f/(1-f)) = log F, the fake *factor* in log space."""
    f = np.clip(f, 1e-300, 1.0 - 1e-16)
    return np.log(f) - np.log1p(-f)


def forms(w, f, is_tight, q=None):
    """Predictions of N(tight) from one per-muon fake rate. See module doc.

    `q = log(f/(1-f))` is the log fake *factor*, and it, not `f`, is the natural
    variable of the matrix form. arXiv:2511.06972 §3.1 makes the point cleanly:
    the object the matrix/fake-factor method applies is the density ratio
    F = p^T/p^L = f/(1-f), and a binary classifier separating tight from
    loose-and-not-tight estimates it directly as F = exp(q) with q the raw
    logit -- sigma(q) = f and exp(q) = F are the same fit read two ways. Passing
    the exact margin therefore matters twice:

      * precision. Recovering F from a float32 probability loses everything once
        f -> 1: at f = 1 - 1e-7 the probability has ~1 significant digit left
        while q is still exact. The leak-demo model lives entirely in that
        regime, so its matrix number is meaningless when computed from f.
      * conditioning. Errors in q are *multiplicative* in F, which is what the
        matrix sum is linear in; errors in f get amplified by 1/(1-f).

    Also reports how well *conditioned* the matrix form is, which turns out to
    be the property that decides whether a sharper FR model is usable at all.
    F diverges as f -> 1, so a fail-side muon the model is confident about
    contributes an enormous term. Two symptoms are tracked:

      matrix_clamped  the sum as bkg.cc would compute it, with f clamped to
                  0.95 first. If it differs from the exact `matrix`, the
                  estimate is being decided by the clamp rather than the physics.
      top1pct     fraction of the exact matrix sum carried by the largest 1% of
                  terms. The flat map sits near a few percent; a value near 1
                  means the whole background estimate rests on a handful of
                  muons and its variance is uncontrolled.
    """
    if q is None:
        q = logit(f)
    else:
        f = 1.0 / (1.0 + np.exp(-q))    # keep f and q exactly consistent
    fail = ~is_tight
    rate = float((w * f).sum())

    terms = w[fail] * np.exp(q[fail])           # exact F, no 1/(1-f) roundoff
    matrix = float(terms.sum())
    fc = np.clip(f[fail], FR_CLAMP_LO, FR_CLAMP_HI)
    matrix_clamped = float((w[fail] * fc / (1.0 - fc)).sum())

    k = max(1, int(round(0.01 * len(terms))))
    top1 = float(np.sort(terms)[-k:].sum() / matrix) if matrix > 0 else float("nan")
    return {"rate": rate, "matrix": matrix, "matrix_clamped": matrix_clamped,
            "n_clamp": int((f[fail] >= FR_CLAMP_HI).sum()), "top1pct": top1}


def matrix_calibration(w, q, is_tight, nbins=10):
    """Per-score-bin audit of the matrix form: predicted vs measured F.

    The rate form needs f to equal the mean tight fraction of its own level set;
    the matrix form needs F to equal n/(d-n) there, and those are *different*
    conditions on the same model once f varies inside the bin (Jensen: the mean
    of F is not F of the mean). A model can therefore close perfectly in `rate`
    and still miss in `matrix`, which is exactly the pattern seen for `direct`
    (1.0007 vs 0.9384). This table localizes the miss instead of leaving it as
    one inclusive number.
    """
    fail = ~is_tight
    edges = weighted_quantile_edges(q, w, nbins)
    ib = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        m = ib == i
        if not m.any():
            continue
        n = float((w[m] * is_tight[m]).sum())
        d = float(w[m].sum())
        wf = float(w[m & fail].sum())
        if wf <= 0:
            continue
        f_meas = n / d
        F_meas = n / max(d - n, 1e-30)
        F_pred = float((w[m & fail] * np.exp(q[m & fail])).sum() / wf)
        rows.append((edges[i], edges[i + 1], d, f_meas,
                     float(np.average(1.0 / (1.0 + np.exp(-q[m])), weights=w[m])),
                     F_meas, F_pred))
    return rows


def weighted_quantile_edges(v, w, nbins):
    """Bin edges cutting `v` into nbins bins of equal total weight."""
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    inner = np.interp(np.linspace(0, 1, nbins + 1)[1:-1], cw, v[o])
    return np.concatenate([[-np.inf], np.unique(inner), [np.inf]])


def merge_top_bins(fr, n, d, cap):
    """Merge the highest score bins until the merged bin's measured FR <= cap.

    Bounding f is what makes the matrix form usable, and merging is the way to
    bound it *without* breaking calibration: the merged value is still the mean
    measured rate of its level set, which is exactly the condition for
    sum_fail w f/(1-f) to be unbiased. Simply clipping f would bound it too, but
    would put a wrong (uncalibrated) value on the level set and bias the result --
    which is precisely what bkg.cc's blind clamp at 0.95 does today.
    """
    k = len(fr)
    while k > 1:
        num, den = n[k - 1:].sum(), d[k - 1:].sum()
        if den <= 0 or num / den > cap:
            k -= 1
        else:
            break
    merged = n[k - 1:].sum() / max(d[k - 1:].sum(), 1e-30)
    out = fr.copy()
    out[k - 1:] = merged
    return out, k - 1, merged


def quantize(q_src, w_src, tight_src, q_tgt, nbins, cap=None):
    """Replace a continuous model score by the FR measured in bins of that score.

    This is the deployment format, not a cosmetic step: what would actually ship
    is a lookup table, and turning the score into one is also the fix for the
    matrix-form conditioning problem. `f` can then take only nbins values, each
    an average over muons that mostly failed, so it is bounded well away from 1
    and `F = f/(1-f)` stays finite. The FR per bin is measured on the *source*
    sample, exactly as the flat FR(conePt,|eta|) map is, which keeps the
    comparison like-for-like.

    Binning is done in q = log F. That changes nothing about *which* muons share
    a bin -- quantile edges are invariant under the monotone map q <-> f -- but
    it does fix what each bin stores. Measuring both n/d and n/(d-n) from the
    same two sums makes the bin simultaneously calibrated for the rate form and
    for the matrix form, which is the property the continuous score does not
    have (see matrix_calibration). Returns the per-target-muon q of its bin.
    """
    edges = weighted_quantile_edges(q_src, w_src, nbins)
    isrc = np.clip(np.searchsorted(edges, q_src, side="right") - 1,
                   0, len(edges) - 2)
    itgt = np.clip(np.searchsorted(edges, q_tgt, side="right") - 1,
                   0, len(edges) - 2)
    n = np.bincount(isrc, weights=w_src * tight_src, minlength=len(edges) - 1)
    d = np.bincount(isrc, weights=w_src, minlength=len(edges) - 1)
    fr = np.where(d > 0, n / np.maximum(d, 1e-30), 0.0)
    if cap is not None:
        fr, kmerged, merged = merge_top_bins(fr, n, d, cap)
        log.info("  merged score bins %d-%d into one, measured FR %.4f (cap %.2f)",
                 kmerged, len(fr) - 1, merged, cap)
    return logit(fr)[itgt], fr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fakeflavor/data/models/flavor_full.json")
    p.add_argument("--target", default="fakeflavor/data/ntuples/TTLJ.parquet",
                   help="process the FR maps are applied to")
    p.add_argument("--source", default="fakeflavor/data/ntuples/QCD_MuEnriched.parquet",
                   help="process the training mixture is taken from (prior 'qcd')")
    p.add_argument("--samples", default="fakeflavor/config/samples.yaml")
    p.add_argument("--fr-sample", default="QCD",
                   help="sample name of the per-flavor FR maps")
    p.add_argument("--quantize", type=int, default=0,
                   help="also report each direct model with its score replaced by "
                        "the FR measured in N equal-weight score bins on --source. "
                        "This is the deployment format and the fix for matrix-form "
                        "conditioning; 0 disables.")
    p.add_argument("--fr-cap", type=float, default=None,
                   help="with --quantize: merge the top score bins until the "
                        "merged bin's measured FR is below this, bounding "
                        "e = f/(1-f) without breaking calibration.")
    p.add_argument("--split-from", default=None,
                   help="restrict the target to the held-out test rows recorded "
                        "in <model>.split.npz. Required for a self-closure "
                        "control on the process a direct model was trained on, "
                        "otherwise the model is scored in-sample. Note the "
                        "map-based rows are in-sample regardless: the FR maps "
                        "are measured on all of it.")
    p.add_argument("--direct-model", action="append", default=[],
                   help="model trained with --target tight, whose output is used "
                        "directly as the fake rate. Repeatable.")
    p.add_argument("--fr-suffix", default="flavor",
                   help="era-dir suffix of the FR maps: 'flavor' (QCD) or "
                        "'flavor_ttlj'. Measuring and applying in the same "
                        "process is the self-closure control and must give 1.0")
    args = p.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    eras = [str(e) for e in load_yaml(args.samples)["eras"]]
    fr, fr_err, edges = load_fr_maps(eras, args.fr_sample, args.fr_suffix)
    ptedges, etaedges = edges
    nx, ny = len(ptedges) - 1, len(etaedges) - 1
    log.info("FR maps from %s over %s: ptcorr %s x |eta| %s",
             args.fr_sample, ",".join(eras), list(ptedges), list(etaedges))
    for flav in list(FLAVOR_MEMBERS) + ["incl"]:
        log.info("  FR_%-4s Mu8 [10,12)x[0,0.9) = %.4f +- %.4f", flav,
                 fr[0][flav][0, 0], fr_err[0][flav][0, 0])

    model, meta = load_model(Path(args.model))
    features = meta["features"]

    arr = ak.from_parquet(args.target)
    if args.split_from:
        idx = np.load(Path(args.split_from).with_suffix(".split.npz"))["test"]
        if idx.max() >= len(arr):
            raise SystemExit(
                f"--split-from {args.split_from} indexes {idx.max() + 1} rows but "
                f"{args.target} has {len(arr)}: the split belongs to a different "
                "ntuple than the target")
        arr = arr[idx]
        log.info("Restricted to %d held-out rows from %s", len(arr), args.split_from)
    X = np.column_stack([ak.to_numpy(arr[f]).astype(np.float32) for f in features])
    y = ak.to_numpy(arr["flavor"]).astype(np.int64)
    w = np.abs(ak.to_numpy(arr["weight"]).astype(np.float64))
    conept = ak.to_numpy(arr["mu_conept"]).astype(np.float64)
    abseta = ak.to_numpy(arr["mu_abseta"]).astype(np.float64)
    is_tight = ak.to_numpy(arr["isTight"]).astype(bool)
    path = ak.to_numpy(arr["path"]).astype(np.int64)
    log.info("Target %s: %d rows", args.target, len(y))

    ix, iy = bin_index(conept, ptedges), bin_index(abseta, etaedges)
    names = [CLASSES[c] for c in sorted(CLASSES)]

    # Sanity: the stitch column must agree with the ptcorr boundary, otherwise
    # the wrong trigger's map gets applied.
    expect = (conept >= STITCH_PTCORR).astype(np.int64)
    if not np.array_equal(expect, path):
        log.error("path column disagrees with the ptcorr %g stitch in %d rows",
                  STITCH_PTCORR, int((expect != path).sum()))

    proba_raw = model.predict_proba(X)

    src = ak.from_parquet(args.source)
    ysrc = ak.to_numpy(src["flavor"]).astype(np.int64)
    wsrc = np.abs(ak.to_numpy(src["weight"]).astype(np.float64))
    prior_src = np.array([wsrc[ysrc == c].sum() for c in sorted(CLASSES)])
    prior_src /= prior_src.sum()
    prior_tgt = np.array([w[y == c].sum() for c in sorted(CLASSES)])
    prior_tgt /= prior_tgt.sum()
    uniform = np.full(N_CLASSES, 1.0 / N_CLASSES)
    log.info("Composition  source(%s) = %s", Path(args.source).stem,
             {n: round(float(v), 4) for n, v in zip(names, prior_src)})
    log.info("Composition  target(%s) = %s", Path(args.target).stem,
             {n: round(float(v), 4) for n, v in zip(names, prior_tgt)})

    variants = {
        "uniform": proba_raw,
        "qcd": prior_correct(proba_raw, uniform, prior_src),
        "ttlj": prior_correct(proba_raw, uniform, prior_tgt),
    }

    # --- per-muon fake rate for each method ------------------------------
    obs = float((w * is_tight).sum())
    obs_err = float(np.sqrt((w[is_tight] ** 2).sum()))

    fr_truth = np.zeros(len(y))
    for c, name in CLASSES.items():
        m = y == c
        if m.any():
            fr_truth[m] = lookup(fr, name, ix[m], iy[m], path[m])
    fr_flat = lookup(fr, "incl", ix, iy, path)

    series = {"truth": fr_truth, "flat": fr_flat}
    # log fake factor q = log(f/(1-f)) where an exact one is available. For the
    # histogram maps f is a ratio of two sums and q is derived from it; for a
    # BDT it is the other way round -- q is the raw margin and f is derived.
    margins: dict[str, np.ndarray] = {}
    for vname, proba in variants.items():
        f = np.zeros(len(y))
        for c, name in CLASSES.items():
            f += proba[:, c] * lookup(fr, name, ix, iy, path)
        series[f"bdt[{vname}]"] = f

    # The direct route: the model output *is* the fake rate, no map lookup and
    # no prior correction (target=tight is trained uncalibrated-free by design).
    for mpath in args.direct_model:
        dmodel, dmeta = load_model(Path(mpath))
        if dmeta.get("target") != "tight":
            raise SystemExit(f"{mpath} has target={dmeta.get('target')!r}, "
                             "expected 'tight' (train.py --target tight)")
        Xd = np.column_stack([ak.to_numpy(arr[f]).astype(np.float32)
                              for f in dmeta["features"]])
        # Raw margin, not predict_proba: this is the log density ratio the
        # matrix form is linear in, and the only form that survives f -> 1.
        q = dmodel.predict(Xd, output_margin=True).astype(np.float64)
        tag = Path(mpath).stem.replace("direct_", "")
        name = f"direct[{tag}]"
        series[name] = 1.0 / (1.0 + np.exp(-q))
        margins[name] = q
        log.info("Direct model %s: feature set '%s' (%d features), test AUC %s",
                 mpath, dmeta["feature_set"], len(dmeta["features"]),
                 dmeta.get("test_auc_ovr", {}).get("tight"))
        log.info("  margin q: min %.2f  median %.2f  max %.2f  -> F up to %.3g",
                 q.min(), float(np.median(q)), q.max(), float(np.exp(q.max())))
        if args.quantize:
            Xs = np.column_stack([ak.to_numpy(src[f]).astype(np.float32)
                                  for f in dmeta["features"]])
            qsrc = dmodel.predict(Xs, output_margin=True).astype(np.float64)
            qq, frbins = quantize(
                qsrc, wsrc, ak.to_numpy(src["isTight"]).astype(np.float64), q,
                args.quantize, args.fr_cap)
            qname = f"{name}~q{args.quantize}"
            series[qname] = 1.0 / (1.0 + np.exp(-qq))
            margins[qname] = qq
            log.info("  quantized to %d score bins, FR range %.4f - %.4f "
                     "(F %.3f - %.3f)", args.quantize, frbins.min(), frbins.max(),
                     float(np.exp(logit(frbins.min()))),
                     float(np.exp(logit(frbins.max()))))

    # --- map statistical errors (rate form; map-based methods only) -------
    err = {
        "truth": np.sqrt(sum(
            map_stat_err(w * (y == c), fr_err, CLASSES[c], ix, iy, path, nx, ny) ** 2
            for c in CLASSES)),
        "flat": map_stat_err(w, fr_err, "incl", ix, iy, path, nx, ny),
    }
    for vname, proba in variants.items():
        err[f"bdt[{vname}]"] = np.sqrt(sum(
            map_stat_err(w * proba[:, c], fr_err, CLASSES[c],
                         ix, iy, path, nx, ny) ** 2 for c in CLASSES))

    # --- inclusive closure, both forms -----------------------------------
    log.info("--- inclusive closure on %s (FR from %s)",
             Path(args.target).stem, args.fr_sample)
    log.info("  %-18s %12s %8s %9s | %9s %10s %7s %8s",
             "method", "rate", "+-map", "rate/obs", "matrix/obs",
             "bkg.cc-clm", "n_clamp", "top1pct")
    log.info("  %-18s %12.2f %8.2f %9s | %9s %10s %7s %8s",
             "OBSERVED", obs, obs_err, "1 (ref)", "-", "-", "-", "-")
    for name, f in series.items():
        r = forms(w, f, is_tight, margins.get(name))
        e = err.get(name, float("nan"))
        log.info("  %-18s %12.2f %8.2f %9.4f | %9.4f %10.4f %7d %8.3f",
                 name, r["rate"], e, r["rate"] / obs, r["matrix"] / obs,
                 r["matrix_clamped"] / obs, r["n_clamp"], r["top1pct"])

    # --- where the matrix form loses what the rate form keeps ---------------
    for name in series:
        if not name.startswith("direct[") or "~q" in name:
            continue
        log.info("--- matrix-form calibration of %s (bins of q = log F)", name)
        log.info("  %8s %8s %12s %8s %8s | %8s %8s %8s",
                 "q_lo", "q_hi", "sum w", "f_meas", "<f_pred>",
                 "F_meas", "<F_pred>", "F ratio")
        for lo, hi, d, f_meas, f_pred, F_meas, F_pred in matrix_calibration(
                w, margins[name], is_tight):
            log.info("  %8.2f %8.2f %12.4g %8.4f %8.4f | %8.4f %8.4f %8.3f",
                     lo, hi, d, f_meas, f_pred, F_meas, F_pred,
                     F_pred / F_meas if F_meas > 0 else float("nan"))

    # --- differential closure --------------------------------------------
    # **No pT-like axis is neutral here, and the inclusive number is the only
    # unambiguous one.** The differential identity sum_{i in bin} w_i f(x_i) =
    # E[sum_{i in bin} w_i tight_i] needs E[tight | x, axis in bin] = f(x), i.e.
    # the axis must add nothing about tightness beyond the method's own inputs.
    # Since conePt = pt x (1 + max(0, miniiso - 0.1)), the pair (pt, conePt)
    # determines miniiso and hence half the WP, so:
    #   * a conePt axis favours the map-based methods and penalizes a conePt-blind
    #     model (measured: `direct` swings 0.82-1.37 in conePt bins while closing
    #     to 1.0007 inclusively);
    #   * a mu_pt axis does the exact reverse (measured: `flat` swings 1.41-0.38
    #     in mu_pt bins in its own self-closure, `direct` stays within 1%).
    # Both tables are shown because the contrast is the point, but neither is a
    # verdict. |eta| is the one axis present in every parametrization compared
    # here, so it is the only fair one, and it is reported first.
    axes = [("|eta|", abseta, etaedges, "%5.2f"),
            ("conePt [map-favouring]", conept, ptedges, "%5.0f"),
            ("mu_pt [direct-favouring]",
             ak.to_numpy(arr["mu_pt"]).astype(np.float64), ptedges, "%5.0f")]
    centers, ratios = [], {k: [] for k in series}
    for label, values, edges_ax, fmt in axes:
        log.info("--- rate-form pred/obs vs %s", label)
        log.info("  %-16s %12s  %s", label.split()[0], "obs",
                 "  ".join(f"{k:>10s}" for k in series))
        ib = bin_index(values, edges_ax)
        for i in range(len(edges_ax) - 1):
            m = ib == i
            o = float((w[m] * is_tight[m]).sum())
            if o <= 0:
                continue
            cells = []
            for k, f in series.items():
                r = float((w[m] * f[m]).sum()) / o
                cells.append(f"{r:10.4f}")
                if label.startswith("conePt"):
                    ratios[k].append(r)
            if label.startswith("conePt"):
                centers.append(0.5 * (edges_ax[i] + edges_ax[i + 1]))
            log.info(("  [" + fmt + "," + fmt + ") %12.2f  %s"),
                     edges_ax[i], edges_ax[i + 1], o, "  ".join(cells))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, v in ratios.items():
        ax.plot(centers, v, marker="o", label=k)
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"muon $p_T^{corr}$ [GeV]")
    ax.set_ylabel("predicted / observed N(tight)")
    ax.set_title(f"Closure on {Path(args.target).stem}, FR from {args.fr_sample}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / f"closure_{Path(args.target).stem}_{meta['feature_set']}.png"
    fig.savefig(out, dpi=300)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
