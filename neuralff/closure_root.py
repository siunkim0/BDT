"""Closure-test plots drawn with the project's own ROOT plotter.

Same numbers as `deck_figs.py --transfer`, drawn through
`/data6/Users/snuintern2/hi/track/Tools/Plotter.py::ComparisonCanvas` so they
match the house style used everywhere else (padUp/padDown, filled stack, hatched
stat band, "Data / Pred" ratio pad).

The mapping onto that class:

    drawData(...)        <- Observed  = sum of w over TIGHT muons (TTLJ truth)
    drawBackgrounds(...) <- Expected  = sum of w*F over LOOSE-and-NOT-TIGHT muons
    drawRatio()          -> Observed / Expected

Two environments are needed and neither has both halves, so this runs in two
passes over the same file:

    # 1. torch + awkward live in the venv; ROOT does not
    .venv-fakeflavor/bin/python -m neuralff.closure_root --dump

    # 2. ROOT lives in the container; torch does not. LCG's ROOT is a CentOS-7
    #    build and dies on ai-tamsa1 with libssl.so.10, hence singularity.
    singularity exec /data6/Users/snuintern2/private-el9.sif bash -c '...
        micromamba activate Nano; python -m neuralff.closure_root --draw'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PLOTTER_DIR = "/data6/Users/snuintern2/hi/track/Tools"

# Observable -> (bin edges, x-axis title). Same six as the matplotlib version.
VARS = {
    "mu_conept": (np.array([10, 12, 14, 17, 20, 25, 30, 40, 50, 70, 100, 200.]),
                  "p_{T}^{corr}(#mu^{fake}) [GeV]"),
    "mu_abseta": (np.linspace(0, 2.4, 13), "|#eta(#mu^{fake})|"),
    "ev_met": (np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 130, 170, 400.]),
               "PUPPI E_{T}^{miss} [GeV]"),
    "ev_mt": (np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 130, 170, 400.]),
              "M_{T}(#mu, E_{T}^{miss}) [GeV]"),
    "ev_njet40": (np.arange(0.5, 7.5), "N_{jets} (p_{T} > 40 GeV)"),
    "ev_awayjet_pt": (np.array([40, 50, 60, 70, 85, 100, 120, 150, 200, 300, 600.]),
                      "p_{T}(away jet) [GeV]"),
}
ESTIMATORS = ("flat", "neural")
EST_LABEL = {"flat": "QCD binned map", "neural": "QCD neural FR"}


def dump(args) -> None:
    """Pass 1: compute Observed/Expected per bin and save to npz."""
    from src.utils import get_logger, load_yaml
    from . import data as D
    from .fakefactor import FakeFactor
    from .transfer import flat_fake_rate, EXTRA_COLUMNS

    log = get_logger("neuralff.closure_root")
    cfg = load_yaml(args.config)
    ff = FakeFactor(Path(args.models) / f"nff_{args.source}_{args.arm}.pt")
    cols = D.load(Path(args.ntuples), cfg, ff.feature_set, extra=EXTRA_COLUMNS)

    w = np.asarray(cols["weight"], dtype=np.float64)
    tight = np.asarray(cols["isTight"]).astype(bool)
    fail = ~tight

    f_flat = flat_fake_rate(cols, cfg, args.source_sample)
    F = {"flat": f_flat / (1.0 - f_flat),
         "neural": np.exp(ff.logit({k: v for k, v in cols.items()}).astype(np.float64))}

    out = {}
    for var, (edges, _) in VARS.items():
        x = np.clip(np.asarray(cols[var], dtype=np.float64), edges[0],
                    edges[-1] - 1e-9)
        out[f"{var}__edges"] = edges
        # Observed: the truth, identical for both estimators.
        out[f"{var}__obs"] = np.histogram(x[tight], bins=edges,
                                          weights=w[tight])[0]
        out[f"{var}__obs_err"] = np.sqrt(
            np.histogram(x[tight], bins=edges, weights=w[tight] ** 2)[0])
        for est in ESTIMATORS:
            ww = w[fail] * F[est][fail]
            out[f"{var}__exp_{est}"] = np.histogram(x[fail], bins=edges,
                                                    weights=ww)[0]
            out[f"{var}__exp_{est}_err"] = np.sqrt(
                np.histogram(x[fail], bins=edges, weights=ww ** 2)[0])

    # Totals, so the drawing pass can assert the histograms integrate to the
    # transfer table rather than trusting that they do.
    out["truth_total"] = np.array([w[tight].sum()])
    for est in ESTIMATORS:
        out[f"total_{est}"] = np.array([(w[fail] * F[est][fail]).sum()])

    dest = Path(args.npz)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **out)
    log.info("Wrote %s  (truth %.1f, flat %.1f, neural %.1f)", dest,
             out["truth_total"][0], out["total_flat"][0], out["total_neural"][0])


def draw(args) -> None:
    """Pass 2: build TH1Ds and draw them with the project's ComparisonCanvas."""
    sys.path.insert(0, PLOTTER_DIR)
    import ROOT
    ROOT.gROOT.SetBatch(True)
    from Plotter import ComparisonCanvas

    d = np.load(args.npz)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Sanity: the binned histograms must reproduce the inclusive numbers.
    for est in ESTIMATORS:
        for var in VARS:
            s = d[f"{var}__exp_{est}"].sum()
            if abs(s / d[f"total_{est}"][0] - 1) > 1e-6:
                raise SystemExit(f"{var}/{est}: histogram sums to {s:.2f} but the "
                                 f"total is {d[f'total_{est}'][0]:.2f} — a bin is "
                                 f"dropping rows")

    def th1(name, edges, vals, errs):
        h = ROOT.TH1D(name, "", len(edges) - 1, np.asarray(edges, dtype=float))
        for i, (v, e) in enumerate(zip(vals, errs), start=1):
            h.SetBinContent(i, float(v))
            h.SetBinError(i, float(e))
        return h

    keep = []
    for est in ESTIMATORS:
        for var, (edges, xtitle) in VARS.items():
            config = {
                "xTitle": xtitle,
                "yTitle": "Weighted fake muons",
                "xRange": [float(edges[0]), float(edges[-1])],
                "ratio": [0.4, 1.4],       # common range for both estimators
                "logy": args.logy,
                "era": "STACK/Run2",
                "lumiInfo": "L_{int} = 138 fb^{-1} (13TeV)",
            }
            obs = th1(f"Observed_{est}_{var}", edges, d[f"{var}__obs"],
                      d[f"{var}__obs_err"])
            exp = th1("Expected", edges, d[f"{var}__exp_{est}"],
                      d[f"{var}__exp_{est}_err"])
            keep += [obs, exp]

            c = ComparisonCanvas(config=config)
            # One "background" only: the matrix-method prediction. The stack and
            # its `systematics` clone are then the same histogram, so the hatched
            # band is the prediction's own statistical error — which is what the
            # closure test wants shown.
            c.drawBackgrounds({"Expected": exp}, {"Expected": ROOT.kAzure + 7})
            c.drawData(obs)
            c.drawRatio()

            # Plotter.drawRatio() clones the RATIO into ratioSyst, so its band
            # follows the points instead of sitting at 1 — and TH1::Divide also
            # folds the prediction's error into the point error, double-counting
            # it. The AN-25-154 Fig. 50 convention is the other way round:
            #   points : obs/exp  +-  (observed stat)/exp
            #   band   : 1        +-  (expected stat)/exp
            # so the band shows how well the prediction itself is known.
            # ratioSyst is a clone of ratio, so it inherits the marker too and
            # Draw("e2&f&same") stamps a second set of points at 1 — which reads
            # as if the data sat at 1 and something else fell below it.
            c.ratioSyst.SetMarkerSize(0)
            c.ratioSyst.SetLineWidth(0)

            ob, oe = d[f"{var}__obs"], d[f"{var}__obs_err"]
            ex, ee = d[f"{var}__exp_{est}"], d[f"{var}__exp_{est}_err"]
            for i in range(1, len(edges)):
                e_i = ex[i - 1]
                if e_i > 0:
                    c.ratio.SetBinContent(i, ob[i - 1] / e_i)
                    c.ratio.SetBinError(i, oe[i - 1] / e_i)
                    c.ratioSyst.SetBinContent(i, 1.0)
                    c.ratioSyst.SetBinError(i, ee[i - 1] / e_i)
                else:
                    for h in (c.ratio, c.ratioSyst):
                        h.SetBinContent(i, 0.)
                        h.SetBinError(i, 0.)

            c.drawLegend()
            # Plotter.py hardcodes "Data" and "Data / Pred", which are wrong here:
            # the points are MC truth, not data. It is shared with other projects,
            # so relabel from the caller instead of editing it.
            c.legend.Clear()
            c.legend.AddEntry(obs, "Observed (t#bar{t} truth)", "lep")
            c.legend.AddEntry(exp, f"Expected ({EST_LABEL[est]})", "f")
            c.legend.AddEntry(c.systematics, "stat", "f")
            c.ratio.GetYaxis().SetTitle("Obs / Exp")
            c.finalize()

            # Plotter.finalize() draws the ratio with "p&hist" — markers only,
            # so the points get no bin-width bar and no error bar, unlike the
            # Observed points in the upper pad which are drawn with "e1". Redraw
            # it on top with "e1" so both pads read the same way.
            c.padDown.cd()
            c.ratio.Draw("e1&same")
            c.padDown.RedrawAxis()
            c.cvs.cd()
            # Plotter.savefig() hardcodes a Higgs/ path, so save directly.
            name = f"closure_{est}_{var}.png"
            c.cvs.SaveAs(str(outdir / name))
            print(f"[saved] {outdir / name}")
            del c


def summary(args) -> None:
    """All six observables on one canvas: Obs/Exp for both estimators.

    The per-variable slides show the full closure; this is the one-glance
    comparison — 2x3 pads, both estimators overlaid, common y range, so the
    difference between them can be read across every variable at once.
    """
    sys.path.insert(0, PLOTTER_DIR)
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    d = np.load(args.npz)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cvs = ROOT.TCanvas("summary", "", 1800, 1150)
    cvs.Divide(3, 2, 0.004, 0.006)
    COL = {"flat": ROOT.kRed + 1, "neural": ROOT.kAzure + 2}
    keep = []
    for k, (var, (edges, xtitle)) in enumerate(VARS.items(), start=1):
        pad = cvs.cd(k)
        pad.SetGrid()
        pad.SetLeftMargin(0.13)
        pad.SetBottomMargin(0.13)
        edges = np.asarray(edges, dtype=float)
        # Same convention as the per-variable plots: a band at 1 carrying the
        # prediction's own statistical error. Drawn first, under the points.
        band = ROOT.TH1D(f"band_{var}", "", len(edges) - 1, edges)
        for i in range(1, len(edges)):
            e_i = d[f"{var}__exp_neural"][i - 1]
            if e_i > 0:
                band.SetBinContent(i, 1.0)
                band.SetBinError(i, d[f"{var}__exp_neural_err"][i - 1] / e_i)
        band.SetFillColorAlpha(12, 0.6)
        band.SetFillStyle(3144)
        band.SetMarkerSize(0)
        band.GetYaxis().SetRangeUser(0.4, 1.4)
        band.GetYaxis().SetTitle("Observed / Expected")
        band.GetYaxis().SetTitleSize(0.05)
        band.GetYaxis().SetTitleOffset(1.15)
        band.GetYaxis().SetLabelSize(0.045)
        band.GetXaxis().SetTitle(xtitle)
        band.GetXaxis().SetTitleSize(0.05)
        band.GetXaxis().SetTitleOffset(1.1)
        band.GetXaxis().SetLabelSize(0.045)
        band.Draw("e2")
        keep.append(band)

        series = {}
        for est in ESTIMATORS:
            h = ROOT.TH1D(f"r_{est}_{var}", "", len(edges) - 1, edges)
            for i in range(1, len(edges)):
                e = d[f"{var}__exp_{est}"][i - 1]
                o = d[f"{var}__obs"][i - 1]
                if e > 0:
                    h.SetBinContent(i, o / e)
                    h.SetBinError(i, d[f"{var}__obs_err"][i - 1] / e)
            h.SetLineColor(COL[est])
            h.SetMarkerColor(COL[est])
            h.SetMarkerStyle(20)
            h.SetMarkerSize(1.0)
            h.SetLineWidth(2)
            h.GetYaxis().SetRangeUser(0.4, 1.4)
            h.GetYaxis().SetTitle("Observed / Expected")
            h.GetYaxis().SetTitleSize(0.05)
            h.GetYaxis().SetTitleOffset(1.15)
            h.GetYaxis().SetLabelSize(0.045)
            h.GetXaxis().SetTitle(xtitle)
            h.GetXaxis().SetTitleSize(0.05)
            h.GetXaxis().SetTitleOffset(1.1)
            h.GetXaxis().SetLabelSize(0.045)
            # The band is already drawn, so every series goes on top of it.
            h.Draw("e1&same")
            series[est] = h
            keep.append(h)
        line = ROOT.TLine(edges[0], 1.0, edges[-1], 1.0)
        line.SetLineStyle(2)
        line.SetLineWidth(2)
        line.Draw("same")
        keep.append(line)
        if k == 1:
            leg = ROOT.TLegend(0.45, 0.72, 0.88, 0.88)
            leg.SetFillStyle(0)
            leg.SetBorderSize(0)
            leg.SetTextSize(0.045)
            for est in ESTIMATORS:
                leg.AddEntry(series[est], EST_LABEL[est], "lep")
            leg.AddEntry(band, "stat (Expected)", "f")
            leg.Draw()
            keep.append(leg)
        pad.RedrawAxis()

    dest = outdir / "summary_ratio_all.png"
    cvs.SaveAs(str(dest))
    print(f"[saved] {dest}")


def pair(args) -> None:
    """Pass 3: stitch flat|neural side by side, one image per observable.

    The deck convention for a two-estimator comparison is two-up on one slide,
    so the pairing happens here rather than in the slide list. Needs only PIL,
    so it runs in the venv like the dump pass.
    """
    from PIL import Image
    outdir = Path(args.out)
    # --pair runs in the venv and --draw in the container, so it is easy to fix
    # a plot, redraw, and forget to re-stitch — the deck then shows the OLD
    # figure with no error anywhere. Refuse to build from stale inputs.
    npz_mtime = Path(args.npz).stat().st_mtime
    for var in VARS:
        for est in ESTIMATORS:
            src = outdir / f"closure_{est}_{var}.png"
            if not src.exists():
                raise SystemExit(f"{src} missing — run --draw first")
            if src.stat().st_mtime < npz_mtime:
                raise SystemExit(
                    f"{src} is older than {args.npz} — re-run --draw before "
                    f"--pair, or the deck will show the previous figures")

    for var in VARS:
        ims = [Image.open(outdir / f"closure_{est}_{var}.png")
               for est in ESTIMATORS]
        h = max(i.height for i in ims)
        w = sum(i.width for i in ims)
        canvas = Image.new("RGB", (w, h), "white")
        x = 0
        for i in ims:
            canvas.paste(i, (x, 0))
            x += i.width
        dest = outdir / f"pair_{var}.png"
        canvas.save(dest)
        print(f"[saved] {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", action="store_true", help="pass 1, needs the venv")
    p.add_argument("--draw", action="store_true", help="pass 2, needs ROOT")
    p.add_argument("--pair", action="store_true",
                   help="pass 3, stitch flat|neural two-up for the deck")
    p.add_argument("--summary", action="store_true",
                   help="all six observables' Obs/Exp on one canvas (needs ROOT)")
    p.add_argument("--config", default="neuralff/config/nff.yaml")
    p.add_argument("--ntuples",
                   default="fakeflavor/data/ntuples_nff/TTLJ.parquet")
    p.add_argument("--models", default="neuralff/data/models")
    p.add_argument("--source", default="qcd")
    p.add_argument("--source-sample", default="QCD_MuEnriched")
    p.add_argument("--arm", default="paper5")
    # Default under plots/deck/transfer/ so make_decks.py finds them without a
    # copy step — that is where deck_figs.py --transfer writes too.
    p.add_argument("--npz",
                   default="neuralff/plots/deck/transfer/closure_hists.npz")
    p.add_argument("--out", default="neuralff/plots/deck/transfer/root")
    p.add_argument("--logy", action="store_true")
    args = p.parse_args()
    if args.dump:
        dump(args)
    elif args.draw:
        draw(args)
    elif args.pair:
        pair(args)
    elif args.summary:
        summary(args)
    else:
        raise SystemExit("pick --dump, --draw, --pair or --summary")


if __name__ == "__main__":
    main()
