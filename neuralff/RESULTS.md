# neuralff — results

State as of **2026-08-12**, on the `nff` ntuples (no MET/MT cut, Inclusive
region — the region the reference maps are measured in). `README.md` is the
*what and how*; this file is the *what came out*.

Decks: `plots/deck/neural_fake_factor_{qcd,ttlj}.{pptx,pdf}`, 31 slides each,
built by `plots/deck/make_decks.py` from `neuralff/deck_figs.py`.

## 0. Gate 1 — the pipeline reproduces the reference exactly

The binned FR computed from `ntuples_nff/QCD_MuEnriched.parquet` equals
`fakerate_muon_QCD_MuEnriched` in `fake/fr/*.json` **to four decimals in all 21
cells of all four eras**. Without this the comparison would be measuring the
pipeline rather than the method. Two things had to be right for it:

- **the region** — the official MC maps are Inclusive (away jet > 40, no MET/MT
  cut). `fake.cc::fillTreeRow` used to hard-apply MET < 25 and MT < 25; ntuples
  built with it sit 2–20% low. Removed 2026-08-11, tag `nff`.
- **the overflow** — the reference **drops** conePt > 100; it does not fold it
  into the last bin, which is what `fake.cc`'s own histograms do
  (`min(ptCorr, 99.9)`). Measured: clipping it in makes the last bin
  0.0584/0.0733/0.1083 against the official 0.0576/0.0725/0.1075 (+1.4/+1.1/
  +0.7%); dropping it reproduces all three. Now `binning.overflow: drop`.

## 1. The headline

`paper5` = (conePt, |eta|, MET, M_T, N_jet40, era), lambda = 0, test split.

| sample | rows | test AUC | closure | median \|ML−binned\| | worst | cells <5% |
|---|---|---|---|---|---|---|
| **QCD_MuEnriched** | 13,572,470 | 0.7179 | **1.0091** | **1.6%** | 7.5% | 18 / 21 |
| **TTLJ_powheg** | 3,693,791 | 0.7535 | **0.9931** | **1.5%** | 4.2% | 21 / 21 |

Per era (`paper5`, closure / median):

| | 2016preVFP | 2016postVFP | 2017 | 2018 |
|---|---|---|---|---|
| QCD | 1.011 / 1.4% | 1.016 / 2.6% | 1.009 / 2.3% | 1.007 / 1.2% |
| TTLJ | 1.001 / 2.6% | 1.001 / 1.2% | 0.998 / 1.5% | 0.984 / 1.8% |

An unbinned per-event estimator, given the same information, returns the
histogram to ~1.5% while closing to within 1%. That is the precondition for
trusting it anywhere the histogram cannot go.

The within-cell spread of `f(x)` is large — 0.10 on 0.51 in the first conePt
bin, and comparable elsewhere. That variation is what the binned method
averages away and this method keeps; it is not an error bar.

QCD's residual is a **coherent +1 to +2%** rather than scatter (closure 1.0091),
i.e. the network sits slightly high overall. It is not a bias of the estimator —
lambda is 0 — but leftover optimization error. A one-parameter recalibration of
`q` would remove it; not applied, because it is not in the paper and 1% is below
the systematics this map carries.

## 2. Which pT the model gets is the biggest single choice

| arm | pT feature | AUC | closure | median | worst | cells <5% |
|---|---|---|---|---|---|---|
| `paper5` QCD | `mu_conept` | 0.7179 | 1.0091 | 1.6% | 7.5% | 18/21 |
| `paper5_rawpt` QCD | `mu_pt` | 0.6037 | 0.9977 | **32.5%** | 84.5% | 1/21 |
| `paper5` TTLJ | `mu_conept` | 0.7535 | 0.9931 | 1.5% | 4.2% | 21/21 |
| `paper5_rawpt` TTLJ | `mu_pt` | 0.6600 | 1.0003 | 14.0% | 41.2% | 8/21 |

Both arms **close** — the raw-pT model is an honest fake rate given what it can
see — but it cannot follow the map along an axis it has never seen. The gap is
the measurement: `0.7179 − 0.6037 = 0.114` of AUC, and essentially all of the
conePt slope, enters only through conePt. Since
`conePt = pt x (1 + max(0, miniiso − 0.1))`, that is **the working point showing
up inside the reference map's own binning axis**.

The rule that forces the choice: `conePt == pt` for 31.4% of rows, `miniiso <=
0.1` for 31.4%, and `P(tight | conePt > pt) = 0.000000` exactly — so the *pair*
is a lossless WP tag and must never both be inputs. Either alone is admissible,
and both leave the fail side well conditioned (nothing at the 0.95 clamp).

## 3. Deviations from the paper, and why

- **lambda = 0, not 0.01.** Their squared-logit term shrinks `|q|`, i.e. pulls
  `f` toward 0.5; the muon fake rate is ~0.26, so it biases the answer **up 6%**
  (closure 1.063 vs 0.982, identical runs). Right trade for their CR→SR
  extrapolation, wrong for a measurement inside the region.
- **The regularizer's sign.** Eq. 12 prints `− lambda q^2`, which under
  minimization rewards exploding logits. Applied as a `+` penalty.
- **Winsorization at the 0.999 quantile.** Not in the paper, but forced by the
  data: with the MET cut gone, PUPPI MET reaches 1.0e7 GeV. One such row drives
  the standardization sigma to ~9000 and collapses every physical MET onto
  z ~ 0. At 0.9999 the cut is still 9.9 TeV (sigma 134, bulk z-span 2.6); at
  0.999 it is 690 GeV (sigma 66, bulk z-span 5.4). Rows are clipped, never
  dropped, so they keep their weight in the loss and the closure sum.
- **AUC uses |w|.** TTLJ is NLO and 0.30% of its rows carry negative weights,
  which makes the ROC non-monotonic and sklearn raise. Diagnostic only; every
  fake-rate number keeps signed weights.

## 4. Infrastructure notes

- The 3 condor jobs that went on hold were the DAG's **hadd** nodes, not
  analysis jobs (`hadd -f -j 8` over 40 × ~320 MB files with ~105 branches
  exceeds the 8 GB cgroup; the submit file also carries a self-contradicting
  `RequestMemory = 16384 MB` next to `request_memory = 8192`). All 120 analysis
  jobs had finished — the merges were done by hand inside `private-el9.sif`
  (LCG's `hadd` is a CentOS-7 build and dies on ai-tamsa1 with `libssl.so.10`).
- **`skim.py` now reads in 2M-row chunks.** TTLJ is 310M rows over four eras
  (134M in 2018 alone) and one `arrays()` call on that is ~56 GB before awkward
  copies anything. Every cut is row-wise so chunking is exactly equivalent, and
  that was verified rather than assumed: re-skimming QCD chunked gives
  13,572,470 rows and a byte-identical parquet.

## 4b. Transfer: QCD → ttbar. **The neural fake rate transfers worse.**

Run 2026-08-14, `neuralff/transfer.py`, deck
`plots/deck/neural_fake_factor_transfer.{pptx,pdf}`. Both estimators measured on
QCD, applied to all 3,693,791 TTLJ rows, against TTLJ truth `N_tight` = 26,107.

| estimator | rate / truth | matrix / truth | n_clamp | top 1% |
|---|---|---|---|---|
| `flat` (the original binned map) | **1.0765** | **1.1074** | 0 | 6.6% |
| `neural` (QCD-trained paper5) | 1.1787 | 1.2976 | 27 | 11.2% |

Sanity floor passed first: the QCD map applied to **QCD itself** returns 0.9999 in
both forms, so the lookup, era matching and overflow handling are not the story.
On QCD the network closes to 1.007 — the excess appears only on transfer.

This is the **opposite sign** to the earlier xgboost `direct` result (1.023 vs the
flat map's 1.095). Those numbers came from a different population (MeasReg, four
flavours) so they are not directly comparable, but the conclusion reverses.

### Why: out-of-distribution extrapolation, following conePt and jet activity

| region | QCD weight | ttbar weight | flat/truth | neural/truth |
|---|---|---|---|---|
| N_jet = 1 | 88.8% | 30.2% | 1.050 | 1.120 |
| N_jet ≥ 2 | 11.2% | 69.8% | ~1.09 | 1.19–1.23 |
| conePt 50–100 | 2.9% | 18.7% | 1.139 | **1.498** |
| conePt > 100 | 0.2% | 3.1% | 1.067 | **1.717** |

The network is fitted where QCD has events and asked about where ttbar lives, and
those are different places. Notably it is **not** a MET extrapolation: broken down
in MET the over-prediction is flat at ~1.18 across the whole range, including where
QCD has almost no support. The coarse map is robust precisely *because* it is
coarse — two variables where the QCD measurement still has events, and no
MET/M_T/N_jet dependence available to get wrong.

### What this does and does not retract

Nothing in §0–§3 changes: Gate 1 passes, both models close on their own samples,
and the within-cell variation (factor 1.7–3, validated against measurement) is
real. What fails is the *useful* version of the claim — the extra information is
QCD's, not ttbar's. Either restrict inputs to variables whose meaning is
process-independent, or measure the FR in a region covering the application phase
space.

Closure distributions are drawn in the style of **AN-25-154 Fig. 50** (p. 97):
Observed = TTLJ tight muons, Expected = matrix-method sum over loose-not-tight,
`Obs/Exp` pad, rate and shape χ²/ndf per panel. Note the χ² values are **not**
comparable to the AN's: ours are MC with large effective statistics, so the
denominators are tiny and any few-% mismatch gives a huge χ². Read the rate/shape
split instead (N_jet: rate 4115, shape 130).

### A bug this test caught

`compare.py::official_fr` looked up `era_tag[era]` with a string while the YAML
parsed unquoted `2017`/`2018` as **ints**; the `KeyError` was caught and logged as
"no official map", so **the official overlay was silently skipped for 2017 and
2018** and Gate 1 had only ever been verified for the two 2016 eras. Fixed by
coercing both sides to `str` (and quoting the YAML keys). Re-verified: all four
eras, 21/21 cells, max |binned/official − 1| = 0.0000%.

## 5. Open

- The QCD +1% coherent offset (above). Cheap to remove, deliberately not removed.
- No transfer test (QCD model applied to ttbar) — out of scope by decision, but
  both models now exist and `compare.py --model <other>` would do it.
- No data arm; that needs the paper's two subtraction classifiers and prompt-MC
  ntuples (DY/WJets with `--userflags MakeTree`), neither of which exists.
