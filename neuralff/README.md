# neuralff — neural fake factor vs the binned muon fake rate

A PyTorch implementation of **arXiv:2511.06972**, *Neural Fake Factor Estimation
Using Data-Based Inference* (Gavranovič, Čalić, Debevec, Lytken, Kerševan),
applied to the SNU muon fake rate and compared against the existing binned map.

`RESULTS.md` records what came out. This file is the *what and how*.

## The method in three lines

Train a binary classifier to separate **tight** from **loose-not-tight** muons.
By the likelihood-ratio trick the logit is the log of the density ratio, so

```
F(x) = exp(q(x)) = p_T(x) / p_L(x)          # the fake factor  (Eq. 10-12)
f(x) = F/(1+F)  = sigmoid(q(x))             # the fake rate, what fr/*.json stores
```

The classifier is not rebalanced between classes, because the class proportions
*are* the answer: `exp(q) = N_T(x)/N_L(x)`. Read the other way, this is exactly
`P(tight | loose, x)` — the same fit as `fakeflavor/train.py --target tight`,
which is why the two agree by construction and not by accident.

## Scope of this study (decided 2026-08-11)

- **MC only**: `QCD_MuEnriched` and `TTLJ_powheg`, compared against
  `fakerate_muon_QCD_MuEnriched` and `fakerate_muon_TT` in
  `/data6/Users/snuintern2/fake/fr/fakerate_muon_*.json`.
- **The ratio step only.** The paper's two *subtraction* classifiers (their
  Fig. 3, upper half) remove real-lepton contamination from data. Our samples
  are MC with gen truth, so `skim.py` selects `isPrompt == False` directly and
  there is nothing to subtract. `model.py` still implements the `softabs` head
  those classifiers need, so a data arm can be added without touching it.
- **The paper's five features**: a pT, `mu_abseta`, `ev_met`, `ev_mt`,
  `ev_njet40`, plus `era` as a categorical so one model covers all four Run 2
  eras. Which pT is the one real choice here — see below.
- Not in scope: the data arm, the QCD→TT transfer test, the extended feature set.

## The one rule that cannot be broken — and which pT it leaves us

**No input may be a function of `miniPFRelIso` or of `SIP3D`**, because those two
define the tight WP. `mu_conept = pt x (1 + max(0, miniiso - 0.1))` is such a
function *in combination with any raw-pT proxy*: measured on QCD,
`conePt == pt` for 31.4% of rows, `miniiso <= 0.1` for 31.4%, and
`P(tight | conePt > pt) = 0.000000` exactly. Hand a model both and its AUC goes
to ~1 while the fake factor collapses to 0 or infinity on the fail side, where
the matrix method needs it. Details in `fakeflavor/features.py::TIGHT_WP_VARS`.

**Either one alone is admissible** — with no second pT to divide by, miniiso is
not recoverable — so the config carries two arms:

| set | pT feature | xgb reference AUC | read it as |
|---|---|---|---|
| `paper5` | `mu_conept` | 0.695 | **the headline.** Same variable the reference map is binned in, which is also what the paper does (their binned map and their network both use the electron pT) |
| `paper5_rawpt` | `mu_pt` | 0.577 | the WP-free control |

The difference between them is not a detail: a model that has never seen conePt
cannot resolve the axis it is being plotted against, so the `rawpt` arm's curve
comes out much flatter than the binned map in conePt bins (+130% in the last
bin, −40% in the first) while still closing to 1%. That gap measures **how much
of the map's steep conePt dependence is the working point rather than physics**,
which is worth reporting but is not a statement about the ML method. Both arms
keep the fail side well-conditioned (`paper5`: mean f = 0.20, p99 = 0.62,
nothing above 0.95).

A test AUC near 1 is a bug report, not a result.

## Data flow

```
SKNanoAnalyzer fake.cc  (--userflags MakeTree --tag nff)
  -> /gv0/Users/snuintern2/SKNanoOutput/fake/MakeTree/{era}_nff/*.root
  -> fakeflavor/skim.py --config fakeflavor/config/samples_nff.yaml
                        --out fakeflavor/data/ntuples_nff --keep-all-flavors
  -> neuralff/train.py    -> neuralff/data/models/nff_{qcd,ttlj}.pt
  -> neuralff/compare.py  -> neuralff/plots/{qcd,ttlj}/
```

Two things about the `nff` tag matter and are easy to get wrong:

1. **It has no MET/MT cut.** The official MC maps are measured in the
   *Inclusive* region (away jet only) — verified: the stitched
   `Mu8|Mu17 / Inclusive / {loose,tight} / fake_ptcorr_abseta` histogram equals
   `fakerate_muon_QCD_MuEnriched` to 1.000 in all 21 bins of 2016preVFP. The
   older `bdtflav*` ntuples hard-applied `MET<25 && MT<25` in `fillTreeRow` and
   sit 2–20% below the official map. `MeasReg` is still recoverable offline from
   the `ev_met` / `ev_mt` columns.
2. **`--keep-all-flavors` is required.** The default skim drops pileup and
   jet-unmatched fakes (~3.3%) because the flavor classifier has no class for
   them; a fake *rate* must keep them or its denominator is wrong.
3. **It is much bigger, very unevenly.** QCD grows ~3x (16.1M rows over the four
   eras); TTLJ grows 23-29x, because ttbar has real MET and a real W and the old
   cut was throwing nearly all of it away — 2016preVFP TTLJ alone is 36.8M rows
   / 5.7 GB. Most of those are prompt muons that the skim's `isPrompt` cut
   removes; what survives is ~1M fakes per era, which finally makes the ttbar
   arm trainable (it was ~21k/era before).
4. **PUPPI MET has a pathological tail that the old cut hid.** In one QCD pT bin
   0.011% of rows have MET > 1 TeV and the maximum is 5.9e6 GeV. Standardization
   cannot survive that — one such row inflates the std to ~9000 and collapses
   every physical MET onto z ~ 0 — so `fit_encoding` winsorizes the numerical
   features at `clip_quantile` (0.9999) fitted on the training split. Rows are
   clipped, never dropped, so they keep their weight in the loss and the closure
   sum. It logs every clipped feature with its bound and row count.

## Running

```bash
cd /data6/Users/snuintern2/BDT && . .venv-fakeflavor/bin/activate

python -m fakeflavor.skim --config fakeflavor/config/samples_nff.yaml \
    --out fakeflavor/data/ntuples_nff --keep-all-flavors

# Gate 1 — does the parquet reproduce fr/*.json? Run this BEFORE training.
python -m neuralff.compare --sample QCD_MuEnriched \
    --ntuples fakeflavor/data/ntuples_nff/QCD_MuEnriched.parquet \
    --out neuralff/plots/gate        # bin/off column must read 1.000

python -m neuralff.train --sample QCD_MuEnriched --feature-set paper5 \
    --ntuples fakeflavor/data/ntuples_nff/QCD_MuEnriched.parquet \
    --out neuralff/data/models/nff_qcd.pt

python -m neuralff.compare --model neuralff/data/models/nff_qcd.pt \
    --ntuples fakeflavor/data/ntuples_nff/QCD_MuEnriched.parquet \
    --sample QCD_MuEnriched --out neuralff/plots/qcd
```

`--sample TTLJ` / `nff_ttlj.pt` for the ttbar arm. `torch` lives in the existing
`.venv-fakeflavor` (installing it downgraded only `nvidia-nccl-cu12`, which
xgboost uses for multi-GPU and this project does not).

## Model

`model.py`, following the paper's §3.2 and Fig. 4:

- every feature is embedded into `embed_dim` (numerical: a per-feature affine map
  from the scalar; categorical: a lookup table), the embeddings are stacked and
  **mean-pooled** across the feature dimension, then projected to `hidden_dim`;
- **4 pre-activation residual blocks**, 2 x 128 neurons each, ReLU, **no**
  BatchNorm or dropout (the paper found neither helped);
- one output neuron = `q`. Head is linear for the ratio classifier; `softabs`
  (Eq. 19) exists for the subtraction classifiers, which force `q >= 0`.

Training: AdamW, lr 3e-4, weight decay 1e-5, batch 1024, up to 100 epochs,
`ReduceLROnPlateau(0.5, patience 5)`, early stopping on validation loss, and the
paper's **50/25/25** split — group-aware on `(sample, era, run, lumi, evt)`, so
no event straddles it.

### Two deliberate deviations, both documented in code

- **The regularization sign.** Eq. 12 writes the squared term as
  `- lambda * q^2`, which under minimization *rewards* exploding logits. It is
  applied here as a `+` penalty (`train.py::loss_fn`).
- **`reg_lambda` defaults to the paper's 0.01, but the headline model uses 0.**
  Shrinking `|q|` pulls `f` toward 0.5, and the muon fake rate sits at ~0.26, so
  the penalty biases it *upward*: measured on QCD, closure `sum(w f)/sum(w tight)`
  = 1.063 at lambda = 0.01 versus 0.982 at lambda = 0 for the same 3 epochs. The
  paper accepts that trade because it buys CR→SR extrapolation; we are comparing
  *inside* the measurement region, where a 6% normalization bias is the whole
  effect being measured. `--reg-lambda` sets it per run.

Weights are globally rescaled so the training mean is 1. That cancels in the
ratio and in the BCE optimum, and keeps the paper's learning rate meaningful
despite QCD pT-bin weights spanning many decades.

## What `compare.py` produces

Per era (and pooled), three numbers per `(conePt, |eta|)` cell:

| curve | definition |
|---|---|
| `binned` | `sum w(tight) / sum w(loose)` over **all** rows — the measurement itself |
| `official` | the same cell read from `fr/fakerate_muon_*.json` |
| `ML` | `sum_{i in cell} w_i f(x_i) / sum_{i in cell} w_i` on the **test split**, band = weighted std of `f(x_i)` in the cell |

The band is the paper's Fig. 10/11 definition: it is the structure the map
averages over, **not** a statistical error.

The identity behind the comparison: if `f` is the true conditional then
`sum_cell w f = sum_cell w 1[tight]` exactly, so the ML and binned points
estimate the same quantity and any difference is model error. `compare.py`
prints that ratio inclusively and per `|eta|` slice as the **closure** gate.

Plots: `FR_vs_ptcorr_*`, `FR_vs_eta_*`, `FR_map_*` (binned | ML | ratio), and
`FF_surface_*` — the learned `f(x)` over `(pt, MET)` and `(pt, |eta|)` grids at
fixed other features, the paper's Fig. 7/8 view of what a binned map cannot hold.
