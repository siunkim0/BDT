# fakeflavor — muon fake-rate source-jet flavor classifier

A 4-class (b / c / uds / g) XGBoost classifier for the jet that produced a fake
muon, trained on reconstructed quantities only so it can be applied to data.

**See `RESULTS.md` for what came out** — every trained model with its AUCs, the
settled physics conclusions, and the open items. This file is the *what and
how*; that one is the *what happened*.

## Why

`bkg.cc` estimates the fake background with a single 2D map
`FR(conePt, |eta|)`. The per-flavor study in `/data6/Users/snuintern2/fake/Tools/TODO.md`
(v3/v4) showed that

- FR depends strongly on the source-jet flavor — Run 2 QCD MuEnriched at
  conePt ∈ [50, 100]: `FR(uds) ≈ 0.20` vs `FR(b) ≈ 0.05`;
- the flavor composition differs wildly between processes — QCD muon fakes are
  61–66 % b, TTLJ 88–89 % b;
- but `FR(flavor, pT, |eta|)` is approximately **sample-universal**.

So a flavor-weighted FR is well defined and should transfer between processes
better than the flat 2D map. The blocker is that flavor is MC truth. This
subproject asks whether it can be predicted from data-observable quantities.

## Data flow

```
SKNanoAnalyzer  fake.cc  (--userflags MakeTree --tag bdtflav)
   -> /gv0/Users/snuintern2/SKNanoOutput/fake/MakeTree/{era}_bdtflav/*.root
      flat TTree "mu", one row per loose muon in the measurement region
   -> fakeflavor/skim.py    -> fakeflavor/data/ntuples/*.parquet
   -> fakeflavor/train.py   -> fakeflavor/data/models/flavor_{full,notag}.json
   -> fakeflavor/evaluate.py, fakeflavor/cv.py -> fakeflavor/plots/
```

Eras: 2016preVFP, 2016postVFP, 2017, 2018 (full Run 2 — single-era `uds` and
gluon statistics were uninterpretable in the v4 study).
Samples: `QCD_Pt-*_MuEnriched` (12 pt bins), `TTLJ_powheg`, and `DoubleMuon`
(data — never trained on, only for input-feature data/MC comparison).

## Labels

Written by `fake::FlavorCode` from the jet matched via `Muon_jetIdx`
(`Jet::OriginalIndex()` on the pre-lepton-cleaning collection):

| code | class | rule |
|---|---|---|
| 0 | b | `hadronFlavour == 5` |
| 1 | c | `hadronFlavour == 4` |
| 2 | uds | `|partonFlavour| ∈ {1,2,3}` |
| 3 | g | `|partonFlavour| == 21` |
| 4 | pileup | `genJetIdx < 0` — checked first, **dropped** from training |
| −1 | unmatched | `Muon_jetIdx < 0` — **dropped**, but observable in data via `mu_hasJet` |
| −2 | (data) | no truth flavor |

Pileup is 0.36 % of loose fakes (~2.1k rows/era, ~100:1 against b), so it is a
systematic on the final flavor-weighted FR, not a model class.

## Running

```bash
# from the BDT project root
bash fakeflavor/bootstrap.sh && . .venv-fakeflavor/bin/activate

python -m fakeflavor.skim --config fakeflavor/config/samples.yaml --data
python -m fakeflavor.train    --feature-set full  --out fakeflavor/data/models/flavor_full.json
python -m fakeflavor.evaluate --model fakeflavor/data/models/flavor_full.json
python -m fakeflavor.cv       --feature-set full

python -m fakeflavor.train    --feature-set notag --out fakeflavor/data/models/flavor_notag.json
python -m fakeflavor.evaluate --model fakeflavor/data/models/flavor_notag.json
```

The original three feature sets are below. `features.py` now defines **eight**
(`full`, `notag`, `noml`, `slim`, `nowp`, `direct`, `direct_map`, `local`) and
`train.py` **four** targets (`flavor`, `genflavor`, `tight`, `origin`) —
`RESULTS.md` is the authoritative list of both, and of every trained model.

| set | what it drops | what it measures |
|---|---|---|
| `full` | nothing | best achievable separation |
| `notag` | DeepJet b/c/QG + QGL on the muon's **own** jet | those encode nearly the same information as `Jet_hadronFlavour` — i.e. very nearly the label — and are the most likely inputs to be mismodeled in data, so `full` − `notag` is the data/MC risk |
| `nowp` | every isolation / IP-significance variable, the lepton MVAs, and `conePt` | these define the tight WP (the FR numerator), so `full` − `nowp` is the size of the leakage bias below |

## The two things that decide whether this is useful

1. **AUC inside each `(conePt, |eta|)` FR cell**, printed by `evaluate.py`.
   The flat FR map already knows those two variables. A classifier that only
   rediscovers the existing binning adds exactly nothing.

2. **The leakage diagnostic**, `evaluate.py::leakage_check`.
   `mu_miniiso`, `mu_sip3d`, `mu_tkreliso` and `mu_conept` are the variables
   that *define* the tight WP, i.e. the FR numerator — `conePt` literally
   contains `miniiso`. Predicting flavor from them is legitimate, but a
   flavor-weighted FR built on the prediction is only valid if, at fixed true
   flavor, the prediction carries no information about tightness. The
   diagnostic plots FR against the predicted probability of the true class
   within each true flavor; it must be flat. If it is not, retrain without the
   WP-defining variables and take the difference as the bias.

## Conventions

Inherited from the parent project (`BDT/CLAUDE.md`): `pathlib`, parquet
intermediates, `logging` (no `print`), seed 42 via `src/utils.py`, matplotlib
PNG at 300 dpi, never `plt.show()`. `FEATURES_FULL` order is load-bearing —
append, never reorder.
