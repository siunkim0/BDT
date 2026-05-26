# CLAUDE.md — H→ZZ→4μ BDT (Run3 2023)

Project guide for Claude Code. Read this first.

**Update this file when making significant changes to the codebase (new files, architecture changes, new dependencies, changed conventions).**

## Project goal

Train a binary classifier (signal vs background) on H→ZZ→4μ events using
Monte Carlo samples at √s = 13.6 TeV. Input samples are NanoAOD-style ROOT
files produced by the companion [madgraph](https://github.com/siunkim0/madgraph)
pipeline (MadGraph5 + Pythia8 + Delphes).

## Phases

| Phase | Output | Status |
|-------|--------|--------|
| 1. Skim   | flat parquet ntuples in `data/ntuples/`            | DONE |
| 2. Features | engineered columns appended to the ntuples        | DONE |
| 3. Train  | xgboost model in `data/models/`, training plots     | DONE — `bdt_v4_sr.json` (23 feats, SR-restricted training) is the recommended model. |
| 4. Evaluate | ROC, feature importance, overtraining check, m4l-vs-score 2D | DONE |
| 5. Planing | `bdt_v3_planed.json`, mass decorrelation study | DONE — planing alone did not hit the \|r(score, m4l)\| < 0.1 target. |
| 6. Path 1 — SR-restricted training | `bdt_v4_sr.json`, m₄ℓ ∈ [105, 140] GeV pre-selection | DONE — AUC = 0.925 (in-window), \|r\| = **0.094** ✓ target met. |

## Environment

- **Python**: 3.10+ via venv. See `requirements.txt`.
- **GPU usage**: xgboost `tree_method='hist'` with `device='cuda'` if a GPU is
  visible. Falls back to CPU silently.
- **No ROOT dependency** — uproot reads NanoAOD directly.

## Data

### Signal
- `ggH_ZZ_4l` — ggH → ZZ → 4μ (HEFT, LO)

### Background
- `qqZZ_4l` — qq̄ → ZZ → 4μ (irreducible)
- `DY_M50` — Z+jets (reducible)
- `TTto2L2Nu` — tt̄ dilepton (reducible)

Sample paths, cross sections, and luminosity are configured in
`config/samples.yaml`. **Do not hardcode paths in scripts.**

`config/samples.yaml` is **gitignored** (it contains machine-specific paths).
The committed template is `config/samples.yaml.example` — new users copy it
and fill in their paths. Do not remove the real `samples.yaml` or try to
commit it.

## Selection (HZZ standard, simplified)

See `config/selection.yaml` for the canonical values. Summary:

- HLT: OR of `HLT_TripleMu_10_5_5_DZ`, `HLT_TripleMu_12_10_5`,
  `HLT_DoubleMu4_3_LowMass`, `HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8`
- 4 muons, charge sum = 0, all loose ID + relIso < 0.35
- Build Z₁ (closest to mZ, mass ∈ [40, 120]), Z₂ (mass ∈ [12, 120])
- 70 < m₄ℓ < 1000 GeV (training window)
- pT thresholds: leading μ > 20, subleading > 10, others > 5

## Features

Phase 2 produces these columns from the skim. The **training input list lives
in `src/features.py::FEATURES`** and currently contains all 23 columns below
(restored to the full set for v4_sr; the SR cut at the data level provides
the decorrelation, not feature removal).

Stored columns (all in `FEATURES`):

- Mass: `m4l`, `mZ1`, `mZ2`
- Kinematics: `pt4l`, `eta4l`, `pt_Z1`, `pt_Z2`, `dR_Z1Z2`
- Per-lepton: `pt_mu1..4`, `eta_mu1..4` (sorted by pT)
- Ratios: `pt_Z1/m4l`, `pt_Z2/m4l`
- Helicity angles: `cos_theta_star`, `cos_theta1`, `cos_theta2`, `Phi`, `Phi1`
  (5 HZZ angular discriminants — see arXiv:1208.4018 for definitions)

v2 used a 17-variable subset that dropped `m4l`, `mZ1`, `mZ2`, the two pT/m₄ℓ
ratios, and `dR_Z1Z2`; see README iteration history for why removing features
turned out to be a worse decorrelation strategy than restricting the m₄ℓ
window at the data level.

#### Mass decorrelation — Path 1 (SR-restricted training) is the solution

Phase 5 attempted to push |r| < 0.1 by adding mass features back to FEATURES
and reweighting each class to a flat m4l distribution at training time
(per-class 1/density, 50 bins in [70, 250] GeV). |r| went **up** monotonically
from v1 → v2 → v3 because per-class normalization combined with single-mass-point
signal support constructed a binary mass-region indicator the BDT learned
preferentially (full diagnosis in README).

Phase 6 (Path 1) succeeded: pre-select to m₄ℓ ∈ [105, 140] GeV at the
dataframe boundary, retrain on the surviving 84,821 events with the full
23-feature set, no planing. The kinematic supports of signal and background
overlap by construction within the SR, so m₄ℓ remains in the input set
without inducing v1-style sculpting. Result: |r(score, m4l)| on background
**= 0.094** (target met), AUC = 0.925 in-window, KS p(sig) = 0.65,
p(bkg) = 0.93 → no overtraining.

**Recommendation**: `bdt_v4_sr.json` (23 features, m₄ℓ ∈ [105, 140] GeV
training cut) is the current default for any analysis that needs both
classification power and score-mass independence. `bdt_v2.json` (17
mass-free features, full m₄ℓ window) remains the reference baseline for the
mass-removal approach. Apply the same SR cut at evaluation time —
`src/evaluate.py` reads `train.signal_region` from `selection.yaml` and
mirrors the mask automatically.

The planing implementation and v3 model are kept in the repo as a documented
negative result (`config/selection.yaml::train.planing.enabled`).

## Repo structure

```
config/samples.yaml.example  ← committed template (FILL_ME paths)
config/samples.yaml           ← real paths (gitignored, never committed)
config/selection.yaml         ← cuts + BDT hyperparameters
src/{skim,features,train,evaluate,cv,utils}.py
scripts/export_onnx.py        ← xgboost → ONNX (--onnx-out flag for deploy path)
scripts/root_feature_stats.py
notebooks/bdt_tutorial.ipynb   ← interactive walk-through
docs/bdt_guide_ko.md          ← Korean-language learning guide
data/{ntuples,models}/         ← gitignored outputs (.gitkeep only)
plots/, logs/, reports/        ← gitignored outputs (.gitkeep only)
```

## Conventions

- All paths via `pathlib.Path`, never string concatenation.
- Use `awkward` arrays end-to-end in skim; convert to `pandas`/`numpy` only at
  the train.py boundary.
- Save intermediates as **parquet** (not ROOT, not pickle).
- Random seed: 42 everywhere. Set in one place (`src/utils.py::SEED`).
- Logging: `logging` module, INFO level by default. No `print()` in src/.
- Plots: matplotlib, save as PNG (300 dpi) to `plots/`. Never call `plt.show()`.

## Cross sections (13.6 TeV, used for MC weighting)

Source: LHC Higgs WG + GenXSecAnalyzer.

| Sample          | σ (pb)  | Notes |
|-----------------|---------|-------|
| ggH→ZZ→4l       | 0.00637 | σ(ggH) × BR(H→ZZ→4l), all flavors |
| qqZZ→4l         | 1.39    | full ZZ→4l |
| ggZZ→4l         | 0.0148  | optional, often folded into qqZZ k-factor |
| DY M>50         | 6225    | inclusive |
| TT 2L2Nu        | 98.04   | dilepton |

## Common gotchas

1. **Negative MC weights**: `genWeight` can be negative for amcatnlo samples
   (DY). Use `np.sign(genWeight)` for normalization, not just sum.
2. **NanoAOD muon collection**: `Muon_pt`, `Muon_eta`, etc. — flat arrays
   indexed by event. Use `awkward` to handle the jagged structure.
3. **Z pairing ambiguity**: with 4 same-flavor muons, there are 2 valid OS
   pairings. Pick the one where Z₁ is closest to mZ.
4. **PU reweighting**: not applied. Known omission for this MC-only study.

## Known limitations

- MC only — no collision data.
- No PU reweighting or lepton scale factors.
- Reducible backgrounds (DY, tt̄) produce 4μ only through fakes, which
  Delphes models poorly.
- LO cross sections throughout.
