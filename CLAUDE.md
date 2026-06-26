# CLAUDE.md — H→ZZ→4μ BDT

Project guide for Claude Code. Read this first.

**Update this file when making significant changes to the codebase (new files, architecture changes, new dependencies, changed conventions).**

## Project goal

Train a binary classifier (signal vs background) on H→ZZ→4μ events using MC
samples from the companion [madgraph](https://github.com/siunkim0/madgraph)
pipeline (MadGraph5 + Pythia8 + Delphes, NanoAOD-style output).

## Current status

| Iter | Description | Status |
|------|-------------|--------|
| v1 / v2 / v3 | mass-included / mass-removed / planing | historical — see README iteration history |
| **Path 1 (v4_sr)** | `bdt_v4_sr.json`, m₄ℓ ∈ [105, 140] GeV, 23 feats, 13.6 TeV / 2023 MC | in-repo targets met (AUC 0.925, \|r\| = 0.094 ✓); superseded by v5_run2 for deployment. The original "unsatisfactory downstream" symptom was traced to a C++ feature-ordering bug, not a BDT problem — see README integration history |
| **v5_run2** *(production)* | `bdt_v5_run2.json`, Path 1 on Run2 / 2018 MC (L = 59.83 fb⁻¹, √s = 13 TeV) | production default. AUC 0.926, \|r\| = 0.091 ✓; deployed in SKNanoAnalyzer, **+33% Asimov Z over matched-selection cut-based** downstream |
| v6 | Path 1 + per-class m₄ℓ planing stacked (Run2 / 2018) | negative result, not adopted — best decorrelation (\|r\| = 0.076) but planing weight pathology returns (unstable training, w/u AUC divergence). See README v6 entry |

**`bdt_v5_run2.json` is the production default.** The Path 1 *procedure* (SR pre-selection + full 23-feature input + in-region training) is the standing workflow. Apply the SR cut at evaluation time too — `src/evaluate.py` reads `train.signal_region` from `selection.yaml` and mirrors the mask automatically.

Downstream significance is evaluated in SKNanoAnalyzer with a **matched event selection** between the cut-based and BDT analyzers (window m₄ℓ ∈ [120, 130] GeV). An earlier mismatched-selection comparison overstated the gain as +53%; the fair number is +33%. Re-run results live at `/data6/Users/snuintern2/Higgs/Tools/ye/`.

## Environment

- **Python**: 3.10+ via venv. See `requirements.txt`.
- **GPU**: xgboost `tree_method='hist'` with `device='cuda'` if visible, silent CPU fallback otherwise.
- **No ROOT dependency** — uproot reads NanoAOD directly.

## Data

- Signal: `ggH_ZZ_4l` (ggH → ZZ → 4μ, HEFT, LO)
- Background: `qqZZ_4l` (irreducible), `DY_M50`, `TTto2L2Nu` (reducible)

Sample paths, cross sections, and luminosity are in `config/samples.yaml`. **Do not hardcode paths in scripts.**

`config/samples.yaml` is **gitignored** (machine-specific paths). The committed template is `config/samples.yaml.example`. Never remove or commit the real `samples.yaml`.

## Selection (HZZ standard, simplified)

See `config/selection.yaml`. Summary:

- HLT: OR list (currently just `HLT_IsoMu24`; multi-muon paths are in the file, commented out)
- 4 muons, charge sum = 0, loose ID + relIso < 0.35
- Z₁ closest to mZ (∈ [40, 120]), Z₂ ∈ [12, 120]
- 70 < m₄ℓ < 1000 GeV training window; SR cut to [105, 140] GeV applied at train/eval time
- pT thresholds: leading μ > 20, subleading > 10, others > 5

## Features

23 columns, all in `src/features.py::FEATURES`. Used by Path 1 / v4_sr and v5 unchanged.

- Mass: `m4l`, `mZ1`, `mZ2`
- Kinematics: `pt4l`, `eta4l`, `pt_Z1`, `pt_Z2`, `dR_Z1Z2`
- Per-lepton: `pt_mu1..4`, `eta_mu1..4` (pT-sorted)
- Ratios: `pt_Z1/m4l`, `pt_Z2/m4l`
- Helicity angles: `cos_theta_star`, `cos_theta1`, `cos_theta2`, `Phi`, `Phi1` (arXiv:1208.4018)

v2 used a 17-variable mass-free subset; see README iteration history.

## Conventions

- All paths via `pathlib.Path`, never string concatenation.
- Use `awkward` end-to-end in skim; convert to `pandas`/`numpy` only at the train.py boundary.
- Save intermediates as **parquet** (not ROOT, not pickle).
- Random seed: 42 everywhere (`src/utils.py::SEED`).
- Logging: `logging` module, INFO level. No `print()` in src/.
- Plots: matplotlib, PNG (300 dpi) to `plots/`. Never `plt.show()`.

## Cross sections

The production model (v5_run2) trains on 13 TeV / 2018 MC; the live σ and `lumi_fb`
values are in `samples.yaml`. The table below records the original **13.6 TeV / 2023**
(v4_sr) values for reference — when switching eras, update both `lumi_fb` and these σ
values in `samples.yaml` to the matching √s equivalents.

| Sample | σ (pb), 13.6 TeV | Notes |
|--------|--------|-------|
| ggH→ZZ→4l | 0.00637 | σ(ggH) × BR(H→ZZ→4l), all flavors |
| qqZZ→4l | 1.39 | full ZZ→4l |
| DY M>50 | 6225 | inclusive |
| TT 2L2Nu | 98.04 | dilepton |

Reminder: the BDT decision boundary is invariant to `lumi_fb` and absolute σ
(`build_train_weight` renormalizes per-class to mean = 1); changing these only affects
downstream expected yields, not the trained model. See "Common gotchas" below.

## Common gotchas

1. **Negative MC weights** (`genWeight` < 0 for amcatnlo samples like DY): use `np.sign(genWeight)` for normalization.
2. **Z pairing**: 4 same-flavor muons → 2 valid OS pairings; pick Z₁ closest to mZ.
3. **PU reweighting**: not applied (known omission, MC-only study).
4. **BDT is lumi-invariant**: `build_train_weight` renormalizes per-class to mean=1, so changing `lumi_fb` or `xsec` in `samples.yaml` does *not* change the trained model. The model only changes if the *input MC distributions* change.

## Known limitations

- MC only — no collision data.
- No PU reweighting or lepton scale factors.
- Reducible backgrounds (DY, tt̄) reach 4μ only through fakes, which Delphes models poorly.
- LO cross sections throughout.
