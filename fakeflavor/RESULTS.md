# fakeflavor — results and handoff

State of the muon source-jet flavour BDT as of **2026-08-09**. Written so the
work can be picked up in a fresh session without re-deriving anything.

> **2026-08-09, later the same day.** Three things landed after the text below
> was first written: `mu_pogTight` was removed from every feature set, the
> 4-class model was measured on ttbar, and a new per-flavour QCD-vs-ttbar
> "origin" classifier was built. See **§12, §13, §14**. Where §1–§11 quote a
> feature count or an AUC, they describe the `_v2` generation; the current
> models are `_v3`, and the difference is ≤0.0002.

`README.md` (this directory) explains *what* the subproject is and how to run
it. This file records *what came out* and *what is still open*.

---

## 1. Where things stand in one paragraph

The 4-class (b/c/uds/g) classifier works and is fully characterised. The
headline number is **mean one-vs-rest AUC 0.811** with all 86 features on QCD
MuEnriched. Removing every ML-derived input (DeepJet, QGL, PU-Id, the three
muon MVAs) costs **−0.054** — by far the largest single effect measured in the
whole study — and trimming the survivors from 76 to a hand-picked 33 costs a
further **−0.013**. The remaining separation is essentially b-vs-light; uds
against g sits at a *kinematic* ceiling (binary AUC ≈ 0.69) because our jets are
soft. What is **not** done: the classifier has never been calibrated and has never
been applied to a fake-rate estimate. (It *has* now been tested on ttbar — §13,
where it reproduces the QCD numbers to 0.005.)

The original motivation — a *flavour-weighted* fake rate — is **dead**, killed
earlier and independently (see §7). The current models are useful as a
measurement of how much flavour information data-observable quantities carry,
and as feature-selection input for the direct `P(tight | loose, x)` route, which
is the live line of work.

---

## 2. Every trained model

All in `data/models/`, `<name>.json` + `<name>.meta.json` + `<name>.split.npz`.
Each `meta.json` records its own `train_config`, so the models are
self-describing. AUCs are **class-balanced one-vs-rest on the test split**.

| model | feat | sample | split | b | c | uds | g | **mean** | best iter |
|---|---|---|---|---|---|---|---|---|---|
| `flavor_full` | 74 | QCD+TTLJ | 64/16/20 | .8789 | .8048 | .7749 | .7798 | 0.80959 | 534 |
| `flavor_notag` | 69 | QCD+TTLJ | 64/16/20 | .8722 | .7925 | .7704 | .7737 | 0.80220 | 602 |
| `flavor_nowp` | 58 | QCD+TTLJ | 64/16/20 | .8586 | .7585 | .7466 | .7526 | 0.77907 | 650 |
| `flavor_recolabel_sv` | 86 | QCD+TTLJ | 64/16/20 | .8801 | .8071 | .7752 | .7798 | 0.81054 | 576 |
| `flavor_genlabel_sv` | 86 | QCD+TTLJ | 64/16/20 | .8848 | .8081 | .7839 | .7913 | **0.81704** | 549 |
| `flavor_nowp_sv` | 70 | QCD+TTLJ | 64/16/20 | .8605 | .7626 | .7462 | .7527 | 0.78053 | 776 |
| `flavor_full_qcd` | 86 | QCD | 64/16/20 | .8804 | .8054 | .7776 | .7797 | 0.81076 | 622 |
| `flavor_noml_qcd` | 76 | QCD | 64/16/20 | .8191 | .7454 | .7236 | .7379 | 0.75651 | 762 |
| `flavor_slim_qcd` | 33 | QCD | 64/16/20 | .8080 | .7293 | .7105 | .7251 | 0.74321 | 522 |
| `flavor_slim_qcd_paper` | 33 | QCD | 50/25/25 | .8073 | .7293 | .7124 | .7232 | 0.74304 | 483 |
| `flavor_noml_qcd_paper` | 76 | QCD | 50/25/25 | .8175 | .7449 | .7227 | .7355 | 0.75516 | 656 |
| `flavor_full_qcd_paper` | 86 | QCD | 50/25/25 | .8802 | .8049 | .7774 | .7781 | 0.81014 | 512 |
| `flavor_slim_qcd_v2` | 33 | QCD | 64/16/20 | .8080 | .7293 | .7105 | .7251 | 0.74321 | 522 |
| `flavor_noml_qcd_v2` | 76 | QCD | 64/16/20 | .8191 | .7454 | .7236 | .7379 | 0.75651 | 762 |
| `flavor_full_qcd_v2` | 86 | QCD | 64/16/20 | .8800 | .8050 | .7775 | .7798 | 0.81057 | 531 |
| **`flavor_slim_qcd_v3`** | 32 | QCD | 64/16/20 | .8081 | .7290 | .7112 | .7250 | **0.74335** | 992 |
| **`flavor_noml_qcd_v3`** | 75 | QCD | 64/16/20 | .8188 | .7448 | .7237 | .7378 | **0.75628** | 762 |
| **`flavor_full_qcd_v3`** | 85 | QCD | 64/16/20 | .8800 | .8050 | .7771 | .7793 | **0.81036** | 547 |
| **`flavor_slim_ttlj_v3`** | 32 | **TTLJ** | 64/16/20 | .8267 | .7035 | .7149 | .7485 | **0.74838** | 83 |
| **`flavor_noml_ttlj_v3`** | 75 | **TTLJ** | 64/16/20 | .8359 | .7154 | .7148 | .7693 | **0.75885** | 93 |
| **`flavor_full_ttlj_v3`** | 85 | **TTLJ** | 64/16/20 | .8903 | .7835 | .7852 | .7734 | **0.80811** | 105 |

The six `_v3` models are the **current headline set** — three on QCD, three on
ttbar, all without `mu_pogTight` (§12). `_v2` is the same thing with
`mu_pogTight` still in, kept as the before/after pair; `_paper` is the 50/25/25
experiment. The 15 `origin_*` models are in §14, not this table — they are a
different target and their AUC is a single binary number, not four.

Feature sets are defined in `features.py`: `full` (85) ⊃ `noml` (75) ⊃ `slim`
(32), plus `notag` (68), `nowp` (69), `direct` (64), `direct_map` (2) and
`local` (71, new in §14). `FEATURES_SLIM` is a **literal** list (top-25 by gain
from `flavor_noml_qcd` minus `mu_pogTight`, plus `mu_pt`, `mu_abseta`,
`mu_dxy`, `jet_nconst`, `jet_abseta`, `jet_dr_mu`, `ev_met`, `ev_mt`) —
deliberately not computed from a model file, so the module has no runtime
dependency on a trained artefact. Two asserts guard the count (32) and the
subset relation to `noml`.

---

## 3. The headline result: what the features are worth

All QCD MuEnriched, 64/16/20, class-balanced 25% per class, nested sets so the
chain decomposes cleanly:

| step | mean AUC | Δ |
|---|---|---|
| with-ML, 86 | 0.81057 | — |
| no-ML, 76 | 0.75651 | **−0.0541** |
| slim, 33 | 0.74321 | **−0.0133** |

**The taggers were carrying the model.** The 10 removed variables are
`jet_deepjet_{b,cvb,cvl,qg}`, `jet_qgl`, `jet_puIdDisc`, `jet_loosePuId`,
`mu_mvaTTH`, `mu_mvaLowPt`, `mu_softMva`. With them gone, `sv_masssum` becomes
rank 1 by gain — i.e. the model falls back to reconstructing the secondary
vertex by hand.

Other measured levers, for scale:
- gen-jet label instead of reco-jet label: **+0.0054** (`genlabel_sv` vs
  `recolabel_sv`)
- adding the `sv_*` block: **+0.0008**
- dropping TTLJ from training: **+0.0002** (it is 6% of rows but **0.0005%** of
  the training weight — about 12 effective events; it contributes nothing)
- every architecture / hyper-parameter knob tried: **≤ 0**

---

## 4. Train / val / test split — 64/16/20 was kept

`train.py` splits `test_size` off first, then `val_size` **of the remainder**,
so `0.20 / 0.20` gives 64/16/20. Verified on QCD (2,348,746 rows): train
1,503,198 / val 375,799 / test 469,749.

50/25/25 was tried on 2026-08-04 to follow `ref/2511.06972v2-2.pdf` §3.2 and
then reverted:

| set | test AUC 64/16/20 → 50/25/25 | train−test gap |
|---|---|---|
| slim / 33 | 0.7434 → 0.7432 | +0.0653 → +0.0730 |
| no-ML / 76 | 0.7567 → 0.7553 | +0.0977 → +0.1034 |
| with-ML / 86 | 0.8107 → 0.8103 | +0.0611 → +0.0705 |

64/16/20 wins on both, but only by +0.0002…+0.0013 in AUC and ~0.008 in the
gap. **The real argument is that 50/25/25 was adopted specifically to shrink the
train−test gap and the gap grew on all three sets.** The useful side finding:
test AUC is unchanged to within 0.0014 despite 22% fewer training rows, so the
quoted numbers are not statistics-limited by the split choice.

---

## 5. The LR-on-plateau schedule is a structural no-op

`train.py` has a `ReduceLROnPlateau(xgb.callback.TrainingCallback)`, gated on
`bdt.lr_plateau.enabled` in the config. It fires (3× per model, ending at
lr 0.00625) and changes nothing. Why:

| set | best_iteration | first LR drop |
|---|---|---|
| slim | 522 | 542 |
| no-ML | 762 | 780 |
| with-ML | 531 | 551 |

The first drop always lands `patience` (20) rounds **after** `best_iteration`,
because the plateau detector and the best-iteration tracker watch the same
metric — a plateau is by definition only detected after the best value has
already passed. The saved model is truncated at `best_iteration`, so every tree
the schedule touches is discarded.

Confirmed empirically: `flavor_slim_qcd_v2` and `flavor_noml_qcd_v2` reproduce
the pre-schedule `flavor_slim_qcd` / `flavor_noml_qcd` **to five decimals**.

For it to do anything it would need `patience > early_stopping_rounds`, which is
a different and probably worse setup. **Recommendation: set
`bdt.lr_plateau.enabled: false`** rather than leave an inert knob that logs as
if it were working. Not done yet — one-line change, never authorised.
`config/direct.yaml` has no `lr_plateau` key at all, so the direct-FR route is
unaffected (verified repeatedly).

---

## 6. Overfitting is the rare classes, and it is a statistics problem

Training split, from `plots/qcd_v2/T_training_table.png`:

| class | rows | weight % | effective N = (Σw)²/Σw² |
|---|---|---|---|
| b | 973,907 | 25.0% | 338,128 |
| c | 403,703 | 25.0% | 142,096 |
| uds | 54,887 | 25.0% | **19,907** |
| g | 70,701 | 25.0% | **21,031** |

All four carry identical *weight*; statistical *power* spans **17×**. The
train−test gap tracks it exactly: b +0.017, c +0.040, uds +0.109, g +0.096.

So the lever for overfitting is capacity (`max_depth`, `min_child_weight`), not
the split and not the learning rate. Neither has been scanned.

---

## 7. Physics conclusions that are settled — do not re-derive

**uds vs g is at a kinematic ceiling, not a model ceiling.** Binary AUC is 0.672
with QGL alone and 0.693 with all 86 — i.e. 84 extra features buy 0.02. It rises
0.667 → 0.737 with jet pT. Measured in our own ntuple, the gluon−uds
constituent-multiplicity gap is only **+0.96** at jet pT 20-30 GeV and **+1.81**
at 30-40, rising to +7.28 above 200. **Our median jet is 31.5 GeV and 76% of
rows are below 50 GeV** — the far-left region where the colour-factor gap is
1-2 constituents on distributions several units wide. Confirmed independently
against `hi/track/Tools/Plots/v21_2016/profile_flavour_nConstituents_vs_Pt_QCD_fit.png`
(same ordering g > c ≈ b > uds, same pT trend; its absolute scale sits 5-6 above
ours because of a different constituent definition).

**Neither flavour label is truth, and the gen-vs-reco argument was overstated.**
Both call the same `FlavorTag()`. b and c come from `hadronFlavour`
(ghost-matched B/C hadron) and the two labellings agree 99.8%. The entire
disagreement is uds vs g and rests on `partonFlavour`, which is a *convention*
and is IR-unsafe for gluons. Measured: jets that migrate between the two
labellings sit *midway* between the pure populations — evidence that neither
label is better, not that gen is. The +0.0054 that `genlabel_sv` gains is a
cleaner *target*, not a more correct one.

**WP leakage is physical and out of scope.** For true c the fake rate collapses
0.336 → 0.007 across the predicted-probability range (56×). The `nowp` control
(16 WP-defining variables removed, −0.0300 AUC) left every swing essentially
unchanged (56.4× → 53.7×), so the leakage is not carried by the WP variables —
it is the physical correlation between flavour and tightness. **The user has
declared this closed: no leakage test, no leakage slides** (memory:
`feedback_leakage_out_of_scope.md`).

**Flavour-weighted FR is dead** (memory: `project_flavor_bdt_leakage.md`). The
closure ceiling using *truth* flavour is 1.127 on the QCD→TTLJ transfer, which
is *worse* than the flat 2D FR map's 1.095 — per-flavour FR is not
sample-universal after all, so no classifier, however good, can help. This is
why the live route is the **direct** `P(tight | loose, x)` estimator
(memory: `project_direct_fr_model.md`), which scores 1.023 on the same transfer.
`ref/2511.06972v2-2.pdf` (Neural Fake Factor) is the reference for that route.

**AUC 0.88 vs a 0.66 confusion diagonal is not a contradiction.** The prior odds
against b are 14× (uds) and 15× (g); prior-corrected, the predicted label mix is
b 0.812 / c 0.180 / uds 0.0045 / g 0.0041 and b's diagonal goes 0.543 → 0.905.
That is normal argmax behaviour under a skewed prior with a weak discriminant —
and it is irrelevant to a *weighted* FR, which never calls argmax. The metric
that would actually matter there is calibration, and it has never been measured.

---

## 8. Training population vs the input plots

The training ntuples (`data/ntuples_sv/QCD_MuEnriched.parquet`) **do** carry the
fake-rate measurement selection, baked in by `fake.cc`: verified directly on the
parquet — `ev_met` max 25.00, `ev_mt` max 25.00, zero rows above either,
`ev_awayjet_pt` min exactly 40.

This was the subject of a long confusion. The *plots* were the ones without
selection: in `track.cc` the entire selection block (lines 334-397) is inside a
`/* */` comment and the histograms are filled before it. The `v20_2022` and
`v21_2016` plot sets are therefore unselected and have a completely different
mixture (b 39.5% vs training's 64.8%).

The corrected set is **`hi/track/Tools/Plots/fakesel_2016/QCD/fakeSel/`** —
MET < 25, M_T < 25, away jet pT > 40, four classes only. Its mixture agrees with
training to about a percentage point:

| | b | c | uds | g |
|---|---|---|---|---|
| plots (`fakesel_2016`) | 66.4% | 25.9% | 4.0% | 3.7% |
| training | 64.8% | 26.8% | 3.7% | 4.7% |

The residual is era coverage — the plots are 2016 only (36.33 fb⁻¹), training
adds 2017 and 2018. All 33 slim features are covered and the file stems are
literally the feature names, so the mapping is exact (`sv_chi2` is also in the
directory but is not in the slim set).

---

## 9. Decks and where each one lives

Under `plots/`. Every deck is 16:9 Times New Roman house style, `.pptx` **and**
`.pdf` built from one shared slide spec, page number bottom-right. **No
LibreOffice on these machines — the `.pptx` files have never been visually
rendered**; they were verified structurally (python-pptx reopen: slide count,
runs, shape geometry) and the PDFs were rendered page-by-page with `pymupdf`.

| dir | slides | what |
|---|---|---|
| `summary/` | 14 | the original report deck |
| `reco_sv/` | 19 | reco label + 86 features, with the `nowp` leakage control |
| `qcd_noml/` | 16 | no-ML 76 |
| `qcd_slim/` | 13 | slim 33 |
| `qcd_paper/` | 48 | the §3.2 / 50-25-25 protocol run |
| `qcd_v2/` | 48 | three arms at 64/16/20 + input appendix |
| **`flavor_v3/`** | **46 / 46 / 64** | **current: three decks in the qcd_v2 format — QCD, ttbar, QCD-vs-ttbar (§12-14)** |
| **`origin_cells/`** | **184** | **current: QCD-vs-ttbar one FR cell at a time (§15). The full body block runs for both global arms and all 24 (flavour, cell) pairs, which is most of the page count. Built by `flavor_v3/deck_figs.py` + `make_decks.py`, lands here** |

Per-model plots moved out of the flat `plots/` directory on 2026-08-09: one
directory per model, `<run>/models/<model>/{roc,confusion,importance,scores,
training_history}.png` (+ `leakage.png` for flavour targets). 21 models x 6 PNGs
loose in `plots/` was unreadable, and 21 model folders loose in `plots/` was
worse. **`train.py` and `evaluate.py` take `--plot-dir`** — point it at
`plots/<run>/models` so a multi-model run adds exactly one folder to `plots/`,
the same as every deck directory. The default is still bare `plots/`, so pass it.

`flavor_v3/` holds all three `_v3` decks in one directory, built by
`deck_figs.py` + `make_decks.py` from a single shared slide list so the QCD,
ttbar and origin decks stay in the same format. Body plots are the **slim / 32**
arm (**local / 71** for origin) — the with-ML arm appears only at the top of the
ablation chain in `A_auc_summary`. Input distributions come from
`hi/track/Tools/Plots/v22_2016/<sample>/pfMatchedJet_fake/`; the origin deck
pairs QCD and TTLJ two-up on each appendix slide.

`qcd_v2/` structure: 1 title · 2 the three arms · 3 `A_auc_summary` · 4
`S_split` · 5 `T_training_table` · 6 training history · 7 ROC · 8
`C_confusion_both` · 9 scores · 10 `I_importance_all33` · 11 `B_cell_auc` · 12
how to read this · 13 name→branch appendix · 14 `D_features` · 15 appendix intro
· 16-48 one slide per feature in gain order (`sv_masssum` #1 → `mu_dxy` #33).

Each deck directory has its own `README.md` with the run's numbers, and the
one-off generator scripts are **copied into the directory next to their output**
(`qcd_v2_extra.py`, `make_qcd_v2_deck.py`, `split_fig.py`, `table_v2.py`,
`fakesel_order.py`, and the `qcd_paper` / `reco_sv` equivalents). They are not
part of the package — regenerate by copying and repointing, per house
convention.

Gotchas learned building these, worth not rediscovering:
- U+25B6 (▶) and similar glyphs are **missing from Nimbus Roman**, which is what
  matplotlib resolves "Times New Roman" to here. Use ASCII.
- A tall single-column table or bar chart renders unreadably small on a 16:9
  slide — split into two panels.
- `I_importance_all33` needs **one common axis** (capped at rank 2 × 1.35 with
  the `sv_masssum` outlier clipped and labelled). Per-panel scales are
  misleading.
- `D_features.png`'s "from" column must rank against the **original**
  `flavor_noml_qcd` — that is the model `_SLIM_TOP25` was actually selected
  from, and later retrains swap `sv_dxysig` / `mu_pt` across the rank-25
  boundary.

---

## 10. Current config state

`config/train.yaml`:

```yaml
train:  test_size: 0.20      # -> 64 / 16 / 20
        val_size:  0.20
        physical_weight: true, class_balance: true
        weight_clip_quantile: 0.999, split_by_event: true
bdt:    n_estimators 1000, max_depth 6, learning_rate 0.05,
        subsample 0.8, colsample_bytree 0.8, min_child_weight 10,
        gamma 0.0, reg_lambda 1.0, eval_metric mlogloss,
        early_stopping_rounds: 60
        lr_plateau: {enabled: true, factor: 0.5, patience: 20, min_lr: 0.005}
```

`reg_lambda` is deliberately **not** set to the paper's 1e-5: that is AdamW
weight decay on network parameters and has no correspondence to L2 on leaf
weights; copying the number would effectively remove regularisation.

`config/direct.yaml` is untouched and has no `lr_plateau` key.

Training runs on the **GPU server** (`ai-tamsa1`), reading `/data6` in place —
no rsync, no bootstrap (memory: `project_gpu_server_env.md`). Roughly 5 minutes
for all three arms:

```bash
for FS in slim noml full; do
  CUDA_VISIBLE_DEVICES=0 .venv-fakeflavor/bin/python -m fakeflavor.train \
    --target flavor --config fakeflavor/config/train.yaml \
    --ntuples fakeflavor/data/ntuples_sv/QCD_MuEnriched.parquet \
    --feature-set $FS --out fakeflavor/data/models/flavor_${FS}_qcd_v2.json
done
```

Two known environment bugs (memory): `detect_gpu()` in `BDT/src/utils.py` always
returns True (xgboost only warns when CUDA is absent, it does not raise), and
xgboost 2.1.4 silently resets `base_score` to 0.5 on load — AUC is unaffected,
absolute probabilities inflate 1.53×; worked around in `fakeflavor/evaluate.py`.

---

## 11. Open items

**Cheap and recommended**
1. Disable `bdt.lr_plateau` (§5). One line, proven inert.
2. Two stale slides in `plots/flavor_opt/` were never regenerated.

**The real gaps**
3. **Calibration has never been measured.** For any weighted use of this
   classifier it is *the* metric, and neither a reliability diagram nor a
   post-hoc calibration exists. Everything quoted here is ranking-only.
4. ~~**ttbar behaviour is untested.**~~ **Closed 2026-08-09, see §13.** The
   4-class model was retrained on TTLJ: mean OvR AUC 0.8081 / 0.7589 / 0.7484
   against QCD's 0.8104 / 0.7563 / 0.7434. What remains open is the *transfer*
   (train QCD, apply to ttbar), which is a different measurement — §14 is the
   first evidence on it and it is not encouraging.
5. **`sv_*` data/MC agreement is unverified.** The `sv_` block is now rank 1 by
   gain in the no-ML and slim models, so it carries the model, and it has never
   been compared against data. Closing this needs the four commented-out
   `DoubleMuon --tag bdtflav2` lines in `fake.sh` submitted (tamsa2 only — the
   user's job, ask first).
6. A `max_depth` / `min_child_weight` scan, aimed at the uds/g gap (§6). Never
   run.
7. `jet_genJetIdx` and gen-jet kinematics are not written out of `fake.cc`, so
   the gen-label study cannot be extended without a re-skim.

**Explicitly closed — do not reopen**
- The leakage test (user's decision).
- Flavour-weighted FR as a method (loses to the flat map).
- The 50/25/25 split (measured, reverted).

---

## 12. `mu_pogTight` removed — and it was worth nothing

Dropped from `features.py` on 2026-08-09. It is the POG **tight** muon ID flag,
and it was in *every* feature set except `direct_map` — including `nowp` and
`direct`, the two sets whose entire stated purpose is to contain no tight-WP
information. It is not the analysis tight WP (`isTight` = POG medium + |SIP3D| <
3 + miniPFRelIso < 0.1), but a hard ID flag correlated with it has no business
in those two sets, and it ranked **9th by gain** in `slim`, so it looked
load-bearing.

It was not. Same data, same config, `_v2` → `_v3`:

| arm | with `mu_pogTight` | without | Δ |
|---|---|---|---|
| slim | 0.74321 (33) | 0.74335 (32) | **+0.00014** |
| no-ML | 0.75651 (76) | 0.75628 (75) | −0.00023 |
| with-ML | 0.81057 (86) | 0.81036 (85) | −0.00021 |

All three are inside the noise. A rank-9 gain of 105 units against `sv_masssum`'s
920 was a *shared* contribution, not a unique one — the other 30 features
reconstruct it. **Do not put it back**, and read this as a general warning about
gain rankings: they measure use, not irreplaceability.

Two consequences that must not be lost:

- The positional feature contract in `features.py` was **broken deliberately** —
  removing a mid-list column shifts every later index. Every model on disk from
  before 2026-08-09 must be read through its own `.meta.json` (which
  `evaluate.py` does) or retrained. Verified: `flavor_slim_qcd_v2` still
  evaluates to its recorded .8080/.7294/.7102/.7259.
- **Every `direct_*` model is stale.** They were trained with `mu_pogTight`
  inside `direct`, which is exactly the kind of input the direct route's own
  rule forbids. It is a hard ID flag rather than a function of miniPFRelIso or
  SIP3D, so this is probably a small effect — but it is unmeasured, and the
  direct route is the live one. Retraining them is the cheapest open item there
  is.

---

## 13. The 4-class model on ttbar — it transfers, and it is not the QCD prior

`--ntuples ntuples_sv/TTLJ.parquet`, otherwise identical to the QCD arms.

| arm | QCD `_v3` | **TTLJ `_v3`** | Δ |
|---|---|---|---|
| slim (32) | 0.74335 | **0.74838** | +0.0050 |
| no-ML (75) | 0.75628 | **0.75885** | +0.0026 |
| with-ML (85) | 0.81036 | **0.80811** | −0.0023 |

**The separation is a property of jets, not of the QCD sample.** All three arms
land within 0.005 of their QCD twins, on a completely different process with a
16× smaller sample and an almost inverted flavour prior (ttbar fakes are 89% b,
QCD 65%). The old `flavor_full_ttlj` (2026-07-30, old ntuple, 74 features,
0.8076) said the same thing and is now superseded.

The tagger ablation survives too: `full` − `noml` is **−0.0493** in ttbar
against **−0.0541** in QCD. The §3 conclusion — the taggers carry the model — is
not a QCD artefact.

Per class, the interesting part is that the two processes trade:

| | b | c | uds | g |
|---|---|---|---|---|
| QCD `full` | .8800 | **.8050** | .7771 | **.7793** |
| TTLJ `full` | **.8903** | .7835 | **.7852** | .7734 |

ttbar b is *easier* (+0.010) and ttbar c is *harder* (−0.022), which is what a
sample whose b jets come from top decay rather than gluon splitting should look
like. Do not over-read uds/g: TTLJ has **1,463 uds and 1,076 g rows in all of
Run 2**, i.e. ~290 and ~215 in the test split, so those two columns carry
roughly ±0.03 and neither Δ above is significant.

Two things to be aware of before quoting these:

- `best_iteration` is **83–105** on ttbar against 547–992 on QCD. With 149,636
  rows early stopping fires almost immediately; these are far smaller models,
  and the config's `min_child_weight: 10` / `weight_clip_quantile: 0.999` were
  tuned for QCD's 6-decade weight spread, which TTLJ does not have.
- The gain ranking changes character entirely. QCD `noml` leads with
  `sv_masssum` (1220); ttbar `noml` leads with **`sv_dr_jet` (313)** and
  `mu_jetNDauCharged` (129). Same AUC, different route to it — a hint that the
  two samples are not the same problem, which §14 makes explicit.

---

## 14. **NEW** — QCD vs ttbar at fixed flavour: they are different objects

The new `--target origin` model. Binary QCD-vs-ttbar, trained **inside one true
flavour class** (`--flavor b|c|uds|g`), on `train.parquet` with `class_balance`
undoing the 1.1e5 physical-weight ratio between the samples. Run on the **same
three nested feature sets as the flavour decks** so the two studies are directly
comparable, plus two extra control arms.

AUC for the TT class, held-out test split:

| slice | with-ML 85 | no-ML 75 | **slim 32** | *local 71* | *conePt,\|eta\|* |
|---|---|---|---|---|---|
| b | .9532 | .9532 | **.7581** | .7773 | .7082 |
| c | .9556 | .9563 | **.6717** | .6630 | .6091 |
| uds | .9431 | .9433 | **.5904** | .5899 | .5483 |
| g | .9578 | .9579 | **.7238** | .7571 | .5606 |
| **mean** | 0.9524 | 0.9527 | **0.6860** | 0.6968 | 0.6065 |

The ablation reads completely differently from the flavour target:

- **the ML outputs are worth +0.0003, i.e. nothing.** On the flavour target they
  were worth −0.0541. The taggers tell you the *flavour*, which is held fixed
  here, so they have nothing left to say;
- **the 75 → 32 step costs −0.2667**, and that step is the **event block**.
  `full` and `noml` both keep `ev_ht`, `ev_njet*` and `ev_awayjet_*`; their gain
  ranking is `ev_ht` (3241) and `ev_njet30` (2171) in every slice. At 0.95 they
  are identifying the *event* — ttbar events trivially have more jets — not the
  fake. They are the top of the ablation chain, not a result.

**`slim` is the physics arm**, and `local` (71 = `full` minus every `ev_*`)
agrees with it to 0.02–0.03 on every slice, which is the cross-check that the
`ev_met`/`ev_mt` still in `slim` are not doing the work — both are capped at
25 GeV by the selection.

**It says the fake object itself differs.** At fixed true flavour, a b-jet fake
from ttbar is distinguishable from a b-jet fake from QCD at AUC 0.758 on muon
and jet variables alone. That is a mechanism for the §7 finding that per-flavour
FR is **not** sample-universal (FR_b 0.4831 QCD vs 0.4501 TTLJ; FR_c 0.4507 vs
0.3540) — the populations being averaged are not the same population.

**And most of it is not absorbed by the existing map.** The decisive number is
the per-FR-cell table, not the inclusive AUC. Inside each (conePt, |eta|) cell
of the map `bkg.cc` actually uses, `origin_b_local` scores:

```
conePt  [10,12) [12,14) [14,17) [17,20) [20,30) [30,50) [50,100)
|eta| <0.9   .668    .698    .691    .705    .708    .716    .672
 0.9-1.6     .682    .661    .708    .697    .713    .718    .665
 1.6-2.4     .670    .678    .697    .693    .699    .714    .678
```

Flat at **0.66–0.72 across all 21 cells** — barely below the inclusive 0.777.
The QCD/ttbar difference is therefore *not* mostly the conePt spectrum; it
survives conditioning on the exact binning the fake rate is measured in.
`origin_b_slim` gives the same picture (mean cell AUC 0.662), and
`origin_c_slim` 0.630. **uds and g have no per-cell table at all**: fewer than
50 ttbar rows land in any single cell, which is the statistics limit of those
two slices (training-split effective N is 903 and 689 on the ttbar side).

Two controls, both passed:

1. **`origin_b_direct_map` inside a cell gives 0.486–0.551 in the five narrow
   conePt bins**, i.e. 0.5 as it must, since that model sees only the two
   variables that define the cell. It rises to 0.579–0.604 in `[30,50)` and
   `[50,100)` — not a failure of the control but a statement about the binning:
   those two bins are wide enough that conePt still discriminates *inside*
   them. Either way it is far below `local`'s flat 0.66–0.72, and the machinery
   is not manufacturing separation.
2. **Era is not the discriminant.** Weighted era composition of the b slice is
   QCD 17.8/14.3/26.8/41.0% vs TTLJ 16.9/14.0/26.9/42.3% across
   2016preVFP/postVFP/2017/2018 — identical to a percentage point, so no
   era-sensitive variable can be carrying this.

Gluon is the outlier: 0.724 (`slim`) / 0.757 (`local`) against a conePt-|eta|
baseline of only 0.561, the largest residual of the four, driven by
`mu_jetNDauCharged` (547 in `slim`). Gluon jets in the two processes really do
look different. It rests on 1,076 ttbar rows, so treat the value as indicative
and the *ordering* as the result.

**What this means for the live work.** It is a warning for the direct
`P(tight | loose, x)` route, not a refutation of it: the route's whole claim is
its QCD→TTLJ transfer (1.023 against the flat map's 1.095), and this says the
two domains are separable at 0.78 in the very variables the model is trained on.
A classifier that can tell the domains apart can also have learned
domain-specific structure. The natural follow-up is the reverse test — train the
direct model on ttbar and transfer to QCD — and a check of whether the transfer
degrades in the cells where `origin_*_local` is largest.

**What it does not mean.** It does not revive the flavour-weighted FR. §7 stands
untouched: that route died on its *ceiling* with truth flavour (1.127 vs the
flat map's 1.095), and better flavour information cannot fix a ceiling. §14 is
an explanation of *why* the per-flavour maps failed to transfer, not a route
back.

**Followed up in §15**, which asks the same question one FR cell at a time with
a model trained inside each cell, so it cannot borrow separation from the cell
boundary.

---

## 15. **NEW** — the same question, one fake-rate cell at a time

§14 conditioned on the analysis binning by *evaluating* the inclusive model
inside each cell. That leaves one loophole: `mu_conept` is that model's top
feature by a factor of four, so part of what it reads inside a wide cell is
still conePt. §15 closes it by **training a separate model inside every cell**,
on a coarse binning chosen so each cell is an exact union of analysis cells:

- conePt **[10,20) [20,30) [30,50) [50,100)** — `[20,30)`, `[30,50)`, `[50,100)`
  are analysis bins verbatim, `[10,20)` is the union of the four below 20.
- |eta| **[0,0.9) [0.9,1.6) [1.6,2.4)**
- **b and c only.** uds and g were dropped for statistics before the run: no
  cell holds 50 ttbar rows of either.

48 models (2 flavours x 12 cells x {`slim` 32, `direct_map` 2}) plus a
re-evaluation of the six inclusive arms in the new bins. New machinery:
`train.py --cell 'ptlo,pthi,etalo,etahi'`, applied **before** `build_weights`
and the split so the class balance and the weight clip are computed within the
cell, and replayed by `evaluate.py` from `meta["cell"]` before the saved test
indices are applied. `config/origin_cells.yaml` differs from `origin.yaml`
**only** in the `evaluate` binning.

AUC for the TT class, physically weighted (the same normalisation as every
per-cell table — **not** comparable to the class-balanced headline numbers of
§14):

| | b: trained | b: global | b: floor | c: trained | c: global | c: floor |
|---|---|---|---|---|---|---|
| [10,20) x [0,0.9) | .663 | .674 | .564 | .606 | .650 | .546 |
| [10,20) x [0.9,1.6) | .667 | .668 | .561 | .626 | .635 | .519 |
| [10,20) x [1.6,2.4) | .651 | .674 | .563 | .596 | .612 | .525 |
| [20,30) x [0,0.9) | .661 | .666 | .541 | .620 | .625 | .497 |
| [20,30) x [0.9,1.6) | .658 | .675 | .537 | .599 | .634 | .490 |
| [20,30) x [1.6,2.4) | .650 | .663 | .547 | .612 | .640 | .517 |
| [30,50) x [0,0.9) | .655 | .672 | .569 | .601 | .633 | .495 |
| [30,50) x [0.9,1.6) | .661 | .683 | .580 | .595 | .638 | .537 |
| [30,50) x [1.6,2.4) | .670 | .675 | .588 | .579 | .621 | .526 |
| [50,100) x [0,0.9) | .633 | .650 | .594 | **.485** | .586 | .501 |
| [50,100) x [0.9,1.6) | .626 | .655 | .566 | **.485** | .663 | .530 |
| [50,100) x [1.6,2.4) | .651 | .645 | .574 | — | — | — |

`trained` = a model fitted only on that cell's rows. `global` = the inclusive
`origin_{fl}_slim` of §14 scored on the same rows. `floor` = the same fit given
only (conePt, |eta|), i.e. what the flat FR map already knows.

**The answer, for b: yes, in all twelve cells.** 0.626–0.670, mean 0.654,
against a floor of 0.537–0.594 (mean 0.565). The two processes differ inside
every bin the fake rate is measured in, and the difference is not the conePt
spectrum.

**For c: yes below 50 GeV; above it the in-cell fits run out of statistics.**
0.579–0.626 in the nine lower cells (mean gap over the floor +0.087). In
`[50,100)` the in-cell fit returns 0.485 — i.e. nothing — but that is the fit
failing, not a physics statement: those two models trained on **167 and 331**
ttbar rows and early-stopped at boosting round **3** and **13**. The third cell
of that row has 137 rows in all of Run 2 and cannot be scored at all. The global
model's 0.586 / 0.663 stands there, with its own ±0.07 from 52–103 test rows.

Three things worth keeping:

1. **The gap over the floor is largest where the fake background lives.** For b
   it runs +0.098 / +0.115 / +0.083 / +0.059 across the four conePt rows. The
   QCD/ttbar difference is concentrated at low conePt, which is the *worse* case
   for the matrix method.
2. **The global model reads the same inside a cell** — trained minus global
   averages **-0.013** on b, largest single deviation 0.029. So §14's 21-cell
   table was not an artefact of fitting across cells, and the inclusive model
   transfers to any binning. On c the same figure is -0.049, driven entirely by
   `[50,100)`. **That deficit is training statistics, not leakage**:
   corr(log ttbar training rows, trained − global) = **+0.641** over the 23
   measurable cells, cells above 2,000 ttbar training rows average -0.013 and
   cells below average -0.049, and `best_iteration` collapses from 46–159 in
   the b cells to 13 and 3 in the two thin c cells. Where the cell-trained arm
   has statistics it confirms the global number; where it does not, it is a
   statistics readout and the global number is the better estimate.
   *(Corrected 2026-08-12 — this was first written up as the global model
   importing separation across the cell boundary, which the evidence above does
   not support.)*
3. **The floor is not 0.5 here, and that is the binning.** conePt/|eta| alone
   gives 0.537–0.594 on b, against 0.486–0.551 in the narrow analysis bins.
   Merging four bins into `[10,20)` leaves conePt with room to discriminate
   inside the cell. Quote `trained − floor`, not `trained`.

Checks that were run, all passed: every one of the 48 filter lines reproduces
the parquet row count for its cell; every cell model's `.meta.json` carries
`target`/`flavor_slice`/`cell` and a scalar `base_score`; every cell model's own
per-cell table has exactly one populated row matching its inclusive AUC to
<2e-3; and the nine coarse cells that are unchanged analysis bins **reproduce
the values already logged in `plots/flavor_v3/eval.log` to three decimals**.

Not covered: `conePt >= 100` (3.0% of the b slice, 3.8% of c), uds and g, and
anything outside MC.

### 15.1 The kinematic ablation chain — you cannot remove a variable by removing its column

Run 2026-08-12 in two steps, and the second is the deck's third series. Both are
inclusive per-flavour fits scored per cell, replacing the conePt/|eta|-only
floor on the A slide.

| arm | features | b (phys) | c (phys) | rank 1 by gain (b / c) |
|---|---|---|---|---|
| global `slim` | 32 | 0.7587 | 0.6725 | `mu_conept` 932 / 286 |
| global `slim_nokin` | 30 | 0.7590 | 0.6713 | `mu_pt` 696 / 328 |
| global `slim_nomukin` | 29 | 0.7583 | 0.6731 | `mu_ptErr` 412 / 177 |
| global `slim_nomukin28` | **28** | **0.7567** | **0.6649** | **`jet_nconst` 386 / 206** |

Each rung removes the variable that had just taken rank 1: `slim_nokin` drops
`mu_conept` and `mu_abseta`, `slim_nomukin` drops `mu_pt`, `slim_nomukin28`
drops `mu_ptErr`. **None of it changes anything.** Over four rungs the inclusive
AUC moves by **-0.0020 on b** and **-0.0076 on c**; per cell, `nomukin28 −
global` averages **-0.0085 on b** (max deviation 0.036) and **-0.0160 on c**
(max 0.049, in the 52-test-row `[50,100)` cell).

What moves is only which variable holds rank 1 by gain: **`mu_conept` → `mu_pt`
→ `mu_ptErr` → `jet_nconst`** (with `sv_pt` right behind it at rank 2). The
successor is always waiting, and the measured correlations on the true-b slice
say in advance who it will be:

| feature | \|r\| vs `mu_conept` | \|r\| vs `mu_abseta` | in the 28? |
|---|---|---|---|
| `mu_pt` | **0.988** (Spearman 0.969) | 0.068 | no |
| `jet_abseta` | 0.078 | **0.996** | **yes** |
| `mu_ptErr` | 0.724 | 0.065 | no |
| `sv_pt` | **0.673** | 0.053 | **yes** |
| `mu_jetNDauCharged` | 0.481 | 0.134 | **yes** |
| `jet_nconst` | 0.387 | 0.154 | **yes** |

**Why the peel can never work, measured directly.** An XGBoost regressor fitted
on whatever survives each rung reconstructs the removed variables this well
(200k rows of the b slice, 70/30 split, 150 trees, depth 6):

| feature set | R²(`mu_conept`) | R²(`mu_abseta`) |
|---|---|---|
| 32 `slim` | 0.974 | 1.000 |
| 30 | 0.971 | 0.997 |
| 29 | 0.964 | 0.996 |
| **28** | **0.959** | 0.996 |
| 27 (also −`sv_pt`) | 0.961 | 0.996 |
| 26 (also −`jet_abseta`) | 0.961 | **0.511** |

**conePt is reconstructible at R² ≈ 0.96 no matter which proxy is removed.** It
is not carried by any single column — it is distributed across the set — so
removing its proxies one at a time cannot blind the model to it, and the flat
AUC is the direct consequence. |eta| behaves oppositely: `jet_abseta` is its
*sole* carrier, and dropping that one column takes R² from 0.996 to 0.511.
(The regression split is by row, not by event, so these R² are marginally
optimistic; the effect is far too large for that to matter.)

So `slim_nomukin28` is **"no muon kinematic columns", not "no kinematics"** —
|eta| is entirely untouched through `jet_abseta`, and conePt is 96%
recoverable — and it must never be labelled otherwise on a plot.

**The transferable lesson: an ablation removes a column, not the information.**
In a feature set this correlated, "we removed X and the AUC did not move" is not
evidence that X was unimportant. It applies to the `notag` and `nowp` ablations
in §3 and §7 as much as here. Measure the correlations, watch what takes over
rank 1, and if the question really is "what does this model know without the
kinematics", **column-dropping is the wrong tool** — that needs decorrelation
or an adversarial/planing approach, not a shorter feature list.

The conePt/|eta|-only `direct_map` arm these replaced is still measured and
still on record — 0.537–0.594 on b in these coarse cells against 0.486–0.551 in
the narrow analysis bins — in `plots/origin_cells/runA.log` and `eval.log`. It
is simply no longer on a slide.
