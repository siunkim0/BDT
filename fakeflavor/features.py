"""Feature definitions for the muon source-jet flavor classifier.

`FEATURES_FULL` order is **load-bearing**: xgboost stores columns positionally,
so any C++/ONNX consumer must feed them in exactly this order. The HZZ side of
this repo was already bitten once by a feature-ordering bug (see BDT/README.md).
Append new features at the end; never reorder.
"""
from __future__ import annotations
# Class codes as written by fake::FlavorCode. 4 (pileup) and -1 (unmatched)
# are filtered out in skim.py; -2 marks DATA (no truth flavor at all).
CLASSES = {0: "b", 1: "c", 2: "uds", 3: "g"}
N_CLASSES = len(CLASSES)

# Columns the skim must carry through but which must never be fed to the BDT.
LABEL_COLUMNS = [
    "flavor", "flavor_fine", "genflavor",   # truth labels
    "isTight", "isPrompt", "leptonType",    # FR numerator / prompt tag
    "isData", "mcweight",
    "pass_mu8", "pass_mu17", "w_mu8", "w_mu17",
    "run", "lumi", "evt",                   # event id — split by these, not by muon
    "path", "weight", "era", "sample",      # added by skim.py
]

# Muon-level inputs. All are reconstructed quantities available in data.
MUON_FEATURES = [
    "mu_pt",
    "mu_conept",          # requested explicitly: FR is binned in this
    "mu_abseta",          # signed eta deliberately dropped — FR is eta-symmetric,
                          # so signed eta only lets the trees chase fluctuations
    "mu_dxy", "mu_dxyerr", "mu_dxysig",
    "mu_dz", "mu_dzerr", "mu_dzsig",
    "mu_ip3d", "mu_sip3d",
    "mu_miniiso", "mu_miniiso_chg", "mu_miniiso_neu",
    "mu_pfiso03", "mu_pfiso03_chg", "mu_pfiso04",
    "mu_tkreliso",
    "mu_ptErr", "mu_ptErrRel", "mu_tunepRelPt",
    "mu_segmentComp", "mu_nStations", "mu_nTrkLayers",
    "mu_mvaTTH", "mu_mvaLowPt", "mu_softMva",
    "mu_tightCharge",
    "mu_isGlobal", "mu_isStandalone", "mu_highPurity", "mu_pogTight",
    "mu_jetPtRelv2", "mu_jetRelIso", "mu_jetNDauCharged",
]

# Matched source jet (Muon_jetIdx -> Jet). Sentinel -999 is converted to NaN in
# the skim so xgboost's native missing-value handling takes over.
#
# Note that after the skim these are never actually missing: dropping the
# `unmatched` class (flavor == -1) is exactly the cut that guarantees a matched
# jet, which is why `mu_hasJet` is not a feature — skim.py reports it as
# CONSTANT = 1. The NaN handling is therefore defensive only.
#
# Consequence for deployment: an unmatched muon in data has no valid jet block
# and the model has never seen one. Such muons must be handled *outside* the
# classifier as their own category (~3.3% of loose fakes), exactly as the
# per-flavor FR study already treats them.
JET_FEATURES = [
    "jet_pt", "jet_rawpt", "jet_abseta", "jet_mass", "jet_area",
    "jet_dr_mu", "jet_ptratio", "jet_ptrel",
    "jet_chHEF", "jet_neHEF", "jet_chEmEF", "jet_neEmEF",
    "jet_muEF", "jet_muonSubtr",
    "jet_nconst", "jet_nmuons", "jet_nelectrons",
    "jet_deepjet_b", "jet_deepjet_cvb", "jet_deepjet_cvl", "jet_deepjet_qg",
    "jet_qgl", "jet_puIdDisc",
    "jet_tightId", "jet_loosePuId",
]

# Event-level context.
EVENT_FEATURES = [
    "ev_met", "ev_mt",          # ev_mt requested explicitly
    "ev_dphi_mu_met", "ev_ht",
    "ev_njet25", "ev_njet30", "ev_njet40", "ev_njet60",
    "ev_awayjet_pt", "ev_awayjet_dr", "ev_awayjet_dphi",
    "ev_nPV", "ev_nPVsGood", "ev_rho",
]

FEATURES_FULL = MUON_FEATURES + JET_FEATURES + EVENT_FEATURES

# Flavor taggers evaluated on the muon's *own* jet encode almost the same
# information as Jet_hadronFlavour, i.e. very nearly the label itself, and are
# the inputs most likely to be mismodeled in data. Train a second model without
# them so the data/MC risk of the headline result can be judged.
TAGGER_FEATURES = [
    "jet_deepjet_b", "jet_deepjet_cvb", "jet_deepjet_cvl", "jet_deepjet_qg",
    "jet_qgl",
]
FEATURES_NOTAG = [f for f in FEATURES_FULL if f not in TAGGER_FEATURES]

# Variables that define the tight WP (the FR numerator) or track it closely.
# The tight WP is |SIP3D| < 3 and miniPFRelIso < 0.1 on top of POG medium ID,
# so every isolation and IP-significance variable is partly the numerator
# itself; `mu_conept` literally contains miniiso
# (pt x (1 + max(0, miniiso - 0.1))); and the lepton MVAs (softMva, mvaTTH,
# mvaLowPt) are trained on isolation + IP and so track it too.
#
# Dropping these costs real discriminating power, but it is the control the
# leakage diagnostic calls for: if FR is not flat vs predicted probability at
# fixed true flavor, the `full` model's flavor-weighted FR is biased, and the
# difference against `nowp` measures by how much. `mu_pt` replaces `mu_conept`.
WP_FEATURES = [
    "mu_conept",
    "mu_sip3d", "mu_ip3d", "mu_dxysig", "mu_dzsig",
    "mu_miniiso", "mu_miniiso_chg", "mu_miniiso_neu",
    "mu_pfiso03", "mu_pfiso03_chg", "mu_pfiso04",
    "mu_tkreliso", "mu_jetRelIso",
    "mu_mvaTTH", "mu_mvaLowPt", "mu_softMva",
]
FEATURES_NOWP = [f for f in FEATURES_FULL if f not in WP_FEATURES]

# ---------------------------------------------------------------------------
# The direct route:  P(tight | loose, x)  used *as* the fake rate.
# ---------------------------------------------------------------------------
# Here the feature set is not a diagnostic choice, it is a correctness
# requirement, and it is the opposite of what "no leakage by construction"
# suggested. `bkg.cc` evaluates the FR on the muon that **failed** the tight WP
# (categories TL and LL, Analyzers/src/bkg.cc:237-252) and applies e = f/(1-f).
# So f must be evaluated on loose-not-tight muons while being measured on all
# loose muons. Any input that is part of the tight WP is therefore fatal, not
# merely inelegant: on every muon bkg.cc asks about, miniiso > 0.1 and the model
# correctly answers f = 0, predicting zero fake background.
#
# Two failure modes have to be separated here.
#
# *Bias* is not the problem. If f(x) is the true conditional then
# sum_{loose,!tight} w f/(1-f) = sum_i w_i f(x_i) = N(tight) identically, for any
# x whatsoever -- the fail-side estimator is unbiased even when x contains the WP.
# What breaks is *efficiency*: if x determines tightness then f is 0 or 1 on
# every fail-side muon, every term is either 0 or (after bkg.cc's clamp at 0.95)
# a meaningless constant, and the estimate collapses to zero. So the requirement
# is that x must leave f genuinely intermediate on the fail side, and a high
# tightness AUC is the symptom of violating it, not a sign of a good model.
#
# The rule that enforces it, stated once: **no input may be a function of
# miniPFRelIso or of SIP3D.** Correlation with them is fine and useful (jet_muEF
# and jet_ptratio reach single-variable AUC 0.65 and stay); functional
# dependence is fatal.
#
# `mu_conept` is such a function, which is why it is excluded here despite being
# what the current map is binned in. Because conePt = pt x (1 + max(0, miniiso -
# 0.1)), conePt == pt *exactly* when miniiso <= 0.1: measured on QCD the
# indicator conePt == pt agrees with miniiso <= 0.1 for 99.9999% of rows and
# P(tight | conePt > pt) = 0.0000 exactly, so conePt/pt alone has AUC 0.933 for
# tightness. Feeding conePt with anything that recovers pt is feeding miniiso,
# and pt is recoverable many ways -- directly, or as jet_ptratio x jet_pt, or as
# mu_ptErr / mu_ptErrRel. Pruning those one at a time is whack-a-mole; dropping
# conePt closes all of them at once and costs nothing, since raw mu_pt carries
# the same kinematic scale WP-independently. (Two measured confirmations that
# the leak is real: keeping both scored AUC 0.9726, and dropping only mu_pt
# still scored 0.9580.) This is also the exact mechanism behind the flavor
# route's leakage-diagnostic failure -- every flavor feature set contains both
# columns, so those models had an exact tightness tag available throughout.
TIGHT_WP_VARS = [
    "mu_conept",          # exact WP tag in combination with any pt proxy

    # miniPFRelIso < 0.1 -- the isolation half of the WP, plus every other
    # isolation flavour and the two muon-jet relative isolations.
    "mu_miniiso", "mu_miniiso_chg", "mu_miniiso_neu",
    "mu_pfiso03", "mu_pfiso03_chg", "mu_pfiso04", "mu_tkreliso",
    "mu_jetRelIso", "mu_jetPtRelv2",
    # |SIP3D| < 3 -- the impact-parameter half. The unnormalized values and
    # their errors go too, since sip3d is recoverable from a (value, error)
    # pair; b/c lifetime information survives on the jet side via the taggers.
    "mu_dxy", "mu_dxyerr", "mu_dxysig",
    "mu_dz", "mu_dzerr", "mu_dzsig",
    "mu_ip3d", "mu_sip3d",
    # lepton MVAs are trained on exactly the variables above.
    "mu_mvaTTH", "mu_mvaLowPt", "mu_softMva",
]
FEATURES_DIRECT = [f for f in FEATURES_FULL if f not in TIGHT_WP_VARS]

# The existing flat map expressed as a feature set: a direct model on these two
# inputs must reproduce FR2D within statistics, which is what makes it the
# baseline for "does adding variables help?" rather than a separate method.
# conePt is admissible *here* precisely because mu_pt is absent and there is
# nothing else to reconstruct it from — measured AUC 0.692 with textbook
# reliability, versus 0.958-0.973 once a pt proxy is added back.
FEATURES_DIRECT_MAP = ["mu_conept", "mu_abseta"]

FEATURE_SETS = {
    "full": FEATURES_FULL,
    "notag": FEATURES_NOTAG,
    "nowp": FEATURES_NOWP,
    "direct": FEATURES_DIRECT,
    "direct_map": FEATURES_DIRECT_MAP,
}

# Branches present in the ntuple that are deliberately NOT features:
#   mu_pogMedium, mu_isTracker, mu_isPFcand  — implied by the loose WP
#                                              (POG medium ID), so constant
#   mu_eta, mu_phi                           — see mu_abseta comment; phi adds
#                                              only detector noise
#   jet_eta                                  — replaced by derived jet_abseta
#   ev_hasAway30/40                          — hasAway30 is the ntuple gate and
#                                              hasAway40 is the Central region
#                                              cut, both constant after the skim
#   ev_hasAway60                             — redundant with ev_awayjet_pt
#   ev_nTrueInt                              — MC truth, does not exist in data
#   mu_hasJet                                — constant after the skim, see above
EXCLUDED = [
    "mu_pogMedium", "mu_isTracker", "mu_isPFcand", "mu_hasJet",
    "mu_eta", "mu_phi", "jet_eta",
    "ev_hasAway30", "ev_hasAway40", "ev_hasAway60",
    "ev_nTrueInt",
]
