"""Phase 1: NanoAOD → flat parquet ntuple.

For each sample listed in config/samples.yaml:
  - read the relevant Muon_* branches with uproot
  - apply trigger + muon ID/iso + 4-muon selection
  - build Z1/Z2 pairing (Z1 closest to mZ)
  - write per-event flat record to parquet

Run from the project root:

    python -m src.skim --config config/samples.yaml --selection config/selection.yaml \
                      --out data/ntuples/ --only ggH_ZZ_4l
"""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import vector

from .utils import get_logger, load_yaml

vector.register_awkward()

log = get_logger("skim")

Z_MASS = 91.1876  # PDG, GeV

# Minimal NanoAOD branch list. Triggers are appended at runtime.
BRANCHES = [
    "run", "luminosityBlock", "event",
    "genWeight",
    "nMuon",
    "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
    "Muon_pfRelIso04_all",
    "Muon_dxy", "Muon_dz", "Muon_sip3d",
    "Muon_looseId",
]


def apply_muon_id(events: ak.Array, sel: dict) -> ak.Array:
    """Per-muon boolean mask (loose HZZ muon ID)."""
    m = sel["muon"]
    return (
        (events.Muon_pt > m["pt_other"])
        & (np.abs(events.Muon_eta) < m["eta_max"])
        & (events.Muon_pfRelIso04_all < m["iso_max"])
        & (np.abs(events.Muon_dxy) < m["dxy_max"])
        & (np.abs(events.Muon_dz) < m["dz_max"])
        & (events.Muon_sip3d < m["sip3d_max"])
        & events.Muon_looseId
    )


def select_4mu_events(events: ak.Array, sel: dict):
    """Keep events with ≥4 good muons, leading/sublead pT cuts, charge sum 0.

    NanoAOD muons are pT-sorted, so taking [:, :4] of the masked collection
    yields the top-4 selected muons.

    Returns the selected (rectangular) per-muon arrays plus an event-level
    pass mask. All output arrays are aligned: row i = event i that passed
    the ≥4-muon precut.
    """
    m = sel["muon"]
    mu_mask = apply_muon_id(events, sel)
    n_sel = ak.sum(mu_mask, axis=1)
    ev_pre = n_sel >= 4

    pt = events.Muon_pt[mu_mask][ev_pre][:, :4]
    eta = events.Muon_eta[mu_mask][ev_pre][:, :4]
    phi = events.Muon_phi[mu_mask][ev_pre][:, :4]
    mass = events.Muon_mass[mu_mask][ev_pre][:, :4]
    charge = events.Muon_charge[mu_mask][ev_pre][:, :4]

    lead_ok = pt[:, 0] > m["pt_lead"]
    sub_ok = pt[:, 1] > m["pt_sublead"]
    qsum_ok = ak.sum(charge, axis=1) == sel["four_lepton"]["charge_sum"]
    good = ak.to_numpy(lead_ok & sub_ok & qsum_ok)

    return ev_pre, good, pt, eta, phi, mass, charge


def build_z_pairs(pt, eta, phi, mass, charge, sel):
    """Pair 4 muons into Z1, Z2.

    Strategy: 4 muons → 3 pairings ((01)(23), (02)(13), (03)(12)). For each
    pairing, both pairs must be OS. Z1 = OS pair with mass closest to mZ;
    Z2 = the other. Among multiple OS pairings, pick the one minimizing
    |mZ1 − mZ|. Apply mass-window cuts on Z1 and Z2.

    Inputs are (N, 4) awkward/numpy arrays of pT-sorted top-4 muons.
    Returns a per-event pass mask plus (mZ1, mZ2, m4l) and the 4-muon
    pair indices (Z1_idx1, Z1_idx2, Z2_idx1, Z2_idx2) into 0..3.
    """
    pt_n = ak.to_numpy(pt)
    eta_n = ak.to_numpy(eta)
    phi_n = ak.to_numpy(phi)
    mass_n = ak.to_numpy(mass)
    q = ak.to_numpy(charge)

    # Build per-slot 4-vectors.
    p4 = [
        vector.array({"pt": pt_n[:, i], "eta": eta_n[:, i],
                      "phi": phi_n[:, i], "M": mass_n[:, i]})
        for i in range(4)
    ]

    N = pt_n.shape[0]
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]

    best_d = np.full(N, np.inf)
    out_mZ1 = np.zeros(N)
    out_mZ2 = np.zeros(N)
    out_idx = np.full((N, 4), -1, dtype=np.int8)  # Z1a, Z1b, Z2a, Z2b

    for (a, b), (c, d) in pairings:
        os_ab = (q[:, a] + q[:, b]) == 0
        os_cd = (q[:, c] + q[:, d]) == 0
        os_both = os_ab & os_cd

        m_ab = (p4[a] + p4[b]).M
        m_cd = (p4[c] + p4[d]).M

        ab_is_z1 = np.abs(m_ab - Z_MASS) < np.abs(m_cd - Z_MASS)
        mZ1 = np.where(ab_is_z1, m_ab, m_cd)
        mZ2 = np.where(ab_is_z1, m_cd, m_ab)

        d_ = np.abs(mZ1 - Z_MASS)
        better = os_both & (d_ < best_d)

        out_mZ1 = np.where(better, mZ1, out_mZ1)
        out_mZ2 = np.where(better, mZ2, out_mZ2)
        best_d = np.where(better, d_, best_d)

        # When this pairing is the new best, store indices.
        # idx layout: Z1 is (a,b) if ab_is_z1 else (c,d)
        z1a = np.where(ab_is_z1, a, c)
        z1b = np.where(ab_is_z1, b, d)
        z2a = np.where(ab_is_z1, c, a)
        z2b = np.where(ab_is_z1, d, b)
        for col, vals in enumerate((z1a, z1b, z2a, z2b)):
            out_idx[:, col] = np.where(better, vals, out_idx[:, col])

    p4_sum = p4[0] + p4[1] + p4[2] + p4[3]
    m4l = p4_sum.M

    z = sel["z_pairing"]
    fl = sel["four_lepton"]
    pass_mask = (
        np.isfinite(best_d)
        & (best_d != np.inf)
        & (out_mZ1 > z["mZ1_min"]) & (out_mZ1 < z["mZ1_max"])
        & (out_mZ2 > z["mZ2_min"]) & (out_mZ2 < z["mZ2_max"])
        & (m4l > fl["m4l_min"]) & (m4l < fl["m4l_max"])
    )
    return pass_mask, out_mZ1, out_mZ2, m4l, out_idx


def _trigger_or(events: ak.Array, trig_names: list[str]) -> ak.Array:
    """Logical OR over the configured HLT paths. Missing paths are False."""
    fired = None
    for name in trig_names:
        if name not in events.fields:
            log.warning("Trigger %s not found in tree, treating as False.", name)
            continue
        f = events[name]
        fired = f if fired is None else (fired | f)
    if fired is None:
        return ak.Array(np.zeros(len(events), dtype=bool))
    return fired


def process_sample(sample_name: str, sample_cfg: dict, sel: dict,
                   is_data: bool, out_dir: Path, lumi_pb: float) -> None:
    """Run the full skim on one sample, write parquet."""
    paths = sample_cfg["path"]
    if paths == "FILL_ME" or paths.startswith("FILL_ME"):
        log.warning("Sample %s has placeholder path, skipping.", sample_name)
        return

    log.info("Processing %s from %s", sample_name, paths)

    branches = list(BRANCHES) + sel["triggers"]
    if is_data:
        branches = [b for b in branches if b != "genWeight"]

    out_chunks: list[ak.Array] = []
    n_in_total = 0
    sum_sgn_gw = 0.0  # for MC normalization

    for chunk in uproot.iterate(
        f"{paths}:Events",
        expressions=branches,
        step_size="200 MB",
    ):
        n_in = len(chunk)
        n_in_total += n_in

        if not is_data:
            sum_sgn_gw += float(ak.sum(np.sign(chunk.genWeight)))

        # Trigger
        trig = _trigger_or(chunk, sel["triggers"])
        chunk = chunk[trig]
        if len(chunk) == 0:
            continue

        ev_pre, good, pt, eta, phi, mass, charge = select_4mu_events(chunk, sel)

        # Restrict to events that passed the precut, then apply event-level cuts.
        chunk_pre = chunk[ev_pre]
        chunk_pre = chunk_pre[good]
        pt = pt[good]
        eta = eta[good]
        phi = phi[good]
        mass = mass[good]
        charge = charge[good]
        if len(chunk_pre) == 0:
            continue

        zmask, mZ1, mZ2, m4l, pair_idx = build_z_pairs(pt, eta, phi, mass, charge, sel)

        chunk_pre = chunk_pre[zmask]
        if len(chunk_pre) == 0:
            continue
        pt = ak.to_numpy(pt)[zmask]
        eta = ak.to_numpy(eta)[zmask]
        phi = ak.to_numpy(phi)[zmask]
        mass = ak.to_numpy(mass)[zmask]
        charge = ak.to_numpy(charge)[zmask]
        mZ1 = mZ1[zmask]
        mZ2 = mZ2[zmask]
        m4l = m4l[zmask]
        pair_idx = pair_idx[zmask]

        rec = {
            "run": ak.to_numpy(chunk_pre.run),
            "luminosityBlock": ak.to_numpy(chunk_pre.luminosityBlock),
            "event": ak.to_numpy(chunk_pre.event),
            "mZ1": mZ1,
            "mZ2": mZ2,
            "m4l": m4l,
            "Z1_idx1": np.ascontiguousarray(pair_idx[:, 0]),
            "Z1_idx2": np.ascontiguousarray(pair_idx[:, 1]),
            "Z2_idx1": np.ascontiguousarray(pair_idx[:, 2]),
            "Z2_idx2": np.ascontiguousarray(pair_idx[:, 3]),
        }
        for i in range(4):
            rec[f"pt_mu{i+1}"] = np.ascontiguousarray(pt[:, i])
            rec[f"eta_mu{i+1}"] = np.ascontiguousarray(eta[:, i])
            rec[f"phi_mu{i+1}"] = np.ascontiguousarray(phi[:, i])
            rec[f"mass_mu{i+1}"] = np.ascontiguousarray(mass[:, i])
            rec[f"charge_mu{i+1}"] = np.ascontiguousarray(charge[:, i]).astype(np.int8)

        if is_data:
            rec["genWeight"] = np.ones(len(chunk_pre), dtype=np.float32)
        else:
            rec["genWeight"] = ak.to_numpy(chunk_pre.genWeight).astype(np.float32)

        out_chunks.append(ak.Array(rec))

    if not out_chunks:
        log.warning("No events passed for %s.", sample_name)
        return

    final = ak.concatenate(out_chunks)

    # Normalization. xsec_weight = (xsec * L * 1000 / sum_sgn_gw) * sign(genWeight)
    if is_data:
        xw = np.ones(len(final), dtype=np.float32)
    else:
        xsec = float(sample_cfg["xsec"])
        if sum_sgn_gw == 0:
            log.warning("sum(sign(genWeight))=0 for %s, setting xsec_weight=0.",
                        sample_name)
            xw = np.zeros(len(final), dtype=np.float32)
        else:
            norm = xsec * lumi_pb / sum_sgn_gw
            xw = (norm * np.sign(ak.to_numpy(final.genWeight))).astype(np.float32)

    final = ak.with_field(final, xw, "xsec_weight")
    if "label" in sample_cfg:
        labels = np.full(len(final), int(sample_cfg["label"]), dtype=np.int8)
        final = ak.with_field(final, labels, "label")

    out_path = out_dir / f"{sample_name}.parquet"
    ak.to_parquet(final, out_path)
    log.info(
        "Wrote %s — input=%d  passed=%d  acc=%.4f  sum_sgn_gw=%.0f",
        out_path, n_in_total, len(final),
        len(final) / max(n_in_total, 1), sum_sgn_gw,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/samples.yaml")
    p.add_argument("--selection", default="config/selection.yaml")
    p.add_argument("--out", default="data/ntuples/")
    p.add_argument("--only", nargs="*", default=None,
                   help="Only process these sample names")
    args = p.parse_args()

    samples = load_yaml(args.config)
    sel = load_yaml(args.selection)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lumi_pb = float(samples.get("lumi_fb", 0.0)) * 1000.0

    for category in ("signal", "background", "data"):
        for name, cfg in samples.get(category, {}).items():
            if args.only and name not in args.only:
                continue
            process_sample(name, cfg, sel,
                           is_data=(category == "data"),
                           out_dir=out_dir,
                           lumi_pb=lumi_pb)


if __name__ == "__main__":
    main()
