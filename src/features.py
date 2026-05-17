"""Phase 2: feature engineering on skimmed ntuples.

Reads the parquet files produced by skim.py, appends engineered columns
(helicity angles, kinematic ratios, etc.), writes to `<name>_features.parquet`.

Run from project root:

    python -m src.features --in data/ntuples/ --out data/ntuples/

Helicity angle definitions follow arXiv:1208.4018 (Gao et al., section 2):
  θ*  : Z1 polar angle in H rest frame, w.r.t. beam axis
  θ1  : ℓ⁻ from Z1 polar angle in Z1 rest frame, w.r.t. -Z2 direction
  θ2  : ℓ⁻ from Z2 polar angle in Z2 rest frame, w.r.t. -Z1 direction
  Φ   : signed angle between Z1 and Z2 decay planes, in H rest frame
  Φ1  : signed angle between Z1 decay plane and (Z1, beam) plane, in H frame
"""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import vector

from .utils import get_logger

vector.register_awkward()

log = get_logger("features")

# The canonical training feature list. Order matters for reproducibility.
# Mass-related variables (m4l, mZ1, mZ2, pt_Z*/m4l ratios) are restored for
# Phase 5: with planing-based reweighting (see src/train.py::apply_planing
# and config selection.yaml::train.planing) the m4l distribution is flattened
# at training time, so the BDT cannot use mass to discriminate. This lets us
# keep the full kinematic information while still producing a score that is
# decorrelated from m4l. The diagnostic remains plots/m4l_vs_score_v3.png.
FEATURES: list[str] = [
    # Mass
    "m4l", "mZ1", "mZ2",
    # Kinematics
    "pt4l", "eta4l",
    "pt_Z1", "pt_Z2",
    "dR_Z1Z2",
    # Per-lepton (sorted by pT)
    "pt_mu1", "pt_mu2", "pt_mu3", "pt_mu4",
    "eta_mu1", "eta_mu2", "eta_mu3", "eta_mu4",
    # Ratios
    "pt_Z1_over_m4l", "pt_Z2_over_m4l",
    # Helicity angles (HZZ 5-angle set)
    "cos_theta_star",
    "cos_theta1", "cos_theta2",
    "Phi", "Phi1",
]


# --- 3-vector helpers (numpy, shape (N, 3)) ---------------------------------

def _xyz(p4: vector.MomentumNumpy4D) -> np.ndarray:
    return np.stack([p4.px, p4.py, p4.pz], axis=-1)


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n > 0, n, 1.0)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def _signed_angle(n1: np.ndarray, n2: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Signed angle from n1 to n2 around `axis`. All inputs are unit (N,3)."""
    cos_a = np.clip(_dot(n1, n2), -1.0, 1.0)
    sign = np.sign(_dot(axis, np.cross(n1, n2)))
    # When sign is exactly 0, np.sign gives 0 — fall back to +1 to avoid NaN.
    sign = np.where(sign == 0, 1.0, sign)
    return sign * np.arccos(cos_a)


# --- Per-muon assembly -------------------------------------------------------

def _muon_p4s(events: ak.Array) -> list[vector.MomentumNumpy4D]:
    """Return list of 4 vector arrays, one per pT-sorted muon slot."""
    return [
        vector.array({
            "pt":  ak.to_numpy(events[f"pt_mu{i+1}"]).astype(np.float64),
            "eta": ak.to_numpy(events[f"eta_mu{i+1}"]).astype(np.float64),
            "phi": ak.to_numpy(events[f"phi_mu{i+1}"]).astype(np.float64),
            "M":   ak.to_numpy(events[f"mass_mu{i+1}"]).astype(np.float64),
        })
        for i in range(4)
    ]


def _gather_p4(p4s: list[vector.MomentumNumpy4D], idx: np.ndarray) -> vector.MomentumNumpy4D:
    """Pick one p4 per event from `p4s` based on integer index in 0..3."""
    N = len(idx)
    pt = np.zeros(N); eta = np.zeros(N); phi = np.zeros(N); m = np.zeros(N)
    for k in range(4):
        sel = idx == k
        if not np.any(sel):
            continue
        pt[sel] = p4s[k].pt[sel]
        eta[sel] = p4s[k].eta[sel]
        phi[sel] = p4s[k].phi[sel]
        m[sel] = p4s[k].M[sel]
    return vector.array({"pt": pt, "eta": eta, "phi": phi, "M": m})


def _gather_int(arrs: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    N = len(idx)
    out = np.zeros(N, dtype=arrs[0].dtype)
    for k in range(4):
        sel = idx == k
        if not np.any(sel):
            continue
        out[sel] = arrs[k][sel]
    return out


def _select_p4(mask: np.ndarray, a: vector.MomentumNumpy4D,
               b: vector.MomentumNumpy4D) -> vector.MomentumNumpy4D:
    """Pick a where mask is True else b, componentwise."""
    return vector.array({
        "pt":  np.where(mask, a.pt,  b.pt),
        "eta": np.where(mask, a.eta, b.eta),
        "phi": np.where(mask, a.phi, b.phi),
        "M":   np.where(mask, a.M,   b.M),
    })


# --- Feature blocks ----------------------------------------------------------

def add_basic_kinematics(events: ak.Array) -> ak.Array:
    """pt4l, eta4l, pt_Z1, pt_Z2, dR_Z1Z2, ratios."""
    mu = _muon_p4s(events)
    z1a_idx = ak.to_numpy(events["Z1_idx1"]).astype(np.int64)
    z1b_idx = ak.to_numpy(events["Z1_idx2"]).astype(np.int64)
    z2a_idx = ak.to_numpy(events["Z2_idx1"]).astype(np.int64)
    z2b_idx = ak.to_numpy(events["Z2_idx2"]).astype(np.int64)

    mu_z1a = _gather_p4(mu, z1a_idx)
    mu_z1b = _gather_p4(mu, z1b_idx)
    mu_z2a = _gather_p4(mu, z2a_idx)
    mu_z2b = _gather_p4(mu, z2b_idx)

    Z1 = mu_z1a + mu_z1b
    Z2 = mu_z2a + mu_z2b
    H4 = mu[0] + mu[1] + mu[2] + mu[3]

    m4l = ak.to_numpy(events["m4l"]).astype(np.float64)

    new = {
        "pt4l":  H4.pt.astype(np.float32),
        "eta4l": H4.eta.astype(np.float32),
        "pt_Z1": Z1.pt.astype(np.float32),
        "pt_Z2": Z2.pt.astype(np.float32),
        "dR_Z1Z2": Z1.deltaR(Z2).astype(np.float32),
        "pt_Z1_over_m4l": (Z1.pt / m4l).astype(np.float32),
        "pt_Z2_over_m4l": (Z2.pt / m4l).astype(np.float32),
    }
    for k, v in new.items():
        events = ak.with_field(events, v, k)
    return events


def add_helicity_angles(events: ak.Array) -> ak.Array:
    """The 5 HZZ helicity angles, computed via explicit boosts."""
    mu = _muon_p4s(events)
    charges = [ak.to_numpy(events[f"charge_mu{i+1}"]).astype(np.int8)
               for i in range(4)]

    z1a_idx = ak.to_numpy(events["Z1_idx1"]).astype(np.int64)
    z1b_idx = ak.to_numpy(events["Z1_idx2"]).astype(np.int64)
    z2a_idx = ak.to_numpy(events["Z2_idx1"]).astype(np.int64)
    z2b_idx = ak.to_numpy(events["Z2_idx2"]).astype(np.int64)

    mu_z1a = _gather_p4(mu, z1a_idx)
    mu_z1b = _gather_p4(mu, z1b_idx)
    mu_z2a = _gather_p4(mu, z2a_idx)
    mu_z2b = _gather_p4(mu, z2b_idx)
    q_z1a = _gather_int(charges, z1a_idx)
    q_z2a = _gather_int(charges, z2a_idx)

    # Pick the negative lepton in each Z (the "a" slot if its charge is -1).
    z1m = _select_p4(q_z1a == -1, mu_z1a, mu_z1b)
    z2m = _select_p4(q_z2a == -1, mu_z2a, mu_z2b)

    Z1 = mu_z1a + mu_z1b
    Z2 = mu_z2a + mu_z2b
    H4 = Z1 + Z2

    N = len(H4)
    # Lab-frame +z beam reference (massless 4-vector along +z). Magnitude is
    # arbitrary — only the spatial direction matters after boost.
    beam_lab = vector.array({
        "px": np.zeros(N), "py": np.zeros(N),
        "pz": np.ones(N),  "E":  np.ones(N),
    })

    # Frame transforms.
    Z1_H = Z1.boostCM_of_p4(H4)
    Z2_H = Z2.boostCM_of_p4(H4)
    z1m_H = z1m.boostCM_of_p4(H4)
    z2m_H = z2m.boostCM_of_p4(H4)
    beam_H = beam_lab.boostCM_of_p4(H4)

    z1m_inZ1 = z1m.boostCM_of_p4(Z1)
    Z2_inZ1 = Z2.boostCM_of_p4(Z1)
    z2m_inZ2 = z2m.boostCM_of_p4(Z2)
    Z1_inZ2 = Z1.boostCM_of_p4(Z2)

    # Unit 3-vectors in the relevant frames.
    n_Z1_H = _unit(_xyz(Z1_H))
    n_beam_H = _unit(_xyz(beam_H))
    n_z1m_H = _unit(_xyz(z1m_H))
    n_z2m_H = _unit(_xyz(z2m_H))
    n_Z2_H = _unit(_xyz(Z2_H))

    # cos(θ*): Z1 vs beam, in H rest frame.
    cos_theta_star = _dot(n_Z1_H, n_beam_H)

    # cos(θ1): ℓ1⁻ vs -Z2, both in Z1 rest frame.
    n_l1m_Z1 = _unit(_xyz(z1m_inZ1))
    n_negZ2_Z1 = -_unit(_xyz(Z2_inZ1))
    cos_theta1 = _dot(n_l1m_Z1, n_negZ2_Z1)

    # cos(θ2): ℓ2⁻ vs -Z1, both in Z2 rest frame.
    n_l2m_Z2 = _unit(_xyz(z2m_inZ2))
    n_negZ1_Z2 = -_unit(_xyz(Z1_inZ2))
    cos_theta2 = _dot(n_l2m_Z2, n_negZ1_Z2)

    # Plane normals in H rest frame.
    n_plane_Z1 = _unit(np.cross(n_Z1_H, n_z1m_H))
    n_plane_Z2 = _unit(np.cross(n_Z2_H, n_z2m_H))
    n_plane_sc = _unit(np.cross(n_Z1_H, n_beam_H))  # "scattering" plane

    # Φ: signed angle between Z1 and Z2 decay planes, around Z1 axis in H frame.
    Phi = _signed_angle(n_plane_Z1, n_plane_Z2, n_Z1_H)
    # Φ1: signed angle between Z1 decay plane and (Z1, beam) plane.
    Phi1 = _signed_angle(n_plane_Z1, n_plane_sc, n_Z1_H)

    new = {
        "cos_theta_star": cos_theta_star.astype(np.float32),
        "cos_theta1":     cos_theta1.astype(np.float32),
        "cos_theta2":     cos_theta2.astype(np.float32),
        "Phi":            Phi.astype(np.float32),
        "Phi1":           Phi1.astype(np.float32),
    }
    for k, v in new.items():
        events = ak.with_field(events, v, k)
    return events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default="data/ntuples/")
    p.add_argument("--out", dest="out_dir", default="data/ntuples/")
    p.add_argument("--only", nargs="*", default=None,
                   help="Only process these sample stems (no extension)")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(in_dir.glob("*.parquet")):
        if fp.stem.endswith("_features"):
            continue
        if args.only and fp.stem not in args.only:
            continue
        log.info("Adding features to %s", fp.name)
        events = ak.from_parquet(fp)
        n_in = len(events)
        events = add_basic_kinematics(events)
        events = add_helicity_angles(events)
        out_fp = out_dir / f"{fp.stem}_features.parquet"
        ak.to_parquet(events, out_fp)
        log.info("Wrote %s — %d events, %d fields", out_fp, n_in,
                 len(events.fields))


if __name__ == "__main__":
    main()
