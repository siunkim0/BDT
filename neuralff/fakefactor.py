"""Load a trained ratio model and evaluate the fake factor on rows.

    F(x) = exp(q(x))          fake factor   (paper Eq. 10-11)
    f(x) = F/(1+F) = sigma(q) fake rate     (what fr/*.json stores)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import model as M


class FakeFactor:
    def __init__(self, bundle_path: str | Path, device: str | None = None):
        # weights_only=False: the bundle carries the config dict and the
        # encoding, not just tensors. It is a file we wrote ourselves.
        b = torch.load(Path(bundle_path), map_location="cpu", weights_only=False)
        self.cfg = b["config"]
        self.feature_set = b.get("feature_set", "paper5")
        self.enc = D.Encoding.from_json(b["encoding"])
        self.splits = {k: np.asarray(v) for k, v in b["splits"].items()}
        self.weight_scale = float(b.get("weight_scale", 1.0))
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = M.build(self.cfg, n_numerical=len(self.enc.numerical),
                           cat_sizes=self.enc.cat_sizes).to(self.device)
        self.net.load_state_dict(b["state_dict"])
        self.net.eval()

    @torch.no_grad()
    def logit(self, cols: dict, batch: int = 65536) -> np.ndarray:
        """q(x) for every row of `cols` (a dict of raw columns, as from data.load)."""
        Xn, Xc = D.encode(cols, self.enc)
        t_xn = torch.from_numpy(Xn).to(self.device)
        t_xc = torch.from_numpy(Xc).to(self.device)
        out = []
        for i in range(0, len(Xn), batch):
            out.append(self.net(t_xn[i:i + batch], t_xc[i:i + batch]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def fake_rate(self, cols: dict) -> np.ndarray:
        """f(x) = sigmoid(q) — directly comparable to the binned FR map."""
        q = self.logit(cols).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-q))

    def fake_factor(self, cols: dict) -> np.ndarray:
        """F(x) = exp(q) = f/(1-f)."""
        return np.exp(self.logit(cols).astype(np.float64))


def grid_columns(base: dict, **overrides) -> dict:
    """Build a synthetic column dict for scanning F over a feature grid.

    `base` supplies the value used for every feature that is not overridden;
    each override must broadcast to a common length. Used for the paper's Fig. 7
    and 8 style 2D projections of the learned fake factor.
    """
    n = max(len(np.atleast_1d(v)) for v in overrides.values())
    out = {}
    for k, v in base.items():
        out[k] = np.repeat(np.asarray(v).reshape(1), n) if np.ndim(v) == 0 \
            else np.asarray(v)
    for k, v in overrides.items():
        v = np.atleast_1d(v)
        out[k] = np.broadcast_to(v, (n,)).copy() if v.size in (1, n) else v
    return out
