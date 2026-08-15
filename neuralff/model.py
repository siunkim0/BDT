"""The paper's classifier: feature embeddings -> pre-activation ResNet -> q.

arXiv:2511.06972 §3.2 and Fig. 4. Numerical and categorical features are each
embedded into the same width, the embeddings are stacked and mean-pooled across
the feature dimension, and the result is fed to a residual stack of 128-neuron
blocks. The single output neuron is the logit `q`, and the density ratio is
`r = exp(q)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SoftAbs(nn.Module):
    """Eq. 19: quadratic near zero, |x| - 1/2 outside. Non-negative everywhere.

    Used only by the two *subtraction* classifiers, where the data correction
    weight (1 - 1/r) must stay positive, i.e. r = exp(q) > 1, i.e. q >= 0. The
    ratio classifier that produces the fake factor uses a linear head instead,
    since F is free to be below 1 (and for muon fakes it usually is).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x.abs() < 1.0, 0.5 * x * x, x.abs() - 0.5)


class FeatureEmbedding(nn.Module):
    """Per-feature embedding, then mean pooling over the feature dimension.

    Numerical features get their own affine map from a scalar into `dim`
    (a weight and a bias per feature); categorical features get a lookup table.
    Mean pooling rather than concatenation is what the paper specifies, and it
    has the useful property that the input width does not grow with the number
    of features.
    """

    def __init__(self, n_numerical: int, cat_sizes: list[int], dim: int):
        super().__init__()
        self.n_numerical = n_numerical
        self.dim = dim
        if n_numerical:
            # x[:, i] * w[i] + b[i], broadcast over the embedding width.
            self.num_w = nn.Parameter(torch.randn(n_numerical, dim) * 0.1)
            self.num_b = nn.Parameter(torch.zeros(n_numerical, dim))
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(n, dim) for n in cat_sizes]
        )

    def forward(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.n_numerical:
            # [B, F, D]
            parts.append(xn.unsqueeze(-1) * self.num_w + self.num_b)
        for i, emb in enumerate(self.cat_emb):
            parts.append(emb(xc[:, i]).unsqueeze(1))
        stacked = torch.cat(parts, dim=1)          # [B, F_total, D]
        return stacked.mean(dim=1)                 # [B, D]


class ResBlock(nn.Module):
    """Pre-activation residual block: (norm -> act -> linear) x 2, plus identity.

    The paper omits batch normalization and dropout, having found neither
    improved performance; `norm='batch'` puts BN back if it is ever wanted.
    """

    def __init__(self, dim: int, norm: str = "none"):
        super().__init__()
        mk = (lambda: nn.BatchNorm1d(dim)) if norm == "batch" else (lambda: nn.Identity())
        self.n1, self.n2 = mk(), mk()
        self.f1 = nn.Linear(dim, dim)
        self.f2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.f1(self.act(self.n1(x)))
        h = self.f2(self.act(self.n2(h)))
        return x + h


class NFFNet(nn.Module):
    """Embeddings -> optional projection -> N residual blocks -> scalar logit."""

    def __init__(self, n_numerical: int, cat_sizes: list[int],
                 embed_dim: int = 32, hidden_dim: int = 128,
                 n_blocks: int = 4, norm: str = "none", head: str = "linear"):
        super().__init__()
        self.embed = FeatureEmbedding(n_numerical, cat_sizes, embed_dim)
        self.proj = (nn.Identity() if embed_dim == hidden_dim
                     else nn.Linear(embed_dim, hidden_dim))
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, norm) for _ in range(n_blocks)]
        )
        self.act = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)
        if head not in ("linear", "softabs"):
            raise ValueError(f"unknown head {head!r}")
        self.head = SoftAbs() if head == "softabs" else nn.Identity()

    def forward(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.embed(xn, xc))
        h = self.blocks(h)
        q = self.out(self.act(h)).squeeze(-1)
        return self.head(q)


def build(cfg: dict, n_numerical: int, cat_sizes: list[int]) -> NFFNet:
    m = cfg["model"]
    return NFFNet(n_numerical=n_numerical, cat_sizes=cat_sizes,
                  embed_dim=int(m["embed_dim"]), hidden_dim=int(m["hidden_dim"]),
                  n_blocks=int(m["n_blocks"]), norm=str(m.get("norm", "none")),
                  head=str(m.get("head", "linear")))
