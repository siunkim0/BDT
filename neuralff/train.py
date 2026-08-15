"""Train the ratio classifier — the fake factor itself.

    python -m neuralff.train --sample QCD_MuEnriched \
        --ntuples fakeflavor/data/ntuples_nff/QCD_MuEnriched.parquet \
        --out neuralff/data/models/nff_qcd.pt

The classifier separates tight (y=1) from loose-not-tight (y=0) muons. By the
likelihood-ratio trick the logit is

    q(x) = log [ p(x|T) N_T / p(x|L) N_L ] = log F(x),

so the fake factor is F = exp(q) and the fake rate is f = F/(1+F) = sigmoid(q).
That identity is why nothing here rebalances the classes: the class proportions
*are* the answer. It also means this is the same fit as `P(tight | loose, x)`,
read in the other variable — see fakeflavor/config/direct.yaml.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import get_logger, load_yaml
from . import data as D
from . import model as M

log = get_logger("neuralff.train")


def weighted_auc(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    """Weighted ROC AUC = P(score(tight) > score(loose)), ties counted as 1/2.

    Uses |w|: TTLJ_powheg is NLO and 0.30% of its rows carry a negative weight,
    which makes the ROC's false-positive rate non-monotonic and sklearn raises
    "x is neither increasing nor decreasing". AUC is a diagnostic, not part of
    the estimate, so the sign is dropped here and only here — the loss and every
    fake-rate number keep the signed weights, as the paper requires.
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score, sample_weight=np.abs(w)))


def loss_fn(q: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
            reg_lambda: float) -> torch.Tensor:
    """Eq. 12: weighted binary cross-entropy plus a squared-logit penalty.

    The paper writes the regularization term with a minus sign, which under
    minimization would reward exploding logits; it is applied as a penalty here.
    """
    bce = F.binary_cross_entropy_with_logits(q, y, weight=w, reduction="sum")
    denom = w.sum()
    out = bce / denom
    if reg_lambda:
        out = out + reg_lambda * (w * q.pow(2)).sum() / denom
    return out


@torch.no_grad()
def evaluate(net, xn, xc, y, w, reg_lambda, batch=65536):
    net.eval()
    qs = []
    for i in range(0, len(y), batch):
        qs.append(net(xn[i:i + batch], xc[i:i + batch]))
    q = torch.cat(qs)
    return loss_fn(q, y, w, reg_lambda).item(), q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="neuralff/config/nff.yaml")
    p.add_argument("--ntuples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample", default=None,
                   help="keep only this `sample` value (the parquets are already "
                        "per-sample, so this is a guard, not a selection)")
    p.add_argument("--feature-set", default=None,
                   help="named set from config.feature_sets (default: "
                        "config.default_set). 'paper5' uses conePt, the axis the "
                        "reference map is binned in; 'paper5_rawpt' uses raw "
                        "mu_pt and is the WP-free control.")
    p.add_argument("--epochs", type=int, default=None, help="override config")
    p.add_argument("--reg-lambda", type=float, default=None,
                   help="override train.reg_lambda. The paper's 0.01 shrinks |q| "
                        "toward 0, i.e. f toward 0.5, which biases a fake rate "
                        "that sits at ~0.26 upward; 0 is the unbiased setting.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    seed = int(tcfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    fset, _ = D.feature_spec(cfg, args.feature_set)
    log.info("Feature set: %s", fset)
    cols = D.load(Path(args.ntuples), cfg, fset)
    if args.sample:
        keep = cols["sample"] == args.sample
        if not keep.all():
            log.info("Keeping %d/%d rows with sample == %s",
                     keep.sum(), len(keep), args.sample)
            cols = {k: v[keep] for k, v in cols.items()}

    splits = D.make_splits(cols, cfg)
    train_mask = np.zeros(len(cols["isTight"]), dtype=bool)
    train_mask[splits["train"]] = True

    enc = D.fit_encoding(cols, cfg, train_mask, fset)
    Xn, Xc = D.encode(cols, enc)
    y = np.asarray(cols["isTight"]).astype(np.float32)
    w = np.asarray(cols["weight"], dtype=np.float64)

    # Global rescale only. It cancels in the ratio (and in the BCE optimum) but
    # keeps the loss and its gradients at O(1) so the paper's learning rate means
    # the same thing here as there. QCD pT-bin weights span many decades.
    w_scale = w[splits["train"]].mean()
    w = (w / w_scale).astype(np.float32)
    log.info("Weights rescaled by 1/%.6g (train mean -> 1)", w_scale)

    for name, idx in splits.items():
        ww, yy = w[idx], y[idx]
        log.info("  %-5s: sum(w)=%.4g  tight frac (weighted)=%.4f",
                 name, ww.sum(), (ww * yy).sum() / ww.sum())

    device = torch.device(tcfg.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    t_xn = torch.from_numpy(Xn).to(device)
    t_xc = torch.from_numpy(Xc).to(device)
    t_y = torch.from_numpy(y).to(device)
    t_w = torch.from_numpy(w).to(device)

    i_tr = torch.from_numpy(splits["train"]).to(device)
    i_va = torch.from_numpy(splits["val"]).to(device)

    net = M.build(cfg, n_numerical=len(enc.numerical),
                  cat_sizes=enc.cat_sizes).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    log.info("Model: %d parameters, cat sizes %s", n_par, enc.cat_sizes)

    opt = torch.optim.AdamW(net.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=int(tcfg["lr_patience"]))
    reg = float(args.reg_lambda if args.reg_lambda is not None
                else tcfg["reg_lambda"])
    cfg["train"] = dict(tcfg, reg_lambda=reg)   # what is saved must be what ran
    batch = int(tcfg["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else tcfg["epochs"])
    patience = int(tcfg["early_stopping_patience"])

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    history = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        perm = i_tr[torch.randperm(len(i_tr), generator=g).to(device)]
        tot, denom = 0.0, 0.0
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            q = net(t_xn[j], t_xc[j])
            l = loss_fn(q, t_y[j], t_w[j], reg)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sw = t_w[j].sum().item()
            tot += l.item() * sw
            denom += sw
        tr_loss = tot / denom
        va_loss, _ = evaluate(net, t_xn[i_va], t_xc[i_va], t_y[i_va],
                              t_w[i_va], reg)
        sched.step(va_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": va_loss, "lr": opt.param_groups[0]["lr"]})
        flag = ""
        if va_loss < best["loss"] - 1e-6:
            best = {"loss": va_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}}
            flag = " *"
        log.info("epoch %3d  train %.6f  val %.6f  lr %.2e%s",
                 epoch, tr_loss, va_loss, opt.param_groups[0]["lr"], flag)
        if epoch - best["epoch"] >= patience:
            log.info("Early stopping: no val improvement for %d epochs", patience)
            break

    net.load_state_dict(best["state"])
    log.info("Best epoch %d, val loss %.6f (%.1f s)",
             best["epoch"], best["loss"], time.time() - t0)

    # Test metrics. AUC is a sanity check, not the goal: a value near 1 means a
    # tight-WP variable leaked into the inputs (see nff.yaml).
    i_te = torch.from_numpy(splits["test"]).to(device)
    te_loss, q_te = evaluate(net, t_xn[i_te], t_xc[i_te], t_y[i_te], t_w[i_te], reg)
    q_te = q_te.cpu().numpy()
    idx = splits["test"]
    auc = weighted_auc(y[idx], q_te, w[idx])
    f_te = 1.0 / (1.0 + np.exp(-q_te))
    closure = float((w[idx] * f_te).sum() / (w[idx] * y[idx]).sum())
    log.info("Test: loss %.6f  AUC %.4f  closure sum(w f)/sum(w tight) = %.4f",
             te_loss, auc, closure)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "encoding": enc.to_json(),
                "feature_set": fset,
                "config": cfg,
                "splits": {k: v for k, v in splits.items()},
                "weight_scale": float(w_scale)}, out)
    meta = {"ntuples": str(args.ntuples), "sample": args.sample,
            "n_rows": int(len(y)), "n_parameters": int(n_par),
            "best_epoch": best["epoch"], "val_loss": best["loss"],
            "test_loss": te_loss, "test_auc": auc, "test_closure": closure,
            "weight_scale": float(w_scale),
            "feature_set": fset,
            "features": {"numerical": enc.numerical,
                         "categorical": enc.categorical},
            "train_config": cfg["train"], "model_config": cfg["model"],
            "history": history}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s and %s", out, out.with_suffix(".meta.json"))


if __name__ == "__main__":
    main()
