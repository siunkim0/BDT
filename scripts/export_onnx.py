"""Export the trained xgboost BDT to ONNX for SKNanoAnalyzer.

Produces three files (default location: `data/models/`):
  - bdt_v1.onnx              — what MLHelper loads at run time
  - bdt_v1_features.txt      — the feature order the C++ side must reproduce
  - bdt_v1_validation.csv    — 100 events with (FEATURES, py_score, onnx_score)
                                used as the C++/Python parity baseline

Run from project root:

    python -m scripts.export_onnx
    python -m scripts.export_onnx --onnx-out /path/to/your/SKNanoAnalyzer/data/.../bdt_v1.onnx

or

    python scripts/export_onnx.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/export_onnx.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import awkward as ak
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from src.features import FEATURES
from src.utils import SEED, get_logger, load_yaml

log = get_logger("export_onnx")

JSON_PATH = PROJECT_ROOT / "data" / "models" / "bdt_v1.json"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "data" / "models" / "bdt_v1.onnx"
FEAT_TXT  = PROJECT_ROOT / "data" / "models" / "bdt_v1_features.txt"
VAL_CSV   = PROJECT_ROOT / "data" / "models" / "bdt_v1_validation.csv"
NTUPLES   = PROJECT_ROOT / "data" / "ntuples"
SAMPLES   = PROJECT_ROOT / "config" / "samples.yaml"


def strip_zipmap(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove the ZipMap node so the probability output is a plain (N,2) tensor.

    The default xgboost binary classifier ONNX graph ends in
        Cast/TreeEnsembleClassifier -> probabilities (Map<int64,float>)
    via a ZipMap node. SKNanoAnalyzer's MLHelper only reads FloatArray
    outputs, so we rewire the ZipMap's input directly to the graph output.
    """
    nodes = list(model.graph.node)
    zipmap_idxs = [i for i, n in enumerate(nodes) if n.op_type == "ZipMap"]
    if not zipmap_idxs:
        return model
    for i in reversed(zipmap_idxs):
        zip_node = nodes[i]
        # The probability tensor feeding ZipMap is its only input.
        prob_tensor_name = zip_node.input[0]
        zipped_name = zip_node.output[0]
        # Rewire any graph output that was the ZipMap's output to the raw tensor.
        for out in model.graph.output:
            if out.name == zipped_name:
                out.name = prob_tensor_name
                # Replace the type with a float tensor of shape [N, 2].
                tt = out.type.tensor_type
                tt.elem_type = onnx.TensorProto.FLOAT
                tt.shape.Clear()
                tt.shape.dim.add().dim_param = "N"
                tt.shape.dim.add().dim_value = 2
        del model.graph.node[i]
    return model


def load_features_from_ntuples() -> np.ndarray:
    """Stack the FEATURES columns from every *_features.parquet for the parity check."""
    samples_cfg = load_yaml(SAMPLES)
    frames = []
    for category in ("signal", "background"):
        for name in samples_cfg.get(category, {}):
            fp = NTUPLES / f"{name}_features.parquet"
            if not fp.exists():
                log.warning("missing %s, skipping", fp.name)
                continue
            arr = ak.from_parquet(fp)
            frames.append(
                pd.DataFrame({k: ak.to_numpy(arr[k]) for k in FEATURES if k in arr.fields})
            )
    if not frames:
        raise RuntimeError("no *_features.parquet files found in data/ntuples/")
    df = pd.concat(frames, ignore_index=True)
    return df[FEATURES].to_numpy(dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--onnx-out",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help=(
            "Path to write the ONNX model. Default: data/models/bdt_v1.onnx "
            "inside the repo. Override this to point at your SKNanoAnalyzer "
            "data directory if you want to deploy directly."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path: Path = args.onnx_out

    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"{JSON_PATH} not found — train the BDT first (python -m src.train)."
        )

    # 1. Load the trained xgboost model.
    model = xgb.XGBClassifier()
    model.load_model(JSON_PATH)
    log.info("loaded xgboost model from %s (%d features)", JSON_PATH, len(FEATURES))

    # 2. Convert to ONNX.
    #    - single named input "features" of shape [batch, 23]
    #    - this onnxmltools.convert_xgboost has no options= kwarg, so if it
    #      emits a ZipMap node we strip it post-conversion. MLHelper only
    #      handles FloatArray outputs, not Map<int64,float>.
    initial_type = [("features", FloatTensorType([None, len(FEATURES)]))]
    onnx_model = convert_xgboost(
        model, initial_types=initial_type, target_opset=15,
    )
    onnx_model = strip_zipmap(onnx_model)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())
    FEAT_TXT.write_text("\n".join(FEATURES))
    log.info("wrote %s (%d bytes)", onnx_path, onnx_path.stat().st_size)
    log.info("wrote %s", FEAT_TXT)

    # 3. Parity check: run both models on every available event.
    X = load_features_from_ntuples()
    log.info("parity check on %d events", len(X))

    p_xgb = model.predict_proba(X)[:, 1]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    log.info("ONNX input='%s', outputs=%s", in_name, out_names)
    outs = sess.run(None, {in_name: X})
    # With zipmap=False, the (N,2) probability tensor is one of the outputs;
    # pick it by shape rather than position to be defensive.
    p_onnx = None
    for arr in outs:
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape == (len(X), 2):
            p_onnx = arr[:, 1]
            break
    if p_onnx is None:
        raise RuntimeError(f"could not find (N,2) probability output among {[type(o) for o in outs]}")

    max_abs = float(np.max(np.abs(p_xgb - p_onnx)))
    log.info("max |Δ probability| = %.2e", max_abs)
    assert max_abs < 1e-5, f"ONNX disagrees with xgboost (max Δ = {max_abs:.2e}) — DO NOT ship"

    # 4. Validation CSV: 100 random events, used as the C++ parity baseline.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(100, len(X)), replace=False)
    val = pd.DataFrame(X[idx], columns=FEATURES)
    val["py_score"] = p_xgb[idx]
    val["onnx_score"] = p_onnx[idx]
    val.to_csv(VAL_CSV, index=False)
    log.info("wrote %s (%d events)", VAL_CSV, len(val))

    log.info("done — copy %s to $SKNANO_DATA/<DataEra>/HZZ4mu/", onnx_path)


if __name__ == "__main__":
    main()
