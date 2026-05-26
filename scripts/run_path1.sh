#!/usr/bin/env bash
# Path 1 — signal-region-restricted training (bdt_v4_sr).
#
# Trains and evaluates the BDT with m4l ∈ [105, 140] GeV pre-selection
# (controlled by config/selection.yaml::train.signal_region). Run from the
# project root.
#
#     bash scripts/run_path1.sh           # train + evaluate
#     bash scripts/run_path1.sh cv        # also run 5-fold CV
#
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=data/models/bdt_v4_sr.json
mkdir -p data/models logs plots reports

echo "[Path 1] Training bdt_v4_sr.json ..."
python -m src.train \
    --samples config/samples.yaml \
    --config  config/selection.yaml \
    --ntuples data/ntuples/ \
    --out     "$MODEL" \
    2>&1 | tee logs/train_v4_sr.log

echo "[Path 1] Evaluating bdt_v4_sr.json ..."
python -m src.evaluate \
    --model   "$MODEL" \
    --ntuples data/ntuples/ \
    --samples config/samples.yaml \
    --config  config/selection.yaml \
    2>&1 | tee logs/evaluate_v4_sr.log

if [[ "${1:-}" == "cv" ]]; then
    echo "[Path 1] 5-fold cross-validation ..."
    python -m src.cv \
        --samples config/samples.yaml \
        --config  config/selection.yaml \
        --ntuples data/ntuples/ \
        2>&1 | tee logs/cv_v4_sr.log
fi

echo "[Path 1] Done."
echo "  model:   $MODEL"
echo "  report:  reports/phase4_summary_v4_sr.md"
echo "  plots:   plots/roc_v4_sr.png, plots/feature_importance_v4_sr.png,"
echo "           plots/overtraining_v4_sr.png, plots/m4l_vs_score_v4_sr.png"
