#!/usr/bin/env bash
# Train + evaluate one version of the BDT.
#
# The signal-region cut (m4l ∈ [105, 140] GeV) is controlled by
# config/selection.yaml::train.signal_region, not by this script.
#
# To train a new version, override TAG — everything downstream (model file,
# logs, plots, reports) is derived from it:
#
#     bash scripts/run_path1.sh                  # default TAG=v4_sr
#     bash scripts/run_path1.sh cv               # also run 5-fold CV
#     TAG=v5 bash scripts/run_path1.sh           # train bdt_v5.json
#     TAG=v5 bash scripts/run_path1.sh cv        # train + evaluate + CV
#
# Or just edit the TAG=... line below.
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-v4_sr}"
MODEL="data/models/bdt_${TAG}.json"

mkdir -p data/models logs plots reports

echo "[run] TAG=${TAG}  MODEL=${MODEL}"

echo "[run] Training ${MODEL} ..."
python -m src.train \
    --samples config/samples.yaml \
    --config  config/selection.yaml \
    --ntuples data/ntuples/ \
    --out     "$MODEL" \
    2>&1 | tee "logs/train_${TAG}.log"

echo "[run] Evaluating ${MODEL} ..."
python -m src.evaluate \
    --model   "$MODEL" \
    --ntuples data/ntuples/ \
    --samples config/samples.yaml \
    --config  config/selection.yaml \
    2>&1 | tee "logs/evaluate_${TAG}.log"

if [[ "${1:-}" == "cv" ]]; then
    echo "[run] 5-fold cross-validation ..."
    python -m src.cv \
        --samples config/samples.yaml \
        --config  config/selection.yaml \
        --ntuples data/ntuples/ \
        2>&1 | tee "logs/cv_${TAG}.log"
fi

echo "[run] Done."
echo "  model:   ${MODEL}"
echo "  logs:    logs/train_${TAG}.log, logs/evaluate_${TAG}.log"
echo "  report:  reports/phase4_summary_${TAG}.md  (suffix may be stripped by the v-regex in src/evaluate.py)"
echo "  plots:   plots/{roc,feature_importance,overtraining,m4l_vs_score}_${TAG}.png"
