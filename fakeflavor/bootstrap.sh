#!/usr/bin/env bash
# Create the fakeflavor venv. Run from the BDT project root:
#
#     bash fakeflavor/bootstrap.sh && . .venv-fakeflavor/bin/activate
#
# If the GPU box already has a suitable conda/venv environment, skip this and
# just `pip install -r fakeflavor/requirements.txt` into that one instead.
set -euo pipefail

cd "$(dirname "$0")/.."
VENV=.venv-fakeflavor

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
. "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r fakeflavor/requirements.txt

python - <<'EOF'
import xgboost, numpy, sklearn, uproot, awkward
print("xgboost ", xgboost.__version__)
print("uproot  ", uproot.__version__)
print("awkward ", awkward.__version__)
import sys
sys.path.insert(0, ".")
from src.utils import detect_gpu
print("CUDA visible to xgboost:", detect_gpu())
EOF
