"""Shared utilities. No analysis logic here."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"
LOG_DIR = PROJECT_ROOT / "logs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger configured for this project."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def detect_gpu() -> bool:
    """Return True if a CUDA device is visible to xgboost."""
    try:
        import xgboost as xgb  # noqa
        # Quick probe: try building a tiny DMatrix on cuda.
        import numpy as np
        dm = xgb.DMatrix(np.array([[1.0]]), label=np.array([0]))
        booster = xgb.train(
            {"device": "cuda", "tree_method": "hist", "verbosity": 0},
            dm, num_boost_round=1,
        )
        del booster
        return True
    except Exception:
        return False
