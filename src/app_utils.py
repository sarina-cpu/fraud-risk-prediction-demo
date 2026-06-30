import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_MEDIUM_RISK_THRESHOLD,
)


def load_json(path: Path, default=None):
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return default
    return default


def get_thresholds(metadata: dict) -> tuple[float, float]:
    high = float(metadata.get("high_risk_threshold", metadata.get("threshold", DEFAULT_HIGH_RISK_THRESHOLD)))
    medium = float(metadata.get("medium_risk_threshold", DEFAULT_MEDIUM_RISK_THRESHOLD))
    return high, medium


def assign_risk_band(prob: float, high_threshold: float, medium_threshold: float) -> str:
    if pd.isna(prob):
        return "Unknown"
    if prob >= high_threshold:
        return "High"
    if prob >= medium_threshold:
        return "Medium"
    return "Low"


def recommended_action(prob: float, high_threshold: float, medium_threshold: float) -> str:
    if pd.isna(prob):
        return "Unable to score"
    if prob >= high_threshold:
        return "Review immediately"
    if prob >= medium_threshold:
        return "Review if capacity allows"
    return "No immediate review required"


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col not in df.columns]


def clean_probability_series(values) -> pd.Series:
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=1)


def format_pct(value) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"
