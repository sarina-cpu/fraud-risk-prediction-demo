from __future__ import annotations

import numpy as np
import pandas as pd

from src.app_utils import assign_risk_band, recommended_action, get_thresholds
from src.model_io import (
    load_manual_feature_engineer,
    load_manual_metadata,
    load_manual_model,
    load_manual_model_features,
)


BOOLEAN_INPUT_COLUMNS = {
    "missing_addr1",
    "missing_addr2",
    "missing_dist1",
    "has_identity_info",
}

NUMERIC_INPUT_COLUMNS = {
    "TransactionAmt",
    "TransactionHour",
    "TransactionDayOfWeek",
}

MISSING_STRINGS = {"", "missing", "none", "nan", "<na>", "null"}


def _clean_manual_value(column: str, value):
    """
    Convert Streamlit form values into the style expected by
    ManualSimulationFeatureEngineer.
    """
    if value is None:
        return np.nan

    if isinstance(value, str) and value.strip().lower() in MISSING_STRINGS:
        # Keep selected categorical missing buckets as explicit strings.
        if column in {"distance_signal_bucket", "payment_identifier_familiarity_bucket"}:
            return "missing"
        return np.nan

    if column in BOOLEAN_INPUT_COLUMNS:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"yes", "true", "1", "available", "present", "missing"}:
                return 1
            if cleaned in {"no", "false", "0", "not available", "not missing"}:
                return 0
        try:
            return int(value)
        except Exception:
            return np.nan

    if column in NUMERIC_INPUT_COLUMNS:
        try:
            return float(value)
        except Exception:
            return np.nan

    return value


def build_manual_input_dataframe(form_values: dict) -> pd.DataFrame:
    """
    Convert a manual form dictionary into a one-row dataframe.
    """
    cleaned = {
        col: _clean_manual_value(col, value)
        for col, value in form_values.items()
    }
    return pd.DataFrame([cleaned])


def build_manual_model_features(
    manual_input_df: pd.DataFrame,
    feature_engineer=None,
    model_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate manual simulation model features.

    Returns:
    - X_model: final model feature matrix in expected order
    - generated_all: full generated dataframe for explainability display
    """
    if manual_input_df is None or manual_input_df.empty:
        raise ValueError("Manual input dataframe is empty.")

    if feature_engineer is None:
        feature_engineer = load_manual_feature_engineer()
    if model_features is None:
        model_features = load_manual_model_features()

    generated_all = feature_engineer.transform(
        manual_input_df,
        return_features_only=False,
    )

    X_model = feature_engineer.transform(
        manual_input_df,
        return_features_only=True,
    )

    if model_features:
        X_model = X_model.reindex(columns=model_features)

    X_model = X_model.replace([np.inf, -np.inf], np.nan)

    return X_model, generated_all


def score_manual_transaction(
    form_values: dict | None = None,
    manual_input_df: pd.DataFrame | None = None,
    model=None,
    feature_engineer=None,
    model_features: list[str] | None = None,
    metadata: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Score one manual scenario.

    Returns:
    - scored_df: one-row input dataframe plus fraud score/risk/action
    - X_model: exact model feature matrix
    - generated_all: full generated feature dataframe
    - result: summary dictionary for metric cards
    """
    if manual_input_df is None:
        if form_values is None:
            raise ValueError("Either form_values or manual_input_df must be provided.")
        manual_input_df = build_manual_input_dataframe(form_values)

    if model is None:
        model = load_manual_model()
    if feature_engineer is None:
        feature_engineer = load_manual_feature_engineer()
    if model_features is None:
        model_features = load_manual_model_features()
    if metadata is None:
        metadata = load_manual_metadata()

    X_model, generated_all = build_manual_model_features(
        manual_input_df=manual_input_df,
        feature_engineer=feature_engineer,
        model_features=model_features,
    )

    fraud_score = float(model.predict_proba(X_model)[:, 1][0])
    high_threshold, medium_threshold = get_thresholds(metadata)

    risk_band = assign_risk_band(fraud_score, high_threshold, medium_threshold)
    action = recommended_action(fraud_score, high_threshold, medium_threshold)

    scored_df = manual_input_df.copy()
    scored_df["fraud_score"] = fraud_score
    scored_df["risk_band"] = risk_band
    scored_df["recommended_action"] = action

    result = {
        "fraud_score": fraud_score,
        "risk_band": risk_band,
        "recommended_action": action,
        "high_threshold": high_threshold,
        "medium_threshold": medium_threshold,
        "n_model_features": X_model.shape[1],
    }

    return scored_df, X_model, generated_all, result
