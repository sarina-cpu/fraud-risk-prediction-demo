import pandas as pd

from src.app_utils import (
    assign_risk_band,
    recommended_action,
    missing_columns,
    get_thresholds,
)
from src.config import CORE_INPUT_COLUMNS
from src.model_io import (
    load_feature_engineer,
    load_metadata,
    load_model,
    load_selected_features,
)
from src.top500_feature_builder import build_top500_features


def validate_upload_columns(raw_df: pd.DataFrame) -> dict:
    """
    Validate source-system upload columns.

    Missing core columns produce warnings, not hard failures, because the model
    pipeline can impute missing values. However, too many missing core columns
    means predictions may be less reliable.
    """
    missing_core = missing_columns(raw_df, CORE_INPUT_COLUMNS)

    return {
        "missing_core_columns": missing_core,
        "n_missing_core_columns": len(missing_core),
        "n_columns_uploaded": raw_df.shape[1],
        "n_rows_uploaded": raw_df.shape[0],
    }


def score_uploaded_transactions(
    raw_df: pd.DataFrame,
    model=None,
    feature_engineer=None,
    selected_features: list[str] | None = None,
    metadata: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Score uploaded raw transactions using the Top 500 upload model.

    Returns:
    - scored_df: raw rows plus fraud_score/rank/risk fields
    - X_model: exact Top 500 model matrix used for scoring
    - validation_info: missing core-column warning details
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("Uploaded dataframe is empty.")

    if model is None:
        model = load_model()
    if feature_engineer is None:
        feature_engineer = load_feature_engineer()
    if selected_features is None:
        selected_features = load_selected_features()
    if metadata is None:
        metadata = load_metadata()

    validation_info = validate_upload_columns(raw_df)

    X_model = build_top500_features(
        raw_df=raw_df,
        feature_engineer=feature_engineer,
        selected_features=selected_features,
    )

    fraud_scores = model.predict_proba(X_model)[:, 1]
    high_threshold, medium_threshold = get_thresholds(metadata)

    scored_df = raw_df.copy()
    scored_df["_row_position"] = range(len(scored_df))
    scored_df["fraud_score"] = fraud_scores
    scored_df["fraud_rank"] = (
        scored_df["fraud_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    scored_df["risk_band"] = scored_df["fraud_score"].apply(
        lambda x: assign_risk_band(x, high_threshold, medium_threshold)
    )
    scored_df["recommended_action"] = scored_df["fraud_score"].apply(
        lambda x: recommended_action(x, high_threshold, medium_threshold)
    )

    scored_df = scored_df.sort_values("fraud_score", ascending=False).reset_index(drop=True)

    return scored_df, X_model, validation_info


def get_review_queue(scored_df: pd.DataFrame, review_pct: int = 5) -> pd.DataFrame:
    if scored_df is None or scored_df.empty:
        return pd.DataFrame()

    n_review = max(1, int(len(scored_df) * review_pct / 100))
    return scored_df.head(n_review).copy()
