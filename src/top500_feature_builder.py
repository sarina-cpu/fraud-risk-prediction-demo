import re
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_feature_prefix(col: str) -> str:
    return (
        str(col)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def _selected_set(selected_features: Iterable[str]) -> set[str]:
    return set(map(str, selected_features))


def _feature_engineer_has(feature_engineer, method_name: str) -> bool:
    return callable(getattr(feature_engineer, method_name, None))


def _run_row_level_feature_steps(raw_df: pd.DataFrame, feature_engineer) -> pd.DataFrame:
    """
    Create the reusable row-level features needed before selected Top 500 features
    can be assembled.

    This intentionally does NOT call feature_engineer.transform(), because that
    would build the full training feature matrix. Instead, it calls the cheaper
    staged methods up to the point where row-level dependencies exist.

    It relies on the fitted feature engineer carrying the same helper methods
    used in training.
    """
    df = raw_df.copy()

    if _feature_engineer_has(feature_engineer, "_create_basic_features"):
        df = feature_engineer._create_basic_features(df)

    if _feature_engineer_has(feature_engineer, "_add_threshold_risk_flags"):
        df = feature_engineer._add_threshold_risk_flags(df)

    if _feature_engineer_has(feature_engineer, "_add_cross_domain_interactions"):
        df = feature_engineer._add_cross_domain_interactions(df)

    return df


def _add_selected_group_amount_stats(
    df: pd.DataFrame,
    feature_engineer,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Add only the selected group amount statistic features.

    Expected naming pattern from training:
    {safe_group_col}_TransactionAmt_mean
    {safe_group_col}_TransactionAmt_std
    {safe_group_col}_TransactionAmt_median
    {safe_group_col}_TransactionAmt_count
    {safe_group_col}_TransactionAmt_dev
    {safe_group_col}_TransactionAmt_abs_dev
    {safe_group_col}_TransactionAmt_ratio
    {safe_group_col}_TransactionAmt_zscore
    """
    selected = _selected_set(selected_features)
    amount_stat_maps = getattr(feature_engineer, "amount_stat_maps_", None) or {}

    if not amount_stat_maps or "TransactionAmt" not in df.columns:
        return df

    out = {}
    amount = pd.to_numeric(df["TransactionAmt"], errors="coerce").astype("float32")

    stat_suffixes = [
        "mean",
        "std",
        "median",
        "count",
        "dev",
        "abs_dev",
        "ratio",
        "zscore",
    ]

    for group_col, stat_maps in amount_stat_maps.items():
        if group_col not in df.columns:
            continue

        safe_col = _safe_feature_prefix(group_col)
        possible_output_cols = [
            f"{safe_col}_TransactionAmt_{suffix}"
            for suffix in stat_suffixes
        ]

        needed_output_cols = [col for col in possible_output_cols if col in selected]
        if not needed_output_cols:
            continue

        group_values = df[group_col]

        group_mean = group_values.map(stat_maps.get("mean", {})).astype("float32")
        group_std = group_values.map(stat_maps.get("std", {})).astype("float32")
        group_median = group_values.map(stat_maps.get("median", {})).astype("float32")
        group_count = group_values.map(stat_maps.get("count", {})).fillna(0).astype("float32")

        amount_dev = (amount - group_mean).astype("float32")
        amount_abs_dev = amount_dev.abs().astype("float32")
        amount_ratio = (amount / (group_mean + 1)).replace([np.inf, -np.inf], np.nan).astype("float32")

        safe_std = group_std.replace(0, np.nan)
        amount_zscore = (amount_dev / safe_std).replace([np.inf, -np.inf], np.nan).astype("float32")

        candidates = {
            f"{safe_col}_TransactionAmt_mean": group_mean,
            f"{safe_col}_TransactionAmt_std": group_std,
            f"{safe_col}_TransactionAmt_median": group_median,
            f"{safe_col}_TransactionAmt_count": group_count,
            f"{safe_col}_TransactionAmt_dev": amount_dev,
            f"{safe_col}_TransactionAmt_abs_dev": amount_abs_dev,
            f"{safe_col}_TransactionAmt_ratio": amount_ratio,
            f"{safe_col}_TransactionAmt_zscore": amount_zscore,
        }

        for output_col in needed_output_cols:
            out[output_col] = candidates[output_col]

    if out:
        df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

    return df


def _add_selected_frequency_features(
    df: pd.DataFrame,
    feature_engineer,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Add only the selected fitted frequency and rare-value flag features.

    Expected naming pattern from training:
    {safe_col}_freq
    rare_{safe_col}_flag

    Also recreates common alias features if selected:
    rare_device_flag
    rare_email_domain_flag
    rare_card_region_flag
    """
    selected = _selected_set(selected_features)
    frequency_maps = getattr(feature_engineer, "frequency_maps_", None) or {}
    rare_threshold = getattr(feature_engineer, "rare_threshold", 50)

    if not frequency_maps:
        return df

    out = {}

    for col, freq_map in frequency_maps.items():
        if col not in df.columns:
            continue

        safe_col = _safe_feature_prefix(col)
        freq_feature = f"{safe_col}_freq"
        rare_feature = f"rare_{safe_col}_flag"

        alias_features = []
        if rare_feature == "rare_DeviceInfo_clean_flag":
            alias_features.append("rare_device_flag")
        if rare_feature == "rare_P_emaildomain_flag":
            alias_features.append("rare_email_domain_flag")
        if rare_feature == "rare_card1_addr1_flag":
            alias_features.append("rare_card_region_flag")

        needed = (
            freq_feature in selected
            or rare_feature in selected
            or any(alias in selected for alias in alias_features)
        )

        if not needed:
            continue

        values = df[col].astype("string").fillna("missing")
        freq_values = values.map(freq_map).fillna(0).astype("int32")
        rare_values = (freq_values < rare_threshold).astype("int8")

        if freq_feature in selected:
            out[freq_feature] = freq_values
        if rare_feature in selected:
            out[rare_feature] = rare_values
        for alias in alias_features:
            if alias in selected:
                out[alias] = rare_values

    if out:
        df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

    return df


def _clean_model_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Keep pandas dtypes model-friendly without over-changing categories.
    The model pipeline handles imputation/encoding.
    """
    X = X.copy()

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    return X


def build_top500_features(
    raw_df: pd.DataFrame,
    feature_engineer,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Build exactly the features required by the Top 500 upload model.

    This is the app-side feature builder:
    - accepts raw/source-system columns
    - ignores extra uploaded columns
    - creates needed row-level dependencies
    - creates selected fitted statistical/frequency features
    - returns exactly selected_features, in the model's expected order
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("Uploaded dataframe is empty.")

    if not selected_features:
        raise ValueError("selected_features list is empty.")

    df = _run_row_level_feature_steps(raw_df, feature_engineer)
    df = _add_selected_group_amount_stats(df, feature_engineer, selected_features)
    df = _add_selected_frequency_features(df, feature_engineer, selected_features)

    X_model = df.reindex(columns=selected_features)
    X_model = _clean_model_matrix(X_model)

    return X_model
