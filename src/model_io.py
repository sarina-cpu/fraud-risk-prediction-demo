import json
import joblib
import pandas as pd
import __main__

from src.config import (
    FEATURE_ENGINEER_PATH,
    MODEL_PATH,
    SELECTED_FEATURES_PATH,
    PERFORMANCE_PATH,
    METADATA_PATH,
    SAMPLE_UPLOAD_PATH,
    MANUAL_FEATURE_ENGINEER_PATH,
    MANUAL_MODEL_PATH,
    MANUAL_MODEL_FEATURES_PATH,
    MANUAL_PERFORMANCE_PATH,
    MANUAL_METADATA_PATH,
    MANUAL_INPUT_SCHEMA_PATH,
    MANUAL_INPUT_OPTIONS_PATH,
    MANUAL_FEATURE_IMPORTANCE_PATH,
    MANUAL_SAMPLE_INPUTS_PATH,
    DEFAULT_MANUAL_HIGH_RISK_THRESHOLD,
    DEFAULT_MANUAL_MEDIUM_RISK_THRESHOLD,
)

from src.feature_engineering import FraudFeatureEngineer
from src.manual_feature_engineering import ManualSimulationFeatureEngineer


# ============================================================
# Pickle compatibility fixes
# ============================================================
# These feature engineers were saved from notebooks, where the classes may have
# been stored as __main__.<ClassName>.
__main__.FraudFeatureEngineer = FraudFeatureEngineer
__main__.ManualSimulationFeatureEngineer = ManualSimulationFeatureEngineer


# ============================================================
# Generic helpers
# ============================================================

def _load_json(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _load_feature_list(path):
    df = pd.read_csv(path)
    if "feature" in df.columns:
        return df["feature"].tolist()
    return df.iloc[:, 0].tolist()


# ============================================================
# Top 500 upload model loaders
# ============================================================

def load_feature_engineer():
    """
    Load fitted Top 500 upload feature engineering pipeline.
    """
    return joblib.load(FEATURE_ENGINEER_PATH)


def load_model():
    """
    Load trained Top 500 upload model pipeline.
    """
    return joblib.load(MODEL_PATH)


def load_selected_features():
    """
    Load selected Top 500 feature list.
    """
    return _load_feature_list(SELECTED_FEATURES_PATH)


def load_performance():
    """
    Load Top 500 model performance metrics if available.
    """
    return _load_csv(PERFORMANCE_PATH)


def load_metadata():
    """
    Load Top 500 model metadata if available.
    """
    return _load_json(
        METADATA_PATH,
        default={
            "model_name": "fraud_lgbm_upload_full_feature_reduction_top_500",
            "model_type": "LightGBM",
            "n_features": 500,
            "threshold": 0.686122,
            "medium_threshold": 0.30,
        },
    )


def load_sample_upload():
    """
    Load optional sample upload CSV for demo/testing.
    """
    return _load_csv(SAMPLE_UPLOAD_PATH)


# ============================================================
# Manual simulation model loaders
# ============================================================

def load_manual_feature_engineer():
    """
    Load fitted manual simulation feature engineer.
    """
    return joblib.load(MANUAL_FEATURE_ENGINEER_PATH)


def load_manual_model():
    """
    Load trained manual simulation model pipeline.
    """
    return joblib.load(MANUAL_MODEL_PATH)


def load_manual_model_features():
    """
    Load final manual simulation model feature list.
    """
    return _load_feature_list(MANUAL_MODEL_FEATURES_PATH)


def load_manual_performance():
    """
    Load manual model performance metrics if available.
    """
    return _load_csv(MANUAL_PERFORMANCE_PATH)


def load_manual_metadata():
    """
    Load manual model metadata if available.
    """
    return _load_json(
        MANUAL_METADATA_PATH,
        default={
            "model_name": "fraud_lgbm_manual_simulation_v2",
            "model_type": "LightGBM",
            "purpose": "Manual risk simulation using business-understandable inputs.",
            "n_features": None,
            "threshold": DEFAULT_MANUAL_HIGH_RISK_THRESHOLD,
            "medium_threshold": DEFAULT_MANUAL_MEDIUM_RISK_THRESHOLD,
        },
    )


def load_manual_input_schema():
    """
    Load Streamlit form schema for the manual simulation page.
    """
    return _load_csv(MANUAL_INPUT_SCHEMA_PATH)


def load_manual_input_options():
    """
    Load dropdown options for the manual simulation page.
    """
    return _load_json(MANUAL_INPUT_OPTIONS_PATH, default={})


def load_manual_feature_importance():
    """
    Load manual model feature importance if available.
    """
    return _load_csv(MANUAL_FEATURE_IMPORTANCE_PATH)


def load_sample_manual_inputs():
    """
    Load optional sample manual inputs for demo/testing.
    """
    return _load_csv(MANUAL_SAMPLE_INPUTS_PATH)
