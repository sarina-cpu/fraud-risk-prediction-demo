from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Artifact folders
# ============================================================

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

TOP500_ARTIFACT_DIR = ARTIFACT_ROOT / "top_500_upload_model"
MANUAL_ARTIFACT_DIR = ARTIFACT_ROOT / "manual_simulation_model"

# Backwards compatibility: existing upload code imports ARTIFACT_DIR.
ARTIFACT_DIR = TOP500_ARTIFACT_DIR


# ============================================================
# Top 500 upload-scoring model artifacts
# ============================================================

MODEL_PATH = TOP500_ARTIFACT_DIR / "model.pkl"
FEATURE_ENGINEER_PATH = TOP500_ARTIFACT_DIR / "feature_engineer.pkl"
SELECTED_FEATURES_PATH = TOP500_ARTIFACT_DIR / "selected_features.csv"
PERFORMANCE_PATH = TOP500_ARTIFACT_DIR / "performance.csv"
FEATURE_IMPORTANCE_PATH = TOP500_ARTIFACT_DIR / "feature_importance.csv"
METADATA_PATH = TOP500_ARTIFACT_DIR / "model_metadata.json"
SAMPLE_UPLOAD_PATH = TOP500_ARTIFACT_DIR / "sample_upload.csv"


# ============================================================
# Manual simulation model artifacts
# ============================================================

MANUAL_MODEL_PATH = MANUAL_ARTIFACT_DIR / "model.pkl"
MANUAL_FEATURE_ENGINEER_PATH = MANUAL_ARTIFACT_DIR / "feature_engineer.pkl"
MANUAL_MODEL_FEATURES_PATH = MANUAL_ARTIFACT_DIR / "model_features.csv"
MANUAL_PERFORMANCE_PATH = MANUAL_ARTIFACT_DIR / "performance.csv"
MANUAL_VALID_PROBS_PATH = MANUAL_ARTIFACT_DIR / "valid_probs.npy"
MANUAL_INPUT_SCHEMA_PATH = MANUAL_ARTIFACT_DIR / "input_schema.csv"
MANUAL_INPUT_OPTIONS_PATH = MANUAL_ARTIFACT_DIR / "input_options.json"
MANUAL_METADATA_PATH = MANUAL_ARTIFACT_DIR / "model_metadata.json"
MANUAL_FEATURE_IMPORTANCE_PATH = MANUAL_ARTIFACT_DIR / "feature_importance.csv"
MANUAL_SAMPLE_INPUTS_PATH = MANUAL_ARTIFACT_DIR / "sample_manual_inputs.csv"


# ============================================================
# Threshold defaults
# ============================================================

DEFAULT_HIGH_RISK_THRESHOLD = 0.686122
DEFAULT_MEDIUM_RISK_THRESHOLD = 0.30

# Manual v2 threshold from validation, used only if model_metadata.json is missing.
DEFAULT_MANUAL_HIGH_RISK_THRESHOLD = 0.736669
DEFAULT_MANUAL_MEDIUM_RISK_THRESHOLD = 0.30


# ============================================================
# Upload validation/display settings
# ============================================================

# These are not all strictly required for model execution, because the model
# can impute missing values. They are used to warn the user if the upload is
# missing important source-system columns.
CORE_INPUT_COLUMNS = [
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
]

REVIEW_DISPLAY_COLUMNS = [
    "fraud_rank",
    "TransactionID",
    "fraud_score",
    "risk_band",
    "recommended_action",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card4",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "isFraud",
]


# ============================================================
# Manual simulation display settings
# ============================================================

MANUAL_DIRECT_INPUT_COLUMNS = [
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "TransactionHour",
    "TransactionDayOfWeek",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_23",
    "id_30",
    "id_31",
    "id_33",
    "missing_addr1",
    "missing_addr2",
    "missing_dist1",
    "distance_signal_bucket",
    "has_identity_info",
    "payment_identifier_familiarity_bucket",
]

MANUAL_GENERATED_FEATURE_PRIORITY = [
    "TransactionAmt_log",
    "amount_bucket",
    "is_high_amount",
    "is_round_amount",
    "has_3_decimal_amount",
    "is_weekend",
    "is_business_hours",
    "is_night",
    "time_of_day_bucket",
    "email_relationship_status",
    "email_domain_mismatch",
    "email_domain_familiarity_bucket",
    "rare_email_domain_flag",
    "identity_completeness_bucket",
    "identity_missing_count",
    "rare_device_flag",
    "device_familiarity_bucket",
    "network_proxy_type",
    "id_23_proxy_flag",
    "id_30_os_family",
    "id_31_browser_family",
    "screen_size_bucket",
    "unusual_screen_ratio",
    "address_signal_available",
    "distance_signal_bucket",
    "is_large_dist1",
    "address_distance_status",
    "payment_identifier_familiarity_bucket",
    "rare_payment_identifier_flag",
    "common_payment_identifier_flag",
    "high_amount_missing_identity",
    "high_amount_missing_email",
    "high_amount_off_hours",
    "high_amount_large_distance",
    "high_amount_proxy_flag",
    "high_amount_rare_device",
    "high_amount_rare_email_domain",
    "high_amount_rare_payment_identifier",
    "email_mismatch_proxy_flag",
    "email_mismatch_rare_payment_identifier",
    "rare_payment_identifier_proxy_flag",
]
