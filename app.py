import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import traceback
import time
import html

from src.app_utils import existing_columns, format_pct, get_thresholds
from src.config import (
    MANUAL_DIRECT_INPUT_COLUMNS,
    MANUAL_GENERATED_FEATURE_PRIORITY,
    REVIEW_DISPLAY_COLUMNS,
)
from src.model_io import (
    load_metadata,
    load_model,
    load_feature_engineer,
    load_selected_features,
    load_performance,
    load_sample_upload,
    load_manual_metadata,
    load_manual_model,
    load_manual_feature_engineer,
    load_manual_model_features,
    load_manual_performance,
    load_manual_input_schema,
    load_manual_input_options,
    load_manual_feature_importance,
    load_sample_manual_inputs,
)
from src.scoring_upload import score_uploaded_transactions, get_review_queue
from src.scoring_manual import score_manual_transaction
from src.shap_explainer import explain_single_transaction

from pathlib import Path

st.set_page_config(
    page_title="Fraud Risk Assistant",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# Cached Top 500 upload artifacts
# ============================================================

@st.cache_resource
def cached_model():
    return load_model()


@st.cache_resource
def cached_feature_engineer():
    return load_feature_engineer()


@st.cache_data
def cached_selected_features():
    return load_selected_features()


@st.cache_data
def cached_metadata():
    return load_metadata()


@st.cache_data
def cached_performance():
    return load_performance()


@st.cache_data
def cached_sample_upload():
    return load_sample_upload()


# ============================================================
# Cached manual simulation artifacts
# ============================================================

@st.cache_resource
def cached_manual_model():
    return load_manual_model()


@st.cache_resource
def cached_manual_feature_engineer():
    return load_manual_feature_engineer()


@st.cache_data
def cached_manual_model_features():
    return load_manual_model_features()


@st.cache_data
def cached_manual_metadata():
    return load_manual_metadata()


@st.cache_data
def cached_manual_performance():
    return load_manual_performance()


@st.cache_data
def cached_manual_input_schema():
    return load_manual_input_schema()


@st.cache_data
def cached_manual_input_options():
    return load_manual_input_options()


@st.cache_data
def cached_manual_feature_importance():
    return load_manual_feature_importance()


@st.cache_data
def cached_sample_manual_inputs():
    return load_sample_manual_inputs()


MODEL_INSIGHTS_DIR = Path("artifacts") / "model_insights"


@st.cache_data
def load_model_insight_image_path(model_folder: str, file_name: str):
    path = MODEL_INSIGHTS_DIR / model_folder / file_name
    return path if path.exists() else None


def render_saved_model_insights(
    model_folder: str,
    model_label: str,
    shap_file_name: str = "global_shap_summary.png",
    importance_file_name: str = "feature_importance_gain.png",
):
    st.subheader("Global model explanation")

    st.markdown(
        """
        These plots explain how the model behaves across a representative set of transactions,
        rather than explaining only one selected transaction.
        """
    )

    shap_path = load_model_insight_image_path(model_folder, shap_file_name)
    importance_path = load_model_insight_image_path(model_folder, importance_file_name)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Global SHAP summary")

        if shap_path is not None:
            st.image(str(shap_path), use_container_width=True)
            st.caption(
                "Each point represents a scenario. Points to the right increase predicted fraud risk; "
                "points to the left reduce predicted fraud risk. Color shows whether the feature value is high or low."
            )
        else:
            st.info(
                f"No saved Global SHAP image found for {model_label}. "
                "Run the model insights notebook first."
            )

    with col2:
        st.markdown("#### Feature importance")

        if importance_path is not None:
            st.image(str(importance_path), use_container_width=True)
            st.caption(
                "Shows which signals the model relied on most overall, based on LightGBM gain importance."
            )
        else:
            st.info(
                f"No saved feature importance image found for {model_label}. "
                "Run the model insights notebook first."
            )
# ============================================================
# UI copy, styling, and display helpers
# ============================================================

EXPLANATION_SECTION_TITLE = "Transaction-Level Risk Explanation"
RISK_RED = "#C62828"
RISK_GREEN = "#2E7D32"
PRIMARY_BLUE = "#1F4E79"
MID_BLUE = "#2F6F9F"
SOFT_BLUE = "#EAF2F8"
SOFT_RED = "#FDECEC"
SOFT_GREEN = "#EEF7F0"
TEXT_DARK = "#1F2937"
TEXT_MUTED = "#64748B"
BORDER = "#D8DEE9"

FORM_LABEL_OVERRIDES = {
    "missing_addr1": "Address available",
    "missing_addr2": "Country code available",
    "missing_dist1": "Distance signal available",
    "distance_signal_bucket": "Distance / location profile",
}

FORM_DESCRIPTION_OVERRIDES = {
    "missing_addr1": "Whether the main address/location signal is available for this transaction.",
    "missing_addr2": "Whether the country-like address signal is available for this transaction.",
    "missing_dist1": "Whether the distance/location signal is available for this transaction.",
    "distance_signal_bucket": "Business-level description of the distance/location pattern.",
}

FEATURE_DISPLAY_NAME_OVERRIDES = {
    "TransactionAmt": "Transaction amount",
    "TransactionAmt_log": "Log-scaled transaction amount",
    "amount_bucket": "Transaction amount band",
    "amount_decimal": "Decimal amount component",
    "amount_cents": "Cents in transaction amount",
    "amount_decimal_places": "Number of decimal places",
    "is_round_amount": "Round amount indicator",
    "has_3_decimal_amount": "Three-decimal amount indicator",
    "is_high_amount": "High amount indicator",
    "TransactionHour": "Transaction hour",
    "TransactionDayOfWeek": "Transaction day of week",
    "ProductCD": "Product / transaction type",
    "card4": "Card network",
    "card6": "Card type",
    "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain",
    "missing_addr1": "Address available",
    "missing_addr2": "Country code available",
    "missing_dist1": "Distance signal available",
    "distance_signal_bucket": "Distance / location profile",
    "has_identity_info": "Identity / device information available",
    "id_23": "Proxy / network signal",
    "id_30": "Operating system",
    "id_31": "Browser",
    "id_33": "Screen resolution",
    "id_30_os_version_available": "Operating system version available",
    "id_31_browser_version_available": "Browser version available",
    "identity_missing_count": "Missing identity field count",
    "addr_missing_count": "Missing address/location field count",
    "device_missing": "Device information missing",
    "is_mobile_device": "Mobile device indicator",
    "is_desktop_device": "Desktop device indicator",
    "DeviceInfo_brand": "Device brand / family",
    "id_23_proxy_flag": "Proxy/network risk signal",
    "id_23_anonymous_flag": "Anonymous proxy signal",
    "id_23_transparent_flag": "Transparent proxy signal",
    "id_23_hidden_flag": "Hidden proxy signal",
    "screen_size_bucket": "Screen size band",
    "unusual_screen_ratio": "Unusual screen ratio indicator",
    "address_signal_available": "Any address signal available",
    "is_large_dist1": "Large distance/location indicator",
    "address_distance_status": "Address and distance status",
    "missing_address_and_distance": "Missing address and distance signals",
    "payment_identifier_familiarity_bucket": "Payment identifier familiarity",
    "rare_payment_identifier_flag": "Rare payment identifier pattern",
    "common_payment_identifier_flag": "Common payment identifier pattern",
    "high_amount_missing_identity": "High amount with missing identity",
    "high_amount_missing_email": "High amount with missing email",
    "high_amount_off_hours": "High amount outside business hours",
    "high_amount_large_distance": "High amount with large distance signal",
    "high_amount_proxy_flag": "High amount with proxy/network signal",
    "high_amount_rare_device": "High amount with rare device pattern",
    "high_amount_rare_email_domain": "High amount with rare email domain",
    "high_amount_rare_payment_identifier": "High amount with rare payment identifier",
    "night_transaction_missing_identity": "Night transaction with missing identity",
    "night_transaction_large_distance": "Night transaction with large distance signal",
    "email_mismatch_missing_identity": "Email mismatch with missing identity",
    "email_mismatch_proxy_flag": "Email mismatch with proxy/network signal",
    "email_mismatch_rare_payment_identifier": "Email mismatch with rare payment identifier",
    "rare_payment_identifier_proxy_flag": "Rare payment identifier with proxy/network signal",
    "rare_payment_identifier_large_distance": "Rare payment identifier with large distance signal",
    "proxy_and_rare_device": "Proxy signal with rare device pattern",
    "P_email_provider": "Purchaser email provider",
    "P_email_suffix": "Purchaser email suffix",
    "R_email_provider": "Recipient email provider",
    "R_email_suffix": "Recipient email suffix",
    "same_email_provider": "Same email provider",
    "same_email_suffix": "Same email suffix",
    "both_emails_missing": "Both email domains missing",
    "only_recipient_email_missing": "Only recipient email missing",

    "amount_decimal": "Decimal amount component",
    "amount_cents": "Cents in transaction amount",
    "amount_decimal_places": "Number of decimal places",

    "id_23_missing": "Proxy / network signal missing",
    "screen_missing": "Screen resolution missing",
    "id_33_screen_width": "Screen width",
    "id_33_screen_height": "Screen height",
    "id_33_screen_area": "Screen area",
    "id_33_aspect_ratio": "Screen aspect ratio",

    "payment_familiarity_email_status": "Payment familiarity and email status",
    "high_amount_ProductCD": "High amount and transaction type",
    "night_ProductCD": "Night transaction and transaction type",
    "email_mismatch_ProductCD": "Email mismatch and transaction type",
    "distance_bucket_ProductCD": "Distance profile and transaction type",
    "proxy_ProductCD": "Proxy signal and transaction type",
    "DeviceType_ProductCD": "Device type and transaction type",
    "card4_card6": "Card network and card type",
    "amount_bucket_ProductCD": "Amount band and transaction type",
    "amount_bucket_time_of_day": "Amount band and time of day",
    "payment_familiarity_ProductCD": "Payment familiarity and transaction type",
    "device_familiarity_ProductCD": "Device familiarity and transaction type",
    "identity_completeness_ProductCD": "Identity completeness and transaction type",
}

FEATURE_EXPLANATIONS = {
    "TransactionAmt_log": "Log transformation that helps the model compare small and very large transaction values more fairly.",
    "amount_bucket": "Groups the transaction amount into a business-friendly size band.",
    "amount_decimal": "Captures whether the amount has an unusual decimal component.",
    "amount_cents": "Shows the cents portion of the amount, which can signal non-standard pricing patterns.",
    "amount_decimal_places": "Counts how many decimal places appear in the amount.",
    "is_round_amount": "Flags whether the amount is a clean round number.",
    "has_3_decimal_amount": "Flags an amount with three decimal places, which may be unusual for normal payment flows.",
    "is_high_amount": "Flags whether the amount is high relative to the training data.",
    "time_of_day_bucket": "Groups the transaction time into a simple daypart.",
    "is_business_hours": "Flags whether the transaction occurred during normal business hours.",
    "is_night": "Flags whether the transaction occurred late at night.",
    "is_weekend": "Flags whether the transaction occurred during the weekend.",
    "missing_P_email": "Flags whether the purchaser email domain is unavailable.",
    "missing_R_email": "Flags whether the recipient email domain is unavailable.",
    "P_email_provider": "Extracts the purchaser email provider family.",
    "P_email_suffix": "Extracts the purchaser email suffix, such as .com or .net.",
    "R_email_provider": "Extracts the recipient email provider family.",
    "R_email_suffix": "Extracts the recipient email suffix, such as .com or .net.",
    "emails_match": "Flags whether purchaser and recipient email domains match.",
    "same_email_provider": "Flags whether the email provider families are the same.",
    "same_email_suffix": "Flags whether the email suffixes are the same.",
    "email_domain_mismatch": "Flags a mismatch between purchaser and recipient email domains.",
    "email_relationship_status": "Summarizes the relationship between purchaser and recipient email domains.",
    "email_domain_familiarity_bucket": "Groups the email domain into a familiarity band based on model training data.",
    "rare_email_domain_flag": "Flags whether the email domain pattern is uncommon.",
    "DeviceInfo_brand": "Extracts a cleaner device brand or family from the raw device string.",
    "device_missing": "Flags whether device information is unavailable.",
    "is_mobile_device": "Flags whether the transaction is associated with a mobile device.",
    "is_desktop_device": "Flags whether the transaction is associated with a desktop device.",
    "device_familiarity_bucket": "Groups the device pattern into a familiarity band.",
    "rare_device_flag": "Flags whether the device pattern is uncommon.",
    "id_30_os_family": "Extracts the operating system family.",
    "id_30_os_version_available": "Flags whether an operating system version is available.",
    "id_31_browser_family": "Extracts the browser family.",
    "id_31_browser_version_available": "Flags whether a browser version is available.",
    "os_browser_combo": "Combines operating system and browser into one compatibility signal.",
    "network_proxy_type": "Converts raw proxy information into a simpler network-signal category.",
    "id_23_proxy_flag": "Flags whether a proxy/network signal is present.",
    "id_23_anonymous_flag": "Flags an anonymous proxy signal.",
    "id_23_transparent_flag": "Flags a transparent proxy signal.",
    "id_23_hidden_flag": "Flags a hidden proxy signal.",
    "identity_missing_count": "Counts how many identity/device fields are missing.",
    "identity_completeness_bucket": "Groups the identity/device information into a completeness band.",
    "screen_size_bucket": "Groups screen resolution into a size band.",
    "unusual_screen_ratio": "Flags an unusual screen aspect ratio.",
    "address_signal_available": "Flags whether at least one address/location signal is available.",
    "missing_addr1": "Shows whether the main address/location signal is available.",
    "missing_addr2": "Shows whether the country-like address signal is available.",
    "missing_dist1": "Shows whether the distance/location signal is available.",
    "distance_signal_bucket": "Groups the distance/location pattern into a simple business band.",
    "is_large_dist1": "Flags whether the distance/location pattern is large or far.",
    "addr_missing_count": "Counts how many address/location fields are missing.",
    "address_distance_status": "Summarizes whether address and distance signals are available.",
    "missing_address_and_distance": "Flags transactions where both address and distance information are unavailable.",
    "payment_identifier_familiarity_bucket": "Groups the masked payment identifier into a familiarity band without exposing raw card data.",
    "rare_payment_identifier_flag": "Flags whether the payment identifier pattern is uncommon.",
    "common_payment_identifier_flag": "Flags whether the payment identifier pattern is common.",
}

AVAILABILITY_FROM_MISSING_FLAG_FEATURES = {"missing_addr1", "missing_addr2", "missing_dist1"}
BINARY_TRUE_WHEN_ONE_FEATURES = {
    "is_high_amount", "is_round_amount", "has_3_decimal_amount", "is_weekend", "is_business_hours", "is_night",
    "missing_P_email", "missing_R_email", "emails_match", "same_email_provider", "same_email_suffix",
    "email_domain_mismatch", "rare_email_domain_flag", "device_missing", "is_mobile_device", "is_desktop_device",
    "rare_device_flag", "id_30_os_version_available", "id_31_browser_version_available", "id_23_proxy_flag",
    "id_23_anonymous_flag", "id_23_transparent_flag", "id_23_hidden_flag", "unusual_screen_ratio",
    "address_signal_available", "is_large_dist1", "missing_address_and_distance", "rare_payment_identifier_flag",
    "common_payment_identifier_flag", "high_amount_missing_identity", "high_amount_missing_email", "high_amount_off_hours",
    "high_amount_large_distance", "high_amount_proxy_flag", "high_amount_rare_device", "high_amount_rare_email_domain",
    "high_amount_rare_payment_identifier", "night_transaction_missing_identity", "night_transaction_large_distance",
    "email_mismatch_missing_identity", "email_mismatch_proxy_flag", "proxy_and_rare_device",
    "email_mismatch_rare_payment_identifier", "rare_payment_identifier_proxy_flag", "rare_payment_identifier_large_distance",
    "has_identity_info", "missing_identity_flag",
}

BUCKET_LABELS = {
    "missing": "Missing / not available",
    "small": "Close / small distance",
    "medium": "Standard / medium distance",
    "large": "Far / large distance",
    "very_low": "Very low",
    "low": "Low",
    "medium_low": "Medium low",
    "medium_high": "Medium high",
    "high": "High",
    "very_high": "Very high",
    "rare_unfamiliar": "Rare / unfamiliar",
    "somewhat_familiar": "Somewhat familiar",
    "common_frequently_observed": "Common / frequently observed",
    "business_hours": "Business hours",
    "off_hours": "Outside business hours",
    "night": "Night",
    "same_domain": "Same domain",
    "same_provider": "Same provider",
    "different": "Different",
    "missing_address_and_distance": "Missing address and distance",
    "missing_address_only": "Missing address only",
    "missing_distance_only": "Missing distance only",
    "available": "Available",
    "complete": "Complete",
    "partial": "Partial",
    "incomplete": "Incomplete",
    
    "recipient_missing": "Recipient email missing",
    "purchaser_missing": "Purchaser email missing",
    "both_missing": "Both email domains missing",
    "different_domains": "Different email domains",
    "missing_or_incomplete": "Missing or incomplete email information",

    "rare_unfamiliar_recipient_missing": "Rare payment pattern with recipient email missing",
    "rare_unfamiliar_purchaser_missing": "Rare payment pattern with purchaser email missing",
    "rare_unfamiliar_both_missing": "Rare payment pattern with both email domains missing",
    "rare_unfamiliar_different_domains": "Rare payment pattern with different email domains",
    "somewhat_familiar_recipient_missing": "Somewhat familiar payment pattern with recipient email missing",
    "somewhat_familiar_different_domains": "Somewhat familiar payment pattern with different email domains",
    "common_frequently_observed_same_domain": "Common payment pattern with same email domain",
    "common_frequently_observed_different_domains": "Common payment pattern with different email domains",
}


def apply_app_theme():
    st.markdown(
        f"""
        <style>
            .block-container {{
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1320px;
            }}
            h1, h2, h3 {{
                color: {TEXT_DARK};
                letter-spacing: -0.02em;
            }}
            .hero-panel {{
                padding: 1.6rem 1.8rem;
                border: 1px solid {BORDER};
                border-radius: 18px;
                background: linear-gradient(135deg, #F8FBFF 0%, #EDF5FB 100%);
                margin: 0.8rem 0 1.2rem 0;
                box-shadow: 0 6px 18px rgba(31, 78, 121, 0.08);
            }}
            .hero-title {{
                font-size: 1.75rem;
                font-weight: 750;
                color: {TEXT_DARK};
                margin-bottom: 0.35rem;
            }}
            .hero-subtitle {{
                font-size: 1rem;
                line-height: 1.55;
                color: {TEXT_MUTED};
                max-width: 950px;
            }}
            .section-card {{
                padding: 1.05rem 1.1rem;
                border: 1px solid {BORDER};
                border-radius: 14px;
                background: #FFFFFF;
                height: 100%;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            }}
            .card-eyebrow {{
                font-size: 0.72rem;
                font-weight: 700;
                color: {PRIMARY_BLUE};
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
            }}
            .card-title {{
                font-size: 1rem;
                font-weight: 750;
                color: {TEXT_DARK};
                margin-bottom: 0.35rem;
            }}
            .card-body {{
                font-size: 0.92rem;
                line-height: 1.45;
                color: {TEXT_MUTED};
            }}
            .feature-card {{
                padding: 0.85rem 0.95rem;
                border: 1px solid {BORDER};
                border-left: 4px solid {PRIMARY_BLUE};
                border-radius: 12px;
                background: #FFFFFF;
                margin-bottom: 0.7rem;
                box-shadow: 0 1px 5px rgba(15, 23, 42, 0.035);
            }}
            .feature-name {{
                font-weight: 750;
                color: {TEXT_DARK};
                margin-bottom: 0.2rem;
            }}
            .feature-value {{
                display: inline-block;
                margin: 0.15rem 0 0.35rem 0;
                padding: 0.16rem 0.48rem;
                border-radius: 999px;
                background: {SOFT_BLUE};
                color: {PRIMARY_BLUE};
                font-weight: 700;
                font-size: 0.86rem;
            }}
            .feature-explanation {{
                color: {TEXT_MUTED};
                font-size: 0.88rem;
                line-height: 1.4;
            }}
            .driver-card {{
                padding: 0.9rem 1rem;
                border-radius: 13px;
                margin-bottom: 0.7rem;
                border: 1px solid {BORDER};
                background: #FFFFFF;
                box-shadow: 0 1px 6px rgba(15, 23, 42, 0.04);
            }}
            .driver-up {{
                border-left: 5px solid {RISK_RED};
                background: linear-gradient(90deg, {SOFT_RED} 0%, #FFFFFF 35%);
            }}
            .driver-down {{
                border-left: 5px solid {RISK_GREEN};
                background: linear-gradient(90deg, {SOFT_GREEN} 0%, #FFFFFF 35%);
            }}
            .driver-title {{
                font-weight: 750;
                color: {TEXT_DARK};
                margin-bottom: 0.3rem;
            }}
            .driver-impact-up {{
                color: {RISK_RED};
                font-size: 1.15rem;
                font-weight: 800;
            }}
            .driver-impact-down {{
                color: {RISK_GREEN};
                font-size: 1.15rem;
                font-weight: 800;
            }}
            .driver-meta {{
                color: {TEXT_MUTED};
                font-size: 0.84rem;
                margin-top: 0.25rem;
            }}
            .small-muted {{
                color: {TEXT_MUTED};
                font-size: 0.9rem;
                line-height: 1.45;
            }}
            div[data-testid="stMetric"] {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 14px;
                padding: 0.85rem 0.95rem;
                box-shadow: 0 1px 6px rgba(15, 23, 42, 0.04);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _h(value) -> str:
    return html.escape(str(value))


def _humanize_token(value) -> str:
    if value is None:
        return "Missing / not available"
    try:
        if pd.isna(value):
            return "Missing / not available"
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "missing", "nan", "none", "null", "<na>"}:
        return "Missing / not available"
    if text in BUCKET_LABELS:
        return BUCKET_LABELS[text]
    text = text.replace("_", " ").replace("-", " ")
    acronym_map = {"ios": "iOS", "os": "OS", "ip": "IP", "id": "ID"}
    words = []
    for word in text.split():
        low = word.lower()
        if low in acronym_map:
            words.append(acronym_map[low])
        elif word.isupper() and len(word) <= 4:
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def _friendly_option_label(column, value):
    if column in {"missing_addr1", "missing_addr2", "missing_dist1"}:
        try:
            return "True" if int(value) == 0 else "False"
        except Exception:
            return _humanize_token(value)

    if column == "has_identity_info":
        try:
            return "True" if int(value) == 1 else "False"
        except Exception:
            return _humanize_token(value)

    text = str(value).strip() if value is not None else "missing"
    low = text.lower()

    if low in {"missing", "", "nan", "none", "null", "<na>"}:
        return "Missing / not available"

    if column == "card4":
        return {
            "visa": "Visa",
            "mastercard": "Mastercard",
            "american express": "American Express",
            "discover": "Discover",
        }.get(low, _humanize_token(text))

    if column == "card6":
        return {
            "debit": "Debit",
            "credit": "Credit",
            "debit or credit": "Debit or credit",
            "charge card": "Charge card",
        }.get(low, _humanize_token(text))

    if column == "DeviceType":
        return {"desktop": "Desktop", "mobile": "Mobile"}.get(low, _humanize_token(text))

    if column == "id_23":
        return {
            "ip_proxy:transparent": "Transparent proxy signal",
            "ip_proxy:anonymous": "Anonymous proxy signal",
            "ip_proxy:hidden": "Hidden proxy signal",
        }.get(low, _humanize_token(text))

    if column in {"distance_signal_bucket", "payment_identifier_familiarity_bucket"}:
        return _humanize_token(text)

    if column in {"P_emaildomain", "R_emaildomain"}:
        return text.lower()

    if column in {"id_30", "id_31", "id_33", "DeviceInfo"}:
        return text

    return _humanize_token(text)


def _friendly_bucket(value):
    return _humanize_token(value)


def get_display_map(feature_engineer=None) -> dict:
    display_map = {}
    if feature_engineer is not None and hasattr(feature_engineer, "get_generated_feature_display_map"):
        try:
            display_map.update(feature_engineer.get_generated_feature_display_map())
        except Exception:
            pass
    display_map.update(FEATURE_DISPLAY_NAME_OVERRIDES)
    return display_map


def _format_boolean(value, true_when_one=True):
    try:
        val = int(float(value))
        is_true = val == 1 if true_when_one else val == 0
        return "True" if is_true else "False"
    except Exception:
        return _humanize_token(value)


def _format_business_value(feature, value):
    if feature in AVAILABILITY_FROM_MISSING_FLAG_FEATURES:
        return _format_boolean(value, true_when_one=False)
    if feature in BINARY_TRUE_WHEN_ONE_FEATURES:
        return _format_boolean(value, true_when_one=True)
    if feature == "TransactionDayOfWeek":
        return _format_day(value)
    if feature == "TransactionHour":
        try:
            return f"{int(float(value)):02d}:00"
        except Exception:
            return _humanize_token(value)
    if feature == "amount_cents":
        try:
            return f"{int(round(float(value)))} cents"
        except Exception:
            return _humanize_token(value)
    if feature == "amount_decimal":
        try:
            return f"{float(value):.2f}"
        except Exception:
            return _humanize_token(value)
    if feature == "TransactionAmt":
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return _humanize_token(value)
    if feature == "model_input_value":
        return _humanize_token(value)

    try:
        if pd.isna(value):
            return "Missing / not available"
    except Exception:
        pass

    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"

    if isinstance(value, (int, np.integer, float, np.floating)):
        val = float(value)
        if val in {0.0, 1.0} and ("flag" in feature or feature.startswith("is_") or feature.startswith("has_") or feature.startswith("missing_")):
            return "True" if val == 1.0 else "False"
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        if abs(val) >= 10:
            return f"{val:.2f}"
        return f"{val:.4g}"

    return _humanize_token(value)


def _feature_explanation(feature, display_name=None):
    if feature in FEATURE_EXPLANATIONS:
        return FEATURE_EXPLANATIONS[feature]
    if "_ProductCD" in feature:
        return "Interaction signal showing how this pattern behaves within the selected product or transaction type."
    if "high_amount" in feature:
        return "Combined signal that checks whether a high-value transaction appears together with another risk indicator."
    if "email_mismatch" in feature:
        return "Combined signal that checks whether email mismatch appears together with another risk indicator."
    return "Model-ready signal generated from the business inputs for scenario scoring and explanation."


def _html_card(title, body, eyebrow=None):
    eyebrow_html = f'<div class="card-eyebrow">{_h(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="section-card">
            {eyebrow_html}
            <div class="card-title">{_h(title)}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards(feature_table, features, columns=2):
    if feature_table is None or feature_table.empty:
        st.caption("No generated features available for this group.")
        return

    rows = []
    for feature in features:
        match = feature_table[feature_table["feature"] == feature]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(row)

    if not rows:
        st.caption("No generated features available for this group.")
        return

    col_objs = st.columns(columns)
    for idx, row in enumerate(rows):
        feature = row["feature"]
        display_name = row.get("display_name", feature)
        business_value = row.get("business_value", _format_business_value(feature, row.get("value")))
        explanation = row.get("explanation", _feature_explanation(feature, display_name))
        with col_objs[idx % columns]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-name">{_h(display_name)}</div>
                    <div class="feature-value">{_h(business_value)}</div>
                    <div class="feature-explanation">{_h(explanation)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _format_shap_impact(value):
    try:
        return f"{float(value):+.4f}"
    except Exception:
        return str(value)


def _format_model_input_for_driver(value):
    try:
        if pd.isna(value):
            return "Missing / not available"
    except Exception:
        pass
    if isinstance(value, (int, np.integer, float, np.floating)):
        val = float(value)
        if val in {0.0, 1.0}:
            return "Present / active" if val == 1.0 else "Not present"
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        if abs(val) >= 10:
            return f"{val:.2f}"
        return f"{val:.4g}"
    return _humanize_token(value)


def render_shap_driver_cards(shap_df, max_cards=10):
    if shap_df is None or shap_df.empty:
        st.caption("No explanation drivers available.")
        return

    display_col = "display_name" if "display_name" in shap_df.columns else "feature"
    card_df = shap_df.sort_values("abs_shap_value", ascending=False).head(max_cards).copy()
    col_up, col_down = st.columns(2)

    def _render_group(df, title, empty_text, container):
        with container:
            st.markdown(f"**{title}**")
            if df.empty:
                st.caption(empty_text)
                return
            for _, row in df.iterrows():
                is_up = float(row["shap_value"]) >= 0
                css_class = "driver-up" if is_up else "driver-down"
                impact_class = "driver-impact-up" if is_up else "driver-impact-down"
                direction = "Increases risk" if is_up else "Reduces risk"
                value_text = _format_model_input_for_driver(row.get("model_input_value", "n/a"))
                st.markdown(
                    f"""
                    <div class="driver-card {css_class}">
                        <div class="driver-title">{_h(row[display_col])}</div>
                        <div class="{impact_class}">{_h(_format_shap_impact(row['shap_value']))}</div>
                        <div class="driver-meta">{_h(direction)} · Model value: {_h(value_text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    up_df = card_df[card_df["shap_value"] >= 0]
    down_df = card_df[card_df["shap_value"] < 0]
    _render_group(up_df, "Drivers increasing predicted fraud risk", "No increasing drivers in the top results.", col_up)
    _render_group(down_df, "Drivers reducing predicted fraud risk", "No reducing drivers in the top results.", col_down)


def render_shap_bar_chart(shap_df, title="Top transaction-level risk drivers"):
    if shap_df is None or shap_df.empty:
        return
    display_col = "display_name" if "display_name" in shap_df.columns else "feature"
    plot_df = shap_df.sort_values("abs_shap_value", ascending=True).copy()
    fig_height = max(4.2, 0.38 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    colors = [RISK_RED if v >= 0 else RISK_GREEN for v in plot_df["shap_value"]]
    ax.barh(plot_df[display_col], plot_df["shap_value"], color=colors, alpha=0.88)
    ax.axvline(0, color="#344054", linewidth=0.9)
    ax.set_xlabel("Local model contribution to fraud risk")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_fraud_score_distribution(scored_df):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.hist(scored_df["fraud_score"], bins=35, color=PRIMARY_BLUE, alpha=0.86, edgecolor="white", linewidth=0.5)
    ax.axvline(high_threshold, color=RISK_RED, linestyle="--", linewidth=1.6, label="High-risk threshold")
    ax.axvline(medium_threshold, color="#F59E0B", linestyle="--", linewidth=1.3, label="Medium-risk threshold")
    ax.set_xlabel("Predicted fraud score")
    ax.set_ylabel("Transaction count")
    ax.set_title("Distribution of predicted fraud risk", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_review_capacity_chart(review_perf_df, metric="Recall"):
    chart_df = review_perf_df.copy()
    chart_df[metric] = pd.to_numeric(chart_df[metric], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(chart_df["Review capacity"], chart_df[metric], color=PRIMARY_BLUE, alpha=0.86)
    ax.set_ylim(0, max(1.0, chart_df[metric].max(skipna=True) * 1.15 if chart_df[metric].notna().any() else 1.0))
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by review capacity", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def format_explanation_method(method):
    method_map = {
        "tree_shap": "SHAP TreeExplainer",
        "lightgbm_pred_contrib": "LightGBM contribution fallback",
        "local_delta_fallback": "Local sensitivity fallback",
    }
    return method_map.get(str(method), str(method))


apply_app_theme()


# ============================================================
# Helper functions
# ============================================================

def load_upload_artifacts_or_stop():
    try:
        model = cached_model()
        feature_engineer = cached_feature_engineer()
        selected_features = cached_selected_features()
        metadata = cached_metadata()
        return model, feature_engineer, selected_features, metadata
    except Exception as e:
        st.error("Could not load Top 500 model artifacts. Check artifacts/top_500_upload_model and src/feature_engineering.py.")
        st.exception(e)
        st.stop()


def load_manual_artifacts_or_stop():
    try:
        manual_model = cached_manual_model()
        manual_feature_engineer = cached_manual_feature_engineer()
        manual_model_features = cached_manual_model_features()
        manual_metadata = cached_manual_metadata()
        manual_schema = cached_manual_input_schema()
        manual_options = cached_manual_input_options()
        manual_performance = cached_manual_performance()
        manual_feature_importance = cached_manual_feature_importance()
        sample_manual_inputs = cached_sample_manual_inputs()
        return {
            "model": manual_model,
            "feature_engineer": manual_feature_engineer,
            "model_features": manual_model_features,
            "metadata": manual_metadata,
            "schema": manual_schema,
            "options": manual_options,
            "performance": manual_performance,
            "feature_importance": manual_feature_importance,
            "sample_inputs": sample_manual_inputs,
        }
    except Exception as e:
        st.error(
            "Could not load manual simulation artifacts. Check that this folder exists and contains the files saved by the manual training notebook: "
            "artifacts/manual_simulation_model/"
        )
        st.exception(e)
        st.stop()


def _clean_missing_value(value):
    if pd.isna(value):
        return "missing"
    value = str(value)
    if value.strip() == "":
        return "missing"
    return value


def _to_number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _get_schema_default(schema_row, sample_row=None):
    col = schema_row["input_column"]
    if sample_row is not None and col in sample_row.index:
        return sample_row[col]
    return schema_row.get("default_value", None)


def _is_missing_like(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "missing", "none", "nan", "<na>", "null"}


def _format_sample_value_for_state(column, value):
    if _is_missing_like(value):
        if column == "TransactionAmt":
            return ""
        return "missing"
    if column in {"TransactionHour", "TransactionDayOfWeek"}:
        return int(float(value))
    if column in {"missing_addr1", "missing_addr2", "missing_dist1", "has_identity_info"}:
        return int(float(value))
    if column == "TransactionAmt":
        return str(float(value)).rstrip("0").rstrip(".") if "." in str(float(value)) else str(float(value))
    return str(value)


def prefill_manual_form_from_sample(schema_df, options_dict, sample_row):
    """
    Push selected sample values into Streamlit session state.

    A checkbox alone does not reliably update already-created widgets because
    Streamlit widgets keep their previous state by key. This function updates
    the actual widget keys, then the app reruns.
    """
    if sample_row is None:
        return

    for _, row in schema_df.iterrows():
        column = row["input_column"]
        input_type = row.get("input_type", "text")
        value = sample_row[column] if column in sample_row.index else row.get("default_value", None)
        value = _format_sample_value_for_state(column, value)

        if input_type == "text_or_select":
            options = _get_options(options_dict, column, default_value=value)
            if value in options:
                st.session_state[f"manual_{column}_select"] = value
            else:
                st.session_state[f"manual_{column}_select"] = "Custom value..."
                st.session_state[f"manual_{column}_custom"] = value
        else:
            st.session_state[f"manual_{column}"] = value

    st.session_state["manual_sample_loaded"] = True


def clear_manual_form_state(schema_df):
    for _, row in schema_df.iterrows():
        column = row["input_column"]
        for key in [f"manual_{column}", f"manual_{column}_select", f"manual_{column}_custom"]:
            if key in st.session_state:
                del st.session_state[key]
    st.session_state["manual_sample_loaded"] = False


def validate_manual_form_values(form_values: dict) -> list[str]:
    """
    Validate only fields that genuinely cannot be missing.

    Most manual simulation fields can intentionally be Missing because missingness
    is itself a model signal. We therefore do not block scoring just because a
    dropdown is set to Missing.
    """
    errors = []

    amount = form_values.get("TransactionAmt")
    try:
        amount_float = float(amount)
        if amount_float <= 0:
            errors.append("Transaction amount must be greater than 0.")
    except Exception:
        errors.append("Transaction amount must be filled in as a number, e.g. 1000.")

    return errors


def _get_options(options_dict, column, default_value=None):
    raw_options = options_dict.get(column, []) if isinstance(options_dict, dict) else []
    clean_options = []
    for value in raw_options:
        value = _clean_missing_value(value)
        if value not in clean_options:
            clean_options.append(value)

    if default_value is not None:
        default_value = _clean_missing_value(default_value)
        if default_value not in clean_options:
            clean_options.insert(0, default_value)

    if "missing" not in clean_options:
        clean_options.append("missing")

    return clean_options


def _format_day(value):
    day_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    try:
        return f"{int(value)} — {day_map.get(int(value), 'Day ' + str(value))}"
    except Exception:
        return str(value)


def _format_yes_no(value, yes_label="Yes", no_label="No"):
    try:
        return yes_label if int(value) == 1 else no_label
    except Exception:
        return str(value)


def _friendly_bucket(value):
    return _humanize_token(value)


def _render_text_or_select(column, label, default_value, options, description, key):
    cleaned_default = _clean_missing_value(default_value)
    options = [x for x in options if str(x).strip() != ""]
    custom_option = "Custom value..."
    select_options = options.copy()

    if cleaned_default not in select_options:
        select_options.insert(0, cleaned_default)
    if custom_option not in select_options:
        select_options.append(custom_option)

    default_index = select_options.index(cleaned_default) if cleaned_default in select_options else 0

    selected = st.selectbox(
        label,
        options=select_options,
        index=default_index,
        format_func=lambda x, col=column: "Custom value..." if x == custom_option else _friendly_option_label(col, x),
        help=description,
        key=f"{key}_select",
    )

    if selected == custom_option:
        return st.text_input(
            f"Enter {label.lower()}",
            value="",
            key=f"{key}_custom",
        )

    return selected


def render_manual_form(schema_df, options_dict, sample_row=None):
    if schema_df.empty:
        st.error("Manual input_schema.csv is missing or empty.")
        st.stop()

    form_values = {}
    sections = schema_df["section"].dropna().unique().tolist()

    with st.form("manual_simulation_form"):
        st.markdown("### Scenario setup")
        st.caption("Use business-readable fields to test how different transaction conditions affect predicted fraud risk.")

        section_tabs = st.tabs(sections)

        for tab, section in zip(section_tabs, sections):
            with tab:
                section_df = schema_df[schema_df["section"] == section].copy()

                for _, row in section_df.iterrows():
                    column = row["input_column"]
                    label = FORM_LABEL_OVERRIDES.get(column, row.get("input_label", column))
                    input_type = row.get("input_type", "text")
                    description = FORM_DESCRIPTION_OVERRIDES.get(column, row.get("description", ""))
                    default_value = _get_schema_default(row, sample_row=sample_row)
                    key = f"manual_{column}"

                    if input_type == "number":
                        # For the amount field, use a required text input instead of
                        # a numeric widget with a silent default. This avoids scoring
                        # a scenario that the user did not really fill in.
                        if column == "TransactionAmt":
                            if sample_row is None and key not in st.session_state:
                                default_text = ""
                            else:
                                default_text = _clean_missing_value(default_value)
                                if default_text == "missing":
                                    default_text = ""
                            form_values[column] = st.text_input(
                                label + " *",
                                value=default_text,
                                placeholder="e.g. 1000",
                                help=description,
                                key=key,
                            )
                        else:
                            form_values[column] = st.number_input(
                                label,
                                value=float(_to_number(default_value, 0.0)),
                                min_value=0.0,
                                step=1.0,
                                help=description,
                                key=key,
                            )

                    elif input_type == "slider":
                        if column == "TransactionHour":
                            form_values[column] = st.slider(
                                label,
                                min_value=0,
                                max_value=23,
                                value=_to_int(default_value, 12),
                                help=description,
                                key=key,
                            )
                        else:
                            form_values[column] = st.slider(
                                label,
                                min_value=0,
                                max_value=100,
                                value=_to_int(default_value, 0),
                                help=description,
                                key=key,
                            )

                    elif input_type == "select":
                        if column == "TransactionDayOfWeek":
                            select_options = list(range(7))
                            default_int = _to_int(default_value, 2)
                            default_index = select_options.index(default_int) if default_int in select_options else 2
                            form_values[column] = st.selectbox(
                                label,
                                options=select_options,
                                index=default_index,
                                format_func=_format_day,
                                help=description,
                                key=key,
                            )

                        elif column in {"missing_addr1", "missing_addr2", "missing_dist1"}:
                            select_options = [0, 1]
                            default_int = _to_int(default_value, 0)
                            default_index = select_options.index(default_int) if default_int in select_options else 0
                            form_values[column] = st.selectbox(
                                label,
                                options=select_options,
                                index=default_index,
                                format_func=lambda x, col=column: _friendly_option_label(col, x),
                                help=description,
                                key=key,
                            )

                        elif column == "has_identity_info":
                            select_options = [0, 1]
                            default_int = _to_int(default_value, 1)
                            default_index = select_options.index(default_int) if default_int in select_options else 1
                            form_values[column] = st.selectbox(
                                label,
                                options=select_options,
                                index=default_index,
                                format_func=lambda x, col=column: _friendly_option_label(col, x),
                                help=description,
                                key=key,
                            )

                        else:
                            options = _get_options(options_dict, column, default_value=default_value)
                            default_clean = _clean_missing_value(default_value)
                            default_index = options.index(default_clean) if default_clean in options else 0
                            form_values[column] = st.selectbox(
                                label,
                                options=options,
                                index=default_index,
                                format_func=lambda x, col=column: _friendly_option_label(col, x),
                                help=description,
                                key=key,
                            )

                    elif input_type == "text_or_select":
                        options = _get_options(options_dict, column, default_value=default_value)
                        form_values[column] = _render_text_or_select(
                            column=column,
                            label=label,
                            default_value=default_value,
                            options=options,
                            description=description,
                            key=key,
                        )

                    else:
                        form_values[column] = st.text_input(
                            label,
                            value=_clean_missing_value(default_value),
                            help=description,
                            key=key,
                        )

        submitted = st.form_submit_button("Score scenario", type="primary")

    return submitted, form_values


def build_generated_feature_table(X_model, generated_all, feature_engineer):
    display_map = get_display_map(feature_engineer)

    records = []
    generated_row = generated_all.iloc[0]
    model_row = X_model.iloc[0]

    for feature in X_model.columns:
        value = model_row.get(feature)
        raw_generated_value = generated_row.get(feature, value)
        display_name = display_map.get(feature, _humanize_token(feature))
        records.append({
            "feature": feature,
            "display_name": display_name,
            "value": raw_generated_value,
            "business_value": _format_business_value(feature, raw_generated_value),
            "explanation": _feature_explanation(feature, display_name),
            "source": "User input" if feature in MANUAL_DIRECT_INPUT_COLUMNS else "Generated feature",
        })

    feature_df = pd.DataFrame(records)
    feature_df["priority"] = feature_df["feature"].apply(
        lambda x: MANUAL_GENERATED_FEATURE_PRIORITY.index(x)
        if x in MANUAL_GENERATED_FEATURE_PRIORITY
        else 999
    )

    return feature_df.sort_values(["source", "priority", "feature"]).drop(columns=["priority"])


def _feature_value(feature_table, feature, default="n/a"):
    if feature_table is None or feature_table.empty:
        return default
    match = feature_table[feature_table["feature"] == feature]
    if match.empty:
        return default
    row = match.iloc[0]
    if "business_value" in row:
        return row["business_value"]
    return _format_business_value(feature, row.get("value", default))


def _feature_display_name(feature_table, feature):
    match = feature_table[feature_table["feature"] == feature] if feature_table is not None else pd.DataFrame()
    if not match.empty:
        return match.iloc[0].get("display_name", feature)
    return feature


def _render_feature_chips(feature_table, features):
    render_feature_cards(feature_table, features, columns=2)


def render_generated_feature_explainers(feature_table, form_values):
    """
    Present generated features as grouped explanations instead of one large table.
    This is designed to answer: user filled X, so the app generated Y features.
    """
    st.caption("The app converts the business inputs into model-ready signals. Each card shows the generated signal, its interpreted value, and why it matters.")

    with st.expander("Transaction amount features", expanded=True):
        st.markdown(
            f"You entered **Transaction amount = {form_values.get('TransactionAmt', 'n/a')}**. "
            "The app then creates amount pattern features used by the model."
        )
        _render_feature_chips(
            feature_table,
            [
                "TransactionAmt_log",
                "amount_bucket",
                "amount_decimal",
                "amount_cents",
                "is_round_amount",
                "amount_decimal_places",
                "has_3_decimal_amount",
                "is_high_amount",
            ],
        )

    with st.expander("Timing features", expanded=False):
        st.markdown(
            f"You entered **hour = {form_values.get('TransactionHour', 'n/a')}** and "
            f"**day of week = {form_values.get('TransactionDayOfWeek', 'n/a')}**. "
            "The app turns these into timing risk signals."
        )
        _render_feature_chips(
            feature_table,
            [
                "time_of_day_bucket",
                "is_business_hours",
                "is_night",
                "is_weekend",
                "high_amount_off_hours",
                "night_transaction_missing_identity",
                "night_transaction_large_distance",
            ],
        )

    with st.expander("Email features", expanded=False):
        st.markdown(
            f"You entered **Purchaser email domain = {form_values.get('P_emaildomain', 'missing')}** and "
            f"**Recipient email domain = {form_values.get('R_emaildomain', 'missing')}**. "
            "The app derives provider, suffix, missingness, match/mismatch, and familiarity signals."
        )
        _render_feature_chips(
            feature_table,
            [
                "missing_P_email",
                "missing_R_email",
                "P_email_provider",
                "P_email_suffix",
                "R_email_provider",
                "R_email_suffix",
                "emails_match",
                "same_email_provider",
                "same_email_suffix",
                "email_domain_mismatch",
                "email_relationship_status",
                "email_domain_familiarity_bucket",
                "rare_email_domain_flag",
                "high_amount_rare_email_domain",
                "email_mismatch_proxy_flag",
            ],
        )

    with st.expander("Device, browser, and identity features", expanded=False):
        st.markdown(
            f"You entered **Device type = {form_values.get('DeviceType', 'missing')}**, "
            f"**Device info = {form_values.get('DeviceInfo', 'missing')}**, "
            f"**OS = {form_values.get('id_30', 'missing')}**, and "
            f"**Browser = {form_values.get('id_31', 'missing')}**. "
            "The app creates device family, browser/OS, screen, proxy, and identity completeness signals."
        )
        _render_feature_chips(
            feature_table,
            [
                "DeviceType",
                "DeviceInfo_brand",
                "device_missing",
                "is_mobile_device",
                "is_desktop_device",
                "device_familiarity_bucket",
                "rare_device_flag",
                "id_30_os_family",
                "id_30_os_version_available",
                "id_31_browser_family",
                "id_31_browser_version_available",
                "os_browser_combo",
                "network_proxy_type",
                "id_23_proxy_flag",
                "id_23_anonymous_flag",
                "id_23_transparent_flag",
                "id_23_hidden_flag",
                "identity_missing_count",
                "identity_completeness_bucket",
                "screen_size_bucket",
                "unusual_screen_ratio",
            ],
        )

    with st.expander("Address and distance/location features", expanded=False):
        st.markdown(
            f"You selected **Address available = {_friendly_option_label('missing_addr1', form_values.get('missing_addr1', 'n/a'))}**, "
            f"**Country code available = {_friendly_option_label('missing_addr2', form_values.get('missing_addr2', 'n/a'))}**, "
            f"**Distance signal available = {_friendly_option_label('missing_dist1', form_values.get('missing_dist1', 'n/a'))}**, and "
            f"**Distance/location profile = {_friendly_option_label('distance_signal_bucket', form_values.get('distance_signal_bucket', 'missing'))}**. "
            "The app converts these into availability and distance risk indicators."
        )
        _render_feature_chips(
            feature_table,
            [
                "missing_addr1",
                "missing_addr2",
                "address_signal_available",
                "missing_dist1",
                "distance_signal_bucket",
                "is_large_dist1",
                "addr_missing_count",
                "address_distance_status",
                "missing_address_and_distance",
                "high_amount_large_distance",
            ],
        )

    with st.expander("Payment identifier familiarity features", expanded=False):
        st.markdown(
            f"You selected **Payment identifier familiarity = {form_values.get('payment_identifier_familiarity_bucket', 'missing')}**. "
            "This is a business-safe proxy for how familiar the masked payment pattern is; the app does not ask for raw card1."
        )
        _render_feature_chips(
            feature_table,
            [
                "payment_identifier_familiarity_bucket",
                "rare_payment_identifier_flag",
                "common_payment_identifier_flag",
                "high_amount_rare_payment_identifier",
                "email_mismatch_rare_payment_identifier",
                "rare_payment_identifier_proxy_flag",
                "rare_payment_identifier_large_distance",
            ],
        )

    with st.expander("Combined risk flags and model interactions", expanded=False):
        st.markdown(
            "These features combine multiple simple signals, for example high amount + proxy signal, "
            "or email mismatch + rare payment pattern."
        )
        _render_feature_chips(
            feature_table,
            [
                "high_amount_missing_identity",
                "high_amount_missing_email",
                "high_amount_proxy_flag",
                "high_amount_rare_device",
                "email_mismatch_missing_identity",
                "email_mismatch_proxy_flag",
                "proxy_and_rare_device",
                "high_amount_ProductCD",
                "night_ProductCD",
                "email_mismatch_ProductCD",
                "distance_bucket_ProductCD",
                "proxy_ProductCD",
                "amount_bucket_ProductCD",
                "payment_familiarity_ProductCD",
                "device_familiarity_ProductCD",
                "identity_completeness_ProductCD",
            ],
        )

def _friendly_shap_display_name(feature, display_map, known_features):
    feature = str(feature)

    # Sometimes sklearn feature names include a transformer prefix like cat__feature_value.
    if "__" in feature:
        feature = feature.split("__", 1)[1]

    # Direct match: normal model feature name.
    if feature in display_map:
        return display_map[feature]

    # One-hot / encoded match:
    # Example: email_relationship_status_recipient_missing
    # Should become: Email Relationship Status = Recipient email missing
    best_prefix = None

    for candidate in sorted(known_features, key=len, reverse=True):
        candidate = str(candidate)
        if not candidate:
            continue

        if feature.startswith(candidate + "_"):
            best_prefix = candidate
            break

    if best_prefix is not None:
        category = feature[len(best_prefix) + 1:]
        label = display_map.get(best_prefix, _humanize_token(best_prefix))
        return f"{label} = {_humanize_token(category)}"

    # Last fallback: never show raw snake_case if we can avoid it.
    return _humanize_token(feature)


def apply_display_names(shap_df, feature_engineer, X_model=None, feature_table=None):
    """
    Apply the same business-friendly naming logic used by the generated feature explanations
    to SHAP outputs.

    This handles:
    1. Direct model features, e.g. is_high_amount
    2. Generated features from the feature engineer display map
    3. App-level overrides
    4. One-hot encoded SHAP features, e.g. email_relationship_status_recipient_missing
    """
    display_map = get_display_map(feature_engineer)

    # Reuse the exact display names already built for the generated feature explanation cards.
    if feature_table is not None and not feature_table.empty:
        try:
            feature_table_map = dict(
                zip(feature_table["feature"].astype(str), feature_table["display_name"].astype(str))
            )
            display_map.update(feature_table_map)
        except Exception:
            pass

    known_features = (
        set(display_map.keys())
        | set(MANUAL_DIRECT_INPUT_COLUMNS)
        | set(MANUAL_GENERATED_FEATURE_PRIORITY)
    )

    # Add the actual model input columns, which is the most reliable source for scenario SHAP.
    if X_model is not None:
        try:
            known_features.update([str(c) for c in X_model.columns])
        except Exception:
            pass

    # Add feature-engineer attributes if available.
    if feature_engineer is not None:
        for attr in ["feature_cols_", "MODEL_FEATURES"]:
            if hasattr(feature_engineer, attr):
                try:
                    known_features.update([str(c) for c in getattr(feature_engineer, attr)])
                except Exception:
                    pass

    shap_df = shap_df.copy()
    shap_df["display_name"] = shap_df["feature"].apply(
        lambda x: _friendly_shap_display_name(x, display_map, known_features)
    )

    return shap_df


# ============================================================
# Load upload artifacts at app start because existing pages need them
# ============================================================

model, feature_engineer, selected_features, metadata = load_upload_artifacts_or_stop()
performance_df = cached_performance()
sample_upload_df = cached_sample_upload()
high_threshold, medium_threshold = get_thresholds(metadata)


# ============================================================
# Sidebar navigation
# ============================================================

st.sidebar.title("Fraud Risk Assistant")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Upload & Prioritize",
        "Scenario Simulation",
        "Model Performance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Upload scoring: Top 500 LightGBM model")
st.sidebar.caption("Scenario simulation: simplified business-input model")


# ============================================================
# Executive overview
# ============================================================

if page == "Executive Overview":
    st.title("Fraud Risk Analytics Demo")

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">From transaction data to investigation priorities</div>
            <div class="hero-subtitle">
                This demo shows how predictive analytics can help finance, audit, and operations teams move
                from large transaction populations to a focused review queue, while still keeping the model
                explanation understandable for business stakeholders.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary use case", "Prioritize review")
    c2.metric("Upload model features", f"{int(metadata.get('n_features', len(selected_features))):,}")
    c3.metric("High-risk threshold", format_pct(high_threshold))

    st.subheader("What executives should take away")
    takeaway_cols = st.columns(3)
    with takeaway_cols[0]:
        _html_card(
            "Focus scarce review time",
            "Rank transactions by predicted fraud risk so investigators start with the cases most likely to matter.",
            eyebrow="Operational prioritization",
        )
    with takeaway_cols[1]:
        _html_card(
            "Explain the risk drivers",
            "Show the transaction-level factors that increased or reduced risk, using red/green visual explanations.",
            eyebrow="Diagnostic analytics",
        )
    with takeaway_cols[2]:
        _html_card(
            "Test business scenarios",
            "Change transaction conditions manually to demonstrate how model-ready features are generated and scored.",
            eyebrow="Scenario simulation",
        )

    st.subheader("Demo workflow")
    wf1, wf2, wf3, wf4 = st.columns(4)
    with wf1:
        _html_card("1. Ingest", "Upload a transaction file or use the sample dataset.", eyebrow="Step 1")
    with wf2:
        _html_card("2. Score", "Generate model features and calculate fraud-risk scores.", eyebrow="Step 2")
    with wf3:
        _html_card("3. Prioritize", "Create a review queue based on available investigation capacity.", eyebrow="Step 3")
    with wf4:
        _html_card("4. Explain", "Inspect the key drivers behind an individual transaction or scenario.", eyebrow="Step 4")

    st.subheader("How to use this demo")
    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown(
            """
            - Use **Upload & Prioritize** to demonstrate the operational workflow: file upload, risk scoring, review capacity, and transaction inspection.
            - Use **Scenario Simulation** to teach the business logic: how high-level transaction inputs become model features, risk scores, and explanations.
            - Use **Model Performance** to discuss validation results and why review-capacity metrics are more useful than accuracy for fraud detection.
            """
        )
    with col_right:
        st.info(
            "This app supports investigation prioritization and analytics education. It should not be positioned as an automated approve/reject decision engine."
        )


# ============================================================
# Upload page
# ============================================================

elif page == "Upload & Prioritize":
    st.title("Upload & Prioritize Transaction Review")

    st.markdown(
        """
        Upload a transaction CSV to create a risk-ranked review queue. This workflow is designed to show
        how analytics can help investigation teams focus on the highest-risk transactions first. Extra
        columns are allowed and ignored for prediction.
        """
    )

    use_sample = st.checkbox(
        "Use sample upload file",
        value=False,
        disabled=sample_upload_df.empty,
    )

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if use_sample:
        raw_df = sample_upload_df.copy()
    elif uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = None

    if raw_df is not None:
        st.subheader("Input preview")
        st.dataframe(raw_df.head(), use_container_width=True)

        with st.spinner("Creating Top 500 features and scoring transactions..."):
            scored_df, X_model, validation_info = score_uploaded_transactions(
                raw_df=raw_df,
                model=model,
                feature_engineer=feature_engineer,
                selected_features=selected_features,
                metadata=metadata,
            )

        missing_core = validation_info.get("missing_core_columns", [])
        if missing_core:
            st.warning(
                "The upload is missing some important source columns. The app still scored the file, "
                "but predictions may be less reliable. Missing columns: "
                + ", ".join(missing_core)
            )

        st.subheader("Scoring summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions scored", f"{len(scored_df):,}")
        col2.metric("Highest fraud score", format_pct(scored_df["fraud_score"].max()))
        col3.metric("High-risk cases", f"{(scored_df['risk_band'] == 'High').sum():,}")
        col4.metric("Model input features", f"{X_model.shape[1]:,}")

        st.subheader("Fraud score distribution")
        render_fraud_score_distribution(scored_df)

        st.subheader("Fraud review queue")
        review_pct = st.slider(
            "Review capacity (% of uploaded transactions)",
            min_value=1,
            max_value=20,
            value=5,
        )

        review_queue = get_review_queue(scored_df, review_pct=review_pct)
        display_cols = existing_columns(review_queue, REVIEW_DISPLAY_COLUMNS)

        st.info(
            f"Reviewing the top {review_pct}% means prioritizing {len(review_queue):,} transactions."
        )

        st.dataframe(
            review_queue[display_cols],
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Download review queue",
                data=review_queue.to_csv(index=False).encode("utf-8"),
                file_name="fraud_review_queue.csv",
                mime="text/csv",
            )
        with col_b:
            st.download_button(
                "Download all scored transactions",
                data=scored_df.to_csv(index=False).encode("utf-8"),
                file_name="fraud_scored_transactions.csv",
                mime="text/csv",
            )

        st.subheader("Inspect a transaction")
        if not review_queue.empty:
            selected_idx = st.selectbox(
                "Select a ranked transaction",
                options=review_queue.index.tolist(),
                format_func=lambda i: (
                    f"Rank {int(review_queue.loc[i, 'fraud_rank'])} | "
                    f"Score {review_queue.loc[i, 'fraud_score']:.1%}"
                ),
            )

            selected_row = review_queue.loc[selected_idx]

            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud score", format_pct(selected_row["fraud_score"]))
            m2.metric("Risk band", selected_row["risk_band"])
            m3.metric("Recommended action", selected_row["recommended_action"])

            with st.expander("Raw transaction details", expanded=False):
                st.write(selected_row.drop(labels=["_row_position"], errors="ignore"))

            st.subheader(EXPLANATION_SECTION_TITLE)
            st.caption(
                "Shows which transaction-level signals pushed the predicted fraud score up or down. "
                "Red drivers increase predicted risk; green drivers reduce predicted risk."
            )

            if st.button("Generate explanation for this transaction"):
                try:
                    with st.spinner("Generating SHAP explanation for this transaction..."):
                        shap_df = explain_single_transaction(
                            pipeline=model,
                            X_model=X_model,
                            row_position=int(selected_row["_row_position"]),
                            top_n=15,
                        )
                        
                    shap_df = apply_display_names(shap_df, feature_engineer, X_model=X_model,)

                    st.success("Explanation generated.")
                    if "explanation_method" in shap_df.columns and not shap_df.empty:
                        st.caption(f"Explanation method used: `{format_explanation_method(shap_df['explanation_method'].iloc[0])}`")

                    render_shap_bar_chart(shap_df, title="Top transaction-level risk drivers")
                    render_shap_driver_cards(shap_df, max_cards=10)

                    with st.expander("Technical view: explanation values", expanded=False):
                        st.dataframe(
                            shap_df[["display_name", "feature", "model_input_value", "shap_value", "impact_direction"]],
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as e:
                    st.error("Could not generate SHAP explanation for this transaction.")
                    st.exception(e)


# ============================================================
# Manual simulation page
# ============================================================

elif page == "Scenario Simulation":
    st.title("Scenario Simulation")

    st.markdown(
        """
        Build a single transaction scenario using business-readable inputs. The app translates those inputs
        into model-ready signals, scores the scenario, and explains the drivers behind the predicted risk.
        """
    )

    manual_artifacts = load_manual_artifacts_or_stop()
    manual_model = manual_artifacts["model"]
    manual_feature_engineer = manual_artifacts["feature_engineer"]
    manual_model_features = manual_artifacts["model_features"]
    manual_metadata = manual_artifacts["metadata"]
    manual_schema = manual_artifacts["schema"]
    manual_options = manual_artifacts["options"]
    sample_manual_inputs = manual_artifacts["sample_inputs"]

    mh, mm = get_thresholds(manual_metadata)

    c1, c2, c3 = st.columns(3)
    c1.metric("Manual model", manual_metadata.get("model_name", "manual_simulation_model"))
    c2.metric("Model features", f"{len(manual_model_features):,}")
    c3.metric("High-risk threshold", format_pct(mh))

    st.info(
        "This scenario model is intentionally simpler than the upload model. Use it to explain concepts and test scenarios, not as the primary operational detector."
    )

    # ------------------------------------------------------------------
    # Important Streamlit behavior:
    # Buttons cause the script to rerun. Therefore, the scored manual
    # scenario must be stored in session_state. Otherwise, clicking the
    # explanation button makes the scored-result block disappear because
    # the form submit button is no longer True on the rerun.
    # ------------------------------------------------------------------
    if "manual_last_result" not in st.session_state:
        st.session_state["manual_last_result"] = None
    if "manual_explanation_debug" not in st.session_state:
        st.session_state["manual_explanation_debug"] = []
    if "manual_last_explanation_error" not in st.session_state:
        st.session_state["manual_last_explanation_error"] = None

    def _debug_log(message):
        st.session_state.setdefault("manual_explanation_debug", [])
        st.session_state["manual_explanation_debug"].append(
            f"{time.strftime('%H:%M:%S')} | {message}"
        )

    sample_row = None
    with st.expander("Optional: start from a sample validation scenario", expanded=False):
        if sample_manual_inputs.empty:
            st.caption("sample_manual_inputs.csv was not found, so the form will start blank/defaulted.")
        else:
            sample_idx = st.selectbox(
                "Choose sample row",
                options=list(range(len(sample_manual_inputs))),
                format_func=lambda i: (
                    f"Sample {i}"
                    + (f" | Amount {sample_manual_inputs.loc[i, 'TransactionAmt']}" if "TransactionAmt" in sample_manual_inputs.columns else "")
                    + (f" | Actual isFraud {sample_manual_inputs.loc[i, 'isFraud']}" if "isFraud" in sample_manual_inputs.columns else "")
                ),
                key="manual_sample_idx",
            )
            sample_row = sample_manual_inputs.loc[sample_idx]

            if st.button("Prefill form with selected sample", use_container_width=True):
                prefill_manual_form_from_sample(
                    schema_df=manual_schema,
                    options_dict=manual_options,
                    sample_row=sample_row,
                )
                st.rerun()

            st.caption(
                "This fills the form with a real validation scenario. You can still edit any field before scoring."
            )

    submitted, form_values = render_manual_form(
        schema_df=manual_schema,
        options_dict=manual_options,
        sample_row=None,
    )

    if submitted:
        validation_errors = validate_manual_form_values(form_values)
        if validation_errors:
            st.error("Please fix the transaction amount before scoring.")
            for err in validation_errors:
                st.markdown(f"- {err}")
            st.stop()

        try:
            st.session_state["manual_explanation_debug"] = []
            st.session_state["manual_last_explanation_error"] = None
            _debug_log("Scoring submitted manual scenario.")

            with st.spinner("Generating manual features and scoring scenario..."):
                scored_manual_df, X_manual, generated_all, manual_result = score_manual_transaction(
                    form_values=form_values,
                    model=manual_model,
                    feature_engineer=manual_feature_engineer,
                    model_features=manual_model_features,
                    metadata=manual_metadata,
                )

            _debug_log(f"Manual scoring succeeded. X_manual shape = {X_manual.shape}.")
            _debug_log(f"Generated feature dataframe shape = {generated_all.shape}.")

            st.session_state["manual_last_result"] = {
                "scored_manual_df": scored_manual_df,
                "X_manual": X_manual,
                "generated_all": generated_all,
                "manual_result": manual_result,
                "form_values": form_values,
            }

        except Exception as e:
            st.session_state["manual_last_result"] = None
            st.session_state["manual_last_explanation_error"] = traceback.format_exc()
            st.error("Could not score the manual scenario.")
            st.exception(e)

    result_state = st.session_state.get("manual_last_result")

    if result_state is not None:
        scored_manual_df = result_state["scored_manual_df"]
        X_manual = result_state["X_manual"]
        generated_all = result_state["generated_all"]
        manual_result = result_state["manual_result"]
        form_values = result_state["form_values"]

        st.subheader("Simulation result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fraud score", format_pct(manual_result["fraud_score"]))
        r2.metric("Risk band", manual_result["risk_band"])
        r3.metric("Recommended action", manual_result["recommended_action"])
        r4.metric("Model features generated", f"{manual_result['n_model_features']:,}")

        st.subheader("Input scenario")
        st.dataframe(scored_manual_df, use_container_width=True)

        st.subheader("Generated model-ready signals")
        feature_table = build_generated_feature_table(
            X_model=X_manual,
            generated_all=generated_all,
            feature_engineer=manual_feature_engineer,
        )

        render_generated_feature_explainers(
            feature_table=feature_table,
            form_values=form_values,
        )

        with st.expander("Technical view: all final model features", expanded=False):
            st.caption("This is the exact one-row model feature matrix before preprocessing/encoding.")
            st.dataframe(
                feature_table[["display_name", "feature", "business_value", "value", "source", "explanation"]],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(EXPLANATION_SECTION_TITLE)
        st.caption(
            "Shows which generated model signals pushed this scenario's fraud score up or down. "
            "Red drivers increase predicted risk; green drivers reduce predicted risk. "
            "The app uses SHAP when available and falls back to a local sensitivity explanation if SHAP fails."
        )

        col_btn, col_note = st.columns([1, 2])
        with col_btn:
            generate_explanation = st.button(
                "Generate explanation for this scenario",
                type="secondary",
                use_container_width=True,
                key="manual_generate_explanation_button",
            )

        if generate_explanation:
            st.session_state["manual_explanation_debug"] = []
            st.session_state["manual_last_explanation_error"] = None

            try:
                _debug_log("Explanation button clicked.")
                _debug_log(f"Manual model type: {type(manual_model)}")
                _debug_log(f"X_manual shape: {X_manual.shape}")
                _debug_log(f"X_manual columns: {list(X_manual.columns)[:10]} ... total={len(X_manual.columns)}")

                with st.spinner("Generating explanation for this scenario..."):
                    _debug_log("Calling explain_single_transaction().")
                    shap_df = explain_single_transaction(
                        pipeline=manual_model,
                        X_model=X_manual,
                        row_position=0,
                        top_n=15,
                    )
                    _debug_log(f"Explanation returned dataframe shape: {shap_df.shape}.")
                    if "explanation_method" in shap_df.columns:
                        _debug_log(f"Explanation method: {shap_df['explanation_method'].iloc[0]}")

                    shap_df = apply_display_names(shap_df, manual_feature_engineer, X_model=X_manual, feature_table=feature_table,)
                    
                    _debug_log("Display names applied successfully.")

                st.success("Explanation generated.")

                method_label = "unknown"
                if "explanation_method" in shap_df.columns and not shap_df.empty:
                    method_label = format_explanation_method(shap_df["explanation_method"].iloc[0])
                st.caption(f"Explanation method used: `{method_label}`")

                render_shap_bar_chart(shap_df, title="Top scenario-level risk drivers")
                render_shap_driver_cards(shap_df, max_cards=10)

                with st.expander("Technical view: explanation values", expanded=False):
                    st.dataframe(
                        shap_df[["display_name", "feature", "model_input_value", "shap_value", "impact_direction"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as e:
                st.session_state["manual_last_explanation_error"] = traceback.format_exc()
                _debug_log(f"Explanation failed: {repr(e)}")
                st.error("Could not generate an explanation for this manual scenario.")
                st.exception(e)

    else:
        st.caption("Score a manual scenario to see the prediction, generated features, and explanation controls.")


# ============================================================
# Model performance page
# ============================================================

elif page == "Model Performance":
    st.title("Model Performance")

    MODEL_INSIGHTS_DIR = Path("artifacts") / "model_insights"

    def get_model_insight_path(model_folder: str, file_name: str):
        path = MODEL_INSIGHTS_DIR / model_folder / file_name
        return path if path.exists() else None

    def render_saved_model_insights(
        model_folder: str,
        model_label: str,
        shap_file_name: str = "global_shap_summary.png",
        importance_file_name: str = "feature_importance_gain.png",
    ):
        st.subheader("Global model explanation")

        st.markdown(
            """
            These plots explain how the model behaves across a representative set of transactions,
            rather than explaining only one selected transaction.
            """
        )

        shap_path = get_model_insight_path(model_folder, shap_file_name)
        importance_path = get_model_insight_path(model_folder, importance_file_name)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Global SHAP summary")

            if shap_path is not None:
                st.image(str(shap_path), use_container_width=True)
                st.caption(
                    "Each point represents a transaction or scenario. Points to the right increase predicted fraud risk; "
                    "points to the left reduce predicted fraud risk. Color shows whether the feature value is high or low."
                )
            else:
                st.info(
                    f"No saved Global SHAP image found for {model_label}. "
                    "Run the model insights notebook first."
                )

        with col2:
            st.markdown("#### Feature importance")

            if importance_path is not None:
                st.image(str(importance_path), use_container_width=True)
                st.caption(
                    "Shows which signals the model relied on most overall, based on LightGBM gain importance."
                )
            else:
                st.info(
                    f"No saved feature importance image found for {model_label}. "
                    "Run the model insights notebook first."
                )

    tab_upload, tab_manual = st.tabs(["Upload model", "Manual simulation model"])

    with tab_upload:
        st.markdown(
            """
            This page summarizes the saved validation results for the selected Top 500 upload model.
            It shows how well the model separates higher-risk from lower-risk transactions and how much
            fraud could be captured at different review-capacity levels.
            """
        )

        if performance_df.empty:
            st.warning("performance.csv not found in artifacts/top_500_upload_model.")
        else:
            row = performance_df.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC", f"{row.get('roc_auc', 0):.3f}")
            c2.metric("PR-AUC", f"{row.get('pr_auc', 0):.3f}")
            c3.metric("Best threshold", format_pct(row.get("best_threshold", high_threshold)))
            c4.metric("Features", f"{int(row.get('n_features', len(selected_features))):,}")

            st.subheader("Review capacity performance")

            st.markdown(
                """
                This shows the trade-off between review effort and fraud detection.
                For example, reviewing only the highest-risk 5% of transactions may still capture
                a meaningful share of total fraud.
                """
            )

            review_rows = []
            for pct in [1, 3, 5, 10]:
                review_rows.append({
                    "Review capacity": f"Top {pct}%",
                    "Review count": row.get(f"top_{pct}pct_review_count"),
                    "Fraud caught": row.get(f"top_{pct}pct_fraud_caught"),
                    "Recall": row.get(f"top_{pct}pct_recall"),
                    "Precision": row.get(f"top_{pct}pct_precision"),
                })

            review_perf_df = pd.DataFrame(review_rows)
            st.dataframe(review_perf_df, use_container_width=True)

            render_review_capacity_chart(review_perf_df, metric="Recall")

            st.divider()

            render_saved_model_insights(
                model_folder="upload_model",
                model_label="Upload Model",
                shap_file_name="global_shap_summary.png",
                importance_file_name="feature_importance_gain.png",
            )

            with st.expander("Full performance file"):
                st.dataframe(performance_df, use_container_width=True)

    with tab_manual:
        st.markdown(
            """
            This page summarizes the saved validation results for the manual simulation model.
            This model powers the executive-friendly scenario tool, where users can adjust business inputs
            and see how the predicted risk changes.
            """
        )

        try:
            manual_performance_df = cached_manual_performance()
            manual_feature_importance = cached_manual_feature_importance()
            manual_metadata = cached_manual_metadata()
        except Exception as e:
            st.warning("Manual simulation artifacts were not found yet.")
            st.exception(e)
        else:
            if manual_performance_df.empty:
                st.warning("performance.csv not found in artifacts/manual_simulation_model.")
            else:
                row = manual_performance_df.iloc[0]
                mh, _ = get_thresholds(manual_metadata)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ROC-AUC", f"{row.get('roc_auc', 0):.3f}")
                c2.metric("PR-AUC", f"{row.get('pr_auc', 0):.3f}")
                c3.metric("Best threshold", format_pct(row.get("best_threshold", mh)))
                c4.metric("Features", f"{int(row.get('n_features', 0)):,}")

                st.subheader("Review capacity performance")

                st.markdown(
                    """
                    This shows how much fraud the manual simulation model can identify when only a limited
                    share of transactions can be reviewed.
                    """
                )

                review_rows = []
                for pct in [1, 3, 5, 10]:
                    review_rows.append({
                        "Review capacity": f"Top {pct}%",
                        "Review count": row.get(f"top_{pct}pct_review_count"),
                        "Fraud caught": row.get(f"top_{pct}pct_fraud_caught"),
                        "Recall": row.get(f"top_{pct}pct_recall"),
                        "Precision": row.get(f"top_{pct}pct_precision"),
                    })

                review_perf_df = pd.DataFrame(review_rows)
                st.dataframe(review_perf_df, use_container_width=True)

                render_review_capacity_chart(review_perf_df, metric="Recall")

                st.divider()

                render_saved_model_insights(
                    model_folder="manual_model",
                    model_label="Manual Simulation Model",
                    shap_file_name="global_shap_summary_business_names.png",
                    importance_file_name="feature_importance_gain_business_names.png",
                )

                with st.expander("Full manual performance file"):
                    st.dataframe(manual_performance_df, use_container_width=True)

            if not manual_feature_importance.empty:
                with st.expander("Technical manual model feature importance table"):
                    st.dataframe(manual_feature_importance.head(30), use_container_width=True)
