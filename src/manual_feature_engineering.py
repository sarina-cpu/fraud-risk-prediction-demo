import re
import numpy as np
import pandas as pd


class ManualSimulationFeatureEngineer:
    """
    Feature engineer for the manual fraud simulation model.

    Design principles:
    - The manual app should ask users for business-understandable inputs.
    - The model can still use generated features, but those features should come
      from inputs that a user can reasonably understand or select.
    - Avoid anonymous V/C/D fields and unclear identity fields.
    - Avoid raw masked address/distance codes as final model features.
    - Use address/distance only through generated availability and bucket features.

    Intended source/input columns:
    - TransactionID
    - isFraud
    - TransactionDT or TransactionHour + TransactionDayOfWeek
    - TransactionAmt
    - ProductCD
    - card4
    - card6
    - P_emaildomain
    - R_emaildomain
    - DeviceType
    - DeviceInfo
    - id_23   # proxy/network signal
    - id_30   # OS / OS version string
    - id_31   # browser / browser version string
    - id_33   # screen resolution string
    - addr1   # used only to generate missingness
    - addr2   # used only to generate missingness
    - dist1   # used only to generate distance bucket / large distance flag
    - payment_identifier_familiarity_bucket  # form-style familiarity proxy derived from card1 in training

    Important:
    - addr1, addr2, dist1 are NOT kept as final model features.
    - id_12, id_15, id_27, id_28, id_29 are NOT used because their business
      meaning is unclear for a manual simulator.
    """

    # Final model features intentionally kept compact and interpretable.
    # These are the columns returned by transform(..., return_features_only=True).
    MODEL_FEATURES = [
        # ----------------------------------------------------
        # Direct / simple business inputs
        # ----------------------------------------------------
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "DeviceType",

        # ----------------------------------------------------
        # Amount features
        # ----------------------------------------------------
        "TransactionAmt_log",
        "amount_decimal",
        "amount_cents",
        "is_round_amount",
        "amount_decimal_places",
        "has_3_decimal_amount",
        "is_high_amount",
        "amount_bucket",

        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------
        "TransactionHour",
        "TransactionDayOfWeek",
        "is_weekend",
        "is_business_hours",
        "is_night",
        "time_of_day_bucket",

        # ----------------------------------------------------
        # Email features
        # ----------------------------------------------------
        "missing_P_email",
        "missing_R_email",
        "P_email_provider",
        "P_email_suffix",
        "R_email_provider",
        "R_email_suffix",
        "emails_match",
        "same_email_provider",
        "same_email_suffix",
        "both_emails_missing",
        "only_recipient_email_missing",
        "email_domain_mismatch",
        "email_relationship_status",
        "rare_email_domain_flag",
        "email_domain_familiarity_bucket",

        # ----------------------------------------------------
        # Device / identity completeness
        # ----------------------------------------------------
        "has_identity_info",
        "identity_missing_count",
        "missing_identity_flag",
        "device_missing",
        "DeviceInfo_brand",
        "rare_device_flag",
        "device_familiarity_bucket",
        "identity_completeness_bucket",
        "is_mobile_device",
        "is_desktop_device",

        # ----------------------------------------------------
        # Proxy / network signal from id_23
        # ----------------------------------------------------
        "network_proxy_type",
        "id_23_missing",
        "id_23_proxy_flag",
        "id_23_anonymous_flag",
        "id_23_transparent_flag",
        "id_23_hidden_flag",

        # ----------------------------------------------------
        # OS / browser from id_30 and id_31
        # ----------------------------------------------------
        "id_30_os_family",
        "id_30_os_version_available",
        "id_31_browser_family",
        "id_31_browser_version_available",
        "os_browser_combo",

        # ----------------------------------------------------
        # Screen resolution from id_33
        # ----------------------------------------------------
        "screen_missing",
        "id_33_screen_width",
        "id_33_screen_height",
        "id_33_screen_area",
        "id_33_aspect_ratio",
        "screen_size_bucket",
        "unusual_screen_ratio",

        # ----------------------------------------------------
        # Address / distance generated indicators
        # Raw addr1, addr2, dist1 are intentionally excluded.
        # ----------------------------------------------------
        "missing_addr1",
        "missing_addr2",
        "address_signal_available",
        "addr_missing_count",
        "missing_dist1",
        "distance_signal_bucket",
        "is_large_dist1",
        "address_distance_status",

        # ----------------------------------------------------
        # Payment identifier familiarity proxy
        # ----------------------------------------------------
        "payment_identifier_familiarity_bucket",
        "rare_payment_identifier_flag",
        "common_payment_identifier_flag",

        # ----------------------------------------------------
        # Combined risk indicators
        # ----------------------------------------------------
        "high_amount_missing_identity",
        "high_amount_missing_email",
        "high_amount_off_hours",
        "high_amount_large_distance",
        "high_amount_proxy_flag",
        "high_amount_rare_device",
        "high_amount_rare_email_domain",
        "high_amount_rare_payment_identifier",
        "night_transaction_missing_identity",
        "night_transaction_large_distance",
        "email_mismatch_missing_identity",
        "email_mismatch_proxy_flag",
        "email_mismatch_rare_payment_identifier",
        "proxy_and_rare_device",
        "rare_payment_identifier_proxy_flag",
        "rare_payment_identifier_large_distance",
        "missing_address_and_distance",

        # ----------------------------------------------------
        # Simple categorical interactions
        # ----------------------------------------------------
        "high_amount_ProductCD",
        "night_ProductCD",
        "email_mismatch_ProductCD",
        "distance_bucket_ProductCD",
        "proxy_ProductCD",
        "DeviceType_ProductCD",
        "card4_card6",
        "amount_bucket_ProductCD",
        "amount_bucket_time_of_day",
        "payment_familiarity_ProductCD",
        "payment_familiarity_email_status",
        "device_familiarity_ProductCD",
        "identity_completeness_ProductCD",
    ]

    def __init__(
        self,
        rare_threshold=50,
        target_col="isFraud",
        id_cols=None,
    ):
        self.rare_threshold = rare_threshold
        self.target_col = target_col
        self.id_cols = id_cols or ["TransactionID"]

        self.thresholds_ = None
        self.frequency_maps_ = None
        self.feature_cols_ = None

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _safe_log1p(series):
        series = pd.to_numeric(series, errors="coerce")
        series = series.where(series >= 0, np.nan)
        return np.log1p(series)

    @staticmethod
    def _clean_string(series):
        return (
            series
            .astype("string")
            .str.lower()
            .str.strip()
            .fillna("missing")
        )

    @staticmethod
    def _extract_email_provider(series):
        return (
            series
            .astype("string")
            .str.lower()
            .str.strip()
            .str.split(".")
            .str[0]
            .fillna("missing")
        )

    @staticmethod
    def _extract_email_suffix(series):
        return (
            series
            .astype("string")
            .str.lower()
            .str.strip()
            .str.split(".")
            .str[-1]
            .fillna("missing")
        )

    @staticmethod
    def _add_interaction(df, col1, col2, new_col):
        if col1 in df.columns and col2 in df.columns:
            df[new_col] = (
                df[col1].astype("string").fillna("missing")
                + "_"
                + df[col2].astype("string").fillna("missing")
            )
        else:
            df[new_col] = "missing_missing"

        return df

    @staticmethod
    def _normalise_yes_no(value):
        if pd.isna(value):
            return np.nan

        value = str(value).strip().lower()

        if value in ["yes", "y", "true", "1", "available", "present"]:
            return 1

        if value in ["no", "n", "false", "0", "missing", "not available", "none"]:
            return 0

        return np.nan

    @staticmethod
    def _normalise_distance_bucket(value):
        if pd.isna(value):
            return "missing"

        value = str(value).strip().lower()

        mapping = {
            "missing": "missing",
            "no distance signal": "missing",
            "none": "missing",
            "small": "small",
            "low": "small",
            "normal": "medium",
            "medium": "medium",
            "large": "large",
            "high": "large",
            "unusual": "large",
        }

        return mapping.get(value, value)


    @staticmethod
    def _normalise_payment_familiarity_bucket(value):
        """
        Normalise the manual-form payment identifier familiarity proxy.

        This is intentionally not labelled as customer history because card1 is masked.
        The business-safe interpretation is: how familiar/common the payment identifier
        pattern is, based on historical observations.
        """
        if pd.isna(value):
            return "missing"

        value = str(value).strip().lower()

        mapping = {
            "missing": "missing",
            "none": "missing",
            "unknown": "missing",
            "new": "rare_unfamiliar",
            "rare": "rare_unfamiliar",
            "unfamiliar": "rare_unfamiliar",
            "new_or_rare": "rare_unfamiliar",
            "rare_unfamiliar": "rare_unfamiliar",
            "rare / unfamiliar": "rare_unfamiliar",
            "some": "somewhat_familiar",
            "some_history": "somewhat_familiar",
            "somewhat familiar": "somewhat_familiar",
            "somewhat_familiar": "somewhat_familiar",
            "common": "common_frequently_observed",
            "frequent": "common_frequently_observed",
            "frequently_observed": "common_frequently_observed",
            "common_frequently_observed": "common_frequently_observed",
            "common / frequently observed": "common_frequently_observed",
        }

        return mapping.get(value, value)

    # ========================================================
    # Feature creation blocks
    # ========================================================

    def _ensure_input_columns(self, df):
        """
        Ensure optional raw/source columns exist so downstream feature logic
        can run consistently.
        """
        df = df.copy()

        optional_cols = [
            "TransactionDT",
            "TransactionHour",
            "TransactionDayOfWeek",
            "TransactionAmt",
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "DeviceInfo",
            "id_23",
            "id_30",
            "id_31",
            "id_33",
            "addr1",
            "addr2",
            "dist1",

            # Optional manual-app inputs.
            # The app may pass these instead of raw masked fields.
            "missing_addr1",
            "missing_addr2",
            "missing_dist1",
            "distance_signal_bucket",
            "has_identity_info",
            "payment_identifier_familiarity_bucket",
        ]

        for col in optional_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def _create_amount_features(self, df):
        df = df.copy()

        df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce")
        df["TransactionAmt_log"] = self._safe_log1p(df["TransactionAmt"])
        df["amount_decimal"] = df["TransactionAmt"] % 1
        df["amount_cents"] = ((df["TransactionAmt"] * 100) % 100).round(0)

        df["is_round_amount"] = (
            df["amount_cents"].fillna(-1) == 0
        ).astype("int8")

        amt_str = df["TransactionAmt"].astype("string")

        df["amount_decimal_places"] = (
            amt_str
            .str.split(".")
            .str[1]
            .fillna("")
            .str.rstrip("0")
            .str.len()
            .clip(upper=3)
        ).astype("int8")

        df["has_3_decimal_amount"] = (
            df["amount_decimal_places"] >= 3
        ).astype("int8")

        # amount_bucket is added after fitted thresholds are available.
        # It is included here as a placeholder for consistency.
        if "amount_bucket" not in df.columns:
            df["amount_bucket"] = "missing"

        return df

    def _create_time_features(self, df):
        df = df.copy()

        if df["TransactionDT"].notna().any():
            transaction_dt = pd.to_numeric(df["TransactionDT"], errors="coerce")

            df["TransactionDay"] = (
                transaction_dt // (24 * 60 * 60)
            ).astype("float32")

            df["TransactionHour"] = (
                (transaction_dt // (60 * 60)) % 24
            ).astype("float32")

            df["TransactionDayOfWeek"] = (
                df["TransactionDay"] % 7
            ).astype("float32")
        else:
            df["TransactionHour"] = pd.to_numeric(
                df["TransactionHour"],
                errors="coerce",
            )

            df["TransactionDayOfWeek"] = pd.to_numeric(
                df["TransactionDayOfWeek"],
                errors="coerce",
            )

        df["is_business_hours"] = (
            df["TransactionHour"].between(9, 17)
        ).fillna(False).astype("int8")

        df["is_night"] = (
            df["TransactionHour"].between(0, 5)
        ).fillna(False).astype("int8")

        df["is_weekend"] = (
            df["TransactionDayOfWeek"].isin([5, 6])
        ).fillna(False).astype("int8")

        hour = pd.to_numeric(df["TransactionHour"], errors="coerce")
        df["time_of_day_bucket"] = "missing"
        df.loc[hour.between(0, 5), "time_of_day_bucket"] = "late_night"
        df.loc[hour.between(6, 8), "time_of_day_bucket"] = "early_morning"
        df.loc[hour.between(9, 17), "time_of_day_bucket"] = "business_hours"
        df.loc[hour.between(18, 23), "time_of_day_bucket"] = "evening"

        return df

    def _create_email_features(self, df):
        df = df.copy()

        df["missing_P_email"] = df["P_emaildomain"].isna().astype("int8")
        df["missing_R_email"] = df["R_emaildomain"].isna().astype("int8")

        df["P_email_provider"] = self._extract_email_provider(df["P_emaildomain"])
        df["P_email_suffix"] = self._extract_email_suffix(df["P_emaildomain"])

        df["R_email_provider"] = self._extract_email_provider(df["R_emaildomain"])
        df["R_email_suffix"] = self._extract_email_suffix(df["R_emaildomain"])

        p_domain = (
            df["P_emaildomain"]
            .astype("string")
            .str.lower()
            .str.strip()
        )

        r_domain = (
            df["R_emaildomain"]
            .astype("string")
            .str.lower()
            .str.strip()
        )

        df["emails_match"] = (
            p_domain.fillna("missing") == r_domain.fillna("missing")
        ).astype("int8")

        df["same_email_provider"] = (
            df["P_email_provider"].fillna("missing")
            == df["R_email_provider"].fillna("missing")
        ).astype("int8")

        df["same_email_suffix"] = (
            df["P_email_suffix"].fillna("missing")
            == df["R_email_suffix"].fillna("missing")
        ).astype("int8")

        df["both_emails_missing"] = (
            (df["missing_P_email"] == 1)
            & (df["missing_R_email"] == 1)
        ).astype("int8")

        df["only_recipient_email_missing"] = (
            (df["missing_P_email"] == 0)
            & (df["missing_R_email"] == 1)
        ).astype("int8")

        df["email_domain_mismatch"] = (
            (df["missing_P_email"] == 0)
            & (df["missing_R_email"] == 0)
            & (p_domain != r_domain)
        ).astype("int8")

        df["email_relationship_status"] = "missing_or_incomplete"

        df.loc[
            (df["missing_P_email"] == 1) & (df["missing_R_email"] == 1),
            "email_relationship_status",
        ] = "both_missing"

        df.loc[
            (df["missing_P_email"] == 0) & (df["missing_R_email"] == 1),
            "email_relationship_status",
        ] = "recipient_missing"

        df.loc[
            (df["missing_P_email"] == 1) & (df["missing_R_email"] == 0),
            "email_relationship_status",
        ] = "purchaser_missing"

        df.loc[
            (df["missing_P_email"] == 0)
            & (df["missing_R_email"] == 0)
            & (df["emails_match"] == 1),
            "email_relationship_status",
        ] = "same_domain"

        df.loc[
            (df["missing_P_email"] == 0)
            & (df["missing_R_email"] == 0)
            & (df["email_domain_mismatch"] == 1),
            "email_relationship_status",
        ] = "different_domains"

        return df

    def _create_device_identity_features(self, df):
        df = df.copy()

        # ------------------------------
        # Device type
        # ------------------------------
        df["DeviceType"] = (
            df["DeviceType"]
            .astype("string")
            .str.lower()
            .str.strip()
            .fillna("missing")
        )

        df["is_mobile_device"] = (
            df["DeviceType"] == "mobile"
        ).astype("int8")

        df["is_desktop_device"] = (
            df["DeviceType"] == "desktop"
        ).astype("int8")

        # ------------------------------
        # Device information
        # ------------------------------
        df["device_missing"] = df["DeviceInfo"].isna().astype("int8")

        df["DeviceInfo_clean"] = (
            df["DeviceInfo"]
            .astype("string")
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("/", "_", regex=False)
            .fillna("missing")
        )

        df["DeviceInfo_brand"] = (
            df["DeviceInfo_clean"]
            .str.split("_")
            .str[0]
            .fillna("missing")
        )

        # ------------------------------
        # Proxy / network signal from id_23
        # ------------------------------
        id_23_clean = (
            df["id_23"]
            .astype("string")
            .str.lower()
            .str.strip()
        )

        df["id_23_missing"] = df["id_23"].isna().astype("int8")

        df["id_23_proxy_flag"] = (
            id_23_clean.str.contains("proxy", na=False)
        ).astype("int8")

        df["id_23_anonymous_flag"] = (
            id_23_clean.str.contains("anonymous", na=False)
        ).astype("int8")

        df["id_23_transparent_flag"] = (
            id_23_clean.str.contains("transparent", na=False)
        ).astype("int8")

        df["id_23_hidden_flag"] = (
            id_23_clean.str.contains("hidden", na=False)
        ).astype("int8")

        df["network_proxy_type"] = "missing"
        df.loc[df["id_23_transparent_flag"] == 1, "network_proxy_type"] = "transparent_proxy"
        df.loc[df["id_23_anonymous_flag"] == 1, "network_proxy_type"] = "anonymous_proxy"
        df.loc[df["id_23_hidden_flag"] == 1, "network_proxy_type"] = "hidden_proxy"

        # ------------------------------
        # OS from id_30
        # ------------------------------
        id_30_clean = (
            df["id_30"]
            .astype("string")
            .str.lower()
            .str.strip()
        )

        df["id_30_os_family"] = (
            id_30_clean
            .str.split(" ")
            .str[0]
            .fillna("missing")
        )

        df["id_30_os_version_available"] = (
            id_30_clean
            .str.extract(r"(\d+\.?\d*)", expand=False)
            .notna()
        ).astype("int8")

        # ------------------------------
        # Browser from id_31
        # ------------------------------
        id_31_clean = (
            df["id_31"]
            .astype("string")
            .str.lower()
            .str.strip()
        )

        df["id_31_browser_family"] = (
            id_31_clean
            .str.split(" ")
            .str[0]
            .fillna("missing")
        )

        df["id_31_browser_version_available"] = (
            id_31_clean
            .str.extract(r"(\d+\.?\d*)", expand=False)
            .notna()
        ).astype("int8")

        df["os_browser_combo"] = (
            df["id_30_os_family"].astype("string").fillna("missing")
            + "_"
            + df["id_31_browser_family"].astype("string").fillna("missing")
        )

        # ------------------------------
        # Screen from id_33
        # ------------------------------
        df["screen_missing"] = df["id_33"].isna().astype("int8")

        screen = (
            df["id_33"]
            .astype("string")
            .str.lower()
            .str.strip()
            .str.split("x", expand=True)
        )

        if screen.shape[1] >= 2:
            df["id_33_screen_width"] = pd.to_numeric(
                screen[0],
                errors="coerce",
            )

            df["id_33_screen_height"] = pd.to_numeric(
                screen[1],
                errors="coerce",
            )
        else:
            df["id_33_screen_width"] = np.nan
            df["id_33_screen_height"] = np.nan

        df["id_33_screen_area"] = (
            df["id_33_screen_width"] * df["id_33_screen_height"]
        )

        df["id_33_aspect_ratio"] = (
            df["id_33_screen_width"] / df["id_33_screen_height"]
        )

        df["unusual_screen_ratio"] = (
            (df["id_33_aspect_ratio"] < 1.0)
            | (df["id_33_aspect_ratio"] > 2.5)
        ).fillna(False).astype("int8")

        df["screen_size_bucket"] = pd.cut(
            df["id_33_screen_area"],
            bins=[-np.inf, 500_000, 1_500_000, 3_000_000, np.inf],
            labels=["small", "medium", "large", "very_large"],
        ).astype("string").fillna("missing")

        # ------------------------------
        # Identity completeness
        # ------------------------------
        interpretable_identity_cols = [
            "DeviceType",
            "DeviceInfo",
            "id_23",
            "id_30",
            "id_31",
            "id_33",
        ]

        if df["has_identity_info"].notna().any():
            df["has_identity_info"] = (
                df["has_identity_info"]
                .apply(self._normalise_yes_no)
                .fillna(0)
                .astype("int8")
            )
        else:
            df["has_identity_info"] = (
                df[interpretable_identity_cols]
                .notna()
                .any(axis=1)
                .astype("int8")
            )

        df["identity_missing_count"] = (
            df[interpretable_identity_cols]
            .isna()
            .sum(axis=1)
            .astype("int16")
        )

        df["missing_identity_flag"] = (
            df["has_identity_info"] == 0
        ).astype("int8")

        df["identity_completeness_bucket"] = "missing"
        df.loc[df["has_identity_info"] == 0, "identity_completeness_bucket"] = "no_identity_info"
        df.loc[
            (df["has_identity_info"] == 1)
            & (df["identity_missing_count"] >= 4),
            "identity_completeness_bucket",
        ] = "limited_identity_info"
        df.loc[
            (df["has_identity_info"] == 1)
            & (df["identity_missing_count"].between(2, 3)),
            "identity_completeness_bucket",
        ] = "partial_identity_info"
        df.loc[
            (df["has_identity_info"] == 1)
            & (df["identity_missing_count"] <= 1),
            "identity_completeness_bucket",
        ] = "rich_identity_info"

        return df

    def _create_address_distance_base_features(self, df):
        """
        Address/distance logic for manual simulator.

        Raw addr1, addr2, and dist1 are source fields only.
        They are used to create generated features, then excluded from the
        final model feature list.

        The manual app may also pass:
        - missing_addr1
        - missing_addr2
        - missing_dist1
        - distance_signal_bucket

        If those are provided, they are respected.
        """
        df = df.copy()

        # ------------------------------
        # Address availability
        # ------------------------------
        if df["missing_addr1"].notna().any():
            df["missing_addr1"] = (
                df["missing_addr1"]
                .apply(self._normalise_yes_no)
                .fillna(1)
                .astype("int8")
            )
        else:
            df["missing_addr1"] = df["addr1"].isna().astype("int8")

        if df["missing_addr2"].notna().any():
            df["missing_addr2"] = (
                df["missing_addr2"]
                .apply(self._normalise_yes_no)
                .fillna(1)
                .astype("int8")
            )
        else:
            df["missing_addr2"] = df["addr2"].isna().astype("int8")

        df["address_signal_available"] = (
            (df["missing_addr1"] == 0)
            | (df["missing_addr2"] == 0)
        ).astype("int8")

        # Distance availability is handled here; bucket is handled after
        # fitted thresholds are available.
        if df["missing_dist1"].notna().any():
            df["missing_dist1"] = (
                df["missing_dist1"]
                .apply(self._normalise_yes_no)
                .fillna(1)
                .astype("int8")
            )
        else:
            df["missing_dist1"] = df["dist1"].isna().astype("int8")

        df["addr_missing_count"] = (
            df[["missing_addr1", "missing_addr2", "missing_dist1"]]
            .sum(axis=1)
            .astype("int8")
        )

        df["address_distance_status"] = "complete_address_distance_signals"
        df.loc[
            (df["address_signal_available"] == 0) & (df["missing_dist1"] == 1),
            "address_distance_status",
        ] = "missing_address_and_distance"
        df.loc[
            (df["address_signal_available"] == 0) & (df["missing_dist1"] == 0),
            "address_distance_status",
        ] = "missing_address_only"
        df.loc[
            (df["address_signal_available"] == 1) & (df["missing_dist1"] == 1),
            "address_distance_status",
        ] = "missing_distance_only"

        return df

    def _create_basic_features(self, df):
        df = self._ensure_input_columns(df)

        df = self._create_amount_features(df)
        df = self._create_time_features(df)
        df = self._create_email_features(df)
        df = self._create_device_identity_features(df)
        df = self._create_address_distance_base_features(df)

        return df.copy()

    # ========================================================
    # Fitted thresholds / fitted maps
    # ========================================================

    def _fit_thresholds(self, train_df):
        thresholds = {}

        if "TransactionAmt" in train_df.columns:
            amount = pd.to_numeric(
                train_df["TransactionAmt"],
                errors="coerce",
            ).dropna()

            if len(amount) > 0:
                thresholds["high_amount"] = amount.quantile(0.95)

                amount_q20 = amount.quantile(0.20)
                amount_q40 = amount.quantile(0.40)
                amount_q60 = amount.quantile(0.60)
                amount_q80 = amount.quantile(0.80)

                amount_cuts = [amount_q20, amount_q40, amount_q60, amount_q80]

                if (
                    all(pd.notna(x) for x in amount_cuts)
                    and len(set(amount_cuts)) == len(amount_cuts)
                    and amount_q20 < amount_q40 < amount_q60 < amount_q80
                ):
                    thresholds["amount_bucket_cuts"] = amount_cuts

        # dist1 is optional.
        # In the final manual form-style pipeline, dist1 is usually NOT present.
        # If dist1 is not available, the model will rely on distance_signal_bucket
        # provided by the training notebook / Streamlit form.
        if "dist1" in train_df.columns:
            dist1 = pd.to_numeric(train_df["dist1"], errors="coerce").dropna()

            if len(dist1) > 0:
                small_cutoff = dist1.quantile(0.33)
                medium_cutoff = dist1.quantile(0.66)
                large_cutoff = dist1.quantile(0.95)

                # Only save thresholds if they are valid and increasing.
                if (
                    pd.notna(small_cutoff)
                    and pd.notna(medium_cutoff)
                    and pd.notna(large_cutoff)
                    and small_cutoff < medium_cutoff
                ):
                    thresholds["dist1_small_medium_cutoff"] = small_cutoff
                    thresholds["dist1_medium_large_cutoff"] = medium_cutoff
                    thresholds["large_dist1"] = large_cutoff

        return thresholds

    def _add_threshold_and_bucket_features(self, df):
        df = df.copy()
        thresholds = self.thresholds_ or {}

        # ------------------------------
        # High amount flag
        # ------------------------------
        if "high_amount" in thresholds:
            df["is_high_amount"] = (
                pd.to_numeric(df["TransactionAmt"], errors="coerce")
                >= thresholds["high_amount"]
            ).fillna(False).astype("int8")
        else:
            df["is_high_amount"] = 0

        # ------------------------------
        # Amount bucket
        # ------------------------------
        amount = pd.to_numeric(df["TransactionAmt"], errors="coerce")

        if "amount_bucket_cuts" in thresholds:
            q20, q40, q60, q80 = thresholds["amount_bucket_cuts"]

            df["amount_bucket"] = pd.cut(
                amount,
                bins=[-np.inf, q20, q40, q60, q80, np.inf],
                labels=["very_low", "low", "medium", "high", "very_high"],
            ).astype("string").fillna("missing")
        else:
            df["amount_bucket"] = "missing"

        # ------------------------------
        # Distance signal bucket
        # ------------------------------
        # In the manual model, the final app will usually provide
        # distance_signal_bucket directly:
        # missing / small / medium / large
        #
        # If raw dist1 exists, we can compute the bucket.
        # If raw dist1 does not exist, we keep the provided bucket.
        existing_bucket = (
            df["distance_signal_bucket"]
            .apply(self._normalise_distance_bucket)
            .astype("string")
            .fillna("missing")
        )

        has_manual_bucket = (
            existing_bucket.notna()
            & ~existing_bucket.isin(["<NA>", "nan", "none"])
            & (existing_bucket != "missing")
        )

        dist1 = pd.to_numeric(df["dist1"], errors="coerce")

        computed_bucket = pd.Series("missing", index=df.index, dtype="string")

        has_valid_dist_thresholds = (
            "dist1_small_medium_cutoff" in thresholds
            and "dist1_medium_large_cutoff" in thresholds
            and pd.notna(thresholds["dist1_small_medium_cutoff"])
            and pd.notna(thresholds["dist1_medium_large_cutoff"])
            and thresholds["dist1_small_medium_cutoff"] < thresholds["dist1_medium_large_cutoff"]
        )

        if has_valid_dist_thresholds and dist1.notna().any():
            small_cutoff = thresholds["dist1_small_medium_cutoff"]
            medium_cutoff = thresholds["dist1_medium_large_cutoff"]

            computed_bucket = pd.cut(
                dist1,
                bins=[-np.inf, small_cutoff, medium_cutoff, np.inf],
                labels=["small", "medium", "large"],
            ).astype("string").fillna("missing")

        df["distance_signal_bucket"] = np.where(
            has_manual_bucket,
            existing_bucket,
            computed_bucket,
        )

        df["distance_signal_bucket"] = (
            pd.Series(df["distance_signal_bucket"], index=df.index)
            .astype("string")
            .fillna("missing")
        )

        # ------------------------------
        # Large distance flag
        # ------------------------------
        # If the user/form provided "large", treat it as large.
        # If raw dist1 exists and valid large_dist1 threshold exists, also use that.
        large_from_bucket = (
            df["distance_signal_bucket"].astype("string") == "large"
        )

        if (
            "large_dist1" in thresholds
            and pd.notna(thresholds["large_dist1"])
            and dist1.notna().any()
        ):
            large_from_raw = (
                dist1 >= thresholds["large_dist1"]
            ).fillna(False)
        else:
            large_from_raw = pd.Series(False, index=df.index)

        df["is_large_dist1"] = (
            large_from_bucket | large_from_raw
        ).astype("int8")

        # ------------------------------
        # Missing distance flag
        # ------------------------------
        df["missing_dist1"] = (
            (df["missing_dist1"] == 1)
            | (df["distance_signal_bucket"].astype("string") == "missing")
        ).astype("int8")

        return df.copy()

    def _fit_frequency_maps(self, train_df):
        """
        Frequency maps are used only to generate rare/common/familiarity features for
        business-understandable concepts:
        - email domain familiarity
        - device brand familiarity

        Raw masked identifiers are not kept as final model features.
        """
        frequency_cols = [
            "P_emaildomain",
            "R_emaildomain",
            "DeviceInfo_brand",
        ]

        frequency_maps = {}

        for col in frequency_cols:
            if col in train_df.columns:
                frequency_maps[col] = (
                    train_df[col]
                    .astype("string")
                    .str.lower()
                    .str.strip()
                    .fillna("missing")
                    .value_counts(dropna=False)
                )

        return frequency_maps

    def _add_frequency_features(self, df):
        df = df.copy()

        for col, freq_map in (self.frequency_maps_ or {}).items():
            if col not in df.columns:
                continue

            safe_col = col.replace(" ", "_").replace("/", "_").replace(".", "_")

            values = (
                df[col]
                .astype("string")
                .str.lower()
                .str.strip()
                .fillna("missing")
            )

            df[f"{safe_col}_freq"] = (
                values
                .map(freq_map)
                .fillna(0)
                .astype("int32")
            )

            df[f"rare_{safe_col}_flag"] = (
                df[f"{safe_col}_freq"] < self.rare_threshold
            ).astype("int8")

        # ------------------------------
        # Email familiarity bucket
        # ------------------------------
        p_freq = (
            df["P_emaildomain_freq"]
            if "P_emaildomain_freq" in df.columns
            else pd.Series(0, index=df.index)
        )

        r_freq = (
            df["R_emaildomain_freq"]
            if "R_emaildomain_freq" in df.columns
            else pd.Series(0, index=df.index)
        )

        email_freq = pd.concat([p_freq, r_freq], axis=1).max(axis=1)

        df["email_domain_familiarity_bucket"] = "missing_or_unseen"
        df.loc[email_freq >= self.rare_threshold, "email_domain_familiarity_bucket"] = "somewhat_familiar"
        df.loc[email_freq >= self.rare_threshold * 10, "email_domain_familiarity_bucket"] = "common_frequently_observed"

        if "rare_P_emaildomain_flag" in df.columns:
            df["rare_email_domain_flag"] = (
                df["rare_P_emaildomain_flag"]
                .astype("int8")
            )
        else:
            df["rare_email_domain_flag"] = 1

        # Treat unseen/missing recipient domains as an additional rare signal.
        if "rare_R_emaildomain_flag" in df.columns:
            df["rare_email_domain_flag"] = (
                (df["rare_email_domain_flag"] == 1)
                | (df["rare_R_emaildomain_flag"] == 1)
            ).astype("int8")

        # ------------------------------
        # Device familiarity bucket
        # ------------------------------
        device_freq = (
            df["DeviceInfo_brand_freq"]
            if "DeviceInfo_brand_freq" in df.columns
            else pd.Series(0, index=df.index)
        )

        df["device_familiarity_bucket"] = "missing_or_unseen"
        df.loc[device_freq >= self.rare_threshold, "device_familiarity_bucket"] = "somewhat_familiar"
        df.loc[device_freq >= self.rare_threshold * 10, "device_familiarity_bucket"] = "common_frequently_observed"

        if "rare_DeviceInfo_brand_flag" in df.columns:
            df["rare_device_flag"] = (
                df["rare_DeviceInfo_brand_flag"]
                .astype("int8")
            )
        else:
            df["rare_device_flag"] = 1

        # If device info is actually missing, keep the bucket explicit.
        if "device_missing" in df.columns:
            df.loc[df["device_missing"] == 1, "device_familiarity_bucket"] = "missing"

        return df.copy()

    def _add_combined_risk_features(self, df):
        df = df.copy()

        # ------------------------------
        # Payment identifier familiarity proxy
        # ------------------------------
        # This is a business-safe proxy, not a claim that card1 is literally
        # customer history. The notebook derives it from the frequency of a
        # masked card-related identifier; the app can ask the user for this
        # as a simple familiarity bucket.
        df["payment_identifier_familiarity_bucket"] = (
            df["payment_identifier_familiarity_bucket"]
            .apply(self._normalise_payment_familiarity_bucket)
            .astype("string")
            .fillna("missing")
        )

        df["rare_payment_identifier_flag"] = (
            df["payment_identifier_familiarity_bucket"] == "rare_unfamiliar"
        ).astype("int8")

        df["common_payment_identifier_flag"] = (
            df["payment_identifier_familiarity_bucket"] == "common_frequently_observed"
        ).astype("int8")

        df["high_amount_missing_identity"] = (
            (df["is_high_amount"] == 1)
            & (df["has_identity_info"] == 0)
        ).astype("int8")

        df["high_amount_missing_email"] = (
            (df["is_high_amount"] == 1)
            & (
                (df["missing_P_email"] == 1)
                | (df["missing_R_email"] == 1)
            )
        ).astype("int8")

        df["high_amount_off_hours"] = (
            (df["is_high_amount"] == 1)
            & (df["is_business_hours"] == 0)
        ).astype("int8")

        df["high_amount_large_distance"] = (
            (df["is_high_amount"] == 1)
            & (df["is_large_dist1"] == 1)
        ).astype("int8")

        df["high_amount_proxy_flag"] = (
            (df["is_high_amount"] == 1)
            & (df["id_23_proxy_flag"] == 1)
        ).astype("int8")

        df["high_amount_rare_device"] = (
            (df["is_high_amount"] == 1)
            & (df["rare_device_flag"] == 1)
        ).astype("int8")

        df["high_amount_rare_email_domain"] = (
            (df["is_high_amount"] == 1)
            & (df["rare_email_domain_flag"] == 1)
        ).astype("int8")

        df["high_amount_rare_payment_identifier"] = (
            (df["is_high_amount"] == 1)
            & (df["rare_payment_identifier_flag"] == 1)
        ).astype("int8")

        df["night_transaction_missing_identity"] = (
            (df["is_night"] == 1)
            & (df["has_identity_info"] == 0)
        ).astype("int8")

        df["night_transaction_large_distance"] = (
            (df["is_night"] == 1)
            & (df["is_large_dist1"] == 1)
        ).astype("int8")

        df["email_mismatch_missing_identity"] = (
            (df["email_domain_mismatch"] == 1)
            & (df["has_identity_info"] == 0)
        ).astype("int8")

        df["email_mismatch_proxy_flag"] = (
            (df["email_domain_mismatch"] == 1)
            & (df["id_23_proxy_flag"] == 1)
        ).astype("int8")

        df["email_mismatch_rare_payment_identifier"] = (
            (df["email_domain_mismatch"] == 1)
            & (df["rare_payment_identifier_flag"] == 1)
        ).astype("int8")

        df["proxy_and_rare_device"] = (
            (df["id_23_proxy_flag"] == 1)
            & (df["rare_device_flag"] == 1)
        ).astype("int8")

        df["rare_payment_identifier_proxy_flag"] = (
            (df["rare_payment_identifier_flag"] == 1)
            & (df["id_23_proxy_flag"] == 1)
        ).astype("int8")

        df["rare_payment_identifier_large_distance"] = (
            (df["rare_payment_identifier_flag"] == 1)
            & (df["is_large_dist1"] == 1)
        ).astype("int8")

        df["missing_address_and_distance"] = (
            (df["address_signal_available"] == 0)
            & (df["missing_dist1"] == 1)
        ).astype("int8")

        # Simple categorical interactions.
        interaction_pairs = [
            ("is_high_amount", "ProductCD", "high_amount_ProductCD"),
            ("is_night", "ProductCD", "night_ProductCD"),
            ("email_domain_mismatch", "ProductCD", "email_mismatch_ProductCD"),
            ("distance_signal_bucket", "ProductCD", "distance_bucket_ProductCD"),
            ("network_proxy_type", "ProductCD", "proxy_ProductCD"),
            ("DeviceType", "ProductCD", "DeviceType_ProductCD"),
            ("card4", "card6", "card4_card6"),
            ("amount_bucket", "ProductCD", "amount_bucket_ProductCD"),
            ("amount_bucket", "time_of_day_bucket", "amount_bucket_time_of_day"),
            ("payment_identifier_familiarity_bucket", "ProductCD", "payment_familiarity_ProductCD"),
            ("payment_identifier_familiarity_bucket", "email_relationship_status", "payment_familiarity_email_status"),
            ("device_familiarity_bucket", "ProductCD", "device_familiarity_ProductCD"),
            ("identity_completeness_bucket", "ProductCD", "identity_completeness_ProductCD"),
        ]

        for col1, col2, new_col in interaction_pairs:
            df = self._add_interaction(df, col1, col2, new_col)

        return df.copy()

    # ========================================================
    # Public helpers for app/schema generation
    # ========================================================

    def get_source_input_columns(self):
        """
        Source columns that the Streamlit manual form should provide.

        The final app should provide TransactionHour and TransactionDayOfWeek,
        not TransactionDT. During training, the notebook derives these from
        TransactionDT.
        """
        return [
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

    def get_generated_feature_display_map(self):
        """
        Useful later for Streamlit when showing generated features.
        """
        return {
            "TransactionAmt_log": "Log Transaction Amount",
            "is_high_amount": "High Amount Transaction",
            "amount_bucket": "Transaction Amount Bucket",
            "is_round_amount": "Round Amount Transaction",
            "has_3_decimal_amount": "Three-Decimal Amount Indicator",
            "is_weekend": "Weekend Transaction",
            "is_business_hours": "Business Hours Transaction",
            "is_night": "Night Transaction",
            "time_of_day_bucket": "Time of Day Bucket",
            "missing_P_email": "Missing Purchaser Email",
            "missing_R_email": "Missing Recipient Email",
            "emails_match": "Purchaser and Recipient Emails Match",
            "email_domain_mismatch": "Email Domain Mismatch",
            "email_relationship_status": "Email Relationship Status",
            "email_domain_familiarity_bucket": "Email Domain Familiarity",
            "rare_email_domain_flag": "Rare Email Domain Indicator",
            "has_identity_info": "Identity Information Available",
            "missing_identity_flag": "Missing Identity Indicator",
            "device_missing": "Missing Device Information",
            "rare_device_flag": "Rare Device Indicator",
            "device_familiarity_bucket": "Device Familiarity",
            "identity_completeness_bucket": "Identity / Device Completeness",
            "id_23_proxy_flag": "Proxy / Network Indicator",
            "network_proxy_type": "Network / Proxy Type",
            "id_30_os_family": "Operating System Family",
            "id_31_browser_family": "Browser Family",
            "screen_size_bucket": "Screen Size Bucket",
            "unusual_screen_ratio": "Unusual Screen Ratio",
            "missing_addr1": "Address available",
            "missing_addr2": "Country code available",
            "missing_dist1": "Distance signal available",
            "address_signal_available": "Any address signal available",
            "distance_signal_bucket": "Distance / location profile",
            "is_large_dist1": "Large Distance / Location Signal",
            "address_distance_status": "Address and Distance Status",
            "payment_identifier_familiarity_bucket": "Payment Identifier Familiarity",
            "rare_payment_identifier_flag": "Rare Payment Identifier Pattern",
            "common_payment_identifier_flag": "Common Payment Identifier Pattern",
            "high_amount_missing_identity": "High Amount with Missing Identity",
            "high_amount_missing_email": "High Amount with Missing Email",
            "high_amount_off_hours": "High Amount Outside Business Hours",
            "high_amount_large_distance": "High Amount with Large Distance Signal",
            "high_amount_proxy_flag": "High Amount with Proxy Signal",
            "high_amount_rare_device": "High Amount with Rare Device",
            "high_amount_rare_email_domain": "High Amount with Rare Email Domain",
            "night_transaction_missing_identity": "Night Transaction with Missing Identity",
            "night_transaction_large_distance": "Night Transaction with Large Distance Signal",
            "email_mismatch_missing_identity": "Email Mismatch with Missing Identity",
            "email_mismatch_proxy_flag": "Email Mismatch with Proxy Signal",
            "proxy_and_rare_device": "Proxy Signal with Rare Device",
            "missing_address_and_distance": "Missing Address and Distance Signals",
        }

    # ========================================================
    # sklearn-like API
    # ========================================================

    def fit(self, train_df):
        train_base = self._create_basic_features(train_df)

        self.thresholds_ = self._fit_thresholds(train_base)

        train_thresholded = self._add_threshold_and_bucket_features(train_base)

        self.frequency_maps_ = self._fit_frequency_maps(train_thresholded)

        train_fe = self._add_frequency_features(train_thresholded)
        train_fe = self._add_combined_risk_features(train_fe)

        # Keep only the curated manual simulation features.
        # This prevents raw masked fields from leaking into the model.
        self.feature_cols_ = [
            col for col in self.MODEL_FEATURES
            if col in train_fe.columns
        ]

        missing_features = [
            col for col in self.MODEL_FEATURES
            if col not in train_fe.columns
        ]

        if missing_features:
            print("Warning: these configured model features were not created:")
            print(missing_features)

        return self

    def transform(self, df, return_features_only=True):
        if self.thresholds_ is None:
            raise ValueError("ManualSimulationFeatureEngineer is not fitted yet.")

        df_fe = self._create_basic_features(df)
        df_fe = self._add_threshold_and_bucket_features(df_fe)
        df_fe = self._add_frequency_features(df_fe)
        df_fe = self._add_combined_risk_features(df_fe)

        if return_features_only:
            for col in self.feature_cols_:
                if col not in df_fe.columns:
                    df_fe[col] = np.nan

            return df_fe[self.feature_cols_].copy()

        return df_fe.copy()

    def fit_transform(self, train_df, return_features_only=True):
        self.fit(train_df)
        return self.transform(
            train_df,
            return_features_only=return_features_only,
        )