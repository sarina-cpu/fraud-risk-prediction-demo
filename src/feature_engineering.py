import numpy as np
import pandas as pd


class FraudFeatureEngineer:

    def __init__(
        self,
        frequency_cols=None,
        group_amount_cols=None,
        rare_threshold=50,
        target_col="isFraud",
        id_cols=None,
        drop_raw_v=False
    ):
        self.frequency_cols = frequency_cols or []
        self.group_amount_cols = group_amount_cols or []
        self.rare_threshold = rare_threshold
        self.target_col = target_col
        self.id_cols = id_cols or ["TransactionID"]
        self.drop_raw_v = drop_raw_v

        self.thresholds_ = None
        self.frequency_maps_ = None
        self.amount_stat_maps_ = None
        self.raw_v_cols_ = None
        self.feature_cols_ = None

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _safe_log1p(series):
        series = pd.to_numeric(series, errors="coerce")
        series = series.where(series >= 0, np.nan)
        return np.log1p(series)

    @staticmethod
    def _get_raw_v_cols(df):
        return [
            col for col in df.columns
            if col.startswith("V") and col[1:].isdigit()
        ]

    @staticmethod
    def _extract_email_provider(series):
        return (
            series
            .astype("string")
            .str.lower()
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
        return df

    # --------------------------------------------------------
    # Row-level feature engineering
    # --------------------------------------------------------

    def _create_basic_features(self, df):
        df = df.copy()

        # ----------------------------------------------------
        # Raw V columns + V aggregation features
        # ----------------------------------------------------
        raw_v_cols = self._get_raw_v_cols(df)

        if self.raw_v_cols_ is None:
            self.raw_v_cols_ = raw_v_cols

        if raw_v_cols:
            v_df = df[raw_v_cols]

            # Overall row-level V summaries
            df["V_missing_count"] = v_df.isna().sum(axis=1).astype("int16")
            df["V_present_count"] = v_df.notna().sum(axis=1).astype("int16")

            v_filled = v_df.fillna(0)

            df["V_nonzero_count"] = (v_filled != 0).sum(axis=1).astype("int16")
            df["V_zero_count"] = (v_filled == 0).sum(axis=1).astype("int16")

            df["V_sum"] = v_df.sum(axis=1, min_count=1).astype("float32")
            df["V_mean"] = v_df.mean(axis=1).astype("float32")
            df["V_std"] = v_df.std(axis=1).astype("float32")
            df["V_min"] = v_df.min(axis=1).astype("float32")
            df["V_max"] = v_df.max(axis=1).astype("float32")
            df["V_range"] = (df["V_max"] - df["V_min"]).astype("float32")

            # V blocks based on common column ranges used in IEEE-CIS EDA patterns
            v_blocks = {
                "V_block_1": list(range(1, 12)),
                "V_block_2": list(range(12, 35)),
                "V_block_3": list(range(35, 53)),
                "V_block_4": list(range(53, 75)),
                "V_block_5": list(range(75, 95)),
                "V_block_6": list(range(95, 138)),
                "V_block_7": list(range(138, 167)),
                "V_block_8": list(range(167, 217)),
                "V_block_9": list(range(217, 279)),
                "V_block_10": list(range(279, 322)),
                "V_block_11": list(range(322, 340)),
            }

            v_block_features = {}

            for block_name, nums in v_blocks.items():
                block_cols = [
                    f"V{i}" for i in nums
                    if f"V{i}" in df.columns
                ]

                if not block_cols:
                    continue

                block_df = df[block_cols]
                block_filled = block_df.fillna(0)

                v_block_features[f"{block_name}_missing_count"] = (
                    block_df.isna().sum(axis=1).astype("int16")
                )
                v_block_features[f"{block_name}_present_count"] = (
                    block_df.notna().sum(axis=1).astype("int16")
                )
                v_block_features[f"{block_name}_nonzero_count"] = (
                    (block_filled != 0).sum(axis=1).astype("int16")
                )
                v_block_features[f"{block_name}_sum"] = (
                    block_df.sum(axis=1, min_count=1).astype("float32")
                )
                v_block_features[f"{block_name}_mean"] = (
                    block_df.mean(axis=1).astype("float32")
                )
                v_block_features[f"{block_name}_std"] = (
                    block_df.std(axis=1).astype("float32")
                )
                v_block_features[f"{block_name}_max"] = (
                    block_df.max(axis=1).astype("float32")
                )
                v_block_features[f"{block_name}_min"] = (
                    block_df.min(axis=1).astype("float32")
                )

            if v_block_features:
                df = pd.concat(
                    [df, pd.DataFrame(v_block_features, index=df.index)],
                    axis=1
                )

        # For the full upload model, keep raw V columns unless explicitly requested otherwise
        if self.drop_raw_v and raw_v_cols:
            df = df.drop(columns=raw_v_cols)

        # ----------------------------------------------------
        # Transaction amount features
        # ----------------------------------------------------
        if "TransactionAmt" in df.columns:
            df["TransactionAmt_log"] = self._safe_log1p(df["TransactionAmt"])
            df["amount_decimal"] = df["TransactionAmt"] % 1
            df["amount_cents"] = ((df["TransactionAmt"] * 100) % 100).round(0)
            df["is_round_amount"] = (df["amount_cents"] == 0).astype("int8")

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

        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------
        if "TransactionDT" in df.columns:
            df["TransactionDay"] = (df["TransactionDT"] // (24 * 60 * 60)).astype("int16")
            df["TransactionHour"] = ((df["TransactionDT"] // (60 * 60)) % 24).astype("int8")
            df["TransactionDayOfWeek"] = (df["TransactionDay"] % 7).astype("int8")

            df["is_weekend"] = df["TransactionDayOfWeek"].isin([5, 6]).astype("int8")
            df["is_business_hours"] = df["TransactionHour"].between(9, 17).astype("int8")
            df["is_night"] = df["TransactionHour"].between(0, 5).astype("int8")

            df["relative_month_day"] = (df["TransactionDay"] % 30).astype("int8")
            df["is_month_start_proxy"] = df["relative_month_day"].between(0, 3).astype("int8")
            df["is_month_end_proxy"] = df["relative_month_day"].between(26, 29).astype("int8")

        # ----------------------------------------------------
        # Email features
        # ----------------------------------------------------
        if "P_emaildomain" in df.columns:
            df["missing_P_email"] = df["P_emaildomain"].isna().astype("int8")
            df["P_email_provider"] = self._extract_email_provider(df["P_emaildomain"])
            df["P_email_suffix"] = self._extract_email_suffix(df["P_emaildomain"])

        if "R_emaildomain" in df.columns:
            df["missing_R_email"] = df["R_emaildomain"].isna().astype("int8")
            df["R_email_provider"] = self._extract_email_provider(df["R_emaildomain"])
            df["R_email_suffix"] = self._extract_email_suffix(df["R_emaildomain"])

        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            p_domain = df["P_emaildomain"].astype("string").str.lower()
            r_domain = df["R_emaildomain"].astype("string").str.lower()

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
                df["P_emaildomain"].isna() & df["R_emaildomain"].isna()
            ).astype("int8")

            df["only_recipient_email_missing"] = (
                df["P_emaildomain"].notna() & df["R_emaildomain"].isna()
            ).astype("int8")

            df["email_domain_mismatch"] = (
                df["P_emaildomain"].notna()
                & df["R_emaildomain"].notna()
                & (p_domain != r_domain)
            ).astype("int8")

        # ----------------------------------------------------
        # Address and distance features
        # ----------------------------------------------------
        address_cols = [
            col for col in ["addr1", "addr2", "dist1", "dist2"]
            if col in df.columns
        ]

        for col in address_cols:
            df[f"missing_{col}"] = df[col].isna().astype("int8")

        if address_cols:
            df["addr_missing_count"] = df[address_cols].isna().sum(axis=1).astype("int8")

        for col in ["dist1", "dist2"]:
            if col in df.columns:
                df[f"{col}_log"] = self._safe_log1p(df[col])

        df = self._add_interaction(df, "addr1", "addr2", "addr1_addr2")

        # ----------------------------------------------------
        # Identity and device features
        # ----------------------------------------------------
        identity_cols = [
            col for col in df.columns
            if col.startswith("id_") or col in ["DeviceType", "DeviceInfo"]
        ]

        if identity_cols:
            df["has_identity_info"] = df[identity_cols].notna().any(axis=1).astype("int8")
            df["identity_missing_count"] = df[identity_cols].isna().sum(axis=1).astype("int16")

        if "DeviceInfo" in df.columns:
            df["device_missing"] = df["DeviceInfo"].isna().astype("int8")
            df["DeviceInfo_clean"] = (
                df["DeviceInfo"]
                .astype("string")
                .str.lower()
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

        if "DeviceType" in df.columns:
            df["DeviceType_clean"] = df["DeviceType"].astype("string").str.lower().fillna("missing")
            df["is_mobile_device"] = (df["DeviceType_clean"] == "mobile").astype("int8")
            df["is_desktop_device"] = (df["DeviceType_clean"] == "desktop").astype("int8")

        if "id_30" in df.columns:
            id_30_clean = df["id_30"].astype("string").str.lower()
            df["id_30_os_family"] = id_30_clean.str.split(" ").str[0].fillna("missing")
            df["id_30_os_version"] = (
                id_30_clean
                .str.extract(r"(\d+\.?\d*)", expand=False)
                .fillna("missing")
            )

        if "id_31" in df.columns:
            id_31_clean = df["id_31"].astype("string").str.lower()
            df["id_31_browser_family"] = id_31_clean.str.split(" ").str[0].fillna("missing")
            df["id_31_browser_version"] = (
                id_31_clean
                .str.extract(r"(\d+\.?\d*)", expand=False)
                .fillna("missing")
            )

        if "id_33" in df.columns:
            screen = df["id_33"].astype("string").str.lower().str.split("x", expand=True)

            if screen.shape[1] >= 2:
                df["id_33_screen_width"] = pd.to_numeric(screen[0], errors="coerce")
                df["id_33_screen_height"] = pd.to_numeric(screen[1], errors="coerce")
                df["id_33_screen_area"] = df["id_33_screen_width"] * df["id_33_screen_height"]
                df["id_33_aspect_ratio"] = df["id_33_screen_width"] / df["id_33_screen_height"]
                df["screen_missing"] = df["id_33"].isna().astype("int8")

                df["unusual_screen_ratio"] = (
                    (df["id_33_aspect_ratio"] < 1.0)
                    | (df["id_33_aspect_ratio"] > 2.5)
                ).fillna(False).astype("int8")

        if "id_23" in df.columns:
            id_23_clean = df["id_23"].astype("string").str.lower()
            df["id_23_missing"] = df["id_23"].isna().astype("int8")
            df["id_23_proxy_flag"] = id_23_clean.str.contains("proxy", na=False).astype("int8")
            df["id_23_anonymous_flag"] = id_23_clean.str.contains("anonymous", na=False).astype("int8")
            df["id_23_transparent_flag"] = id_23_clean.str.contains("transparent", na=False).astype("int8")

        # New / Found identity flags
        new_found_cols = ["id_12", "id_15", "id_27", "id_28", "id_29"]

        for col in new_found_cols:
            if col in df.columns:
                lower_col = df[col].astype("string").str.lower().fillna("missing")
                df[f"{col}_is_New"] = (lower_col == "new").astype("int8")
                df[f"{col}_is_Found"] = (lower_col == "found").astype("int8")

        new_cols_created = [
            f"{col}_is_New"
            for col in new_found_cols
            if f"{col}_is_New" in df.columns
        ]

        found_cols_created = [
            f"{col}_is_Found"
            for col in new_found_cols
            if f"{col}_is_Found" in df.columns
        ]

        if new_cols_created:
            df["new_device_or_identity_flag"] = (
                df[new_cols_created].sum(axis=1) > 0
            ).astype("int8")

        if found_cols_created:
            df["known_device_or_identity_flag"] = (
                df[found_cols_created].sum(axis=1) > 0
            ).astype("int8")

        # ----------------------------------------------------
        # M match features
        # ----------------------------------------------------
        m_cols = [
            col for col in df.columns
            if col.startswith("M") and col[1:].isdigit()
        ]

        for col in m_cols:
            if col != "M4":
                df[f"{col}_binary"] = df[col].map({"T": 1, "F": 0}).astype("float32")

        m_binary_cols = [
            f"{col}_binary"
            for col in m_cols
            if f"{col}_binary" in df.columns
        ]

        if m_cols:
            df["M_missing_count"] = df[m_cols].isna().sum(axis=1).astype("int8")
            df["all_M_missing"] = (df["M_missing_count"] == len(m_cols)).astype("int8")

        if m_binary_cols:
            df["M_match_sum"] = df[m_binary_cols].sum(axis=1).astype("float32")
            df["M_mismatch_sum"] = (df[m_binary_cols] == 0).sum(axis=1).astype("int8")
            df["M_true_ratio"] = (df["M_match_sum"] / len(m_binary_cols)).astype("float32")
            df["M_false_ratio"] = (df["M_mismatch_sum"] / len(m_binary_cols)).astype("float32")
            df["has_any_mismatch"] = (df["M_mismatch_sum"] > 0).astype("int8")

        if "M4" in df.columns:
            df["M4_clean"] = df["M4"].astype("string").fillna("missing")
            df["M4_is_M0"] = (df["M4_clean"] == "M0").astype("int8")
            df["M4_is_M1"] = (df["M4_clean"] == "M1").astype("int8")
            df["M4_is_M2"] = (df["M4_clean"] == "M2").astype("int8")

        # ----------------------------------------------------
        # C and D row-level aggregations only
        # No V aggregations are created in this explainable version.
        # ----------------------------------------------------
        c_cols = [
            col for col in df.columns
            if col.startswith("C") and col[1:].isdigit()
        ]

        d_cols = [
            col for col in df.columns
            if col.startswith("D") and col[1:].isdigit()
        ]

        if c_cols:
            df["C_sum"] = df[c_cols].sum(axis=1).astype("float32")
            df["C_mean"] = df[c_cols].mean(axis=1).astype("float32")
            df["C_std"] = df[c_cols].std(axis=1).astype("float32")
            df["C_max"] = df[c_cols].max(axis=1).astype("float32")
            df["C_min"] = df[c_cols].min(axis=1).astype("float32")
            df["C_range"] = (df["C_max"] - df["C_min"]).astype("float32")
            df["C_missing_count"] = df[c_cols].isna().sum(axis=1).astype("int8")
            df["C_nonzero_count"] = (df[c_cols].fillna(0) != 0).sum(axis=1).astype("int8")

        if d_cols:
            df["D_mean"] = df[d_cols].mean(axis=1).astype("float32")
            df["D_std"] = df[d_cols].std(axis=1).astype("float32")
            df["D_max"] = df[d_cols].max(axis=1).astype("float32")
            df["D_min"] = df[d_cols].min(axis=1).astype("float32")
            df["D_range"] = (df["D_max"] - df["D_min"]).astype("float32")
            df["D_missing_count"] = df[d_cols].isna().sum(axis=1).astype("int8")
            df["D_nonzero_count"] = (df[d_cols].fillna(0) != 0).sum(axis=1).astype("int8")

            d_log_features = {}

            for col in d_cols:
                d_log_features[f"{col}_log"] = self._safe_log1p(df[col]).astype("float32")

            if d_log_features:
                df = pd.concat(
                    [df, pd.DataFrame(d_log_features, index=df.index)],
                    axis=1
                )
        

        # ----------------------------------------------------
        # Interaction features
        # ----------------------------------------------------
        interactions = [
            ("card1", "card2", "card1_card2"),
            ("card1", "addr1", "card1_addr1"),
            ("card1", "addr2", "card1_addr2"),
            ("card1", "ProductCD", "card1_ProductCD"),
            ("card4", "ProductCD", "card4_ProductCD"),
            ("card6", "ProductCD", "card6_ProductCD"),
            ("ProductCD", "card6", "ProductCD_card6"),
            ("addr1", "ProductCD", "addr1_ProductCD"),
            ("addr2", "ProductCD", "addr2_ProductCD"),
            ("card1", "card4", "card1_card4"),
            ("card1", "card6", "card1_card6"),
            ("card2", "card3", "card2_card3"),
            ("card2", "addr1", "card2_addr1"),
            ("card2", "ProductCD", "card2_ProductCD"),
            ("P_emaildomain", "ProductCD", "P_emaildomain_ProductCD"),
            ("R_emaildomain", "ProductCD", "R_emaildomain_ProductCD"),
            ("P_email_provider", "card4", "P_email_provider_card4"),
            ("P_email_provider", "card6", "P_email_provider_card6"),
            ("DeviceType", "DeviceInfo_clean", "DeviceType_DeviceInfo"),
            ("DeviceType", "ProductCD", "DeviceType_ProductCD"),
            ("id_30_os_family", "id_31_browser_family", "os_browser_combo"),
            ("addr1", "card1", "addr1_card1"),
            ("addr2", "card1", "addr2_card1"),
            ("addr1", "P_email_provider", "addr1_P_email_provider"),
            ("addr2", "P_email_suffix", "addr2_P_email_suffix"),
        ]

        for col1, col2, new_col in interactions:
            df = self._add_interaction(df, col1, col2, new_col)

        # ----------------------------------------------------
        # UID-style identity features inspired by public top solutions
        # ----------------------------------------------------
        if "TransactionDay" in df.columns and "card1_addr1" in df.columns and "D1" in df.columns:
            uid_day = np.floor(df["TransactionDay"] - df["D1"]).astype("float32")

            df["UID"] = (
                df["card1_addr1"].astype("string").fillna("missing")
                + "_"
                + uid_day.astype("string").fillna("missing")
            )

        if "TransactionDay" in df.columns and "card1_addr1" in df.columns and "D2" in df.columns:
            uid2_day = np.floor(df["TransactionDay"] - df["D2"]).astype("float32")

            df["UID2"] = (
                df["card1_addr1"].astype("string").fillna("missing")
                + "_"
                + uid2_day.astype("string").fillna("missing")
            )

        if (
            "TransactionDay" in df.columns
            and "card1_addr1" in df.columns
            and "P_emaildomain" in df.columns
            and "D1" in df.columns
        ):
            uid3_day = np.floor(df["TransactionDay"] - df["D1"]).astype("float32")

            df["UID3"] = (
                df["card1_addr1"].astype("string").fillna("missing")
                + "_"
                + df["P_emaildomain"].astype("string").fillna("missing")
                + "_"
                + uid3_day.astype("string").fillna("missing")
            )
            
        return df.copy()

def _fit_thresholds(self, train_df):
    thresholds = {}

    if "TransactionAmt" in train_df.columns:
        thresholds["high_amount"] = train_df["TransactionAmt"].quantile(0.95)

    if "dist1" in train_df.columns:
        thresholds["large_dist1"] = train_df["dist1"].quantile(0.95)

    if "dist2" in train_df.columns:
        thresholds["large_dist2"] = train_df["dist2"].quantile(0.95)

    if "C_sum" in train_df.columns:
        thresholds["high_C_sum"] = train_df["C_sum"].quantile(0.95)

    if "D_min" in train_df.columns:
        thresholds["recent_D_min"] = train_df["D_min"].quantile(0.05)

    if "D_max" in train_df.columns:
        thresholds["long_gap_D_max"] = train_df["D_max"].quantile(0.95)

    return thresholds


def _add_threshold_risk_flags(self, df):
    df = df.copy()
    thresholds = self.thresholds_ or {}

    if "TransactionAmt" in df.columns and "high_amount" in thresholds:
        df["is_high_amount"] = (
            df["TransactionAmt"] >= thresholds["high_amount"]
        ).astype("int8")

    if "dist1" in df.columns and "large_dist1" in thresholds:
        df["is_large_dist1"] = (
            df["dist1"] >= thresholds["large_dist1"]
        ).astype("int8")

    if "dist2" in df.columns and "large_dist2" in thresholds:
        df["is_large_dist2"] = (
            df["dist2"] >= thresholds["large_dist2"]
        ).astype("int8")

    if "C_sum" in df.columns and "high_C_sum" in thresholds:
        df["C_high_activity_flag"] = (
            df["C_sum"] >= thresholds["high_C_sum"]
        ).astype("int8")

    if "D_min" in df.columns and "recent_D_min" in thresholds:
        df["D_recent_activity_flag"] = (
            df["D_min"] <= thresholds["recent_D_min"]
        ).astype("int8")

    if "D_max" in df.columns and "long_gap_D_max" in thresholds:
        df["D_long_gap_flag"] = (
            df["D_max"] >= thresholds["long_gap_D_max"]
        ).astype("int8")

    # Combined risk flags
    if "is_high_amount" in df.columns and "has_identity_info" in df.columns:
        df["high_amount_missing_identity"] = (
            (df["is_high_amount"] == 1)
            & (df["has_identity_info"] == 0)
        ).astype("int8")

    if (
        "is_high_amount" in df.columns
        and "missing_P_email" in df.columns
        and "missing_R_email" in df.columns
    ):
        df["high_amount_missing_email"] = (
            (df["is_high_amount"] == 1)
            & ((df["missing_P_email"] == 1) | (df["missing_R_email"] == 1))
        ).astype("int8")

    if "is_high_amount" in df.columns and "is_business_hours" in df.columns:
        df["high_amount_off_hours"] = (
            (df["is_high_amount"] == 1)
            & (df["is_business_hours"] == 0)
        ).astype("int8")

    if (
        "has_3_decimal_amount" in df.columns
        and "missing_addr1" in df.columns
        and "missing_addr2" in df.columns
    ):
        df["foreign_like_amount_flag"] = (
            (df["has_3_decimal_amount"] == 1)
            & (df["missing_addr1"] == 1)
            & (df["missing_addr2"] == 1)
        ).astype("int8")

    if "is_high_amount" in df.columns and "is_large_dist1" in df.columns:
        df["high_amount_large_distance"] = (
            (df["is_high_amount"] == 1)
            & (df["is_large_dist1"] == 1)
        ).astype("int8")

    if "is_high_amount" in df.columns and "new_device_or_identity_flag" in df.columns:
        df["high_amount_new_identity"] = (
            (df["is_high_amount"] == 1)
            & (df["new_device_or_identity_flag"] == 1)
        ).astype("int8")

    if "is_high_amount" in df.columns and "has_any_mismatch" in df.columns:
        df["high_amount_any_mismatch"] = (
            (df["is_high_amount"] == 1)
            & (df["has_any_mismatch"] == 1)
        ).astype("int8")

    if "has_identity_info" in df.columns and "is_night" in df.columns:
        df["night_transaction_missing_identity"] = (
            (df["is_night"] == 1)
            & (df["has_identity_info"] == 0)
        ).astype("int8")

    if "is_large_dist1" in df.columns and "is_night" in df.columns:
        df["night_transaction_large_distance"] = (
            (df["is_night"] == 1)
            & (df["is_large_dist1"] == 1)
        ).astype("int8")

    if "email_domain_mismatch" in df.columns and "has_identity_info" in df.columns:
        df["email_mismatch_missing_identity"] = (
            (df["email_domain_mismatch"] == 1)
            & (df["has_identity_info"] == 0)
        ).astype("int8")

    return df.copy()


def _add_cross_domain_interactions(self, df):
    df = df.copy()

    interactions = [
        ("is_high_amount", "ProductCD", "high_amount_ProductCD"),
        ("is_high_amount", "card6", "high_amount_card6"),
        ("is_high_amount", "DeviceType", "high_amount_DeviceType"),
        ("is_high_amount", "P_email_provider", "high_amount_email_provider"),
        ("is_night", "ProductCD", "night_ProductCD"),
        ("is_night", "card6", "night_card6"),
        ("is_night", "DeviceType", "night_DeviceType"),
        ("is_weekend", "ProductCD", "weekend_ProductCD"),
        ("is_weekend", "card6", "weekend_card6"),
        ("new_device_or_identity_flag", "ProductCD", "new_identity_ProductCD"),
        ("new_device_or_identity_flag", "DeviceType", "new_identity_DeviceType"),
        ("email_domain_mismatch", "ProductCD", "email_mismatch_ProductCD"),
        ("email_domain_mismatch", "card6", "email_mismatch_card6"),
        ("is_large_dist1", "ProductCD", "large_distance_ProductCD"),
        ("is_large_dist1", "card6", "large_distance_card6"),
        ("has_identity_info", "ProductCD", "identity_ProductCD"),
        ("has_identity_info", "card6", "identity_card6"),
    ]

    if "has_identity_info" in df.columns:
        df["missing_identity_flag"] = (df["has_identity_info"] == 0).astype("int8")
        interactions.extend([
            ("missing_identity_flag", "ProductCD", "missing_identity_ProductCD"),
            ("missing_identity_flag", "card6", "missing_identity_card6"),
        ])

    for col1, col2, new_col in interactions:
        df = self._add_interaction(df, col1, col2, new_col)

    return df.copy()


def _fit_frequency_maps(self, train_df):
    frequency_maps = {}

    for col in self.frequency_cols:
        if col in train_df.columns:
            frequency_maps[col] = (
                train_df[col]
                .astype("string")
                .fillna("missing")
                .value_counts(dropna=False)
            )

    return frequency_maps


def _add_frequency_features(self, df):
    df = df.copy()

    for col, freq_map in self.frequency_maps_.items():
        if col not in df.columns:
            continue

        safe_col = (
            col
            .replace(" ", "_")
            .replace("/", "_")
            .replace(".", "_")
        )

        values = df[col].astype("string").fillna("missing")

        df[f"{safe_col}_freq"] = values.map(freq_map).fillna(0).astype("int32")
        df[f"rare_{safe_col}_flag"] = (
            df[f"{safe_col}_freq"] < self.rare_threshold
        ).astype("int8")

    if "rare_DeviceInfo_clean_flag" in df.columns:
        df["rare_device_flag"] = df["rare_DeviceInfo_clean_flag"].astype("int8")

    if "rare_P_emaildomain_flag" in df.columns:
        df["rare_email_domain_flag"] = df["rare_P_emaildomain_flag"].astype("int8")

    if "rare_card1_addr1_flag" in df.columns:
        df["rare_card_region_flag"] = df["rare_card1_addr1_flag"].astype("int8")

    return df.copy()


def _fit_group_amount_stats(self, train_df):
    amount_stat_maps = {}

    for col in self.group_amount_cols:
        if col not in train_df.columns:
            continue

        stats = (
            train_df
            .groupby(col, dropna=False)["TransactionAmt"]
            .agg(["mean", "std", "median", "count"])
            .astype("float32")
        )

        amount_stat_maps[col] = {
            "mean": stats["mean"].to_dict(),
            "std": stats["std"].to_dict(),
            "median": stats["median"].to_dict(),
            "count": stats["count"].to_dict()
        }

    return amount_stat_maps


def _add_group_amount_stats(self, df):
    df = df.copy()
    new_features = {}

    for col, stat_maps in self.amount_stat_maps_.items():
        if col not in df.columns:
            continue

        safe_col = (
            col
            .replace(" ", "_")
            .replace("/", "_")
            .replace(".", "_")
        )

        group_values = df[col]

        group_mean = group_values.map(stat_maps["mean"]).astype("float32")
        group_std = group_values.map(stat_maps["std"]).astype("float32")
        group_median = group_values.map(stat_maps["median"]).astype("float32")
        group_count = group_values.map(stat_maps["count"]).fillna(0).astype("float32")

        amount = df["TransactionAmt"].astype("float32")

        amount_dev = (amount - group_mean).astype("float32")
        amount_abs_dev = amount_dev.abs().astype("float32")
        amount_ratio = (amount / (group_mean + 1)).astype("float32")

        safe_std = group_std.replace(0, np.nan)
        amount_zscore = (
            amount_dev / safe_std
        ).replace([np.inf, -np.inf], np.nan).astype("float32")

        new_features[f"{safe_col}_TransactionAmt_mean"] = group_mean
        new_features[f"{safe_col}_TransactionAmt_std"] = group_std
        new_features[f"{safe_col}_TransactionAmt_median"] = group_median
        new_features[f"{safe_col}_TransactionAmt_count"] = group_count
        new_features[f"{safe_col}_TransactionAmt_dev"] = amount_dev
        new_features[f"{safe_col}_TransactionAmt_abs_dev"] = amount_abs_dev
        new_features[f"{safe_col}_TransactionAmt_ratio"] = amount_ratio
        new_features[f"{safe_col}_TransactionAmt_zscore"] = amount_zscore

    if new_features:
        df = pd.concat(
            [df, pd.DataFrame(new_features, index=df.index)],
            axis=1
        )

    return df.copy()


def fit(self, train_df):
    """
    Fit preprocessing artifacts using training data only.
    """
    train_base = self._create_basic_features(train_df)

    self.thresholds_ = self._fit_thresholds(train_base)

    train_thresholded = self._add_threshold_risk_flags(train_base)
    train_thresholded = self._add_cross_domain_interactions(train_thresholded)

    self.amount_stat_maps_ = self._fit_group_amount_stats(train_thresholded)
    self.frequency_maps_ = self._fit_frequency_maps(train_thresholded)

    train_fe = self._add_group_amount_stats(train_thresholded)
    train_fe = self._add_frequency_features(train_fe)

    drop_cols = list(self.id_cols) + [self.target_col]

    self.feature_cols_ = [
        col for col in train_fe.columns
        if col not in drop_cols
    ]

    return self


def transform(self, df, return_features_only=True):
    """
    Transform raw transaction data using fitted preprocessing artifacts.
    """
    if self.thresholds_ is None:
        raise ValueError("FraudFeatureEngineer is not fitted yet.")

    df_fe = self._create_basic_features(df)
    df_fe = self._add_threshold_risk_flags(df_fe)
    df_fe = self._add_cross_domain_interactions(df_fe)
    df_fe = self._add_group_amount_stats(df_fe)
    df_fe = self._add_frequency_features(df_fe)

    if return_features_only:
        for col in self.feature_cols_:
            if col not in df_fe.columns:
                df_fe[col] = np.nan

        return df_fe[self.feature_cols_].copy()

    return df_fe.copy()


def fit_transform(self, train_df, return_features_only=True):
    self.fit(train_df)
    return self.transform(train_df, return_features_only=return_features_only)


# Attach fitted methods to class
FraudFeatureEngineer._fit_thresholds = _fit_thresholds
FraudFeatureEngineer._add_threshold_risk_flags = _add_threshold_risk_flags
FraudFeatureEngineer._add_cross_domain_interactions = _add_cross_domain_interactions
FraudFeatureEngineer._fit_frequency_maps = _fit_frequency_maps
FraudFeatureEngineer._add_frequency_features = _add_frequency_features
FraudFeatureEngineer._fit_group_amount_stats = _fit_group_amount_stats
FraudFeatureEngineer._add_group_amount_stats = _add_group_amount_stats
FraudFeatureEngineer.fit = fit
FraudFeatureEngineer.transform = transform
FraudFeatureEngineer.fit_transform = fit_transform