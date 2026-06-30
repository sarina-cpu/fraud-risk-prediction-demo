import numpy as np
import pandas as pd


def _require_shap():
    try:
        import shap
        return shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with `pip install shap` or add it to requirements.txt."
        ) from exc


def get_pipeline_step(pipeline, step_name: str):
    if not hasattr(pipeline, "named_steps") or step_name not in pipeline.named_steps:
        raise ValueError(f"Expected sklearn Pipeline with a '{step_name}' step.")
    return pipeline.named_steps[step_name]


def get_preprocessed_matrix(pipeline, X_model: pd.DataFrame):
    preprocessor = get_pipeline_step(pipeline, "preprocess")
    return preprocessor.transform(X_model)


def _clean_feature_name(name: str) -> str:
    name = str(name)
    if "__" in name:
        name = name.split("__", 1)[1]
    return name


def get_pipeline_feature_names(pipeline, X_model: pd.DataFrame | None = None) -> list[str]:
    """
    Return post-preprocessing feature names.

    For the manual simulation model, categorical features are one-hot encoded
    before LightGBM. Therefore SHAP contributions are in the encoded feature
    space, not the original raw dataframe feature space.
    """
    preprocessor = get_pipeline_step(pipeline, "preprocess")

    try:
        feature_names = [_clean_feature_name(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        feature_names = []

    if feature_names:
        if X_model is not None:
            try:
                n_out = get_preprocessed_matrix(pipeline, X_model.iloc[:1]).shape[1]
                if len(feature_names) == n_out:
                    return feature_names
            except Exception:
                return feature_names
        else:
            return feature_names

    # Fallback for older sklearn versions.
    feature_names = []
    try:
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            if cols is None:
                continue
            feature_names.extend(list(cols))
    except Exception:
        feature_names = []

    if X_model is not None:
        try:
            n_out = get_preprocessed_matrix(pipeline, X_model.iloc[:1]).shape[1]
            if len(feature_names) != n_out:
                return [f"model_feature_{i}" for i in range(n_out)]
        except Exception:
            pass

    return feature_names


def _to_dense_matrix(matrix):
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _row_to_dense_array(row_preprocessed):
    return _to_dense_matrix(row_preprocessed).reshape(-1)


def _build_shap_df(feature_names, row_values, values, method="tree_shap"):
    values = np.asarray(values).reshape(-1)
    row_values = np.asarray(row_values).reshape(-1)

    if len(feature_names) != len(values):
        feature_names = [f"model_feature_{i}" for i in range(len(values))]

    if len(row_values) != len(values):
        row_values = np.resize(row_values, len(values))

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "model_input_value": row_values,
        "shap_value": values,
    })
    shap_df["abs_shap_value"] = shap_df["shap_value"].abs()
    shap_df["impact_direction"] = np.where(
        shap_df["shap_value"] >= 0,
        "Increases fraud risk",
        "Decreases fraud risk",
    )
    shap_df["explanation_method"] = method
    return shap_df


def _get_shap_values_tree_explainer(lgbm_model, row_preprocessed):
    shap = _require_shap()
    explainer = shap.TreeExplainer(lgbm_model)

    errors = []
    for candidate in [row_preprocessed, _to_dense_matrix(row_preprocessed)]:
        try:
            shap_values = explainer.shap_values(candidate)
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]
            return np.asarray(values).reshape(-1)
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError("TreeExplainer failed: " + " | ".join(errors))


def _get_shap_values_lightgbm_contrib(lgbm_model, row_preprocessed):
    """
    Robust fallback using LightGBM's native prediction contributions.

    LightGBM returns one extra value for the expected value / bias term.
    We drop that last value so the remaining values align to model features.
    """
    errors = []
    candidates = [row_preprocessed, _to_dense_matrix(row_preprocessed)]

    for candidate in candidates:
        try:
            if hasattr(lgbm_model, "booster_"):
                contrib = lgbm_model.booster_.predict(candidate, pred_contrib=True)
            elif hasattr(lgbm_model, "predict"):
                contrib = lgbm_model.predict(candidate, pred_contrib=True)
            else:
                raise ValueError("Could not access LightGBM booster for contribution fallback.")

            contrib = np.asarray(contrib)
            if contrib.ndim == 2:
                contrib = contrib[0]
            contrib = np.asarray(contrib).reshape(-1)

            # Drop expected value / bias term.
            if len(contrib) > 1:
                contrib = contrib[:-1]

            return contrib
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError("LightGBM pred_contrib failed: " + " | ".join(errors))


def _neutral_value_for_feature(series: pd.Series):
    """
    Return a safe neutral value for local sensitivity fallback.

    Numeric features use NaN so the pipeline's imputer handles them.
    Categorical/string features use 'missing'.
    """
    try:
        if pd.api.types.is_numeric_dtype(series):
            return np.nan
    except Exception:
        pass
    return "missing"


def _get_local_delta_explanation(pipeline, X_model: pd.DataFrame, row_position: int, top_n: int):
    """
    Last-resort explanation that does not depend on the SHAP package or the
    LightGBM contribution API.

    It measures how the predicted fraud score changes when each original model
    feature is neutralized one at a time. This is not exact SHAP, but it is a
    stable local explanation and keeps the manual page usable if SHAP fails.
    """
    row = X_model.iloc[[row_position]].copy()
    base_score = float(pipeline.predict_proba(row)[:, 1][0])

    records = []
    for feature in X_model.columns:
        perturbed = row.copy()
        original_value = row.iloc[0][feature]
        perturbed.loc[perturbed.index[0], feature] = _neutral_value_for_feature(X_model[feature])

        try:
            perturbed_score = float(pipeline.predict_proba(perturbed)[:, 1][0])
            impact = base_score - perturbed_score
        except Exception:
            impact = 0.0

        records.append({
            "feature": feature,
            "model_input_value": original_value,
            "shap_value": impact,
            "abs_shap_value": abs(impact),
            "impact_direction": "Increases fraud risk" if impact >= 0 else "Decreases fraud risk",
            "explanation_method": "local_delta_fallback",
        })

    return (
        pd.DataFrame(records)
        .sort_values("abs_shap_value", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def explain_single_transaction(
    pipeline,
    X_model: pd.DataFrame,
    row_position: int,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Return top local risk drivers for one transaction/scenario.

    Order of explanation methods:
    1. SHAP TreeExplainer
    2. LightGBM native pred_contrib
    3. Local delta fallback over the original model features

    The third fallback is intentionally included because the manual simulator
    should never fail just because a SHAP/LightGBM/scipy version combination is
    fussy about sparse one-hot encoded rows.
    """
    if row_position < 0 or row_position >= len(X_model):
        raise IndexError(f"row_position {row_position} is outside X_model length {len(X_model)}")

    lgbm_model = get_pipeline_step(pipeline, "model")

    try:
        X_preprocessed = get_preprocessed_matrix(pipeline, X_model)
        row_preprocessed = X_preprocessed[row_position : row_position + 1]
        row_values = _row_to_dense_array(row_preprocessed)
        feature_names = get_pipeline_feature_names(pipeline, X_model=X_model)

        try:
            values = _get_shap_values_tree_explainer(lgbm_model, row_preprocessed)
            method = "tree_shap"
        except Exception:
            values = _get_shap_values_lightgbm_contrib(lgbm_model, row_preprocessed)
            method = "lightgbm_pred_contrib"

        shap_df = _build_shap_df(
            feature_names=feature_names,
            row_values=row_values,
            values=values,
            method=method,
        )

        return (
            shap_df
            .sort_values("abs_shap_value", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    except Exception:
        return _get_local_delta_explanation(
            pipeline=pipeline,
            X_model=X_model,
            row_position=row_position,
            top_n=top_n,
        )
