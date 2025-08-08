from typing import Any, Dict, List, Optional, Tuple, Union
import shap
import numpy as np
import pandas as pd
from time import perf_counter
from .logger_setup import setup_logger

logger = setup_logger("fir_log.txt", to_console=True)


def get_shap_values(
    model: Any,
    X_test: pd.DataFrame,
    target_class: Optional[Union[int, str]] = None,
) -> Any:
    """
    Compute SHAP values for a tree-based model using SHAP's TreeExplainer.

    This function creates a SHAP TreeExplainer for the provided model and computes SHAP values for the given test data.
    For multi-class models, you can specify a target_class to extract SHAP values for a specific class.
    The function logs progress and timing information.

    :param model: Trained tree-based model.
    :param X_test: Test features as a pandas DataFrame or numpy array.
    :param target_class: (Optional) Integer index or class label for which to extract SHAP values in multi-class problems.

    :raises ValueError: If target_class is specified but not found in model classes.

    :return: SHAP Explanation object containing SHAP values for the test data (for the specified class if multi-class).
    """
    logger.info(f"Getting SHAP values for {type(model).__name__}")
    start = perf_counter()
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer(X_test)
    end = perf_counter()
    logger.info(
        f"✔ Finished getting SHAP values for {type(model).__name__} in {end - start:.2f} seconds"
    )

    # Handle multi-class SHAP value output
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        # Determine class index
        if target_class is not None:
            # If target_class is a label, convert to index
            if hasattr(model, "classes_"):
                class_labels = list(model.classes_)
                if target_class in class_labels:
                    class_idx = class_labels.index(target_class)
                else:
                    raise ValueError(
                        f"target_class {target_class} not found in model.classes_: {class_labels}"
                    )
            else:
                class_idx = int(target_class)
        else:
            class_idx = 0  # Default to first class

        shap_values = shap.Explanation(
            values=shap_values.values[:, :, class_idx],
            base_values=shap_values.base_values[:, class_idx]
            if shap_values.base_values.ndim == 2
            else shap_values.base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    return shap_values


def get_tree_rankings(shap_values: Any) -> Dict[str, float]:
    """
    Computes feature importances based on mean absolute SHAP values from a tree-based model.

    :param shap_values: SHAP Explanation object containing SHAP values, feature names, and associated data.

    :return: Dictionary where keys are feature names (str) and values are the corresponding mean absolute SHAP values (float).
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    rankings = {
        feature_name: mean_abs_shap_val
        for feature_name, mean_abs_shap_val in zip(
            shap_values.feature_names, mean_abs_shap
        )
    }
    logger.info("Computed feature importances based on mean absolute SHAP values.")
    return rankings


def combine_rankings(*rankings_dicts: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Combine multiple feature importance dictionaries into a single normalized ranking.

    :param rankings_dicts: One or more dictionaries where each key is a feature name (str) and each value is a raw importance score (float).

    :return: Sorted list of (feature_name, normalized_importance) tuples representing the top 10 features.
    """
    normalized_rankings = []
    for rankings_dict in rankings_dicts:
        rank_vals_sum = sum(rankings_dict.values())
        normalized_rankings.append(
            {
                feature_name: orig_rank / rank_vals_sum
                for feature_name, orig_rank in rankings_dict.items()
            }
        )

    combined_rankings = []
    for feature_name in normalized_rankings[0].keys():
        normalized_ranking = sum(
            [rankings_dict[feature_name] for rankings_dict in normalized_rankings]
        ) / len(normalized_rankings)
        combined_rankings.append((feature_name, normalized_ranking))
    logger.info("Combined and normalized feature importance rankings.")
    return sorted(combined_rankings, key=lambda x: x[1], reverse=True)[:10]


def find_quantile_ranges(
    shap_vals: Any,
    X: pd.DataFrame,
    features: Optional[List[str]] = None,
    quantile: float = 0.10,
    direction: str = "low",
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    For each feature, find the range of X values whose raw SHAP contributions lie in the specified tail.

    This function identifies, for each feature, the range of values in X where the SHAP values are in the specified quantile tail.
    It can be used to find value ranges associated with the most positive or negative SHAP contributions.

    :param shap_vals: SHAP Explanation object (with .values, .feature_names).
    :param X: DataFrame of the same rows that produced shap_vals.
    :param features: List of feature names to include (defaults to all).
    :param quantile: Quantile threshold (e.g., 0.10 for bottom 10%).
    :param direction: "low" for shap ≤ quantile cutoff, "high" for shap ≥ (1 - quantile) cutoff.

    :return: Dictionary {feature: (min_value, max_value)} for each feature. If no rows in that tail, returns (None, None).
    """
    vals = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
    names = (
        shap_vals.feature_names
        if hasattr(shap_vals, "feature_names")
        else list(X.columns)
    )

    if features is None:
        features = names

    out = {}
    for feat in features:
        j = names.index(feat)
        one_shap = vals[:, j]

        if direction == "low":
            cutoff = np.quantile(one_shap, quantile)
            mask = one_shap <= cutoff
        else:  # "high"
            cutoff = np.quantile(one_shap, 1 - quantile)
            mask = one_shap >= cutoff

        if mask.any():
            xs = X.iloc[mask, j]
            out[feat] = (float(xs.min()), float(xs.max()))
        else:
            out[feat] = (None, None)
    logger.info("Found quantile ranges for SHAP values.")
    return out


def intersect_ranges(
    *ranked_feat_min_max_dicts: Dict[str, Tuple[Optional[float], Optional[float]]]
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Take a list of per-model {feature: (min, max)} dictionaries and return the intersection range for each feature.

    This function computes, for each feature, the intersection of the value ranges across all provided models.
    If the intersection is empty, it returns the full range covered by all models for that feature.

    :param ranked_feat_min_max_dicts: One or more dictionaries mapping feature names to (min, max) tuples.

    :return: Dictionary {feature: (min, max)} representing the intersection range for each feature.
    """

    feats = list(ranked_feat_min_max_dicts[0].keys())

    for rd in ranked_feat_min_max_dicts[1:]:
        if set(rd) != set(feats):
            logger.error("Feature keys differ among models.")
            raise ValueError("Feature keys differ among models.")

    final_dict = {}

    for feat in feats:
        global_feat_min_vals = [
            ranked_dict[feat][0] for ranked_dict in ranked_feat_min_max_dicts
        ]
        global_feat_max_vals = [
            ranked_dict[feat][1] for ranked_dict in ranked_feat_min_max_dicts
        ]

        lo = max(global_feat_min_vals)  # lo
        hi = min(global_feat_max_vals)  # hi

        if lo <= hi:
            final_dict[feat] = (lo, hi)
        else:
            true_min_range = min(global_feat_min_vals)
            true_max_range = max(global_feat_max_vals)
            final_dict[feat] = (true_min_range, true_max_range)

    logger.info("Intersected feature value ranges across models.")
    return final_dict


def display_ranking_and_range(
    rankings,
    range_dict,
    top_n=10,
):
    """
    Displays the top N features based on mean absolute SHAP value, along with their optimal value ranges.

    This function logs a formatted table of the top features, their mean SHAP values, and the corresponding optimal value ranges,
    with columns dynamically sized to fit the longest feature name and aligned decimal points/brackets.

    :param rankings: List of (feature_name, importance) tuples, sorted by importance.
    :param range_dict: Dictionary mapping feature names to (min, max) value ranges.
    :param top_n: Number of top features to display (default: 10).
    """
    # Determine dynamic column widths
    feature_names = [feat for feat, _ in rankings[:top_n]]
    max_feat_len = max(
        len("FEATURE"), max((len(str(f)) for f in feature_names), default=0)
    )
    shap_col_width = 12  # Enough for 'MEAN SHAP'
    range_strs = []
    for feat in feature_names:
        rng = range_dict.get(feat, (None, None))
        rng_str = f"[{rng[0]:.5f}, {rng[1]:.5f}]" if None not in rng else "N/A"
        range_strs.append(rng_str)
    max_range_len = max(
        len("OPTIMAL RANGE"), max((len(r) for r in range_strs), default=0)
    )

    # Prepare header
    header = (
        f"{'RANK':<5} "
        f"{'FEATURE':<{max_feat_len}} "
        f"{'MEAN SHAP':>{shap_col_width}} "
        f"{'OPTIMAL RANGE':>{max_range_len}}"
    )
    logger.info("\n" + header)
    logger.info("-" * len(header))

    # Prepare data rows with aligned decimals and brackets
    for i, (feat, mean_shap) in enumerate(rankings[:top_n], 1):
        rng_str = range_strs[i - 1]
        shap_str = f"{mean_shap:.5f}"
        logger.info(
            f"{i:<5} "
            f"{feat:<{max_feat_len}} "
            f"{shap_str:>{shap_col_width}} "
            f"{rng_str:>{max_range_len}}",
        )
