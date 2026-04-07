"""Structured feature engineering.

Builds clean feature matrices for OLS and XGBoost from the listings DataFrame.
Handles numeric imputation, boolean encoding, and one-hot encoding of
categorical variables. The structured baseline is intentionally lean and
theory-driven so both the baseline and augmented models share the same
research-backed control set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# ── Column groups ──────────────────────────────────────────────────────────────

BASELINE_NUMERIC: list[str] = [
    "log_living_area", "log_lot_size", "property_age",
    "bedrooms", "bathrooms", "stories", "garage_spaces",
]

BASELINE_BOOL: list[str] = [
    "has_fireplace", "has_pool",
]

BASELINE_CATEGORICAL: list[str] = ["home_type", "zip_code"]

TARGET_COL = "log_price"

# LLM score columns (1-5 integers, nullable)
LLM_SCORE_COLS: list[str] = [
    "llm_luxury_score", "llm_uniqueness_score",
    "llm_renovation_quality_score", "llm_curb_appeal_score",
    "llm_spaciousness_score",
]

# LLM boolean flag columns (0/1 integers, nullable → filled with 0)
LLM_FLAG_COLS: list[str] = [
    "llm_is_unique_property", "llm_has_premium_finishes", "llm_is_recently_updated",
    "llm_foreclosure_flag", "llm_auction_flag", "llm_as_is_flag",
    "llm_fixer_upper_flag", "llm_needs_repair_flag", "llm_water_damage_flag",
    "llm_fire_damage_flag", "llm_foundation_issue_flag", "llm_roof_issue_flag",
    "llm_mold_flag", "llm_tenant_occupied_flag", "llm_cash_only_flag",
    "llm_investor_special_flag",
]


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    include_llm: bool = False,
    extra_numeric: list[str] | None = None,
    fill_numeric: str = "median",
) -> pd.DataFrame:
    """Construct a clean feature matrix ready for sklearn / statsmodels.

    Parameters
    ----------
    df            : cleaned listings DataFrame
    include_llm   : whether to append LLM score + flag columns
    extra_numeric : any additional numeric columns to include
    fill_numeric  : "median" (default) or "zero"

    Returns
    -------
    X : pd.DataFrame of shape (n_listings, n_features)
    """
    parts: list[pd.DataFrame] = []

    # ── Numeric ────────────────────────────────────────────────────────────────
    num_cols = [c for c in BASELINE_NUMERIC + (extra_numeric or []) if c in df.columns]
    if num_cols:
        num_df = df[num_cols].copy().apply(pd.to_numeric, errors="coerce")
        if fill_numeric == "median":
            num_df = num_df.fillna(num_df.median())
        else:
            num_df = num_df.fillna(0)
        parts.append(num_df)

    # ── Boolean → int ──────────────────────────────────────────────────────────
    bool_cols = [c for c in BASELINE_BOOL if c in df.columns]
    if bool_cols:
        bool_df = df[bool_cols].copy().fillna(False).astype(int)
        parts.append(bool_df)

    # ── Categorical → one-hot ─────────────────────────────────────────────────
    MIN_CATEGORY_COUNT = 5
    cat_cols = [c for c in BASELINE_CATEGORICAL if c in df.columns]
    if cat_cols:
        cat_raw = df[cat_cols].fillna("UNKNOWN").copy()
        # Merge rare categories (< MIN_CATEGORY_COUNT observations) into "OTHER"
        for col in cat_cols:
            counts = cat_raw[col].value_counts()
            rare = counts[counts < MIN_CATEGORY_COUNT].index
            if len(rare):
                cat_raw[col] = cat_raw[col].replace(rare, "OTHER")
                logger.info(
                    f"Merged {len(rare)} rare {col} categories into OTHER: "
                    f"{list(rare)}"
                )
        cat_df = pd.get_dummies(cat_raw, drop_first=True, dtype=int)
        parts.append(cat_df)

    # ── LLM features ───────────────────────────────────────────────────────────
    if include_llm:
        score_cols = [c for c in LLM_SCORE_COLS if c in df.columns]
        if score_cols:
            sc_df = df[score_cols].copy().apply(pd.to_numeric, errors="coerce")
            sc_df = sc_df.fillna(0)
            parts.append(sc_df)
            # Polynomial term: uniqueness has diminishing/reversing returns
            if "llm_uniqueness_score" in sc_df.columns:
                sc_df_sq = pd.DataFrame(
                    {"llm_uniqueness_score_sq": sc_df["llm_uniqueness_score"] ** 2},
                    index=sc_df.index,
                )
                parts.append(sc_df_sq)

        flag_cols = [c for c in LLM_FLAG_COLS if c in df.columns]
        if flag_cols:
            fl_df = df[flag_cols].copy().fillna(0).astype(int)
            parts.append(fl_df)

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    logger.info(
        f"Feature matrix built: {X.shape[0]:,} rows × {X.shape[1]} features"
        f"{'  [+LLM]' if include_llm else ''}"
    )
    return X


def get_target(df: pd.DataFrame, col: str = TARGET_COL) -> pd.Series:
    return df[col].copy()


def feature_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Quick overview of the feature matrix: dtype, n_nonzero, share of zeros."""
    stats = pd.DataFrame({
        "dtype":     X.dtypes,
        "n_nonzero": (X != 0).sum(),
        "pct_zero":  ((X == 0).sum() / len(X) * 100).round(1),
        "n_missing": X.isnull().sum(),
    })
    return stats


def build_single_feature_row(
    feature_cols: list[str],
    *,
    bedrooms: int,
    bathrooms: float,
    living_area_sqft: float,
    lot_size_sqft: float,
    property_age: int,
    stories: float,
    garage_spaces: float,
    zip_code: str,
    home_type: str,
    has_fireplace: bool = False,
    has_pool: bool = False,
    use_llm: bool = True,
    llm_values: dict[str, float | int | None] | None = None,
) -> pd.DataFrame:
    """Build a single prediction row aligned to an existing feature matrix."""
    row = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    numeric_values = {
        "log_living_area": np.log(max(living_area_sqft, 1)),
        "log_lot_size": np.log(max(lot_size_sqft, 1)),
        "property_age": property_age,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "garage_spaces": garage_spaces,
    }
    for col, val in numeric_values.items():
        if col in row.columns:
            row[col] = float(val)

    for col, val in {
        "has_fireplace": has_fireplace,
        "has_pool": has_pool,
    }.items():
        if col in row.columns:
            row[col] = int(val)

    zip_col = f"zip_code_{zip_code}"
    if zip_col in row.columns:
        row[zip_col] = 1
    elif "zip_code_OTHER" in row.columns:
        row["zip_code_OTHER"] = 1

    ht_col = f"home_type_{home_type}"
    if ht_col in row.columns:
        row[ht_col] = 1
    elif "home_type_OTHER" in row.columns:
        row["home_type_OTHER"] = 1

    score_defaults = {
        "llm_luxury_score": 3.0,
        "llm_uniqueness_score": 3.0,
        "llm_renovation_quality_score": 3.0,
        "llm_curb_appeal_score": 3.0,
        "llm_spaciousness_score": 3.0,
    }
    flag_defaults = {
        "llm_is_unique_property": 0,
        "llm_has_premium_finishes": 0,
        "llm_is_recently_updated": 0,
        "llm_foreclosure_flag": 0,
        "llm_auction_flag": 0,
        "llm_as_is_flag": 0,
        "llm_fixer_upper_flag": 0,
        "llm_needs_repair_flag": 0,
        "llm_water_damage_flag": 0,
        "llm_fire_damage_flag": 0,
        "llm_foundation_issue_flag": 0,
        "llm_roof_issue_flag": 0,
        "llm_mold_flag": 0,
        "llm_tenant_occupied_flag": 0,
        "llm_cash_only_flag": 0,
        "llm_investor_special_flag": 0,
    }
    llm_values = llm_values or {}

    if not use_llm:
        for col in score_defaults:
            if col in row.columns:
                row[col] = 0
        if "llm_uniqueness_score_sq" in row.columns:
            row["llm_uniqueness_score_sq"] = 0
        for col, val in flag_defaults.items():
            if col in row.columns:
                row[col] = int(val)
        return row

    for col, default in score_defaults.items():
        val = llm_values.get(col, default)
        if col in row.columns:
            row[col] = 0 if val is None else float(val)

    # Polynomial term for uniqueness
    if "llm_uniqueness_score_sq" in row.columns:
        u_val = float(row["llm_uniqueness_score"].iloc[0])
        row["llm_uniqueness_score_sq"] = u_val ** 2

    for col, default in flag_defaults.items():
        val = llm_values.get(col, default)
        if col in row.columns:
            row[col] = 0 if val is None else int(val)

    return row
