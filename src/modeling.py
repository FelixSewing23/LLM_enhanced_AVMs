"""Hedonic regression and XGBoost models.

All functions return (model, metrics_dict) or (model, metrics_dict, coef_df)
so the notebook can display tables without containing modelling logic.

Metrics are reported on original (dollar) scale where possible so they are
economically interpretable, but R² is always on the log scale (standard for
hedonic models in the literature).
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import DMatrix, XGBRegressor, cv as xgb_cv

from scipy import stats

from src.config import RANDOM_SEED, TEST_SIZE, CV_FOLDS, CV_REPEATS
from src.utils import get_logger

try:
    import shap as _shap
except ImportError as _e:   # shap is optional
    _shap = None
    _SHAP_IMPORT_ERROR: ImportError | None = _e
else:
    _SHAP_IMPORT_ERROR = None

logger  = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


XGB_NUM_BOOST_ROUND = 500
XGB_EARLY_STOPPING_ROUNDS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    n: Optional[int] = None,
    p: Optional[int] = None,
) -> dict:
    """Compute RMSE, MAE, MAPE, R², Adj. R² with back-transformation from log scale.

    Parameters
    ----------
    p : number of predictors (excluding intercept). When provided, adjusted R²
        is computed as 1 - (1 - R²) * (n - 1) / (n - p - 1).
    """
    y_true_orig = np.exp(y_true)
    y_pred_orig = np.exp(y_pred)

    _n   = n or len(y_true)
    rmse = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
    mae  = float(mean_absolute_error(y_true_orig, y_pred_orig))
    mape = float(np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100)
    r2   = float(r2_score(y_true, y_pred))   # on log scale (standard)

    adj_r2 = None
    if p is not None and _n - p - 1 > 0:
        adj_r2 = float(1 - (1 - r2) * (_n - 1) / (_n - p - 1))

    out = {
        "model":   model_name,
        "n":       _n,
        "R²":      r2,
        "Adj. R²": adj_r2,
        "RMSE":    rmse,
        "MAE":     mae,
        "MAPE (%)": mape,
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared hold-out split
# ─────────────────────────────────────────────────────────────────────────────

def shared_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a single shared 80/20 hold-out split for all models.

    Every model in the comparison must use these exact splits so test-set
    metrics are directly comparable.
    """
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# OLS hedonic
# ─────────────────────────────────────────────────────────────────────────────

def fit_ols(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "OLS",
    run_cv: bool = True,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> tuple[object, dict, pd.DataFrame]:
    """Fit OLS with HC3 robust standard errors.

    Returns
    -------
    model      : fitted statsmodels OLSResults
    metrics    : dict with R², RMSE, MAE, MAPE, optional CV/test metrics
    coef_table : DataFrame of coefficients, se, t, p, CI
    """
    X_c   = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_c).fit(cov_type="HC3")
    n_features = X.shape[1]  # predictors excl. intercept

    in_pred = model.predict(X_c).values
    metrics = regression_metrics(
        y.values, in_pred, model_name=model_name, n=len(y), p=n_features,
    )
    # Prefer statsmodels' own adjusted R² for in-sample (more precise)
    metrics["Adj. R²"] = float(model.rsquared_adj)
    metrics["sample"] = "in-sample"

    # Test-set metrics on shared hold-out
    if X_test is not None and y_test is not None:
        X_te_c = sm.add_constant(X_test, has_constant="add")
        test_pred = model.predict(X_te_c).values
        test_m = regression_metrics(
            y_test.values, test_pred, model_name=model_name,
            n=len(y_test), p=n_features,
        )
        metrics["test_R²"] = test_m["R²"]
        metrics["test_Adj. R²"] = test_m["Adj. R²"]
        metrics["test_RMSE"] = test_m["RMSE"]
        metrics["test_MAE"] = test_m["MAE"]
        metrics["test_MAPE (%)"] = test_m["MAPE (%)"]
        metrics["n_test"] = len(y_test)

    if run_cv:
        cv_m = _repeated_cv_ols(X, y)
        metrics["CV_R²"] = cv_m["CV_R²_mean"]
        metrics["CV_RMSE"] = cv_m["CV_RMSE_mean"]
        metrics["CV_R²_mean"] = cv_m["CV_R²_mean"]
        metrics["CV_R²_sd"] = cv_m["CV_R²_sd"]
        metrics["CV_RMSE_mean"] = cv_m["CV_RMSE_mean"]
        metrics["CV_RMSE_sd"] = cv_m["CV_RMSE_sd"]
        metrics["fold_rmses"] = cv_m["fold_rmses"]
        if "CV_Adj. R²_mean" in cv_m:
            metrics["CV_Adj. R²_mean"] = cv_m["CV_Adj. R²_mean"]
            metrics["CV_Adj. R²_sd"] = cv_m["CV_Adj. R²_sd"]

    coef_table = pd.DataFrame({
        "coef":    model.params,
        "std_err": model.bse,
        "t_stat":  model.tvalues,
        "p_value": model.pvalues,
        "ci_low":  model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    }).round(4)

    logger.info(
        f"[{model_name}] R²={metrics['R²']:.3f}  "
        f"RMSE=${metrics['RMSE']:,.0f}  n={len(y):,}"
    )
    return model, metrics, coef_table


def _repeated_cv_ols(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """5x5 repeated K-fold CV for OLS. Returns mean +/- SD of metrics and per-fold RMSE array."""
    rkf = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    rmses, r2s, adj_r2s = [], [], []
    n_features = X.shape[1]

    for tr, te in rkf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        Xtr_c = sm.add_constant(Xtr, has_constant="add")
        Xte_c = sm.add_constant(Xte, has_constant="add")
        m = sm.OLS(ytr, Xtr_c).fit()
        pred = m.predict(Xte_c).values
        m_d = regression_metrics(yte.values, pred, p=n_features)
        rmses.append(m_d["RMSE"])
        r2s.append(m_d["R²"])
        if m_d["Adj. R²"] is not None:
            adj_r2s.append(m_d["Adj. R²"])

    result = {
        "CV_R²_mean": float(np.mean(r2s)),
        "CV_R²_sd": float(np.std(r2s)),
        "CV_RMSE_mean": float(np.mean(rmses)),
        "CV_RMSE_sd": float(np.std(rmses)),
        "fold_rmses": np.array(rmses),
    }
    if adj_r2s:
        result["CV_Adj. R²_mean"] = float(np.mean(adj_r2s))
        result["CV_Adj. R²_sd"] = float(np.std(adj_r2s))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def fit_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "XGBoost",
    run_cv: bool = True,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> tuple[XGBRegressor, dict]:
    """Fit XGBoost with training-only tuning and hold-out test evaluation.

    Hyperparameter tuning and early stopping are performed only on the
    training data via an internal 5-fold CV search. The external hold-out test
    set is used only once for final evaluation.

    Returns
    -------
    model   : fitted XGBRegressor
    metrics : dict with R², RMSE, MAE, MAPE, n_train, n_test, optional CV
    """
    if X_test is not None and y_test is not None:
        X_tr, X_te, y_tr, y_te = X, X_test, y, y_test
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

    tuning = _tune_xgb_training_cv(X_tr, y_tr)
    selected_rounds = tuning["best_iteration"]
    model = XGBRegressor(
        n_estimators=selected_rounds,
        **tuning["params"],
    )
    model.fit(X_tr, y_tr, verbose=False)
    model._selected_n_estimators = selected_rounds
    model._tuned_params = tuning["params"]

    n_features = X_tr.shape[1]
    pred    = model.predict(X_te)
    metrics = regression_metrics(y_te.values, pred, model_name=model_name,
                                  n=len(y_te), p=n_features)
    metrics["n_train"] = len(X_tr)
    metrics["n_test"]  = len(X_te)
    metrics["sample"]  = "test-set"
    metrics["best_iteration"] = selected_rounds
    metrics["tuning_CV_log_RMSE"] = tuning["cv_rmse"]

    # Store test-set metrics under canonical keys for two-panel table
    metrics["test_R²"] = metrics["R²"]
    metrics["test_Adj. R²"] = metrics["Adj. R²"]
    metrics["test_RMSE"] = metrics["RMSE"]
    metrics["test_MAE"] = metrics["MAE"]
    metrics["test_MAPE (%)"] = metrics["MAPE (%)"]

    if run_cv:
        cv_m = _repeated_cv_xgb(X_tr, y_tr, tuning["params"], selected_rounds)
        metrics["CV_R²"] = cv_m["CV_R²_mean"]
        metrics["CV_RMSE"] = cv_m["CV_RMSE_mean"]
        metrics["CV_R²_mean"] = cv_m["CV_R²_mean"]
        metrics["CV_R²_sd"] = cv_m["CV_R²_sd"]
        metrics["CV_RMSE_mean"] = cv_m["CV_RMSE_mean"]
        metrics["CV_RMSE_sd"] = cv_m["CV_RMSE_sd"]
        metrics["fold_rmses"] = cv_m["fold_rmses"]
        if "CV_Adj. R²_mean" in cv_m:
            metrics["CV_Adj. R²_mean"] = cv_m["CV_Adj. R²_mean"]
            metrics["CV_Adj. R²_sd"] = cv_m["CV_Adj. R²_sd"]

    logger.info(
        f"[{model_name}] R²={metrics['R²']:.3f}  "
        f"RMSE=${metrics['RMSE']:,.0f}  "
        f"n_test={len(X_te):,}  best_iter={selected_rounds}"
    )
    return model, metrics


def _xgb_base_params() -> dict:
    """Return the fixed XGBoost parameters shared across tuning candidates."""
    return {
        "objective": "reg:squarederror",
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }


def _tune_xgb_training_cv(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Tune XGBoost only within the training sample via 5-fold CV.

    Uses a small grid over key tree-complexity parameters and selects the
    candidate with the lowest mean validation RMSE on the log-price scale.
    The boosting round is chosen via early stopping within the same CV.
    """
    dtrain = DMatrix(X, label=y)
    base_params = _xgb_base_params()
    cv_base_params = {
        **base_params,
        "nthread": -1,
        "verbosity": 0,
    }
    search_results = []

    grid = product(
        [4, 5],        # max_depth
        [3, 5],        # min_child_weight
        [0.0, 0.1],    # gamma
    )
    for max_depth, min_child_weight, gamma in grid:
        cv_params = cv_base_params.copy()
        cv_params.update(
            {
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
            }
        )
        cv_results = xgb_cv(
            params=cv_params,
            dtrain=dtrain,
            num_boost_round=XGB_NUM_BOOST_ROUND,
            nfold=CV_FOLDS,
            metrics="rmse",
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            seed=RANDOM_SEED,
            shuffle=True,
            verbose_eval=False,
        )
        best_idx = int(cv_results["test-rmse-mean"].argmin())
        search_results.append(
            {
                "params": {
                    **base_params,
                    "max_depth": max_depth,
                    "min_child_weight": min_child_weight,
                    "gamma": gamma,
                    "random_state": RANDOM_SEED,
                    "n_jobs": -1,
                    "verbosity": 0,
                },
                "best_iteration": best_idx + 1,
                "cv_rmse": float(cv_results.loc[best_idx, "test-rmse-mean"]),
            }
        )

    best = min(search_results, key=lambda row: row["cv_rmse"])
    logger.info(
        "[XGBoost tuning] "
        f"best log-RMSE={best['cv_rmse']:.4f}  "
        f"best_iter={best['best_iteration']}  "
        f"params={{max_depth={best['params']['max_depth']}, "
        f"min_child_weight={best['params']['min_child_weight']}, "
        f"gamma={best['params']['gamma']}}}"
    )
    return best


def _repeated_cv_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_estimators: int,
) -> dict:
    """5x5 repeated K-fold CV for a fixed tuned XGBoost specification."""
    rkf = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    rmses, r2s, adj_r2s = [], [], []
    n_features = X.shape[1]

    for tr, te in rkf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        m = XGBRegressor(n_estimators=n_estimators, **params)
        m.fit(Xtr, ytr, verbose=False)
        pred = m.predict(Xte)
        m_d = regression_metrics(yte.values, pred, p=n_features)
        rmses.append(m_d["RMSE"])
        r2s.append(m_d["R²"])
        if m_d["Adj. R²"] is not None:
            adj_r2s.append(m_d["Adj. R²"])

    result = {
        "CV_R²_mean": float(np.mean(r2s)),
        "CV_R²_sd": float(np.std(r2s)),
        "CV_RMSE_mean": float(np.mean(rmses)),
        "CV_RMSE_sd": float(np.std(rmses)),
        "fold_rmses": np.array(rmses),
    }
    if adj_r2s:
        result["CV_Adj. R²_mean"] = float(np.mean(adj_r2s))
        result["CV_Adj. R²_sd"] = float(np.std(adj_r2s))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(
    model: XGBRegressor,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame."""
    scores = model.get_booster().get_score(importance_type=importance_type)
    df = pd.DataFrame(
        [(f, scores.get(f, 0.0)) for f in feature_names],
        columns=["feature", "importance"],
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    """Combine multiple metrics dicts into a clean comparison table."""
    df   = pd.DataFrame(metrics_list).set_index("model")
    cols = ["R²", "Adj. R²", "RMSE", "MAE", "MAPE (%)", "CV_R²", "CV_RMSE", "n", "sample"]
    cols = [c for c in cols if c in df.columns]
    df   = df[cols]

    # Format for display
    display = df.copy()
    for col in ["RMSE", "MAE", "CV_RMSE"]:
        if col in display:
            display[col] = display[col].map(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
            )
    for col in ["MAPE (%)"]:
        if col in display:
            display[col] = display[col].map(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
            )
    for col in ["R²", "Adj. R²", "CV_R²"]:
        if col in display:
            display[col] = display[col].map(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )
    return display


def build_two_panel_table(metrics_list: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two-panel model comparison table.

    Panel A: Test-set metrics (R2, RMSE, MAE, MAPE) on shared 20% hold-out
    Panel B: CV metrics (mean +/- SD) from repeated 5-fold on 80% training set

    Returns (panel_a, panel_b) as formatted DataFrames.
    """
    rows_a, rows_b = [], []
    for m in metrics_list:
        name = m.get("model", "?")
        # Panel A: test-set
        rows_a.append({
            "Model": name,
            "R\u00b2": m.get("test_R\u00b2", m.get("R\u00b2")),
            "Adj. R\u00b2": m.get("test_Adj. R\u00b2", m.get("Adj. R\u00b2")),
            "RMSE": m.get("test_RMSE", m.get("RMSE")),
            "MAE": m.get("test_MAE", m.get("MAE")),
            "MAPE (%)": m.get("test_MAPE (%)", m.get("MAPE (%)")),
        })
        # Panel B: CV
        cv_r2_mean = m.get("CV_R\u00b2_mean")
        cv_r2_sd = m.get("CV_R\u00b2_sd")
        cv_adj_r2_mean = m.get("CV_Adj. R\u00b2_mean")
        cv_adj_r2_sd = m.get("CV_Adj. R\u00b2_sd")
        cv_rmse_mean = m.get("CV_RMSE_mean")
        cv_rmse_sd = m.get("CV_RMSE_sd")
        rows_b.append({
            "Model": name,
            "CV R\u00b2": f"{cv_r2_mean:.3f} \u00b1 {cv_r2_sd:.3f}" if cv_r2_mean is not None else "\u2014",
            "CV Adj. R\u00b2": f"{cv_adj_r2_mean:.3f} \u00b1 {cv_adj_r2_sd:.3f}" if cv_adj_r2_mean is not None else "\u2014",
            "CV RMSE": f"${cv_rmse_mean:,.0f} \u00b1 ${cv_rmse_sd:,.0f}" if cv_rmse_mean is not None else "\u2014",
        })

    panel_a = pd.DataFrame(rows_a).set_index("Model")
    panel_b = pd.DataFrame(rows_b).set_index("Model")

    # Format Panel A
    for col in ["RMSE", "MAE"]:
        if col in panel_a:
            panel_a[col] = panel_a[col].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "\u2014")
    for col in ["MAPE (%)"]:
        if col in panel_a:
            panel_a[col] = panel_a[col].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "\u2014")
    for col in ["R\u00b2", "Adj. R\u00b2"]:
        if col in panel_a:
            panel_a[col] = panel_a[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014")

    return panel_a, panel_b


def paired_rmse_test(
    fold_rmses_a: np.ndarray,
    fold_rmses_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict:
    """Paired t-test on fold-level RMSE values.

    Tests H0: mean(RMSE_A) = mean(RMSE_B).
    Positive t-statistic means model A has higher (worse) RMSE.
    """
    t_stat, p_value = stats.ttest_rel(fold_rmses_a, fold_rmses_b)
    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "mean_rmse_a": float(np.mean(fold_rmses_a)),
        "mean_rmse_b": float(np.mean(fold_rmses_b)),
        "rmse_diff": float(np.mean(fold_rmses_a) - np.mean(fold_rmses_b)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "n_folds": len(fold_rmses_a),
        "significant_005": p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for all features in X.

    Returns DataFrame with columns: feature, VIF, collinear (VIF > 5).
    """
    X_c = sm.add_constant(X, has_constant="add")
    vif_data = pd.DataFrame(
        {
            "feature": X_c.columns[1:],  # skip constant
            "VIF": [
                variance_inflation_factor(X_c.values, i)
                for i in range(1, X_c.shape[1])
            ],
        }
    )
    vif_data["collinear"] = vif_data["VIF"] > 5
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def partial_f_test(
    model_restricted,
    model_full,
    robust: str | None = "hc3",
) -> pd.DataFrame:
    """Compare nested OLS models via partial F-test.

    Parameters
    ----------
    model_restricted : fitted OLS result (fewer features)
    model_full : fitted OLS result (more features, superset)
    robust : str or None, heteroscedasticity correction

    Returns
    -------
    DataFrame with df_resid, ssr, df_diff, ss_diff, F, Pr(>F)
    """
    return anova_lm(model_restricted, model_full, test="F", robust=robust)


def partial_f_test_ssr(
    ssr_restricted: float,
    ssr_unrestricted: float,
    q: int,
    n: int,
    k_full: int,
) -> dict:
    """Compute partial F-statistic from residual sums of squares.

    Evaluates the classical nested-model F-test
    ``F = ((SSR_r - SSR_u) / q) / (SSR_u / (n - k_full))``
    where ``k_full`` counts all parameters in the unrestricted model
    (including the intercept) and ``q`` is the number of extra regressors
    in the unrestricted specification.

    Parameters
    ----------
    ssr_restricted   : residual sum of squares of the restricted model
    ssr_unrestricted : residual sum of squares of the unrestricted model
    q                : number of restrictions (extra parameters in full model)
    n                : sample size used for fitting both models
    k_full           : total parameters in the unrestricted model (incl. intercept)

    Returns
    -------
    dict with keys F_stat, df1, df2, p_value, SSR_restricted, SSR_unrestricted,
    n, k_full
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if n - k_full <= 0:
        raise ValueError(
            f"residual degrees of freedom must be positive, got n - k_full = {n - k_full}"
        )
    if ssr_unrestricted <= 0:
        raise ValueError(
            f"ssr_unrestricted must be positive, got {ssr_unrestricted}"
        )

    df1 = q
    df2 = n - k_full
    f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
    p_value = float(stats.f.sf(f_stat, df1, df2))

    return {
        "F_stat": float(f_stat),
        "df1": int(df1),
        "df2": int(df2),
        "p_value": p_value,
        "SSR_restricted": float(ssr_restricted),
        "SSR_unrestricted": float(ssr_unrestricted),
        "n": int(n),
        "k_full": int(k_full),
    }


def partial_f_test_from_ssr_models(
    ols_restricted,
    ols_full,
    n_train: int,
    k_full: int,
    q: int,
) -> dict:
    """Partial F-test dict from two fitted statsmodels OLS results.

    Extracts the residual sums of squares from the two fitted models and
    delegates to :func:`partial_f_test_ssr`. The two models must be fitted on
    the same ``n_train`` rows with ``k_full`` total parameters (intercept
    included) in ``ols_full`` and ``q`` additional regressors compared to
    ``ols_restricted``.

    Returns
    -------
    dict identical in shape to :func:`partial_f_test_ssr`, with two extra
    keys ``SSR_structured`` and ``SSR_augmented`` aliasing the restricted and
    unrestricted SSR for call sites that prefer the paper's labels.
    """
    ssr_r = float(ols_restricted.ssr)
    ssr_u = float(ols_full.ssr)
    result = partial_f_test_ssr(
        ssr_restricted=ssr_r,
        ssr_unrestricted=ssr_u,
        q=q,
        n=n_train,
        k_full=k_full,
    )
    # Paper-facing aliases so existing CSV schema is unchanged
    result["SSR_structured"] = result["SSR_restricted"]
    result["SSR_augmented"] = result["SSR_unrestricted"]
    result["n_train"] = result["n"]
    logger.info(
        f"[partial F-test] F={result['F_stat']:.4f}  "
        f"df=({result['df1']}, {result['df2']})  "
        f"p={result['p_value']:.3e}"
    )
    return result


def marginal_feature_contribution(
    X_base: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    base_metrics: dict,
    fit_fn=None,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute marginal R-squared gain for each feature added individually.

    For each column in *feature_cols*, appends that single column to
    *X_base*, fits OLS (run_cv=False), and computes the R-squared gain
    vs the baseline.

    Parameters
    ----------
    X_base       : baseline feature matrix (no LLM features)
    y            : target variable
    feature_cols : list of column names to test one at a time
    base_metrics : metrics dict from the baseline model (must contain 'R\u00b2')
    fit_fn       : fitting function, defaults to fit_ols
    source_df    : DataFrame containing the columns listed in *feature_cols*.
                   If None, columns are taken from X_base itself.

    Returns
    -------
    DataFrame with columns: feature, R2_gain, sorted descending by R2_gain
    """
    if fit_fn is None:
        fit_fn = fit_ols

    src = source_df if source_df is not None else X_base
    base_r2 = base_metrics["R\u00b2"]
    gains = []
    for col in feature_cols:
        if col not in src.columns:
            continue
        X_aug = X_base.copy()
        X_aug[col] = src[col].values
        mask = X_aug[col].notna() & np.isfinite(X_aug[col])
        X_aug, y_clean = X_aug.loc[mask], y.loc[mask]
        _, m_single, _ = fit_fn(X_aug, y_clean, model_name=f"+{col}", run_cv=False)
        gains.append({"feature": col, "R2_gain": m_single["R\u00b2"] - base_r2})

    return pd.DataFrame(gains).sort_values("R2_gain", ascending=False).reset_index(
        drop=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lasso variable selection
# ─────────────────────────────────────────────────────────────────────────────

def fit_lasso_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "LassoCV",
    n_alphas: int = 100,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> tuple[LassoCV, dict, pd.DataFrame]:
    """Fit LassoCV with standardised features for variable selection.

    Returns
    -------
    model      : fitted LassoCV (on standardised X)
    metrics    : dict with R², RMSE, MAE, MAPE, best alpha, optional test metrics
    coef_table : DataFrame of original-scale coefficients with selection flag
    """
    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    model = LassoCV(
        n_alphas=n_alphas, cv=CV_FOLDS, random_state=RANDOM_SEED,
        max_iter=10_000, n_jobs=-1,
    )
    model.fit(X_sc, y)
    n_features = X.shape[1]

    pred = model.predict(X_sc)
    metrics = regression_metrics(
        y.values, pred, model_name=model_name, n=len(y), p=n_features,
    )
    metrics["alpha"] = float(model.alpha_)

    # Test-set metrics on shared hold-out
    if X_test is not None and y_test is not None:
        X_te_sc = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        test_pred = model.predict(X_te_sc)
        test_m = regression_metrics(
            y_test.values, test_pred, model_name=model_name,
            n=len(y_test), p=n_features,
        )
        metrics["test_R²"] = test_m["R²"]
        metrics["test_Adj. R²"] = test_m["Adj. R²"]
        metrics["test_RMSE"] = test_m["RMSE"]
        metrics["test_MAE"] = test_m["MAE"]
        metrics["test_MAPE (%)"] = test_m["MAPE (%)"]
        metrics["n_test"] = len(y_test)

    # Coefficient table: standardised + original scale
    coefs_std = pd.Series(model.coef_, index=X.columns)
    coefs_orig = coefs_std / scaler.scale_  # back to original scale
    coef_table = pd.DataFrame({
        "coef_standardised": coefs_std.round(6),
        "coef_original": coefs_orig.round(6),
        "abs_coef": coefs_std.abs().round(6),
        "selected": coefs_std.abs() > 0,
    }).sort_values("abs_coef", ascending=False)

    n_sel = coef_table["selected"].sum()
    logger.info(
        f"[{model_name}] alpha={model.alpha_:.6f}  "
        f"selected {n_sel}/{len(X.columns)} features  R²={metrics['R²']:.3f}"
    )

    # Attach scaler for downstream use
    model._scaler = scaler
    return model, metrics, coef_table


def lasso_selected_features(coef_table: pd.DataFrame) -> list[str]:
    """Return list of feature names selected by Lasso (non-zero coefficients)."""
    return coef_table.index[coef_table["selected"]].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# PCA on LLM score features
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_scores(
    df: pd.DataFrame,
    score_cols: list[str] | None = None,
    n_components: int | None = None,
) -> tuple[PCA, pd.DataFrame, StandardScaler]:
    """Run PCA on LLM score columns (continuous 1-5 scales).

    Parameters
    ----------
    df           : DataFrame containing the score columns
    score_cols   : columns to include (defaults to 5 core LLM scores)
    n_components : number of PCs to keep (None = keep all)

    Returns
    -------
    pca         : fitted PCA object
    scores_df   : DataFrame of PC scores aligned to df.index
    scaler      : fitted StandardScaler (for reproducibility)
    """
    if score_cols is None:
        score_cols = [
            "llm_luxury_score", "llm_uniqueness_score",
            "llm_renovation_quality_score", "llm_curb_appeal_score",
            "llm_spaciousness_score",
        ]

    available = [c for c in score_cols if c in df.columns]
    X_raw = df[available].copy().apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.fillna(X_raw.median())

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pc_vals = pca.fit_transform(X_sc)

    pc_names = [f"PC{i+1}" for i in range(pc_vals.shape[1])]
    scores_df = pd.DataFrame(pc_vals, index=df.index, columns=pc_names)

    logger.info(
        f"PCA on {len(available)} scores → {pca.n_components_} components, "
        f"explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.1%}"
    )
    return pca, scores_df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# K-means clustering (descriptive)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_listings(
    scores_df: pd.DataFrame,
    n_clusters: int = 4,
    max_k: int = 8,
) -> tuple[KMeans, pd.Series, pd.DataFrame]:
    """K-means on PCA scores for descriptive profiling.

    Returns
    -------
    best_model  : fitted KMeans
    labels      : Series of cluster labels aligned to scores_df.index
    inertia_df  : DataFrame of k vs. inertia (for elbow plot)
    """
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        km.fit(scores_df)
        inertias.append({"k": k, "inertia": km.inertia_})

    inertia_df = pd.DataFrame(inertias)

    best = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED)
    labels = pd.Series(best.fit_predict(scores_df), index=scores_df.index, name="cluster")

    logger.info(f"K-means clustering: k={n_clusters}, inertia={best.inertia_:,.0f}")
    return best, labels, inertia_df


# ─────────────────────────────────────────────────────────────────────────────
# SHAP analysis (tree models)
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_xgb(
    model: XGBRegressor,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute SHAP values for a fitted XGBoost regressor.

    Uses :class:`shap.TreeExplainer` on ``X`` and returns the raw SHAP matrix
    together with per-row base (expected) values and the feature-name list.
    Requires the optional ``shap`` dependency.

    Parameters
    ----------
    model         : fitted XGBRegressor
    X             : feature matrix the SHAP values are computed on
    feature_names : optional override for the feature-name list (defaults to
                    ``X.columns``)

    Returns
    -------
    shap_values   : ndarray, shape (n_rows, n_features)
    base_values   : ndarray, shape (n_rows,) -- base value broadcast per row
    feature_names : list[str]
    """
    if _shap is None:
        raise ImportError(
            "shap package required for compute_shap_xgb; "
            "install with `pip install shap`"
        ) from _SHAP_IMPORT_ERROR

    explainer = _shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = float(np.asarray(base_value).ravel()[0])
    base_values = np.full(len(X), float(base_value), dtype=float)

    names = list(feature_names) if feature_names is not None else list(X.columns)
    logger.info(
        f"[SHAP] computed values on X shape={X.shape}  "
        f"base_value={base_value:.4f}"
    )
    return shap_values, base_values, names


def rank_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP contribution.

    Parameters
    ----------
    shap_values   : 2-D ndarray, shape (n_rows, n_features)
    feature_names : list of feature names matching the columns of *shap_values*

    Returns
    -------
    DataFrame with columns ``rank``, ``feature``, ``mean_abs_shap``,
    ``mean_shap_signed`` sorted by descending mean |SHAP|.
    """
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim != 2:
        raise ValueError(
            f"shap_values must be 2-D, got shape {shap_arr.shape}"
        )
    if shap_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {shap_arr.shape[1]} columns but "
            f"{len(feature_names)} feature names were provided"
        )

    mean_abs = np.abs(shap_arr).mean(axis=0)
    mean_signed = shap_arr.mean(axis=0)
    ranking = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_abs_shap": mean_abs,
                "mean_shap_signed": mean_signed,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


# ─────────────────────────────────────────────────────────────────────────────
# Full OLS coefficient table (appendix-ready)
# ─────────────────────────────────────────────────────────────────────────────

# Thematic ordering for publication table
_FEATURE_GROUP_ORDER = {
    "Intercept": 0,
    "Structural (continuous)": 1,
    "Structural (binary)": 2,
    "LLM scores": 3,
    "LLM soft flags": 4,
    "LLM distress flags": 5,
    "Fixed effects": 6,
}

_STRUCTURAL_CONTINUOUS = {
    "log_living_area", "log_lot_size", "property_age",
    "bedrooms", "bathrooms", "stories", "garage_spaces",
}
_STRUCTURAL_BINARY = {"has_fireplace", "has_pool"}
_LLM_SCORES = {
    "llm_luxury_score", "llm_uniqueness_score", "llm_uniqueness_score_sq",
    "llm_renovation_quality_score", "llm_curb_appeal_score",
    "llm_spaciousness_score",
}
_LLM_SOFT_FLAGS = {
    "llm_is_unique_property", "llm_has_premium_finishes", "llm_is_recently_updated",
}
_LLM_DISTRESS_FLAGS = {
    "llm_foreclosure_flag", "llm_auction_flag", "llm_as_is_flag",
    "llm_fixer_upper_flag", "llm_needs_repair_flag", "llm_water_damage_flag",
    "llm_fire_damage_flag", "llm_foundation_issue_flag", "llm_roof_issue_flag",
    "llm_mold_flag", "llm_tenant_occupied_flag", "llm_cash_only_flag",
    "llm_investor_special_flag",
}


def _classify_feature(name: str) -> str:
    if name == "const":
        return "Intercept"
    if name in _STRUCTURAL_CONTINUOUS:
        return "Structural (continuous)"
    if name in _STRUCTURAL_BINARY:
        return "Structural (binary)"
    if name in _LLM_SCORES:
        return "LLM scores"
    if name in _LLM_SOFT_FLAGS:
        return "LLM soft flags"
    if name in _LLM_DISTRESS_FLAGS:
        return "LLM distress flags"
    return "Fixed effects"


def build_full_coef_table(
    ols_model,
    X: pd.DataFrame,
    include_fe: bool = False,
) -> pd.DataFrame:
    """Build publication-ready full coefficient table with VIF.

    Parameters
    ----------
    ols_model : fitted statsmodels OLSResults (HC3)
    X         : feature matrix used to fit the model (without constant)
    include_fe: if False, ZIP-code and home-type dummies are excluded

    Returns
    -------
    DataFrame with columns: group, coef, std_err, t_stat, p_value,
    ci_low, ci_high, VIF — sorted thematically then by |coef| within group.
    """
    coef_df = pd.DataFrame({
        "feature": ols_model.params.index,
        "coef": ols_model.params.values,
        "std_err": ols_model.bse.values,
        "t_stat": ols_model.tvalues.values,
        "p_value": ols_model.pvalues.values,
        "ci_low": ols_model.conf_int()[0].values,
        "ci_high": ols_model.conf_int()[1].values,
    })

    # Classify features
    coef_df["group"] = coef_df["feature"].apply(_classify_feature)

    # Filter out fixed effects if requested
    if not include_fe:
        coef_df = coef_df[coef_df["group"] != "Fixed effects"].copy()

    # Compute VIF for non-constant, non-FE features
    vif_features = [f for f in coef_df["feature"] if f != "const"]
    vif_cols = [c for c in vif_features if c in X.columns]
    if vif_cols:
        vif_data = compute_vif(X[vif_cols])
        vif_map = dict(zip(vif_data["feature"], vif_data["VIF"]))
    else:
        vif_map = {}
    coef_df["VIF"] = coef_df["feature"].map(vif_map)

    # Sort: group order, then |coef| descending within group
    coef_df["_group_order"] = coef_df["group"].map(_FEATURE_GROUP_ORDER).fillna(99)
    coef_df["_abs_coef"] = coef_df["coef"].abs()
    coef_df = (
        coef_df
        .sort_values(["_group_order", "_abs_coef"], ascending=[True, False])
        .drop(columns=["_group_order", "_abs_coef"])
        .reset_index(drop=True)
    )

    # Round for display
    for col in ["coef", "std_err", "t_stat", "ci_low", "ci_high"]:
        coef_df[col] = coef_df[col].round(4)
    coef_df["p_value"] = coef_df["p_value"].round(4)
    coef_df["VIF"] = coef_df["VIF"].round(2)

    return coef_df
