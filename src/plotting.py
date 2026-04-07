"""Plotting utilities for descriptive statistics and model diagnostics.

All functions:
• Accept a DataFrame and return a matplotlib Figure
• Save to outputs/figures/ with a numbered filename
• Follow a consistent clean aesthetic (seaborn whitegrid)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import OUTPUTS_FIGS
from src.utils import get_logger

logger = get_logger(__name__)

# ── Default style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
_PALETTE  = sns.color_palette("muted")
_BLUE     = _PALETTE[0]
_GREEN    = _PALETTE[1]
_RED      = _PALETTE[2]
_PURPLE   = _PALETTE[4]
DPI       = 150
WIDE      = (12, 5)
STD       = (9, 5)
SQ        = (7, 7)


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGS / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    logger.info(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_histogram(df: pd.DataFrame) -> plt.Figure:
    """Side-by-side: price in $K and log(price)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)

    ax1.hist(df["price"] / 1_000, bins=40, color=_BLUE, edgecolor="white", lw=0.4)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax1.set_xlabel("List Price")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of List Price")

    ax2.hist(df["log_price"], bins=40, color=_GREEN, edgecolor="white", lw=0.4)
    ax2.set_xlabel("log(List Price)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of log(List Price)")

    fig.suptitle("Houston Zillow Sample  —  Price", y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_price_distribution")
    return fig


def plot_sqft_histogram(df: pd.DataFrame) -> plt.Figure:
    """Histogram of living area."""
    valid = df["living_area_sqft"].dropna()
    fig, ax = plt.subplots(figsize=STD)
    ax.hist(valid, bins=40, color=_RED, edgecolor="white", lw=0.4)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Living Area (sq ft)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Living Area")
    fig.tight_layout()
    _save(fig, "02_sqft_distribution")
    return fig


def plot_sqft_vs_price(df: pd.DataFrame) -> plt.Figure:
    """Scatter: living area vs. price (sample up to 600 for clarity)."""
    sample = df.dropna(subset=["living_area_sqft", "price"]).sample(
        min(len(df), 600), random_state=42
    )
    fig, ax = plt.subplots(figsize=STD)
    ax.scatter(sample["living_area_sqft"], sample["price"] / 1_000,
               alpha=0.35, s=14, color=_BLUE, edgecolors="none")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Living Area (sq ft)")
    ax.set_ylabel("List Price")
    ax.set_title("Living Area vs. List Price")
    fig.tight_layout()
    _save(fig, "03_sqft_vs_price")
    return fig


def plot_price_by_home_type(df: pd.DataFrame) -> plt.Figure:
    """Violin plot of price by home type."""
    valid = df.dropna(subset=["home_type", "price"])
    order = (valid.groupby("home_type")["price"]
             .median()
             .sort_values(ascending=False)
             .index.tolist())

    fig, ax = plt.subplots(figsize=WIDE)
    sns.violinplot(data=valid, x="home_type", y="price", order=order,
                   ax=ax, palette="muted", inner="box", cut=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1_000:,.0f}K"))
    ax.set_xlabel("Home Type")
    ax.set_ylabel("List Price")
    ax.set_title("Price Distribution by Home Type")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "04_price_by_home_type")
    return fig


def plot_listings_by_zip(df: pd.DataFrame, top_n: int = 25) -> plt.Figure:
    """Bar chart of listing counts per ZIP code."""
    counts = df["zip_code"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(11, 5))
    counts.plot(kind="bar", ax=ax, color=_BLUE, edgecolor="white")
    ax.set_xlabel("ZIP Code")
    ax.set_ylabel("Listing Count")
    ax.set_title(f"Listing Counts by ZIP Code (top {top_n})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, "05_listings_by_zip")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap for key numeric variables."""
    num_cols = [
        "price", "log_price", "bedrooms", "bathrooms",
        "living_area_sqft", "log_living_area",
        "lot_size_sqft", "log_lot_size",
        "property_age", "stories", "garage_spaces",
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()

    n    = len(available)
    fig, ax = plt.subplots(figsize=(n * 0.9, n * 0.8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap — Key Numeric Variables")
    fig.tight_layout()
    _save(fig, "06_correlation_heatmap")
    return fig


def plot_map(df: pd.DataFrame) -> None:
    """Interactive scatter map: one dot per listing, coloured by price (Plotly)."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("lat/lon not available — skipping map")
        return

    valid = df.dropna(subset=["latitude", "longitude", "price"])
    if valid.empty:
        return

    try:
        import plotly.express as px

        hover = {c: True for c in ["address_full", "bedrooms", "bathrooms",
                                    "living_area_sqft", "home_type"]
                 if c in valid.columns}
        hover["price"] = ":$,.0f"

        try:
            fig = px.scatter_map(
                valid, lat="latitude", lon="longitude",
                color="price", color_continuous_scale="RdYlGn_r",
                size_max=8, zoom=10,
                map_style="open-street-map",
                hover_data=hover, labels={"price": "List Price ($)"},
                title="Houston Zillow For-Sale Listings — Price per Listing",
            )
        except Exception:
            fig = px.scatter_mapbox(
                valid, lat="latitude", lon="longitude",
                color="price", color_continuous_scale="RdYlGn_r",
                size_max=8, zoom=10, mapbox_style="open-street-map",
                hover_data=hover, labels={"price": "List Price ($)"},
                title="Houston Zillow For-Sale Listings — Price per Listing",
            )

        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0},
                          coloraxis_colorbar_tickformat="$,.0f")
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUTS_FIGS / "07_scatter_map.html"))
        try:
            fig.write_image(str(OUTPUTS_FIGS / "07_scatter_map.png"),
                            width=1200, height=700)
        except Exception:
            pass
        logger.info("  Scatter map → 07_scatter_map.html")
    except ImportError:
        logger.warning("plotly not installed — skipping scatter map")


def plot_density_map(df: pd.DataFrame, radius: int = 18) -> None:
    """Plotly density heatmap: shows where listings are concentrated.

    Complements plot_map (which shows price) by showing listing volume per area.
    Saved as HTML (interactive) and PNG (static).
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("lat/lon not available — skipping density map")
        return

    valid = df.dropna(subset=["latitude", "longitude"])
    if valid.empty:
        return

    try:
        import plotly.express as px

        center = {"lat": float(valid["latitude"].median()),
                  "lon": float(valid["longitude"].median())}

        try:
            fig = px.density_map(
                valid, lat="latitude", lon="longitude",
                radius=radius, zoom=10, center=center,
                map_style="open-street-map",
                color_continuous_scale="YlOrRd",
                title=f"Listing Density — {len(valid):,} Houston For-Sale Listings",
            )
        except Exception:
            fig = px.density_mapbox(
                valid, lat="latitude", lon="longitude",
                radius=radius, zoom=10, center=center,
                mapbox_style="open-street-map",
                color_continuous_scale="YlOrRd",
                title=f"Listing Density — {len(valid):,} Houston For-Sale Listings",
            )

        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        fig.update_coloraxes(showscale=False)
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUTS_FIGS / "15_density_map.html"))
        try:
            fig.write_image(str(OUTPUTS_FIGS / "15_density_map.png"),
                            width=1200, height=700)
        except Exception:
            pass
        logger.info("  Density map → 15_density_map.html")
    except ImportError:
        logger.warning("plotly not installed — skipping density map")


def plot_hexbin_map(df: pd.DataFrame, gridsize: int = 35) -> plt.Figure:
    """Static side-by-side hexbin maps — no internet tiles required.

    Left panel : listing count per hexagonal cell (where are the listings?).
    Right panel: median list price per cell (how expensive is each area?).
    Suitable for papers and presentations without an internet connection.
    """
    valid = df.dropna(subset=["latitude", "longitude"])
    if valid.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No lat/lon data", ha="center")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: listing count
    hb = axes[0].hexbin(
        valid["longitude"], valid["latitude"],
        gridsize=gridsize, cmap="YlOrRd", mincnt=1, linewidths=0.2,
    )
    cb = fig.colorbar(hb, ax=axes[0], shrink=0.8)
    cb.set_label("Listing Count", fontsize=9)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Listing Density\n(count per hexagonal cell)", fontweight="bold")

    # Right: median price per cell
    valid_p = valid.dropna(subset=["price"])
    hb2 = axes[1].hexbin(
        valid_p["longitude"], valid_p["latitude"],
        C=valid_p["price"] / 1_000,
        gridsize=gridsize, cmap="RdYlGn_r",
        reduce_C_function=np.median,
        mincnt=1, linewidths=0.2,
    )
    cb2 = fig.colorbar(hb2, ax=axes[1], shrink=0.8,
                       format=mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    cb2.set_label("Median List Price", fontsize=9)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("Median Price by Area\n(median list price per cell)",
                      fontweight="bold")

    fig.suptitle("Houston For-Sale Listings — Geographic Distribution",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "16_hexbin_map")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Model diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_name: str = "actual_vs_predicted",
) -> plt.Figure:
    """Scatter of actual vs predicted log(price) with 45° line."""
    fig, ax = plt.subplots(figsize=SQ)
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color=_BLUE, edgecolors="none")
    lo = min(y_true.min(), y_pred.min()) - 0.1
    hi = max(y_true.max(), y_pred.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="45° line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual log(Price)")
    ax.set_ylabel("Predicted log(Price)")
    ax.set_title(f"Actual vs. Predicted  —  {model_name}")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_name)
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance (XGBoost gain)",
    save_name: str = "feature_importance",
) -> plt.Figure:
    """Horizontal bar chart of top-N features."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, top_n * 0.38 + 1))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=_BLUE)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_name)
    return fig


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    save_name: str = "shap_summary",
) -> Optional[plt.Figure]:
    """SHAP beeswarm plot.  Graceful no-op if shap is not installed."""
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(X)
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(shap_values, X, show=False, max_display=20)
        fig = plt.gcf()
        _save(fig, save_name)
        return fig
    except ImportError:
        logger.warning("shap not installed — skipping SHAP summary plot")
        return None


def plot_llm_score_distributions(df: pd.DataFrame) -> plt.Figure:
    """Histograms for each LLM score column."""
    score_cols = [c for c in [
        "llm_luxury_score", "llm_uniqueness_score",
        "llm_renovation_quality_score", "llm_curb_appeal_score",
        "llm_spaciousness_score",
    ] if c in df.columns]

    if not score_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No LLM score columns found", ha="center")
        return fig

    n    = len(score_cols)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 3))
    axes      = np.array(axes).flatten()

    for i, col in enumerate(score_cols):
        vals = df[col].dropna()
        axes[i].hist(vals, bins=range(0, 7), align="left",
                     color=_PURPLE, edgecolor="white", rwidth=0.8)
        axes[i].set_title(col.replace("llm_", "").replace("_", " ").title())
        axes[i].set_xlabel("Score (1-5)")
        axes[i].set_ylabel("Count")
        axes[i].set_xticks(range(1, 6))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("LLM-Extracted Score Distributions", fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_llm_score_distributions")
    return fig


def plot_llm_flag_rates(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of share of listings with each hard LLM flag = 1."""
    flag_cols = [c for c in df.columns if c.startswith("llm_") and c.endswith("_flag")]
    if not flag_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No LLM flag columns found", ha="center")
        return fig

    rates = (df[flag_cols].fillna(0).mean() * 100).sort_values(ascending=False)
    labels = [c.replace("llm_", "").replace("_flag", "").replace("_", " ") for c in rates.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rates) * 0.4)))
    ax.barh(labels[::-1], rates.values[::-1], color=_RED, edgecolor="white")
    ax.set_xlabel("Share of Listings (%)")
    ax.set_title("LLM Hard Flag Prevalence")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    fig.tight_layout()
    _save(fig, "09_llm_flag_rates")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sample construction & pipeline charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_waterfall(steps: list[tuple[str, int]]) -> plt.Figure:
    """Horizontal waterfall / funnel showing the full data pipeline.

    Parameters
    ----------
    steps : list of (label, n_remaining) tuples in pipeline order.
            Example: [("ZIP search (raw)", 2500), ("Pre-filter home type", 2414), ...]

    Suitable for a Methods slide; shows where and how much is lost at each stage.
    """
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    drops  = [0] + [values[i - 1] - values[i] for i in range(1, len(values))]

    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.65 + 1.2))

    bar_h = 0.55
    y_pos = np.arange(len(labels))[::-1]   # top → bottom

    # Background grey bars (full width = starting value)
    for yi, val in zip(y_pos, values):
        ax.barh(yi, values[0], height=bar_h, color="#E8E8E8", zorder=1)

    # Coloured bars (remaining)
    cmap   = plt.cm.RdYlGn
    colors = [cmap(v / values[0]) for v in values]
    for yi, val, col in zip(y_pos, values, colors):
        ax.barh(yi, val, height=bar_h, color=col, zorder=2)

    # Annotations: n remaining + drop
    for yi, val, drop in zip(y_pos, values, drops):
        ax.text(val + values[0] * 0.01, yi, f"{val:,}", va="center",
                fontsize=9, fontweight="bold", color="#222")
        if drop > 0:
            ax.text(values[0] * 0.99, yi, f"−{drop:,}", va="center", ha="right",
                    fontsize=8, color="#b00020", style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Records")
    ax.set_title("Data Pipeline — Sample Construction", fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(0, values[0] * 1.18)
    sns.despine(left=True, bottom=False)
    fig.tight_layout()
    _save(fig, "00_sample_waterfall")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Additional descriptive charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_by_bedrooms(df: pd.DataFrame) -> plt.Figure:
    """Box plot of list price by bedroom count — demand-side segmentation."""
    valid = df.dropna(subset=["bedrooms", "price"]).copy()
    valid["beds"] = valid["bedrooms"].astype(int).clip(upper=6)
    valid["beds_label"] = valid["beds"].map(lambda x: f"{x}+" if x == 6 else str(x))

    order = sorted(valid["beds_label"].unique(),
                   key=lambda x: int(x.replace("+", "")))

    fig, ax = plt.subplots(figsize=WIDE)
    sns.boxplot(data=valid, x="beds_label", y="price", order=order,
                palette="Blues", flierprops={"marker": ".", "markersize": 3},
                ax=ax, linewidth=0.8)

    # Overlay sample sizes
    counts = valid.groupby("beds_label").size()
    for i, b in enumerate(order):
        ax.text(i, ax.get_ylim()[1] * 0.97, f"n={counts.get(b, 0):,}",
                ha="center", fontsize=7.5, color="#555")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K"))
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("List Price")
    ax.set_title("Price Distribution by Bedroom Count", fontweight="bold")
    fig.tight_layout()
    _save(fig, "10_price_by_bedrooms")
    return fig


def plot_price_per_sqft(df: pd.DataFrame) -> plt.Figure:
    """Price per sq ft by home type — normalises for size in one chart."""
    valid = df.dropna(subset=["price", "living_area_sqft", "home_type"]).copy()
    valid = valid[valid["living_area_sqft"] > 0]
    valid["ppsf"] = valid["price"] / valid["living_area_sqft"]
    valid = valid[valid["ppsf"].between(10, 2000)]

    order = (valid.groupby("home_type")["ppsf"]
             .median().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=WIDE)
    sns.violinplot(data=valid, x="home_type", y="ppsf", order=order,
                   palette="muted", inner="quartile", cut=0, ax=ax, linewidth=0.8)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Home Type")
    ax.set_ylabel("Price per Sq Ft ($)")
    ax.set_title("Price per Square Foot by Home Type", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Median annotations
    for i, ht in enumerate(order):
        med = valid.loc[valid["home_type"] == ht, "ppsf"].median()
        ax.text(i, med, f" ${med:,.0f}", va="center", fontsize=8, color="#111")

    fig.tight_layout()
    _save(fig, "11_price_per_sqft_by_type")
    return fig


def plot_year_built_vs_price(df: pd.DataFrame) -> plt.Figure:
    """Median list price by construction decade — captures depreciation cycle."""
    valid = df.dropna(subset=["year_built", "price"]).copy()
    valid = valid[valid["year_built"].between(1900, 2026)]
    valid["decade"] = (valid["year_built"] // 10 * 10).astype(int)

    agg = (valid.groupby("decade")["price"]
           .agg(median="median", count="count")
           .reset_index())
    agg = agg[agg["count"] >= 5]   # suppress sparse decades

    fig, ax1 = plt.subplots(figsize=WIDE)
    ax2 = ax1.twinx()

    ax1.bar(agg["decade"], agg["count"], width=7, color=_BLUE, alpha=0.25,
            label="Count (right)")
    ax2.plot(agg["decade"], agg["median"] / 1_000, marker="o", color=_RED,
             linewidth=2, markersize=5, label="Median price ($K)")

    ax1.set_xlabel("Construction Decade")
    ax1.set_ylabel("Listing Count", color=_BLUE)
    ax2.set_ylabel("Median List Price ($K)", color=_RED)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax1.set_title("Listing Count & Median Price by Construction Decade",
                  fontweight="bold")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, "12_year_built_vs_price")
    return fig


def plot_days_on_market(df: pd.DataFrame) -> plt.Figure:
    """Days-on-market histogram with median/mean annotations."""
    valid = df["days_on_zillow"].dropna()
    valid = valid[valid.between(0, 90)]

    med  = valid.median()
    mean = valid.mean()

    fig, ax = plt.subplots(figsize=STD)
    ax.hist(valid, bins=30, color=_BLUE, edgecolor="white", lw=0.4)
    ax.axvline(med,  color=_RED,    linestyle="--", lw=1.5, label=f"Median: {med:.0f} d")
    ax.axvline(mean, color=_GREEN,  linestyle=":",  lw=1.5, label=f"Mean: {mean:.0f} d")
    ax.set_xlabel("Days on Zillow")
    ax.set_ylabel("Count")
    ax.set_title("Days on Market Distribution", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "13_days_on_market")
    return fig


def plot_zip_price_bubbles(df: pd.DataFrame, top_n: int = 30) -> plt.Figure:
    """ZIP-code bubble chart: x=median price, y=median sqft, size=count.

    Complements the listing count bar chart (05) by adding price and size
    dimensions simultaneously.
    """
    valid = df.dropna(subset=["zip_code", "price", "living_area_sqft"])
    agg   = (valid.groupby("zip_code")
             .agg(median_price=("price", "median"),
                  median_sqft=("living_area_sqft", "median"),
                  count=("price", "count"))
             .reset_index()
             .sort_values("count", ascending=False)
             .head(top_n))

    sizes  = (agg["count"] / agg["count"].max() * 600).clip(lower=40)
    colors = agg["median_price"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(agg["median_price"] / 1_000, agg["median_sqft"],
                    s=sizes, c=colors, cmap="RdYlGn_r",
                    alpha=0.75, edgecolors="#444", linewidths=0.5)

    # Annotate largest ZIPs
    for _, row in agg.nlargest(12, "count").iterrows():
        ax.annotate(row["zip_code"],
                    xy=(row["median_price"] / 1_000, row["median_sqft"]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    cb = fig.colorbar(sc, ax=ax, format=mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K"))
    cb.set_label("Median Price", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Median List Price")
    ax.set_ylabel("Median Living Area (sq ft)")
    ax.set_title(f"ZIP-Code Price & Size Profile — Top {top_n} ZIPs by Volume\n"
                 "(bubble size = listing count)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "14_zip_price_bubbles")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Spatial analysis — regression diagnostics
# ─────────────────────────────────────────────────────────────────────────────

# Houston CBD coordinates (downtown)
_CBD_LAT = 29.7604
_CBD_LON = -95.3698


def _haversine_km(lat: pd.Series, lon: pd.Series,
                  ref_lat: float, ref_lon: float) -> pd.Series:
    """Vectorised haversine distance in km from a reference point."""
    R = 6371.0
    phi1 = np.radians(lat)
    phi2 = np.radians(ref_lat)
    dphi = np.radians(ref_lat - lat)
    dlam = np.radians(ref_lon - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def plot_price_gradient(df: pd.DataFrame) -> plt.Figure:
    """Price vs. distance from Houston CBD with LOWESS smooth.

    Tests the monocentric city model (Alonso-Muth-Mills): price should
    decline monotonically with distance from the CBD.  The LOWESS curve
    reveals whether the relationship is non-linear, which would justify
    adding a dist_cbd^2 term to the regression.
    """
    valid = df.dropna(subset=["latitude", "longitude", "price", "log_price"]).copy()
    valid["dist_cbd_km"] = _haversine_km(
        valid["latitude"], valid["longitude"], _CBD_LAT, _CBD_LON
    )

    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    specs = [
        ("price",     "List Price ($)",  "Price vs. Distance from CBD"),
        ("log_price", "log(List Price)", "log-Price vs. Distance from CBD"),
    ]
    for ax, (ycol, ylabel, title) in zip(axes, specs):
        y_vals = valid[ycol]
        x_vals = valid["dist_cbd_km"]
        ax.scatter(x_vals, y_vals, alpha=0.15, s=8, color=_BLUE, edgecolors="none")
        smooth = lowess(y_vals.values, x_vals.values, frac=0.25, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color=_RED, lw=2, label="LOWESS")
        if ycol == "price":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K"))
        ax.axvline(10, color="#888", lw=0.8, linestyle=":", alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(10.3, ylim[0] + (ylim[1] - ylim[0]) * 0.04, "10 km",
                fontsize=7.5, color="#666")
        ax.set_xlabel("Distance from CBD (km)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Monocentric City Price Gradient — Houston TX",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "17_price_gradient_cbd")
    return fig


def plot_residual_map(df: pd.DataFrame,
                      residuals: np.ndarray,
                      model_name: str = "OLS-Structured",
                      gridsize: int = 30) -> plt.Figure:
    """Hexbin map of OLS residuals — core spatial autocorrelation diagnostic.

    Left panel  (diverging): signed residuals — red = under-predicted,
                              blue = over-predicted.
    Right panel (sequential): absolute residuals — where is the model least
                              accurate regardless of direction?

    Systematic spatial clustering indicates a missing location effect and
    justifies spatial FEs, SAR, or SEM.

    Parameters
    ----------
    df        : cleaned listings DataFrame with lat/lon (must align with residuals)
    residuals : array of (actual - predicted) in log-price scale
    """
    valid = df.copy().reset_index(drop=True)
    valid["residual"] = residuals
    valid = valid.dropna(subset=["latitude", "longitude", "residual"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmax = float(np.percentile(np.abs(valid["residual"]), 95))

    # Left: signed residuals
    hb = axes[0].hexbin(
        valid["longitude"], valid["latitude"],
        C=valid["residual"],
        gridsize=gridsize, cmap="RdBu_r",
        reduce_C_function=np.mean,
        vmin=-vmax, vmax=vmax, mincnt=1, linewidths=0.2,
    )
    cb = fig.colorbar(hb, ax=axes[0], shrink=0.8)
    cb.set_label("Mean Residual (log scale)", fontsize=9)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title(f"{model_name} Residuals\n"
                      "(red = under-predicted, blue = over-predicted)",
                      fontweight="bold")

    # Right: absolute residuals
    hb2 = axes[1].hexbin(
        valid["longitude"], valid["latitude"],
        C=np.abs(valid["residual"]),
        gridsize=gridsize, cmap="YlOrRd",
        reduce_C_function=np.mean,
        mincnt=1, linewidths=0.2,
    )
    cb2 = fig.colorbar(hb2, ax=axes[1], shrink=0.8)
    cb2.set_label("Mean |Residual| (log scale)", fontsize=9)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title(f"{model_name} |Residuals|\n"
                      "(where is the model least accurate?)",
                      fontweight="bold")

    fig.suptitle(f"Spatial Distribution of {model_name} Residuals",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _save(fig, f"18_residual_map_{safe_name}")
    return fig


def plot_price_surface(df: pd.DataFrame, resolution: int = 200) -> plt.Figure:
    """Interpolated list-price surface using thin-plate spline (scipy RBF).

    Fills the continuous price landscape across Houston from the discrete
    listing observations.  Highlights spatial heterogeneity that ZIP FEs
    only partially capture.  The CBD is marked with a star.
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MplPath

    valid = df.dropna(subset=["latitude", "longitude", "log_price"]).copy()
    if len(valid) < 50:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for interpolation", ha="center")
        return fig

    sample = valid.sample(min(600, len(valid)), random_state=42)
    coords = np.column_stack([sample["longitude"].values,
                               sample["latitude"].values])
    z_pts  = sample["log_price"].values

    lon_min = valid["longitude"].quantile(0.01)
    lon_max = valid["longitude"].quantile(0.99)
    lat_min = valid["latitude"].quantile(0.01)
    lat_max = valid["latitude"].quantile(0.99)

    lon_g, lat_g = np.meshgrid(
        np.linspace(lon_min, lon_max, resolution),
        np.linspace(lat_min, lat_max, resolution),
    )

    interp = RBFInterpolator(coords, z_pts, kernel="thin_plate_spline",
                             smoothing=len(sample) * 0.05)
    z_g = interp(np.column_stack([lon_g.ravel(), lat_g.ravel()])).reshape(lon_g.shape)

    # Mask outside convex hull to suppress extrapolation artefacts
    hull     = ConvexHull(coords)
    hull_pts = coords[hull.vertices]
    in_hull  = MplPath(hull_pts).contains_points(
        np.column_stack([lon_g.ravel(), lat_g.ravel()])
    ).reshape(lon_g.shape)
    z_g[~in_hull] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(lon_g, lat_g, np.exp(z_g) / 1_000,
                     levels=20, cmap="RdYlGn_r", alpha=0.85)
    ax.contour(lon_g, lat_g, np.exp(z_g) / 1_000,
               levels=10, colors="white", linewidths=0.3, alpha=0.4)
    ax.scatter(valid["longitude"], valid["latitude"],
               c="white", s=3, alpha=0.25, edgecolors="none", zorder=3)
    ax.scatter([_CBD_LON], [_CBD_LAT], marker="*", s=200,
               color="white", edgecolors="black", zorder=5, label="CBD")

    cb = fig.colorbar(
        cf, ax=ax, shrink=0.8,
        format=mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"),
    )
    cb.set_label("Interpolated List Price", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Interpolated Price Surface — Houston For-Sale Listings\n"
                 "(thin-plate spline, clipped to data convex hull)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "19_price_surface")
    return fig


def plot_zip_fe_chart(df: pd.DataFrame, coef_table: pd.DataFrame) -> plt.Figure:
    """ZIP fixed-effect coefficients vs. median price — location premium chart.

    x = ZIP-level median list price (structural price signal)
    y = OLS fixed-effect coefficient (pure location premium after controls)
    Bubble size = listing count.

    Shows reviewers which ZIPs are truly premium/discounted once structural
    features are held constant — often surprises compared to raw median prices.
    """
    fe_rows = coef_table[coef_table.index.str.startswith("zip_code_")].copy()
    if fe_rows.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ZIP fixed effects in coef_table", ha="center")
        return fig

    fe_rows = fe_rows.copy()
    fe_rows["zip_code"] = fe_rows.index.str.replace("zip_code_", "")

    zip_stats = (df.groupby("zip_code")["price"]
                 .agg(median_price="median", count="count")
                 .reset_index())

    merged = fe_rows.merge(zip_stats, on="zip_code", how="inner")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        merged["median_price"] / 1_000,
        merged["coef"],
        s=(merged["count"] / merged["count"].max() * 350).clip(lower=25),
        c=merged["coef"],
        cmap="RdYlGn",
        alpha=0.8,
        edgecolors="#333",
        linewidths=0.4,
    )
    ax.axhline(0, color="#aaa", lw=1, linestyle="--")

    # Label extreme ZIPs
    top = pd.concat([merged.nlargest(5, "coef"),
                     merged.nsmallest(5, "coef")]).drop_duplicates()
    for _, row in top.iterrows():
        ax.annotate(
            row["zip_code"],
            xy=(row["median_price"] / 1_000, row["coef"]),
            fontsize=7.5, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
        )

    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("ZIP FE coefficient (log scale)", fontsize=9)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.set_xlabel("ZIP-Level Median List Price")
    ax.set_ylabel("OLS Fixed-Effect Coefficient (log-price units)")
    ax.set_title("Location Premium by ZIP Code\n"
                 "(FE coef = price premium after controlling for structural features; "
                 "bubble size = listing count)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "20_zip_fe_chart")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# New LLM / description charts  (Charts 14–20)
# ─────────────────────────────────────────────────────────────────────────────

def plot_desc_quality_vs_price(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Violin: llm_description_quality (1-5) vs price, split by home_type (single vs multi). Chart #14"""
    plt.style.use("seaborn-v0_8-darkgrid")
    valid = df.dropna(subset=["llm_description_quality", "price", "home_type"]).copy()
    valid["quality"] = valid["llm_description_quality"].astype(int).astype(str)
    valid["type_group"] = valid["home_type"].apply(
        lambda x: "Multi-family" if "MULTI" in str(x).upper() else "Single/Other"
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(
        data=valid, x="quality", y="price", hue="type_group",
        split=True, inner="quartile", palette=["#4C72B0", "#DD8452"],
        cut=0, ax=ax, linewidth=0.8,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1_000:,.0f}K"))
    ax.set_xlabel("LLM Description Quality Score (1–5)")
    ax.set_ylabel("List Price")
    ax.set_title("Description Quality vs. Price — Single vs. Multi-Family", fontweight="bold")
    ax.legend(title="Home Type")
    fig.tight_layout()
    if save_path is not None:
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=DPI, bbox_inches="tight")
        logger.info(f"  Saved → {save_path}")
    return fig


def plot_age_vs_ppsf(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Bar: property age by decade (1920s-2020s) vs median $/sqft with error bars. Chart #15"""
    plt.style.use("seaborn-v0_8-darkgrid")
    valid = df.dropna(subset=["year_built", "price", "living_area_sqft"]).copy()
    valid = valid[valid["living_area_sqft"] > 0]
    valid["ppsf"] = valid["price"] / valid["living_area_sqft"]
    valid = valid[valid["ppsf"].between(10, 2000)]
    valid = valid[valid["year_built"].between(1920, 2026)]
    valid["decade"] = (valid["year_built"] // 10 * 10).astype(int)

    agg = (
        valid.groupby("decade")["ppsf"]
        .agg(median="median", sem=lambda x: x.std() / np.sqrt(len(x)), count="count")
        .reset_index()
    )
    agg = agg[agg["count"] >= 5]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(
        agg["decade"].astype(str), agg["median"],
        yerr=agg["sem"], capsize=4,
        color=_BLUE, edgecolor="white", lw=0.5, error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Construction Decade")
    ax.set_ylabel("Median Price per Sq Ft ($)")
    ax.set_title("Price per Sq Ft by Construction Decade (±1 SEM)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    decade_labels = list(agg["decade"].astype(str))
    for pos, (_, row) in enumerate(agg.iterrows()):
        ax.text(
            pos,
            row["median"] + row["sem"] + 1,
            f"n={row['count']:.0f}", ha="center", va="bottom", fontsize=7, color="#444",
        )

    fig.tight_layout()
    if save_path is not None:
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=DPI, bbox_inches="tight")
        logger.info(f"  Saved → {save_path}")
    return fig


def plot_desc_length_dist(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Hist+KDE: description_length distribution, vertical lines at quartiles. Chart #16"""
    plt.style.use("seaborn-v0_8-darkgrid")
    valid = df["description_length"].dropna()

    q1, q2, q3 = valid.quantile([0.25, 0.50, 0.75])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(valid, bins=40, color=_PURPLE, edgecolor="white", lw=0.4,
            density=True, alpha=0.6, label="Histogram (density)")
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(valid)
        xs = np.linspace(valid.min(), valid.max(), 400)
        ax.plot(xs, kde(xs), color=_RED, lw=2, label="KDE")
    except ImportError:
        pass

    for q, label, color in [(q1, "Q1", "#F5A623"), (q2, "Median", "#E74C3C"), (q3, "Q3", "#2ECC71")]:
        ax.axvline(q, color=color, lw=1.5, linestyle="--", label=f"{label}: {q:.0f}")

    ax.set_xlabel("Description Length (characters)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Listing Description Length", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=DPI, bbox_inches="tight")
        logger.info(f"  Saved → {save_path}")
    return fig


def plot_llm_scores_vs_price(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """2x3 scatter grid: each of the 5 LLM numeric scores vs log_price, with regression line + r² annotation. Chart #17"""
    plt.style.use("seaborn-v0_8-darkgrid")
    score_cols = [
        "llm_luxury_score", "llm_uniqueness_score",
        "llm_renovation_quality_score", "llm_curb_appeal_score",
        "llm_spaciousness_score",
    ]
    available = [c for c in score_cols if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(available):
        ax = axes[i]
        sub = df[[col, "log_price"]].dropna()
        ax.scatter(sub[col], sub["log_price"], alpha=0.2, s=10,
                   color=_BLUE, edgecolors="none")
        m, b = np.polyfit(sub[col], sub["log_price"], 1)
        xs = np.linspace(sub[col].min(), sub[col].max(), 100)
        ax.plot(xs, m * xs + b, color=_RED, lw=1.5)
        r2 = np.corrcoef(sub[col], sub["log_price"])[0, 1] ** 2
        ax.text(0.05, 0.92, f"R²={r2:.3f}", transform=ax.transAxes,
                fontsize=9, color=_RED, fontweight="bold")
        ax.set_xlabel(col.replace("llm_", "").replace("_", " ").title())
        ax.set_ylabel("log(Price)")
        ax.set_title(col.replace("llm_", "").replace("_score", "").replace("_", " ").title())

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("LLM Numeric Scores vs. log(Price)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=DPI, bbox_inches="tight")
        logger.info(f"  Saved → {save_path}")
    return fig


def plot_llm_coverage_by_zip(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Horizontal bar: for top-15 ZIPs by count, show % of listings with non-null llm_luxury_score. Chart #18"""
    plt.style.use("seaborn-v0_8-darkgrid")
    valid = df.dropna(subset=["zip_code"]).copy()
    top_zips = valid["zip_code"].value_counts().head(15).index.tolist()
    sub = valid[valid["zip_code"].isin(top_zips)].copy()

    coverage = (
        sub.groupby("zip_code")
        .apply(lambda g: g["llm_luxury_score"].notna().mean() * 100)
        .reindex(top_zips)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#E74C3C" if v < 50 else "#27AE60" for v in coverage.values]
    ax.barh(coverage.index.astype(str), coverage.values, color=colors, edgecolor="white")
    ax.axvline(80, color="#888", lw=1, linestyle="--", label="80% threshold")
    ax.set_xlabel("LLM Coverage (%)")
    ax.set_title("LLM Luxury Score Coverage — Top 15 ZIPs by Listing Count", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=9)
    for i, (z, v) in enumerate(coverage.items()):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=DPI, bbox_inches="tight")
        logger.info(f"  Saved → {save_path}")
    return fig


def plot_luxury_vs_ppsf_zip(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Bubble scatter: x=median llm_luxury_score per ZIP, y=median price_per_sqft, size=listing count, color=count. Chart #19"""
    plt.style.use("seaborn-v0_8-darkgrid")
    valid = df.dropna(subset=["zip_code", "llm_luxury_score", "price", "living_area_sqft"]).copy()
    valid = valid[valid["living_area_sqft"] > 0]
    valid["ppsf"] = valid["price"] / valid["living_area_sqft"]
    valid = valid[valid["ppsf"].between(10, 2000)]

    agg = (
        valid.groupby("zip_code")
        .agg(
            median_luxury=("llm_luxury_score", "median"),
            median_ppsf=("ppsf", "median"),
            count=("ppsf", "count"),
        )
        .reset_index()
    )
    agg = agg[agg["count"] >= 5]

    sizes = (agg["count"] / agg["count"].max() * 500).clip(lower=30)

    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(
        agg["median_luxury"], agg["median_ppsf"],
        s=sizes, c=agg["count"], cmap="YlOrRd",
        alpha=0.75, edgecolors="#444", linewidths=0.5,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Listing Count", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Median LLM Luxury Score")
    ax.set_ylabel("Median Price per Sq Ft ($)")
    ax.set_title("Luxury Score vs. Price/SqFt by ZIP Code\n(bubble size = listing count)",
                 fontweight="bold")

    for _, row in agg.nlargest(10, "count").iterrows():
        ax.annotate(
            str(row["zip_code"]),
            xy=(row["median_luxury"], row["median_ppsf"]),
            fontsize=7, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
        )

    fig.tight_layout()
    if save_path is not None:
        OUTPUTS_FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=DPI, bbox_inches="tight")
        logger.info(f"  Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Correlation analysis (LLM × LLM, LLM × structured, luxury × hard facts)
# ─────────────────────────────────────────────────────────────────────────────

def plot_llm_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap of all LLM features (scores + flags) with each other."""
    llm_cols = [c for c in df.columns if c.startswith("llm_") and
                df[c].dtype in ["int64", "float64", "Int64", "Float64"]]
    if not llm_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No LLM columns found", ha="center")
        return fig

    corr = df[llm_cols].apply(pd.to_numeric, errors="coerce").corr()
    labels = [c.replace("llm_", "").replace("_", " ").title() for c in llm_cols]

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.3,
        xticklabels=labels, yticklabels=labels, ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.set_title("LLM Feature Inter-Correlation Matrix\n"
                 "(scores 1–5 and binary flags)",
                 fontweight="bold", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    _save(fig, "21_llm_correlation_matrix")
    return fig


def plot_llm_vs_structured_correlation(df: pd.DataFrame) -> plt.Figure:
    """Cross-correlation heatmap: LLM features (rows) vs structured features (cols)."""
    struct_cols = [
        "log_price", "bedrooms", "bathrooms",
        "log_living_area", "log_lot_size",
        "property_age", "stories", "garage_spaces",
    ]
    llm_cols = [c for c in df.columns if c.startswith("llm_") and
                df[c].dtype in ["int64", "float64", "Int64", "Float64"]]

    struct_avail = [c for c in struct_cols if c in df.columns]
    llm_avail = [c for c in llm_cols if c in df.columns]
    if not struct_avail or not llm_avail:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient columns", ha="center")
        return fig

    combined = df[llm_avail + struct_avail].apply(pd.to_numeric, errors="coerce")
    corr_full = combined.corr()
    cross = corr_full.loc[llm_avail, struct_avail]

    row_labels = [c.replace("llm_", "").replace("_", " ").title() for c in llm_avail]
    col_labels = [c.replace("_", " ").title() for c in struct_avail]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cross, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-0.5, vmax=0.5, linewidths=0.3,
        xticklabels=col_labels, yticklabels=row_labels, ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Cross-Correlation: LLM Features vs. Structured Variables\n"
                 "(Do LLM features capture information beyond structured data?)",
                 fontweight="bold", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    _save(fig, "22_llm_vs_structured_correlation")
    return fig


def plot_luxury_vs_hardfacts(df: pd.DataFrame) -> plt.Figure:
    """Bar chart: point-biserial correlation of llm_luxury_score with each hard flag."""
    flag_cols = [c for c in df.columns if c.startswith("llm_") and
                 c.endswith("_flag") and df[c].dtype in ["int64", "float64", "Int64", "Float64"]]
    if "llm_luxury_score" not in df.columns or not flag_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Missing columns", ha="center")
        return fig

    luxury = df["llm_luxury_score"].apply(pd.to_numeric, errors="coerce")
    corrs = {}
    for col in flag_cols:
        vals = df[col].apply(pd.to_numeric, errors="coerce")
        valid = luxury.notna() & vals.notna()
        if valid.sum() > 10:
            corrs[col] = float(np.corrcoef(luxury[valid], vals[valid])[0, 1])

    if not corrs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid correlations", ha="center")
        return fig

    s = pd.Series(corrs).sort_values()
    labels = [c.replace("llm_", "").replace("_flag", "").replace("_", " ").title()
              for c in s.index]
    colors = [_RED if v < 0 else _GREEN for v in s.values]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, s.values, color=colors, edgecolor="white", lw=0.5)
    ax.axvline(0, color="#888", lw=1)
    ax.set_xlabel("Point-Biserial Correlation with Luxury Score")
    ax.set_title("Which Hard Flags Correlate with Luxury Score?\n"
                 "(negative = distress indicators inversely related to perceived luxury)",
                 fontweight="bold")
    for i, (lbl, v) in enumerate(zip(labels, s.values)):
        ax.text(v + 0.005 * np.sign(v), i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, "23_luxury_vs_hardfacts")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Lasso visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_lasso_coefficients(coef_table, model_name: str = "LassoCV") -> plt.Figure:
    """Horizontal bar: Lasso selected features by standardised coefficient magnitude."""
    sel = coef_table[coef_table["selected"]].copy()
    sel = sel.sort_values("coef_standardised")

    labels = [c.replace("llm_", "LLM:").replace("zip_code_", "ZIP:").replace("_", " ").title()
              for c in sel.index]
    colors = [_RED if v < 0 else _BLUE for v in sel["coef_standardised"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sel) * 0.3)))
    ax.barh(labels, sel["coef_standardised"], color=colors, edgecolor="white", lw=0.3)
    ax.axvline(0, color="#888", lw=1)
    ax.set_xlabel("Standardised Coefficient")
    ax.set_title(f"{model_name} — Selected Features\n"
                 f"({len(sel)} of {len(coef_table)} features retained, "
                 f"alpha={coef_table.attrs.get('alpha', '?')})",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "24_lasso_coefficients")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# PCA visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_scree(pca, title_suffix: str = "LLM Scores") -> plt.Figure:
    """Scree plot: explained variance per PC + cumulative line."""
    n = pca.n_components_
    var_ratio = pca.explained_variance_ratio_
    cum = np.cumsum(var_ratio)
    pcs = [f"PC{i+1}" for i in range(n)]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(pcs, var_ratio * 100, color=_BLUE, edgecolor="white", lw=0.5, label="Individual")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_xlabel("Principal Component")

    ax2 = ax1.twinx()
    ax2.plot(pcs, cum * 100, "o-", color=_RED, lw=2, label="Cumulative")
    ax2.set_ylabel("Cumulative (%)")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color="#aaa", linestyle="--", lw=1, alpha=0.7)

    for i, (v, c) in enumerate(zip(var_ratio, cum)):
        ax1.text(i, v * 100 + 1, f"{v:.1%}", ha="center", fontsize=9, color=_BLUE)
        ax2.text(i, c * 100 + 2, f"{c:.0%}", ha="center", fontsize=8, color=_RED)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(f"PCA Scree Plot — {title_suffix}\n"
                  f"({n} components, {cum[-1]:.0%} total variance explained)",
                  fontweight="bold")
    fig.tight_layout()
    _save(fig, "26_pca_scree")
    return fig


def plot_pca_loadings(pca, feature_names: list[str]) -> plt.Figure:
    """Heatmap of PCA loadings — which original features load onto which PC."""
    loadings = pd.DataFrame(
        pca.components_.T,
        index=[c.replace("llm_", "").replace("_score", "").replace("_", " ").title()
               for c in feature_names],
        columns=[f"PC{i+1}\n({v:.0%})" for i, v in enumerate(pca.explained_variance_ratio_)],
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        loadings, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        ax=ax, annot_kws={"fontsize": 10} if len(feature_names) <= 7 else {},
    )
    ax.set_title("PCA Loadings — LLM Score Features\n"
                 "(how each original score maps to principal components)",
                 fontweight="bold")
    ax.set_xlabel("Principal Component (% variance)")
    fig.tight_layout()
    _save(fig, "27_pca_loadings")
    return fig


def plot_pca_scatter(scores_df: pd.DataFrame, labels: pd.Series = None,
                     price: pd.Series = None) -> plt.Figure:
    """2D scatter of PC1 vs PC2, colored by cluster or price."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if labels is not None:
        for cl in sorted(labels.unique()):
            mask = labels == cl
            axes[0].scatter(scores_df.loc[mask, "PC1"], scores_df.loc[mask, "PC2"],
                           s=12, alpha=0.5, label=f"Cluster {cl}")
        axes[0].legend(fontsize=8)
        axes[0].set_title("PC1 vs PC2 — by Cluster", fontweight="bold")
    else:
        axes[0].scatter(scores_df["PC1"], scores_df["PC2"], s=12, alpha=0.3, color=_BLUE)
        axes[0].set_title("PC1 vs PC2", fontweight="bold")

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    if price is not None:
        sc = axes[1].scatter(scores_df["PC1"], scores_df["PC2"],
                             c=price, s=12, alpha=0.5, cmap="RdYlGn_r")
        cb = fig.colorbar(sc, ax=axes[1], shrink=0.8)
        cb.set_label("log(Price)")
        axes[1].set_title("PC1 vs PC2 — by Price", fontweight="bold")
    else:
        axes[1].set_visible(False)

    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.suptitle("PCA of LLM Quality Scores — Latent Property Quality Dimensions",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "28_pca_scatter")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Cluster profiling
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_elbow(inertia_df: pd.DataFrame) -> plt.Figure:
    """Elbow plot for K-means — inertia vs k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(inertia_df["k"], inertia_df["inertia"], "o-", color=_BLUE, lw=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("K-Means Elbow Plot\n(choose k where marginal reduction flattens)",
                 fontweight="bold")
    ax.set_xticks(inertia_df["k"])
    fig.tight_layout()
    _save(fig, "29_cluster_elbow")
    return fig


def plot_cluster_profiles(df: pd.DataFrame, labels: pd.Series,
                          profile_cols: list[str] | None = None) -> plt.Figure:
    """Grouped bar chart: mean of key variables per cluster — descriptive profiling."""
    if profile_cols is None:
        profile_cols = [
            "price", "living_area_sqft", "bedrooms", "bathrooms", "property_age",
            "llm_luxury_score", "llm_renovation_quality_score",
            "llm_spaciousness_score", "llm_curb_appeal_score",
        ]
    avail = [c for c in profile_cols if c in df.columns]
    tmp = df[avail].copy()
    tmp["cluster"] = labels.values

    # Normalise each column to [0,1] for visual comparability
    for col in avail:
        vals = tmp[col].apply(pd.to_numeric, errors="coerce")
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            tmp[col] = (vals - mn) / (mx - mn)

    means = tmp.groupby("cluster")[avail].mean()
    labels_clean = [c.replace("llm_", "").replace("_", " ").title() for c in avail]

    n_clusters = len(means)
    x = np.arange(len(avail))
    width = 0.8 / n_clusters
    colors = sns.color_palette("Set2", n_clusters)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (cl, row) in enumerate(means.iterrows()):
        ax.bar(x + i * width, row.values, width, label=f"Cluster {cl}",
               color=colors[i], edgecolor="white", lw=0.3)

    ax.set_xticks(x + width * (n_clusters - 1) / 2)
    ax.set_xticklabels(labels_clean, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Normalised Mean (0–1)")
    ax.set_title("Cluster Profiles — Structural + LLM Features\n"
                 "(each variable normalised to [0,1] for visual comparability)",
                 fontweight="bold")
    ax.legend(title="Cluster", fontsize=9)
    fig.tight_layout()
    _save(fig, "30_cluster_profiles")
    return fig


def plot_cluster_price_box(df: pd.DataFrame, labels: pd.Series) -> plt.Figure:
    """Box plot of price by cluster — shows economic meaning of clusters."""
    tmp = df[["price"]].copy()
    tmp["Cluster"] = labels.values.astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=tmp, x="Cluster", y="price", palette="Set2", ax=ax,
                showfliers=False)
    sns.stripplot(data=tmp, x="Cluster", y="price", color="#333", size=2,
                  alpha=0.15, ax=ax, jitter=True)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1_000:,.0f}K"))
    ax.set_ylabel("List Price")
    ax.set_title("Price Distribution by Cluster\n"
                 "(do latent quality dimensions predict price segments?)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "31_cluster_price_box")
    return fig


def plot_residual_map_interactive(df: pd.DataFrame, y_pred: np.ndarray, save_path=None):
    """Interactive folium map: scatter dots colored by residual (actual-predicted log_price), blue=under, red=over. Chart #20"""
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.warning("folium not installed — skipping residual map")
        return None

    valid = df.copy().reset_index(drop=True)
    valid["_y_pred"] = y_pred
    valid = valid.dropna(subset=["latitude", "longitude", "log_price"])
    valid["_residual"] = valid["log_price"] - valid["_y_pred"]

    center_lat = float(valid["latitude"].median())
    center_lon = float(valid["longitude"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB dark_matter")

    vmax = float(np.percentile(np.abs(valid["_residual"]), 95))

    def _color(r: float) -> str:
        norm = max(-1.0, min(1.0, r / (vmax + 1e-9)))
        if norm >= 0:
            intensity = int(norm * 200)
            return f"#{200 + intensity // 5:02x}{50:02x}{50:02x}"
        else:
            intensity = int(-norm * 200)
            return f"#{50:02x}{50:02x}{200 + intensity // 5:02x}"

    for _, row in valid.iterrows():
        resid = row["_residual"]
        color = _color(resid)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            opacity=0.8,
            tooltip=f"Residual: {resid:+.3f} | Actual log-price: {row['log_price']:.3f}",
        ).add_to(m)

    if save_path is not None:
        path = str(save_path)
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        m.save(path)
        logger.info(f"  Saved residual map → {path}")
    return m
