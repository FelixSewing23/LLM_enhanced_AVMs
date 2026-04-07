"""
3D interactive map visualisations for Houston listings (pydeck).

Charts:
  A – Listing density + median price      (ZIP polygon extrusion)
  B – Price per sqft by ZIP               (hex grid  +  ZIP polygon version)
  C – LLM luxury score geography          (ZIP polygon extrusion)
  D – K-means clusters on LLM features    (scatter + ZIP outlines)
  E – Distress flag density               (ZIP polygon extrusion)

Charts A, B-ZIP, C, E use GeoJsonLayer extruded ZIP polygons for geographic accuracy.
All charts share the same dark basemap, tooltip style, and colour helpers.
"""
from __future__ import annotations

import copy
import json
import urllib.request
from pathlib import Path

import pandas as pd
import pydeck as pdk

# Cached ZIP boundary file (fetched once from Census TIGER)
_ZIP_CACHE = Path(__file__).parent.parent / "data" / "interim" / "houston_zip_boundaries.geojson"

# Houston bounding box (west, south, east, north)
_HOUSTON_BBOX = (-96.5, 29.2, -94.8, 30.4)


def _fetch_houston_zip_geojson() -> dict:
    """Load Houston-area ZIP boundaries (cached from OpenDataDE Texas GeoJSON)."""
    if _ZIP_CACHE.exists():
        with open(_ZIP_CACHE, encoding="utf-8") as f:
            return json.load(f)

    # Download full Texas ZIP GeoJSON and filter to Houston bounding box
    url = (
        "https://raw.githubusercontent.com/OpenDataDE/"
        "State-zip-code-GeoJSON/master/tx_texas_zip_codes_geo.min.json"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read())

    w, s, e, n = _HOUSTON_BBOX

    def _in_bbox(feat: dict) -> bool:
        def _flat(c):
            if isinstance(c[0], list):
                return [p for sub in c for p in _flat(sub)]
            return [c]
        pts = _flat(feat["geometry"]["coordinates"])
        lngs = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return min(lngs) < e and max(lngs) > w and min(lats) < n and max(lats) > s

    houston = {
        "type": "FeatureCollection",
        "features": [f for f in data["features"] if _in_bbox(f)],
    }

    _ZIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_ZIP_CACHE, "w", encoding="utf-8") as f:
        json.dump(houston, f)

    return houston


def zip_outline_layer() -> pdk.Layer:
    """Thin white wireframe outlines of all Houston-area ZIP codes."""
    geojson = _fetch_houston_zip_geojson()
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        stroked=True,
        filled=True,
        get_line_color=[255, 255, 255, 60],   # subtle white outline
        get_fill_color=[255, 255, 255, 6],     # near-invisible fill so basemap shows through
        line_width_min_pixels=1,
        pickable=False,
    )


# ── colour ramps (RGB triplets for pydeck) ────────────────────────────────────
# Deep-purple → gold  (dramatic, dark-background friendly)
PRICE_COLORMAP = [
    [68,  1,  84],   # 0 %  — deep purple
    [59,  82, 139],  # 25 %
    [33, 145, 140],  # 50 %  — teal
    [94, 201,  98],  # 75 %
    [253, 231,  37],  # 100 % — gold
]

# Blue → red  (luxury)
LUXURY_COLORMAP = [
    [0,   0, 180],
    [0, 120, 220],
    [80, 200, 120],
    [240, 120,   0],
    [220,  20,  20],
]

# Light pink → deep red  (distress)
RED_COLORMAP = [
    [255, 235, 235],
    [255, 180, 180],
    [255, 100, 100],
    [220,  30,  30],
    [140,   0,   0],
]

# Map view centred on Houston
HOUSTON_VIEW = pdk.ViewState(
    latitude=29.76,
    longitude=-95.37,
    zoom=9.8,
    pitch=45,
    bearing=-15,
)

DARK_MAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

_DEFAULT_TOOLTIP = {"html": "{tooltip}", "style": {"color": "white", "background": "rgba(0,0,0,.7)"}}


def _base_deck(layers: list, view: pdk.ViewState | None = None, tooltip: dict | None = None) -> pdk.Deck:
    return pdk.Deck(
        layers=layers,
        initial_view_state=view or HOUSTON_VIEW,
        map_style=DARK_MAP,
        tooltip=tooltip or _DEFAULT_TOOLTIP,
    )


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _extract_neighborhood(df: pd.DataFrame) -> pd.Series:
    """Extract neighborhood name from dict {'name': ...} or plain string column."""
    return df["neighborhood"].apply(
        lambda x: x["name"] if isinstance(x, dict) and "name" in x else (x if isinstance(x, str) else None)
    )


def _mode_agg(s: pd.Series) -> str:
    """Return most common non-null value, or empty string."""
    vals = s.dropna()
    return str(vals.mode()[0]) if not vals.empty else ""


def _color_from_norm(n: float, colormap: list) -> list[int]:
    """Interpolate an RGBA colour from a 0–1 normalised value along a colormap."""
    idx = n * (len(colormap) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(colormap) - 1)
    t = idx - lo
    return [int(colormap[lo][i] * (1 - t) + colormap[hi][i] * t) for i in range(3)] + [190]


def _zip_geojson_extrusion(stats_map: dict) -> pdk.Layer:
    """Build extruded GeoJsonLayer from stats_map {zipcode: {elevation, color, tooltip}}.

    Each value must have keys: elevation (float), color (list[int] ×4), tooltip (str).
    """
    geojson = _fetch_houston_zip_geojson()
    features = []
    for feat in geojson["features"]:
        zipcode = str(feat["properties"].get("ZCTA5CE10", ""))
        if zipcode not in stats_map:
            continue
        s = stats_map[zipcode]
        feat = copy.deepcopy(feat)
        feat["properties"].update({
            "elevation": float(s["elevation"]),
            "color": s["color"],
            "tooltip": s["tooltip"],
        })
        features.append(feat)

    return pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": features},
        extruded=True,
        get_elevation="properties.elevation",
        get_fill_color="properties.color",
        opacity=0.75,
        pickable=True,
        auto_highlight=True,
        stroked=True,
        get_line_color=[255, 255, 255, 80],
        line_width_min_pixels=1,
    )


def _neighborhood_line(r) -> str:
    """Return '<neighborhood><br>' if present, else empty string."""
    name = r.get("neighborhood", "")
    return f"{name}<br>" if name else ""


# ── Chart A: listing density + median price (ZIP extrusion) ───────────────────

def chart_a_density_price(
    df: pd.DataFrame,
    elevation_scale: int = 600,
) -> pdk.Deck:
    """ZIP polygon extrusion — height = listing count, colour = median price."""
    df2 = df.dropna(subset=["latitude", "longitude", "price", "zip_code"]).copy()
    df2["_neighborhood"] = _extract_neighborhood(df2)

    zip_stats = (
        df2.groupby("zip_code")
        .agg(
            median_price=("price", "median"),
            count=("price", "count"),
            neighborhood=("_neighborhood", _mode_agg),
        )
        .reset_index()
        .dropna(subset=["median_price"])
    )

    mn, mx = zip_stats["median_price"].min(), zip_stats["median_price"].max()
    zip_stats["norm"] = (zip_stats["median_price"] - mn) / (mx - mn + 1e-9)

    stats_map = {}
    for _, r in zip_stats.iterrows():
        stats_map[str(r["zip_code"])] = {
            "elevation": float(r["count"]) * elevation_scale,
            "color": _color_from_norm(float(r["norm"]), PRICE_COLORMAP),
            "tooltip": (
                f"<b>ZIP: {r['zip_code']}</b><br>"
                + _neighborhood_line(r)
                + f"listings: {int(r['count'])}<br>"
                f"median price: ${r['median_price']:,.0f}"
            ),
        }

    deck = _base_deck([zip_outline_layer(), _zip_geojson_extrusion(stats_map)])
    deck.description = "Chart A — Listing density & median price"
    return deck


# ── Chart B: $/sqft by ZIP — hex version + column version ────────────────────

def chart_b_ppsf_hex(
    df: pd.DataFrame,
    radius: int = 2_000,
    elevation_scale: int = 50,
    show_zip_grid: bool = True,
) -> pdk.Deck:
    """Hex grid version — height + colour = median $/sqft."""
    pts = (
        df.assign(ppsf=df["price"] / df["living_area_sqft"])
        .query("ppsf > 0 and ppsf < 2000")
        [["latitude", "longitude", "ppsf"]]
        .dropna()
        .rename(columns={"latitude": "lat", "longitude": "lng"})
        .to_dict("records")
    )

    layer = pdk.Layer(
        "HexagonLayer",
        data=pts,
        get_position="[lng, lat]",
        get_elevation_weight="ppsf",
        elevation_aggregation="MEAN",
        get_color_weight="ppsf",
        color_aggregation="MEAN",
        radius=radius,
        elevation_scale=elevation_scale,
        extruded=True,
        pickable=True,
        auto_highlight=True,
        color_range=PRICE_COLORMAP,
        coverage=0.9,
    )

    layers = ([zip_outline_layer()] if show_zip_grid else []) + [layer]
    # HexagonLayer exposes colorValue / elevationValue in tooltip (deck.gl built-ins)
    hex_tooltip = {
        "html": "<b>Hex bin</b><br>Avg $/sqft: <b>${colorValue}</b><br>Listings: {elevationValue}",
        "style": {"color": "white", "background": "rgba(0,0,0,.7)"},
    }
    deck = _base_deck(layers, tooltip=hex_tooltip)
    deck.description = "Chart B (hex) — Price per sqft"
    return deck


def chart_b_ppsf_zip(df: pd.DataFrame, elevation_scale: int = 300, show_zip_grid: bool = False) -> pdk.Deck:
    """ZIP polygon extrusion — height + colour = median $/sqft."""
    df2 = df.assign(ppsf=df["price"] / df["living_area_sqft"]).query("ppsf > 0 and ppsf < 2000").copy()
    df2["_neighborhood"] = _extract_neighborhood(df2)

    zip_stats = (
        df2.groupby("zip_code")
        .agg(
            lat=("latitude", "mean"),
            lng=("longitude", "mean"),
            median_ppsf=("ppsf", "median"),
            median_price=("price", "median"),
            count=("ppsf", "count"),
            neighborhood=("_neighborhood", _mode_agg),
        )
        .reset_index()
        .dropna(subset=["lat", "lng", "median_ppsf"])
    )

    mn, mx = zip_stats["median_ppsf"].min(), zip_stats["median_ppsf"].max()
    zip_stats["norm"] = (zip_stats["median_ppsf"] - mn) / (mx - mn + 1e-9)

    stats_map = {}
    for _, r in zip_stats.iterrows():
        stats_map[str(r["zip_code"])] = {
            "elevation": float(r["median_ppsf"]) * elevation_scale,
            "color": _color_from_norm(float(r["norm"]), PRICE_COLORMAP),
            "tooltip": (
                f"<b>ZIP: {r['zip_code']}</b><br>"
                + _neighborhood_line(r)
                + f"median $/sqft: ${r['median_ppsf']:.0f}<br>"
                f"median price: ${r['median_price']:,.0f}<br>"
                f"listings: {int(r['count'])}"
            ),
        }

    layers = ([zip_outline_layer()] if show_zip_grid else []) + [_zip_geojson_extrusion(stats_map)]
    deck = _base_deck(layers)
    deck.description = "Chart B (ZIP) — Price per sqft by ZIP code"
    return deck


# ── Chart C: LLM luxury score geography (ZIP extrusion) ──────────────────────

def chart_c_luxury(
    df: pd.DataFrame,
    elevation_scale: int = 8_000,
) -> pdk.Deck:
    """ZIP polygon extrusion — height + colour = median LLM luxury score."""
    col = "llm_luxury_score"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found — run LLM pipeline first.")

    df2 = df.dropna(subset=["latitude", "longitude", col, "zip_code"]).copy()
    df2["_neighborhood"] = _extract_neighborhood(df2)

    zip_stats = (
        df2.groupby("zip_code")
        .agg(
            median_luxury=(col, "median"),
            median_price=("price", "median"),
            count=(col, "count"),
            neighborhood=("_neighborhood", _mode_agg),
        )
        .reset_index()
        .dropna(subset=["median_luxury"])
    )

    mn, mx = zip_stats["median_luxury"].min(), zip_stats["median_luxury"].max()
    zip_stats["norm"] = (zip_stats["median_luxury"] - mn) / (mx - mn + 1e-9)

    stats_map = {}
    for _, r in zip_stats.iterrows():
        stats_map[str(r["zip_code"])] = {
            "elevation": float(r["median_luxury"]) * elevation_scale,
            "color": _color_from_norm(float(r["norm"]), LUXURY_COLORMAP),
            "tooltip": (
                f"<b>ZIP: {r['zip_code']}</b><br>"
                + _neighborhood_line(r)
                + f"avg luxury score: {r['median_luxury']:.1f}/5<br>"
                f"median price: ${r['median_price']:,.0f}<br>"
                f"listings: {int(r['count'])}"
            ),
        }

    deck = _base_deck([zip_outline_layer(), _zip_geojson_extrusion(stats_map)])
    deck.description = "Chart C — LLM luxury score geography"
    return deck


# ── Chart D: K-means clusters on LLM features ────────────────────────────────

def chart_d_clusters(df: pd.DataFrame, n_clusters: int = 4, save_path=None) -> pdk.Deck:
    """
    K-means cluster on LLM scores + distress flags → pydeck ScatterplotLayer.
    Cluster on: llm_luxury_score, llm_uniqueness_score, llm_renovation_quality_score,
                llm_curb_appeal_score, llm_spaciousness_score, llm_as_is_flag, llm_fixer_upper_flag
    Impute nulls with column median before clustering.
    Use 4 clusters. Assign descriptive names based on cluster centroids.
    Colour each cluster distinctly (gold, teal, orange, red).
    Tooltip shows cluster name, price, ZIP, and neighborhood.
    Returns pdk.Deck
    """
    from sklearn.cluster import KMeans
    import numpy as np

    cluster_cols = [
        "llm_luxury_score", "llm_uniqueness_score", "llm_renovation_quality_score",
        "llm_curb_appeal_score", "llm_spaciousness_score",
        "llm_as_is_flag", "llm_fixer_upper_flag",
    ]
    available_cols = [c for c in cluster_cols if c in df.columns]

    valid = df.dropna(subset=["latitude", "longitude"]).copy().reset_index(drop=True)
    valid["_neighborhood"] = _extract_neighborhood(valid)

    X = valid[available_cols].copy()
    for col in available_cols:
        X[col] = X[col].fillna(X[col].median())

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    valid["_cluster"] = km.fit_predict(X.values)

    # Assign descriptive names based on centroids
    centroids = km.cluster_centers_
    lux_idx = available_cols.index("llm_luxury_score") if "llm_luxury_score" in available_cols else 0
    reno_idx = available_cols.index("llm_renovation_quality_score") if "llm_renovation_quality_score" in available_cols else 2
    asis_idx = available_cols.index("llm_as_is_flag") if "llm_as_is_flag" in available_cols else -1
    fixer_idx = available_cols.index("llm_fixer_upper_flag") if "llm_fixer_upper_flag" in available_cols else -1

    distress_cols_idx = []
    if asis_idx >= 0:
        distress_cols_idx.append(asis_idx)
    if fixer_idx >= 0:
        distress_cols_idx.append(fixer_idx)

    cluster_names = {}
    ranked_lux = sorted(range(n_clusters), key=lambda i: centroids[i][lux_idx], reverse=True)
    ranked_reno = sorted(range(n_clusters), key=lambda i: centroids[i][reno_idx], reverse=True)

    for k in range(n_clusters):
        distress_score = np.mean([centroids[k][j] for j in distress_cols_idx]) if distress_cols_idx else 0
        if distress_score > 0.3:
            cluster_names[k] = "Distressed"
        elif ranked_lux.index(k) == 0 and ranked_reno.index(k) <= 1:
            cluster_names[k] = "Luxury Move-In"
        elif ranked_reno.index(k) <= 1 and centroids[k][lux_idx] < 3.5:
            cluster_names[k] = "Value-Add Renovator"
        else:
            cluster_names[k] = "Standard"

    # Colours: gold, teal, orange, red
    colour_map = {
        "Luxury Move-In":      [255, 215,   0, 220],
        "Standard":            [ 32, 178, 170, 220],
        "Value-Add Renovator": [255, 140,   0, 220],
        "Distressed":          [220,  30,  30, 220],
    }

    price_col = "price" if "price" in valid.columns else None
    cluster_medians = {}
    if price_col:
        cluster_medians = valid.groupby("_cluster")[price_col].median().to_dict()

    records = []
    for _, row in valid.iterrows():
        k = int(row["_cluster"])
        name = cluster_names.get(k, f"Cluster {k}")
        med_price = cluster_medians.get(k, 0)
        zipcode = str(row.get("zip_code", "")) if pd.notna(row.get("zip_code")) else ""
        hood = str(row["_neighborhood"]) if pd.notna(row["_neighborhood"]) else ""
        records.append({
            "lat": float(row["latitude"]),
            "lng": float(row["longitude"]),
            "color": colour_map.get(name, [128, 128, 128, 200]),
            "tooltip": (
                f"<b>{name}</b><br>"
                + (f"{hood}<br>" if hood else "")
                + (f"ZIP: {zipcode}<br>" if zipcode else "")
                + f"price: ${float(row[price_col]):,.0f}<br>" if price_col and pd.notna(row.get(price_col)) else ""
                + f"cluster median: ${med_price:,.0f}"
            ),
        })

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=records,
        get_position="[lng, lat]",
        get_fill_color="color",
        get_radius=150,
        pickable=True,
        auto_highlight=True,
        opacity=0.8,
    )

    deck = _base_deck([zip_outline_layer(), layer])
    deck.description = "Chart D — LLM K-means clusters"
    return deck


# ── Chart E: Distress flag density (ZIP extrusion) ───────────────────────────

def chart_e_distress(
    df: pd.DataFrame,
    elevation_scale: int = 30_000,
) -> pdk.Deck:
    """ZIP polygon extrusion — height + colour = share of listings with ≥1 distress flag.

    Distress flags: as_is, fixer_upper, foreclosure, water_damage, foundation_issue, needs_repair.
    """
    distress_flag_cols = [
        "llm_as_is_flag", "llm_fixer_upper_flag", "llm_foreclosure_flag",
        "llm_water_damage_flag", "llm_foundation_issue_flag", "llm_needs_repair_flag",
    ]
    available_flags = [c for c in distress_flag_cols if c in df.columns]

    valid = df.dropna(subset=["latitude", "longitude", "zip_code"]).copy()
    if available_flags:
        valid["_distress"] = valid[available_flags].fillna(0).max(axis=1).astype(float)
    else:
        valid["_distress"] = 0.0
    valid["_neighborhood"] = _extract_neighborhood(valid)

    agg_dict = {
        "distress_share": ("_distress", "mean"),
        "count": ("_distress", "count"),
        "neighborhood": ("_neighborhood", _mode_agg),
    }
    if "price" in valid.columns:
        agg_dict["median_price"] = ("price", "median")

    zip_stats = (
        valid.groupby("zip_code")
        .agg(**agg_dict)
        .reset_index()
        .dropna(subset=["distress_share"])
    )

    mn, mx = zip_stats["distress_share"].min(), zip_stats["distress_share"].max()
    zip_stats["norm"] = (zip_stats["distress_share"] - mn) / (mx - mn + 1e-9)

    stats_map = {}
    for _, r in zip_stats.iterrows():
        price_line = f"median price: ${r['median_price']:,.0f}<br>" if "median_price" in r and pd.notna(r["median_price"]) else ""
        stats_map[str(r["zip_code"])] = {
            "elevation": float(r["distress_share"]) * elevation_scale,
            "color": _color_from_norm(float(r["norm"]), RED_COLORMAP),
            "tooltip": (
                f"<b>ZIP: {r['zip_code']}</b><br>"
                + _neighborhood_line(r)
                + f"distressed: {r['distress_share']:.0%}<br>"
                + price_line
                + f"listings: {int(r['count'])}"
            ),
        }

    deck = _base_deck([zip_outline_layer(), _zip_geojson_extrusion(stats_map)])
    deck.description = "Chart E — Distress flag density"
    return deck


# ── Save helper ───────────────────────────────────────────────────────────────

def save_deck(deck: pdk.Deck, path: Path) -> Path:
    """Export pydeck Deck as a standalone HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(path), open_browser=False)
    return path


def show_deck(deck: pdk.Deck, height: int = 600):
    """Render a pydeck Deck inline in any Jupyter environment (including VSCode).

    Falls back from native widget → iframe of to_html() output.
    """
    from IPython.display import HTML, display

    html = deck.to_html(as_string=True)
    display(HTML(
        f'<iframe srcdoc="{html.replace(chr(34), "&quot;")}" '
        f'width="100%" height="{height}" frameborder="0"></iframe>'
    ))
    return None
