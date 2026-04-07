"""Data cleaning and sample construction.

Takes raw Zillow detail-scraper output (list of dicts) and produces:
  1. A wide, normalized DataFrame (all available fields)
  2. A cleaned, analysis-ready DataFrame with derived columns
  3. An attrition table documenting every dropped row and the reason
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from src.config import (
    PRICE_MIN,
    PRICE_MAX,
    DAYS_ON_ZILLOW,
    EXCLUDE_HOME_TYPES,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raw JSON → flat DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast any object column whose non-null values are all bool/int to BooleanDtype.

    PyArrow cannot infer a schema for object columns that mix Python bools and
    ints (e.g. has_fireplace gets True/False from hasFireplace but falls back to
    the integer fireplaces count).  BooleanDtype stores them uniformly as
    nullable booleans and serialises cleanly to parquet.
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        if all(isinstance(v, (bool, int)) and not isinstance(v, float)
               for v in non_null):
            df[col] = df[col].map(
                lambda x: bool(x) if x is not None else pd.NA
            ).astype(pd.BooleanDtype())
    return df


def _parse_money(val) -> float | None:
    """Extract the first number from a raw money value.

    Handles ints, floats, and strings like '$495 bi-annually', '$1,200/mo'.
    Returns None when the value is absent or unparseable.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if not (isinstance(val, float) and val != val) else None
    match = re.search(r"[\d,]+\.?\d*", str(val).replace(",", ""))
    return float(match.group().replace(",", "")) if match else None


def _parse_number(val) -> float | None:
    """Extract a numeric value from ints, floats, or numeric-like strings."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if not (isinstance(val, float) and val != val) else None
    match = re.search(r"-?[\d,]+\.?\d*", str(val).replace(",", ""))
    return float(match.group().replace(",", "")) if match else None


def _extract_photo_urls(photos, max_n: int = 5) -> list[str] | None:
    """Return up to max_n photo URLs from originalPhotos."""
    if not photos:
        return None

    urls = [
        photo.get("mixedSources", {}).get("jpeg", [{}])[0].get("url")
        for photo in photos[:max_n]
        if isinstance(photo, dict) and photo.get("mixedSources", {}).get("jpeg")
    ]
    return urls or None

def normalize_detail_records(records: list[dict]) -> pd.DataFrame:
    """Flatten every field from the Zillow detail scraper response.

    We collect *all* available data (as instructed) and will select columns
    at the modelling stage.  The output schema mirrors the data dictionary
    defined in the project instructions.
    """
    rows = [_flatten_record(r) for r in records]
    df   = pd.DataFrame(rows)
    df   = _coerce_bool_columns(df)
    logger.info(f"Normalized {len(df):,} records → {df.shape[1]} columns")
    return df


def _flatten_record(r: dict) -> dict:
    """Extract and rename fields from one detail-scraper record."""

    def g(d, *keys, default=None):
        """Safe nested getter."""
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        return d

    address       = r.get("address") or {}
    if isinstance(address, str):
        address_full = address
        address      = {}
    else:
        address_full  = g(address, "streetAddress")

    reso          = r.get("resoFacts")    or {}
    tax_hist      = r.get("taxHistory")   or []
    price_hist    = r.get("priceHistory") or []
    attrib        = r.get("attributionInfo") or {}
    listing_agent = r.get("listingAgent")    or {}
    photos        = r.get("originalPhotos") or []
    schools       = r.get("schools") or []
    hd            = r.get("hdpData") or {}
    hi            = g(hd, "homeInfo") or {}

    # ── Helper: first truthy value from ordered candidates ──────────────────
    def first(*vals):
        for v in vals:
            if v is not None and v != "":
                return v
        return None

    row = {
        # ── Identity ─────────────────────────────────────────────────────────
        "zpid":               first(r.get("zpid"), hi.get("zpid")),
        "url":                first(r.get("url"), r.get("hdpUrl")),

        # ── Address ──────────────────────────────────────────────────────────
        "address_full":       address_full,
        "street_address":     g(address, "streetAddress"),
        "city":               first(g(address, "city"),  r.get("city"),  hi.get("city")),
        "state":              first(g(address, "state"), r.get("state"), hi.get("state")),
        "zip_code":           first(g(address, "zipcode"), r.get("zipcode"), hi.get("zipcode")),
        "neighborhood":       first(r.get("neighborhoodRegion"), hi.get("neighborhoodRegion")),
        "latitude":           first(r.get("latitude"),  g(address, "latitude"),  hi.get("latitude")),
        "longitude":          first(r.get("longitude"), g(address, "longitude"), hi.get("longitude")),

        # ── Listing metadata ──────────────────────────────────────────────────
        "home_status":        first(r.get("homeStatus"),     hi.get("homeStatus")),
        "days_on_zillow":     first(r.get("daysOnZillow"),   hi.get("daysOnZillow")),
        "listing_date":       first(r.get("listingDateTimeOnZillow"), r.get("datePosted"), r.get("datePostedString")),
        "price":              first(r.get("price"),          r.get("unformattedPrice"), hi.get("price")),
        "price_per_sqft":     first(r.get("pricePerSquareFoot"), r.get("priceSqFt"), g(reso, "pricePerSquareFoot"), hi.get("pricePerSquareFoot")),
        "zestimate":          first(r.get("zestimate"),      hi.get("zestimate")),
        "rent_zestimate":     first(r.get("rentZestimate"),  hi.get("rentZestimate")),

        # ── Property characteristics ──────────────────────────────────────────
        "bedrooms":           first(r.get("bedrooms"),  r.get("beds"),  hi.get("bedrooms")),
        "bathrooms":          first(r.get("bathrooms"), r.get("baths"), hi.get("bathrooms")),
        "living_area_sqft":   first(r.get("livingArea"), r.get("area"), hi.get("livingArea")),
        "lot_size_value":     r.get("lotAreaValue"),
        "lot_size_unit":      r.get("lotAreaUnits"),
        "lot_size_sqft_raw":  _parse_number(r.get("lotSize")),
        "year_built":         first(r.get("yearBuilt"),  hi.get("yearBuilt")),
        "home_type":          first(r.get("homeType"),   hi.get("homeType")),
        "stories":            first(r.get("stories"),    g(reso, "stories")),
        "new_construction":   first(r.get("newConstruction"), g(reso, "newConstruction"), g(reso, "isNewConstruction")),

        # ── HVAC / systems ────────────────────────────────────────────────────
        "has_cooling":        g(reso, "hasCooling"),
        "has_heating":        g(reso, "hasHeating"),
        "cooling":            g(reso, "cooling"),
        "heating":            g(reso, "heating"),

        # ── Parking ───────────────────────────────────────────────────────────
        "parking_capacity":   g(reso, "parkingCapacity"),
        "garage_spaces":      first(g(reso, "garageSpaces"), g(reso, "garageParkingCapacity")),
        "has_garage":         g(reso, "hasGarage"),
        "has_attached_garage":g(reso, "hasAttachedGarage"),
        "parking_type":       g(reso, "parkingFeatures"),

        # ── Financials ────────────────────────────────────────────────────────
        "hoa_fee":            _parse_money(first(r.get("monthlyHoaFee"), g(reso, "hoaFee"))),
        "annual_insurance":   _parse_money(r.get("annualHomeownersInsurance")),
        "tax_annual":         _parse_money(g(tax_hist[0], "taxPaid")   if tax_hist  else None),
        "tax_year":           g(tax_hist[0], "time")      if tax_hist  else None,
        "prev_sale_price":    _parse_money(g(price_hist[0], "price")   if price_hist else None),
        "prev_sale_date":     g(price_hist[0], "date")    if price_hist else None,
        "price_history_raw":  str(price_hist[:3])         if price_hist else None,

        # ── Amenities ─────────────────────────────────────────────────────────
        "has_pool":           first(g(reso, "hasPrivatePool"), g(reso, "hasPool")),
        "has_spa":            g(reso, "hasSpa"),
        "has_fireplace":      first(g(reso, "hasFireplace"), g(reso, "fireplaces")),
        "has_basement":       first(g(reso, "hasBasement"), g(reso, "basementYN")),
        "flooring":           g(reso, "flooring"),
        "appliances":         g(reso, "appliances"),
        "interior_features":  g(reso, "interiorFeatures"),
        "exterior_features":  g(reso, "exteriorFeatures"),
        "construction_materials": g(reso, "constructionMaterials"),
        "roof":               first(g(reso, "roof"), g(reso, "roofType")),
        "foundation_details": g(reso, "foundationDetails"),
        "water_source":       g(reso, "waterSource"),
        "sewer":              g(reso, "sewer"),
        "utilities":          g(reso, "utilities"),

        # ── Schools ───────────────────────────────────────────────────────────
        "school_count":       len(schools),
        "school_names":       [s.get("name") for s in schools[:3]] if schools else None,

        # ── Agent / broker ────────────────────────────────────────────────────
        "listing_agent_name": first(g(listing_agent, "name"), g(attrib, "agentName")),
        "listing_agent_phone":first(g(listing_agent, "phone"), g(attrib, "agentPhoneNumber")),
        "broker_name":        r.get("brokerageName"),
        "listing_provider":   g(attrib, "providerListingId"),
        "mls_id":             r.get("mlsid"),

        # ── Text ──────────────────────────────────────────────────────────────
        "property_description": r.get("description"),

        # ── Media ─────────────────────────────────────────────────────────────
        "photo_count":        r.get("photoCount"),
        "photo_urls":         _extract_photo_urls(photos),

        # ── Page metadata ─────────────────────────────────────────────────────
        "featured":           r.get("isFeatured"),
        "open_house":         r.get("openHouseSchedule"),
        "virtual_tour_url":   r.get("virtualTourUrl"),
        "listing_type":       r.get("listingTypeDimension"),
    }

    return row


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_listings(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply sample filters and quality controls.

    Returns
    -------
    clean_df     : analysis-ready DataFrame with derived columns added
    attrition_df : table showing rows remaining / dropped at each step
    """
    steps: list[tuple[str, int, int]] = []
    n0 = len(df)

    def checkpoint(label: str, current: pd.DataFrame):
        n     = len(current)
        prev  = steps[-1][1] if steps else n0
        steps.append((label, n, prev - n))
        logger.info(f"  [{label:35s}] {n:>5,} remaining  ({prev - n:>4,} dropped)")

    logger.info(f"Cleaning pipeline: {n0:,} raw records")

    # 1. Deduplicate by zpid (most stable identifier)
    df = df.drop_duplicates(subset=["zpid"], keep="first").reset_index(drop=True)
    checkpoint("dedup_zpid", df)

    # 2. Require price
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    checkpoint("require_price", df)

    # 3. Require living area (needed for all models)
    df = df.dropna(subset=["living_area_sqft"]).reset_index(drop=True)
    checkpoint("require_living_area", df)

    # 4. Price range filter
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)].reset_index(drop=True)
    checkpoint("price_range_filter", df)

    # 5. Home type: exclude lots / land / manufactured / mobile.
    #    Use exact isin() matching (after uppercasing) to avoid regex partial
    #    matches (e.g. "LOT" matching "PILOT").  Also catch any unknown compound
    #    variants that START with an excluded keyword (e.g. "LOT_LAND_OTHER").
    ht_upper = df["home_type"].astype(str).str.upper()
    exact_excl   = ht_upper.isin(EXCLUDE_HOME_TYPES)
    prefix_excl  = ht_upper.str.match(
        r"^(?:" + "|".join(EXCLUDE_HOME_TYPES) + r")\b", na=False
    )
    df = df[~(exact_excl | prefix_excl)].reset_index(drop=True)
    checkpoint("home_type_exclusion", df)

    # 6. Days on Zillow cap  (allow nulls through — Zillow sometimes omits)
    max_days = int(DAYS_ON_ZILLOW)
    df = df[df["days_on_zillow"].isna() | (df["days_on_zillow"] <= max_days)].reset_index(drop=True)
    checkpoint("days_on_zillow_cap", df)

    # 7. Sanity ranges on structural fields
    df = df[df["bedrooms"].isna()         | df["bedrooms"].between(0, 20)].reset_index(drop=True)
    df = df[df["bathrooms"].isna()        | df["bathrooms"].between(0, 20)].reset_index(drop=True)
    df = df[df["living_area_sqft"].isna() | df["living_area_sqft"].between(100, 25_000)].reset_index(drop=True)
    checkpoint("sanity_structural_fields", df)

    # 8. Add derived columns
    df = _add_derived_columns(df)

    attrition = pd.DataFrame(steps, columns=["step", "n_remaining", "n_dropped"])
    logger.info(f"Cleaning complete: {n0:,} → {len(df):,} rows kept")

    return df, attrition


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns used in modelling and charts."""

    # Log price (primary dependent variable)
    df["log_price"] = np.log(df["price"].clip(lower=1))

    # Property age
    cy = pd.Timestamp.now().year
    df["property_age"] = (cy - pd.to_numeric(df["year_built"], errors="coerce")).clip(lower=0, upper=200)

    # Lot size → square feet from lotSize (already standardized by the actor).
    df["lot_size_sqft"] = pd.to_numeric(df["lot_size_sqft_raw"], errors="coerce")
    df["log_lot_size"]  = np.where(df["lot_size_sqft"] > 0, np.log(df["lot_size_sqft"]), np.nan)

    # Log living area
    df["log_living_area"] = np.where(df["living_area_sqft"] > 0,
                                      np.log(df["living_area_sqft"]), np.nan)

    # HOA indicator
    hoa = pd.to_numeric(df["hoa_fee"], errors="coerce")
    df["has_hoa"]  = (hoa > 0).fillna(False)
    df["hoa_fee"]  = hoa  # coerce to numeric

    # Boolean casts for amenity columns
    for col in ["has_pool", "has_garage", "has_spa", "has_fireplace",
                "has_cooling", "has_heating", "has_basement",
                "new_construction", "has_attached_garage"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: False if pd.isna(x) else bool(x))

    # ZIP code zero-padded string
    df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)

    # Description length (proxy for listing quality)
    desc = df["property_description"]
    df["description_length"] = desc.apply(
        lambda x: len(str(x)) if (x is not None and not (isinstance(x, float) and pd.isna(x))
                                  and str(x) not in ("None", "", "nan")) else 0
    )
    df["has_description"] = df["description_length"] > 20

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a publication-ready descriptive statistics table."""
    num_cols = [
        "price", "bedrooms", "bathrooms", "living_area_sqft",
        "lot_size_sqft", "property_age", "stories",
        "garage_spaces", "price_per_sqft",
    ]
    cols = [c for c in num_cols if c in df.columns]
    tbl  = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    tbl  = tbl.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    tbl  = tbl[["count", "mean", "std", "min", "p25", "median", "p75", "max"]]
    return tbl.round(2)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return column-level missingness overview."""
    miss   = df.isnull().sum().rename("n_missing")
    pct    = (miss / len(df) * 100).rename("pct_missing").round(1)
    dtypes = df.dtypes.rename("dtype")
    return pd.concat([dtypes, miss, pct], axis=1).sort_values("pct_missing")


# ─────────────────────────────────────────────────────────────────────────────
# Variable catalog
# ─────────────────────────────────────────────────────────────────────────────

#: Thematic groupings for the variable catalog
_COL_CATEGORY: dict[str, str] = {
    "zpid": "Identity", "url": "Identity",
    "address_full": "Address", "street_address": "Address",
    "city": "Address", "state": "Address", "zip_code": "Address",
    "neighborhood": "Address", "latitude": "Address", "longitude": "Address",
    "home_status": "Listing", "days_on_zillow": "Listing",
    "listing_date": "Listing", "listing_type": "Listing",
    "featured": "Listing", "open_house": "Listing",
    "virtual_tour_url": "Listing",
    "price": "Price", "price_per_sqft": "Price",
    "zestimate": "Price", "rent_zestimate": "Price",
    "log_price": "Price (derived)",
    "bedrooms": "Structural", "bathrooms": "Structural",
    "living_area_sqft": "Structural", "log_living_area": "Structural (derived)",
    "lot_size_value": "Structural", "lot_size_unit": "Structural",
    "lot_size_sqft_raw": "Structural",
    "lot_size_sqft": "Structural (derived)",
    "year_built": "Structural", "property_age": "Structural (derived)",
    "home_type": "Structural", "stories": "Structural",
    "new_construction": "Structural",
    "has_cooling": "Amenities", "has_heating": "Amenities",
    "cooling": "Amenities", "heating": "Amenities",
    "has_attached_garage": "Amenities", "garage_spaces": "Amenities",
    "has_pool": "Amenities", "has_spa": "Amenities",
    "has_fireplace": "Amenities", "has_basement": "Amenities",
    "flooring": "Amenities", "appliances": "Amenities",
    "interior_features": "Amenities", "exterior_features": "Amenities",
    "construction_materials": "Amenities", "roof": "Amenities",
    "foundation_details": "Amenities", "water_source": "Amenities",
    "sewer": "Amenities", "utilities": "Amenities",
    "hoa_fee": "Financials", "has_hoa": "Financials",
    "tax_annual": "Financials", "tax_year": "Financials",
    "prev_sale_price": "History", "prev_sale_date": "History",
    "price_history_raw": "History",
    "listing_agent_name": "Agent / MLS",
    "listing_agent_phone": "Agent / MLS",
    "broker_name": "Agent / MLS",
    "listing_provider": "Agent / MLS", "mls_id": "Agent / MLS",
    "school_count": "Location", "school_names": "Location",
    "property_description": "Text / Media",
    "description_length": "Text / Media (derived)",
    "has_description": "Text / Media (derived)",
    "photo_count": "Text / Media", "photo_urls": "Text / Media",
}
_CAT_ORDER = [
    "Identity", "Address", "Listing", "Price", "Price (derived)",
    "Structural", "Structural (derived)", "Amenities", "Financials",
    "History", "Agent / MLS", "Location", "Text / Media",
    "Text / Media (derived)",
]


def build_variable_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column descriptive table for the raw or clean DataFrame.

    Columns returned
    ----------------
    category        : thematic group
    dtype           : pandas dtype
    n_non_null      : count of non-missing values
    pct_coverage    : % of rows with a value
    stats           : mean ± std (numeric), top-3 values (categorical/bool), or
                      median length (text)
    sample_values   : up to 3 representative non-null values as a string
    """
    rows = []
    n = len(df)

    for col in df.columns:
        series    = df[col]
        n_valid   = series.notna().sum()
        pct_cov   = n_valid / n * 100 if n > 0 else 0
        category  = _COL_CATEGORY.get(col, "Other")
        dtype_str = str(series.dtype)

        # ── stats string ──────────────────────────────────────────────────
        try:
            num = pd.to_numeric(series, errors="coerce")
            if num.notna().sum() > n * 0.3 and dtype_str not in ("object", "string"):
                mean = num.mean()
                std  = num.std()
                med  = num.median()
                stats_str = f"mean={mean:,.1f}  std={std:,.1f}  median={med:,.1f}"
            elif series.dtype == bool or str(series.dtype) == "boolean":
                rate = series.dropna().mean() * 100
                stats_str = f"True: {rate:.1f}%"
            else:
                top = (series.dropna().astype(str)
                       .value_counts().head(3))
                stats_str = "  |  ".join(
                    f"{v} ({c})" for v, c in top.items()
                )
        except Exception:
            stats_str = "—"

        # ── sample values ─────────────────────────────────────────────────
        sample = series.dropna().head(3).tolist()
        sample_str = "  |  ".join(str(v)[:40] for v in sample)

        rows.append({
            "variable":     col,
            "category":     category,
            "dtype":        dtype_str,
            "n_non_null":   int(n_valid),
            "pct_coverage": round(pct_cov, 1),
            "stats":        stats_str,
            "sample_values": sample_str,
        })

    catalog = pd.DataFrame(rows)

    # Sort by category (defined order) then variable name
    cat_rank = {c: i for i, c in enumerate(_CAT_ORDER)}
    catalog["_rank"] = catalog["category"].map(
        lambda c: cat_rank.get(c, len(_CAT_ORDER))
    )
    catalog = (catalog.sort_values(["_rank", "variable"])
               .drop(columns="_rank")
               .reset_index(drop=True))

    return catalog
