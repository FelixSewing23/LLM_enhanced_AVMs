"""Data collection via Apify Zillow actors.

Chosen pipeline
───────────────
Step 1  zillow-zip-search (maxcopell/zillow-zip-search)
        Fan out across all Houston ZIP codes with structured filter parameters
        (price range, for-sale, ≤90 days on Zillow).  Returns summary rows
        and — crucially — the Apify dataset ID for direct use in Step 2.

Step 2  zillow-detail-scraper (maxcopell/zillow-detail-scraper)
        Accepts searchResultsDatasetId from Step 1, avoiding manual URL
        extraction.  Returns 100+ fields per listing including
        property_description needed for the LLM pipeline.

Why this path
─────────────
• ZIP-search actor uses structured params → no URL construction, fully
  config-driven, unaffected by Zillow frontend changes.
• searchResultsDatasetId hand-off is the cleanest possible pipe between
  the two actors (no intermediate URL list to manage).
• Self-scraping Zillow (unofficial API / Firecrawl) was evaluated and
  rejected: Cloudflare protection makes it brittle; ~$1 cost saving for
  500 listings does not justify the maintenance burden.
• Fallback: if ZIP search returns < TARGET_N results, the search scraper
  (maxcopell/zillow-scraper) can be called with a Houston search URL.

All raw API responses are cached to data/raw/.  Re-runs load from cache
so Apify credits are never spent twice.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from apify_client import ApifyClient

from src.config import (
    APIFY_API_TOKEN,
    ACTOR_ZIP_SEARCH,
    ACTOR_DETAIL,
    ACTOR_SEARCH,
    HOUSTON_ZIP_CODES,
    PRICE_MIN,
    PRICE_MAX,
    DAYS_ON_ZILLOW,
    TARGET_N,
    EXCLUDE_HOME_TYPES,
    DATA_RAW,
)
from src.utils import get_logger, save_json, load_json

logger = get_logger(__name__)


# ── Apify client ───────────────────────────────────────────────────────────────
def _client() -> ApifyClient:
    if not APIFY_API_TOKEN:
        raise EnvironmentError("APIFY_API_TOKEN is not set.  Add it to .env")
    return ApifyClient(APIFY_API_TOKEN)


# ── Step 1: ZIP search ─────────────────────────────────────────────────────────
def run_zip_search(
    zip_codes: list[str] = HOUSTON_ZIP_CODES,
    price_min: int = PRICE_MIN,
    price_max: int = PRICE_MAX,
    days_on_zillow: str = DAYS_ON_ZILLOW,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> tuple[list[dict], str]:
    """Run the ZIP search actor across all Houston ZIP codes.

    Returns
    -------
    records : list[dict]   — raw summary rows from Apify dataset
    dataset_id : str       — Apify dataset ID (passed to detail scraper)
    """
    cache_path     = cache_path or DATA_RAW / "zip_search_raw.json"
    meta_path      = DATA_RAW / "zip_search_meta.json"

    if cache_path.exists() and meta_path.exists() and not force_refresh:
        logger.info(f"ZIP search: loading from cache ({cache_path})")
        records    = load_json(cache_path)
        meta       = load_json(meta_path)
        dataset_id = meta["defaultDatasetId"]
        logger.info(f"  {len(records):,} records, dataset_id={dataset_id}")
        return records, dataset_id

    client = _client()

    run_input = {
        "zipCodes":      zip_codes,
        "priceMin":      price_min,
        "priceMax":      price_max,
        "daysOnZillow":  days_on_zillow,
        "forSaleByAgent": True,
        "forSaleByOwner": True,
        "forRent":        False,
        "sold":           False,
    }

    logger.info(f"ZIP search: launching actor across {len(zip_codes)} ZIP codes")
    logger.info(f"  Filters → price ${price_min:,}–${price_max:,}, ≤{days_on_zillow} days")

    run = client.actor(ACTOR_ZIP_SEARCH).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]

    logger.info(f"ZIP search finished. Dataset ID: {dataset_id}")

    records = list(client.dataset(dataset_id).iterate_items())
    logger.info(f"  Retrieved {len(records):,} summary records")

    # Cache raw results and run metadata
    save_json(records,          cache_path)
    save_json(dict(run),        meta_path)
    logger.info(f"  Cached → {cache_path}")

    return records, dataset_id


# ── Pre-filter helper ──────────────────────────────────────────────────────────
def _filter_and_extract_urls(records: list[dict]) -> tuple[list[str], int]:
    """Filter ZIP-search summary records before hitting the detail scraper.

    The ZIP-search actor has no home-type parameter, so we apply the exclusion
    here on the summary data (which includes hdpData.homeInfo.homeType).
    This avoids spending $3/1k detail-scraper credits on lots/land/manufactured
    listings we would discard at the cleaning stage anyway.

    Returns
    -------
    urls     : deduplicated list of detail URLs for accepted listings
    n_dropped: number of records removed by the home-type pre-filter
    """
    seen_zpids: set = set()
    urls:  list[str] = []
    n_dropped = 0

    for r in records:
        hd        = r.get("hdpData") or {}
        hi        = hd.get("homeInfo") or {}
        home_type = str(hi.get("homeType") or r.get("homeType") or "").upper()

        # Drop excluded home types (same logic as cleaning stage)
        if home_type and (
            home_type in EXCLUDE_HOME_TYPES
            or any(home_type.startswith(ex) for ex in EXCLUDE_HOME_TYPES)
        ):
            n_dropped += 1
            continue

        # Deduplicate by zpid
        zpid = r.get("zpid") or hi.get("zpid")
        if zpid and zpid in seen_zpids:
            continue
        if zpid:
            seen_zpids.add(zpid)

        url = r.get("detailUrl") or r.get("url") or ""
        if not url:
            continue
        if not url.startswith("http"):
            url = "https://www.zillow.com" + url
        urls.append(url)

    return urls, n_dropped


# ── Step 2: Detail scraper ─────────────────────────────────────────────────────
def run_detail_scraper(
    start_urls: list[str],
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Run the detail scraper on a pre-filtered list of listing URLs.

    We pass startUrls (not searchResultsDatasetId) so we can feed only
    home-type-accepted URLs and avoid paying for excluded property types.

    Returns a list of full property records (100+ fields each).
    """
    cache_path = cache_path or DATA_RAW / "detail_raw.json"

    if cache_path.exists() and not force_refresh:
        logger.info(f"Detail scraper: loading from cache ({cache_path})")
        records = load_json(cache_path)
        logger.info(f"  {len(records):,} detail records")
        return records

    client = _client()

    run_input = {
        "startUrls":      [{"url": u} for u in start_urls],
        "propertyStatus": "FOR_SALE",
    }

    logger.info(f"Detail scraper: launching on {len(start_urls):,} pre-filtered URLs")
    run = client.actor(ACTOR_DETAIL).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]

    # Save meta (contains dataset_id) BEFORE pulling items.
    # If iterate_items() fails, you can re-fetch via:
    #   list(client.dataset(dataset_id).iterate_items())
    save_json(dict(run), DATA_RAW / "detail_meta.json")
    logger.info(f"  Apify dataset_id saved → detail_meta.json (safe to re-fetch if needed)")

    records = list(client.dataset(dataset_id).iterate_items())
    logger.info(f"  Retrieved {len(records):,} detail records")

    save_json(records, cache_path)
    logger.info(f"  Cached → {cache_path}")

    return records


# ── Fallback: search scraper with a Houston URL ────────────────────────────────
def run_search_scraper_fallback(
    houston_search_url: str,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> tuple[list[dict], str]:
    """Fallback path using zillow-scraper with an explicit search URL.

    Use when the ZIP search returns too few results.  Generate the URL by
    going to zillow.com, applying your filters, and copying the full URL
    (including the ?searchQueryState=... query string).

    Returns (records, dataset_id) matching the same signature as
    run_zip_search() so the caller can feed into run_detail_scraper().
    """
    cache_path = cache_path or DATA_RAW / "search_scraper_raw.json"
    meta_path  = DATA_RAW / "search_scraper_meta.json"

    if cache_path.exists() and meta_path.exists() and not force_refresh:
        logger.info(f"Search scraper: loading from cache ({cache_path})")
        records    = load_json(cache_path)
        meta       = load_json(meta_path)
        dataset_id = meta["defaultDatasetId"]
        return records, dataset_id

    client = _client()

    run_input = {
        "searchUrls":       [{"url": houston_search_url}],
        "extractionMethod": "PAGINATION_WITH_ZOOM_IN",
    }

    logger.info("Search scraper fallback: launching")
    run = client.actor(ACTOR_SEARCH).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]

    records = list(client.dataset(dataset_id).iterate_items())
    logger.info(f"  Retrieved {len(records):,} records")

    save_json(records,   cache_path)
    save_json(dict(run), meta_path)

    return records, dataset_id


# ── Orchestrator ───────────────────────────────────────────────────────────────
def collect_listings(
    force_refresh: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run the full two-step pipeline and return both raw result sets.

    Returns
    -------
    zip_records    : list[dict]   ZIP search summary rows
    detail_records : list[dict]   Full property records (100+ fields)
    """
    zip_records, _ = run_zip_search(force_refresh=force_refresh)

    # Pre-filter: drop excluded home types before hitting the detail scraper
    detail_urls, n_dropped = _filter_and_extract_urls(zip_records)
    logger.info(
        f"Pre-filter: {len(detail_urls):,} URLs kept, "
        f"{n_dropped:,} dropped (excluded home types)"
    )

    # Cap to TARGET_N + 20 % buffer (accounts for cleaning attrition).
    # Without this every Houston ZIP listing in the price range gets sent
    # to the detail scraper — potentially thousands of unnecessary credits.
    cap = int(TARGET_N * 1.2)
    if len(detail_urls) > cap:
        logger.info(f"Capping detail URLs: {len(detail_urls):,} → {cap:,} ({TARGET_N} target + 20% buffer)")
        detail_urls = detail_urls[:cap]

    if len(detail_urls) < TARGET_N:
        logger.warning(
            f"Only {len(detail_urls):,} URLs after pre-filter "
            f"(target {TARGET_N}).  Consider the search-scraper fallback."
        )

    # Persist the URL list so a failed detail-scraper run can be retried
    # without re-running the (slower) ZIP search.
    save_json(detail_urls, DATA_RAW / "detail_urls.json")
    logger.info(f"  Saved {len(detail_urls):,} detail URLs → detail_urls.json")

    detail_records = run_detail_scraper(
        start_urls=detail_urls,
        force_refresh=force_refresh,
    )

    logger.info(
        f"Collection done: {len(zip_records):,} summary rows, "
        f"{len(detail_records):,} detail records"
    )
    return zip_records, detail_records


# ── Normalisers ────────────────────────────────────────────────────────────────
def normalize_zip_records(records: list[dict]) -> pd.DataFrame:
    """Flatten ZIP search summary rows to a DataFrame (for reference / QA)."""
    rows = []
    for r in records:
        hd = r.get("hdpData") or {}
        hi = hd.get("homeInfo") or {}
        rows.append({
            "zpid":             r.get("zpid") or hi.get("zpid"),
            "detail_url":       r.get("detailUrl"),
            "price":            r.get("unformattedPrice") or hi.get("price"),
            "beds":             r.get("beds")  or hi.get("bedrooms"),
            "baths":            r.get("baths") or hi.get("bathrooms"),
            "area":             r.get("area")  or hi.get("livingArea"),
            "address":          r.get("address"),
            "zip_code":         hi.get("zipcode"),
            "latitude":         hi.get("latitude"),
            "longitude":        hi.get("longitude"),
            "home_type":        hi.get("homeType"),
            "home_status":      hi.get("homeStatus"),
            "days_on_zillow":   hi.get("daysOnZillow"),
            "zestimate":        hi.get("zestimate"),
            "rent_zestimate":   hi.get("rentZestimate"),
        })
    return pd.DataFrame(rows)
