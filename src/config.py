"""Central configuration for the Houston Zillow LLM AVM project.

All paths, API identifiers, collection parameters, and modelling constants
live here so every module imports from a single source of truth.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_INTERIM   = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_FIGS   = ROOT / "outputs" / "figures"
OUTPUTS_TABLES = ROOT / "outputs" / "tables"

# ── API keys ───────────────────────────────────────────────────────────────────
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

# ── Apify actor IDs ────────────────────────────────────────────────────────────
ACTOR_ZIP_SEARCH = "maxcopell/zillow-zip-search"
ACTOR_DETAIL     = "maxcopell/zillow-detail-scraper"
ACTOR_SEARCH     = "maxcopell/zillow-scraper"  # alternative / fallback

# ── Houston ZIP codes (all city ZIP codes for fanout) ─────────────────────────
HOUSTON_ZIP_CODES: list[str] = [
    "77002", "77003", "77004", "77005", "77006", "77007", "77008", "77009",
    "77010", "77011", "77012", "77013", "77014", "77015", "77016", "77017",
    "77018", "77019", "77020", "77021", "77022", "77023", "77024", "77025",
    "77026", "77027", "77028", "77029", "77030", "77031", "77033", "77034",
    "77035", "77036", "77037", "77038", "77039", "77040", "77041", "77042",
    "77043", "77044", "77045", "77046", "77047", "77048", "77049", "77050",
    "77051", "77053", "77054", "77055", "77056", "77057", "77058", "77059",
    "77060", "77061", "77062", "77063", "77064", "77065", "77066", "77067",
    "77068", "77069", "77070", "77071", "77072", "77073", "77074", "77075",
    "77076", "77077", "77078", "77079", "77080", "77081", "77082", "77083",
    "77084", "77085", "77086", "77087", "77088", "77089", "77090", "77091",
    "77092", "77093", "77094", "77095", "77096", "77098", "77099",
]

# ── Collection filters ─────────────────────────────────────────────────────────
PRICE_MIN      = 175_000
PRICE_MAX      = 750_000
DAYS_ON_ZILLOW = "90"        # "1","7","14","30","90" or "6months","12months"…
TARGET_N       = 1_500       # desired final sample size

# Home types to EXCLUDE at the cleaning stage (ZIP search has no type filter).
# Covers Zillow's known values and compound variants (e.g. LOT_LAND).
EXCLUDE_HOME_TYPES: set[str] = {
    "LOT", "LAND", "LOT_LAND",
    "MANUFACTURED", "MOBILE", "MOBILE_MANUFACTURED",
}

# ── LLM parameters ─────────────────────────────────────────────────────────────
LLM_MODEL       = "gpt-5-mini"
LLM_RUN_TAG     = "run-1"        # change between runs to filter traces in dashboard
AUDIT_SAMPLE_N  = 20
AUDIT_SEED      = 42

# LLM run mode: "batch" (OpenAI Batch API, 50% cost, up to 24h) or
#               "immediate" (concurrent synchronous calls, instant results)
LLM_RUN_MODE             = "immediate"   # "batch" | "immediate"
# gpt-5-mini limits: 500 RPM / 500k TPM → ~463 effective RPM ceiling (TPM-bound at ~1,080 tok/req)
# 18 workers @ ~2.4s avg latency ≈ 450 RPM ≈ 90% of limit — near-maximum throughput
LLM_IMMEDIATE_MAX_WORKERS = 50     # max concurrent threads in immediate mode

# ── Modelling parameters ───────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.20
CV_FOLDS    = 5
CV_REPEATS  = 5       # number of repeats for RepeatedKFold
