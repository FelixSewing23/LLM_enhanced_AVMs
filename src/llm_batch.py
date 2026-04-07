"""OpenAI Batch API pipeline using the Responses endpoint + Structured Outputs.

Pipeline
────────
1. build_batch_jsonl()   — create JSONL input file (one row per listing)
2. upload_and_submit()   — upload file, create batch job, persist metadata
3. poll_until_done()     — poll status until completed / failed / expired
4. download_results()    — save output and error JSONL to data/interim/
5. parse_results()       — parse into a (features_df, failures_df) pair
6. merge_features()      — left-join LLM features back to listings DataFrame
7. run_llm_pipeline()    — convenience wrapper; skips completed stages

Each step caches its output so re-runs are free.

Notes on the Responses API in batch mode
─────────────────────────────────────────
• endpoint in JSONL body:  "url": "/v1/responses"
• structured output key:   "text": {"format": {"type":"json_schema", ...}}
• result text location in response body:
    output[?type=="message"].content[?type=="output_text"].text
• store: false  → stateless, no server-side context stored
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_RUN_TAG,
    DATA_INTERIM,
    LLM_RUN_MODE,
    LLM_IMMEDIATE_MAX_WORKERS,
)
from src.llm_schema import PropertyFeatures, SYSTEM_PROMPT, get_json_schema
from src.utils import get_logger, save_jsonl, load_jsonl

logger = get_logger(__name__)


def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set.  Add it to .env")
    return OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build batch JSONL
# ─────────────────────────────────────────────────────────────────────────────

def build_batch_jsonl(
    df: pd.DataFrame,
    description_col: str = "property_description",
    id_col: str = "zpid",
    output_path: Optional[Path] = None,
    model: str = LLM_MODEL,
    force_rebuild: bool = False,
) -> Path:
    """Create the batch input JSONL file.

    Each line is one Responses-API request, identified by a custom_id that
    maps back to the listing's zpid.  Rows with missing / trivially short
    descriptions are skipped and logged.
    """
    output_path = output_path or DATA_INTERIM / "batch_input.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_rebuild:
        n = sum(1 for _ in open(output_path))
        logger.info(f"Batch JSONL already exists ({n:,} requests): {output_path}")
        return output_path

    schema  = get_json_schema()
    records: list[dict] = []
    skipped: list[dict] = []

    for _, row in df.iterrows():
        desc       = row.get(description_col)
        listing_id = str(row.get(id_col, row.name))

        if not isinstance(desc, str) or len(desc.strip()) < 20:
            skipped.append({"listing_id": listing_id, "reason": "missing or too short"})
            continue

        request = {
            "custom_id": f"listing-{listing_id}",
            "method":    "POST",
            "url":       "/v1/responses",
            "body": {
                "model":        model,
                "instructions": SYSTEM_PROMPT,
                "input":        (
                    "Extract structured features from this property listing description:\n\n"
                    + desc.strip()
                ),
                "text": {
                    "format": {
                        "type":   "json_schema",
                        "name":   schema["name"],
                        "strict": schema["strict"],
                        "schema": schema["schema"],
                    }
                },
                "store": False,
            },
        }
        records.append(request)

    save_jsonl(records, output_path)

    if skipped:
        save_jsonl(skipped, DATA_INTERIM / "batch_skipped.jsonl")

    logger.info(
        f"Batch JSONL: {len(records):,} requests written, "
        f"{len(skipped):,} skipped → {output_path}"
    )
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Upload and submit
# ─────────────────────────────────────────────────────────────────────────────

def upload_and_submit(
    jsonl_path: Path,
    meta_path: Optional[Path] = None,
) -> str:
    """Upload the JSONL and create a batch job.  Returns batch_id.

    If a previous batch job metadata file exists, returns the stored batch_id
    without re-submitting (idempotent).
    """
    meta_path = meta_path or DATA_INTERIM / "batch_job_meta.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        batch_id = meta.get("id")
        logger.info(f"Found existing batch job: {batch_id}  (status: {meta.get('status')})")
        return batch_id

    client = _client()

    logger.info(f"Uploading {jsonl_path} to OpenAI Files API …")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info(f"  File uploaded: {file_obj.id}")

    logger.info("Creating batch job …")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"project": "houston_zillow_llm_avm"},
    )
    logger.info(f"  Batch created: {batch.id}  status={batch.status}")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(batch.model_dump(), f, indent=2, default=str)

    return batch.id


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Poll
# ─────────────────────────────────────────────────────────────────────────────

def poll_until_done(
    batch_id: str,
    poll_interval: int = 60,
    timeout_hours: float = 25.0,
) -> dict:
    """Block until the batch completes, fails, expires, or times out.

    Prints live progress.  Returns the final batch object as a dict.
    """
    client          = _client()
    deadline        = time.time() + timeout_hours * 3600
    terminal_states = {"completed", "failed", "expired", "cancelled"}

    while time.time() < deadline:
        batch  = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts

        logger.info(
            f"Batch {batch_id}: {status:12s} | "
            f"total={counts.total}  completed={counts.completed}  failed={counts.failed}"
        )

        if status in terminal_states:
            if status != "completed":
                logger.warning(f"Batch ended with status: {status}")
            return batch.model_dump()

        time.sleep(poll_interval)

    raise TimeoutError(f"Batch {batch_id} did not finish within {timeout_hours}h")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Download results
# ─────────────────────────────────────────────────────────────────────────────

def download_results(
    batch: dict,
    output_path: Optional[Path] = None,
    error_path: Optional[Path] = None,
) -> Path:
    """Download completed batch output (and errors) from the OpenAI Files API."""
    output_path = output_path or DATA_INTERIM / "batch_output.jsonl"
    error_path  = error_path  or DATA_INTERIM / "batch_errors.jsonl"

    client = _client()

    if batch.get("output_file_id"):
        content = client.files.content(batch["output_file_id"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content.read())
        logger.info(f"Output downloaded → {output_path}")
    else:
        logger.warning("No output_file_id in batch metadata")

    if batch.get("error_file_id"):
        content = client.files.content(batch["error_file_id"])
        error_path.write_bytes(content.read())
        logger.warning(f"Error file downloaded → {error_path}")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Parse results
# ─────────────────────────────────────────────────────────────────────────────

def parse_results(output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse batch output JSONL into a features DataFrame.

    Returns
    -------
    features_df : pd.DataFrame  indexed by listing_id (zpid)
    failures_df : pd.DataFrame  rows that could not be parsed
    """
    records  = load_jsonl(output_path)
    success: list[dict] = []
    failure: list[dict] = []

    for rec in records:
        custom_id  = rec.get("custom_id", "")
        listing_id = custom_id.removeprefix("listing-")
        response   = rec.get("response", {})
        status     = response.get("status_code", 0)

        if status != 200:
            failure.append({
                "listing_id": listing_id,
                "error":      str(rec.get("error", f"HTTP {status}")),
            })
            continue

        body = response.get("body", {})
        try:
            text = _extract_output_text(body)
            if text is None:
                raise ValueError("output_text not found in response body")

            parsed = PropertyFeatures.model_validate_json(text)
            row    = _flatten_features(listing_id, parsed)
            success.append(row)

        except Exception as exc:
            failure.append({"listing_id": listing_id, "error": str(exc)})

    features_df = (
        pd.DataFrame(success).set_index("listing_id")
        if success else pd.DataFrame()
    )
    failures_df = pd.DataFrame(failure) if failure else pd.DataFrame()

    logger.info(
        f"Parse: {len(success):,} succeeded, {len(failure):,} failed "
        f"({len(failure) / max(len(records), 1) * 100:.1f}% failure rate)"
    )
    return features_df, failures_df


def _extract_output_text(body: dict) -> Optional[str]:
    """Navigate Responses API body to find output_text content."""
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text")
    return None


def _flatten_features(listing_id: str, f: PropertyFeatures) -> dict:
    """Flatten a PropertyFeatures object to a flat dict row."""
    s = f.soft
    h = f.hard

    def b(val) -> Optional[int]:
        """bool → 0/1, None → None."""
        return int(val) if val is not None else None

    return {
        "listing_id":                    listing_id,
        # Soft scores
        "llm_luxury_score":              s.luxury_score,
        "llm_uniqueness_score":          s.uniqueness_score,
        "llm_renovation_quality_score":  s.renovation_quality_score,
        "llm_curb_appeal_score":         s.curb_appeal_score,
        "llm_spaciousness_score":        s.spaciousness_score,
        # Soft booleans
        "llm_is_unique_property":        b(s.is_unique_property),
        "llm_has_premium_finishes":      b(s.has_premium_finishes),
        "llm_is_recently_updated":       b(s.is_recently_updated),
        "llm_soft_evidence":             s.soft_evidence,
        # Hard flags
        "llm_foreclosure_flag":          b(h.foreclosure_flag),
        "llm_auction_flag":              b(h.auction_flag),
        "llm_as_is_flag":                b(h.as_is_flag),
        "llm_fixer_upper_flag":          b(h.fixer_upper_flag),
        "llm_needs_repair_flag":         b(h.needs_repair_flag),
        "llm_water_damage_flag":         b(h.water_damage_flag),
        "llm_fire_damage_flag":          b(h.fire_damage_flag),
        "llm_foundation_issue_flag":     b(h.foundation_issue_flag),
        "llm_roof_issue_flag":           b(h.roof_issue_flag),
        "llm_mold_flag":                 b(h.mold_flag),
        "llm_tenant_occupied_flag":      b(h.tenant_occupied_flag),
        "llm_cash_only_flag":            b(h.cash_only_flag),
        "llm_investor_special_flag":     b(h.investor_special_flag),
        "llm_hard_evidence":             h.hard_evidence,
        # Meta
        "llm_extraction_confidence":     f.extraction_confidence,
        "llm_description_quality":       f.description_quality,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_features(
    listings_df: pd.DataFrame,
    features_df: pd.DataFrame,
    id_col: str = "zpid",
) -> pd.DataFrame:
    """Left-join LLM features into the listings DataFrame."""
    if features_df.empty:
        logger.warning("features_df is empty — returning listings unchanged")
        return listings_df

    feat = features_df.reset_index().rename(columns={"listing_id": id_col})
    feat[id_col] = feat[id_col].astype(str)

    result = listings_df.copy()
    result[id_col] = result[id_col].astype(str)
    result = result.merge(feat, on=id_col, how="left")

    n_matched = result["llm_luxury_score"].notna().sum()
    logger.info(
        f"LLM features merged: {n_matched:,}/{len(result):,} listings matched "
        f"({n_matched / len(result) * 100:.1f}%)"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7a.  Immediate pipeline (synchronous + concurrent)
# ─────────────────────────────────────────────────────────────────────────────

def _call_single(
    client: OpenAI,
    listing_id: str,
    description: str,
    model: str,
    schema: dict,
) -> dict:
    """Make one synchronous Responses API call.  Returns a parsed result dict."""
    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=(
                "Extract structured features from this property listing description:\n\n"
                + description.strip()
            ),
            text={"format": {
                "type":   "json_schema",
                "name":   schema["name"],
                "strict": schema["strict"],
                "schema": schema["schema"],
            }},
            store=True,
            metadata={
                "run_tag":    LLM_RUN_TAG,
                "project":    "houston_zillow_avm",
                "listing_id": str(listing_id),
            },
        )
        # Extract output_text from Responses API output array
        text = None
        for item in response.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        text = c.text
                        break
            if text:
                break

        if text is None:
            raise ValueError("output_text not found in response")

        parsed = PropertyFeatures.model_validate_json(text)
        return {"ok": True, "listing_id": listing_id, "row": _flatten_features(listing_id, parsed)}

    except Exception as exc:
        logger.error(f"_call_single [{listing_id}]: {type(exc).__name__}: {exc}")
        return {"ok": False, "listing_id": listing_id, "error": str(exc)}


def run_immediate_pipeline(
    df: pd.DataFrame,
    description_col: str = "property_description",
    id_col: str = "zpid",
    model: str = LLM_MODEL,
    max_workers: int = LLM_IMMEDIATE_MAX_WORKERS,
    output_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run LLM extraction immediately using concurrent synchronous API calls.

    Faster than batch mode for small samples or when results are needed right
    away.  Uses a ThreadPoolExecutor with ``max_workers`` concurrent threads.

    Returns
    -------
    merged_df   : listings DataFrame enriched with LLM feature columns
    features_df : LLM features only (indexed by zpid)
    failures_df : rows that failed
    """
    output_path = output_path or DATA_INTERIM / "immediate_output.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if force_rebuild and output_path.exists():
        output_path.unlink()
        logger.info("Immediate: cache cleared (force_rebuild=True)")

    # Load already-processed IDs from any existing (possibly partial) file so
    # an aborted run or kernel restart automatically resumes from where it left off.
    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if obj.get("ok"):  # only skip successes; failures are retried
                    done_ids.add(str(obj.get("listing_id", "")))
            except json.JSONDecodeError:
                pass
        if done_ids:
            logger.info(f"Immediate: resuming — {len(done_ids)} already cached")

    # Build pending work items — skip rows without a usable description or already done
    tasks: list[tuple[str, str]] = []
    skipped = 0
    for _, row in df.iterrows():
        desc = row.get(description_col)
        lid  = str(row.get(id_col, row.name))
        if lid in done_ids:
            continue
        if not isinstance(desc, str) or len(desc.strip()) < 20:
            skipped += 1
            continue
        tasks.append((lid, desc))

    if not tasks:
        logger.info("Immediate: all listings already cached — loading from disk")
        all_rows = load_jsonl(output_path)
        success = [r["row"] for r in all_rows if r.get("ok")]
        failure = [{"listing_id": r["listing_id"], "error": r["error"]} for r in all_rows if not r.get("ok")]
        features_df = pd.DataFrame(success).set_index("listing_id") if success else pd.DataFrame()
        failures_df = pd.DataFrame(failure) if failure else pd.DataFrame()
        return merge_features(df, features_df), features_df, failures_df

    logger.info(
        f"Immediate mode: {len(tasks):,} pending, {len(done_ids)} cached, "
        f"{skipped} skipped, {max_workers} workers, model={model}"
    )

    client = _client()
    schema = get_json_schema()
    file_lock = threading.Lock()
    done = 0

    # Write each result to disk immediately so partial progress survives
    # a kernel restart or KeyboardInterrupt.
    with open(output_path, "a", encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_call_single, client, lid, desc, model, schema): lid
            for lid, desc in tasks
        }
        for fut in as_completed(futures):
            res = fut.result()
            with file_lock:
                fh.write(json.dumps(res, default=str) + "\n")
                fh.flush()
            done += 1
            if done % 50 == 0 or done == len(tasks):
                logger.info(f"  {done}/{len(tasks)} completed")

    # Parse full file (cached + newly written rows)
    all_rows = load_jsonl(output_path)
    success = [r["row"] for r in all_rows if r.get("ok")]
    failure = [{"listing_id": r["listing_id"], "error": r["error"]} for r in all_rows if not r.get("ok")]

    if failure:
        logger.warning(
            f"{len(failure)}/{len(all_rows)} calls failed. "
            f"First error: {failure[0]['error']}"
        )

    features_df = (
        pd.DataFrame(success).set_index("listing_id")
        if success else pd.DataFrame()
    )
    failures_df = pd.DataFrame(failure) if failure else pd.DataFrame()

    logger.info(
        f"Immediate: {len(success):,} succeeded, {len(failure):,} failed "
        f"({len(failure) / max(len(all_rows), 1) * 100:.1f}% failure rate)"
    )

    merged_df = merge_features(df, features_df)
    return merged_df, features_df, failures_df


# ─────────────────────────────────────────────────────────────────────────────
# 7b.  Full pipeline dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_pipeline(
    df: pd.DataFrame,
    force_rebuild: bool = False,
    mode: str = LLM_RUN_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the LLM feature-extraction pipeline end-to-end.

    Dispatches to the batch or immediate runner based on ``mode``
    (defaults to ``LLM_RUN_MODE`` from config).

    Parameters
    ----------
    df            : cleaned listings DataFrame
    force_rebuild : ignore all caches and re-run from scratch
    mode          : ``"batch"`` (OpenAI Batch API, 50% cost, ≤24h) or
                    ``"immediate"`` (concurrent sync calls, instant results)

    Returns
    -------
    merged_df   : listings DataFrame enriched with LLM feature columns
    features_df : LLM features only (indexed by zpid)
    failures_df : rows that failed parsing
    """
    if mode == "immediate":
        logger.info("LLM pipeline mode: immediate (concurrent synchronous calls)")
        return run_immediate_pipeline(df, force_rebuild=force_rebuild)

    # ── batch mode ────────────────────────────────────────────────────────────
    logger.info("LLM pipeline mode: batch (OpenAI Batch API)")

    jsonl_path  = DATA_INTERIM / "batch_input.jsonl"
    output_path = DATA_INTERIM / "batch_output.jsonl"
    meta_path   = DATA_INTERIM / "batch_job_meta.json"

    build_batch_jsonl(df, output_path=jsonl_path, force_rebuild=force_rebuild)

    batch_id = upload_and_submit(jsonl_path, meta_path=meta_path)

    if not output_path.exists():
        batch = poll_until_done(batch_id)
        download_results(batch, output_path=output_path)
    else:
        logger.info(f"Using cached batch output: {output_path}")

    features_df, failures_df = parse_results(output_path)
    merged_df = merge_features(df, features_df)

    return merged_df, features_df, failures_df
