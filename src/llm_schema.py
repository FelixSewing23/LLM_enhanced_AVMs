"""Pydantic schema for LLM property feature extraction.

Used with the OpenAI Responses API + Batch API + Structured Outputs.

Design decisions
────────────────
• All fields have NO Python default value so they appear in JSON Schema
  `required` — OpenAI strict mode mandates every property be required.
• Nullable via Optional[...] — allows the model to output null when the
  description contains no supporting evidence (preventing hallucination).
• Scores use plain Optional[int] with a description bounding 1-5, avoiding
  IntEnum which can confuse strict schema validation in some edge cases.
• Nested SoftFeatures / HardFeatures → PropertyFeatures so the schema is
  well-organized and the JSON output is self-documenting.
• Evidence snippets capped at 100 chars to keep token cost low.
"""

from __future__ import annotations

import copy
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Schema models
# ─────────────────────────────────────────────────────────────────────────────

class SoftFeatures(BaseModel):
    """Subjective quality signals inferred from the property description."""

    luxury_score: Optional[int] = Field(
        description=(
            "Overall luxury / premium positioning of the property. "
            "1 = basic/entry-level, 5 = ultra-luxury. "
            "Null if not inferable from the text."
        )
    )
    uniqueness_score: Optional[int] = Field(
        description=(
            "How architecturally or aesthetically distinctive the property is. "
            "1 = generic tract home, 5 = one-of-a-kind custom design. "
            "Null if not inferable."
        )
    )
    renovation_quality_score: Optional[int] = Field(
        description=(
            "Quality and recency of renovations or upgrades mentioned. "
            "1 = dated/poor condition, 5 = extensive high-end recent remodel. "
            "Null if renovations are not mentioned."
        )
    )
    curb_appeal_score: Optional[int] = Field(
        description=(
            "Exterior presentation and street presence. "
            "1 = poor, 5 = exceptional. Null if not inferable."
        )
    )
    spaciousness_score: Optional[int] = Field(
        description=(
            "Sense of space, openness, or large rooms mentioned. "
            "1 = described as cramped/small, 5 = very spacious/open layout. "
            "Null if not mentioned."
        )
    )
    is_unique_property: Optional[bool] = Field(
        description=(
            "True if the property is explicitly described as architecturally "
            "unique, custom-built, one-of-a-kind, or unlike typical homes."
        )
    )
    has_premium_finishes: Optional[bool] = Field(
        description=(
            "True if premium, high-end, or luxury finishes are explicitly "
            "mentioned (e.g. marble, quartz, hardwood, custom cabinetry)."
        )
    )
    is_recently_updated: Optional[bool] = Field(
        description=(
            "True if recent updates, remodel, renovation, or new construction "
            "elements are mentioned."
        )
    )
    soft_evidence: Optional[str] = Field(
        description=(
            "A direct quote (≤100 chars) from the description that best supports "
            "the scores above. Null if no strong evidence."
        )
    )


class HardFeatures(BaseModel):
    """Verifiable or risk-related flags extracted from the description."""

    foreclosure_flag:      Optional[bool] = Field(description="True if foreclosure is explicitly mentioned.")
    auction_flag:          Optional[bool] = Field(description="True if auction sale is explicitly mentioned.")
    as_is_flag:            Optional[bool] = Field(description="True if sold 'as-is' is explicitly stated.")
    fixer_upper_flag:      Optional[bool] = Field(description="True if described as a fixer-upper or needing significant work.")
    needs_repair_flag:     Optional[bool] = Field(description="True if specific repair needs are mentioned.")
    water_damage_flag:     Optional[bool] = Field(description="True if water damage, flooding, or water intrusion is mentioned.")
    fire_damage_flag:      Optional[bool] = Field(description="True if fire or smoke damage is mentioned.")
    foundation_issue_flag: Optional[bool] = Field(description="True if foundation issues, cracks, or settling are mentioned.")
    roof_issue_flag:       Optional[bool] = Field(description="True if roof problems, age, or needed replacement is mentioned.")
    mold_flag:             Optional[bool] = Field(description="True if mold or mildew is mentioned.")
    tenant_occupied_flag:  Optional[bool] = Field(description="True if the property is currently tenant-occupied.")
    cash_only_flag:        Optional[bool] = Field(description="True if a cash-only purchase is required or strongly implied.")
    investor_special_flag: Optional[bool] = Field(description="True if the listing is explicitly marketed as an investor opportunity.")
    hard_evidence:         Optional[str]  = Field(
        description=(
            "A direct quote (≤100 chars) from the description supporting any "
            "True flag above. Null if all flags are False/null."
        )
    )


class PropertyFeatures(BaseModel):
    """Complete structured extraction from one Zillow property description."""

    soft: SoftFeatures
    hard: HardFeatures
    extraction_confidence: Optional[int] = Field(
        description=(
            "Overall confidence that the description had enough information for "
            "a reliable extraction. 1 = near-empty / unusable, 5 = very detailed. "
            "Null only if the field itself cannot be assessed."
        )
    )
    description_quality: Optional[int] = Field(
        description=(
            "Quality and informativeness of the property description itself. "
            "1 = near-empty/boilerplate, 5 = comprehensive and specific. "
            "Null only if description is absent."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional real estate analyst extracting structured features from
property listing descriptions.

Rules:
- Return null for ANY field not explicitly stated or not reasonably inferable
  from the provided text.  Do NOT guess hard facts.
- For soft scores (1-5): use professional judgment but lean toward null when
  the evidence is weak or ambiguous.
- Evidence snippets must be direct quotes from the text, ≤100 characters.
- If the description is empty, very short (<20 chars), or pure boilerplate
  (e.g. only the address), return null for all fields and set
  description_quality to 1.
- Output ONLY schema-compliant JSON.  No commentary.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Schema preparation for OpenAI strict mode
# ─────────────────────────────────────────────────────────────────────────────

def _make_strict(schema: dict) -> None:
    """Recursively add additionalProperties:false and complete required lists.

    OpenAI strict mode requires:
    • additionalProperties: false on every object
    • all properties listed in required
    """
    if schema.get("type") == "object" or "properties" in schema:
        schema.setdefault("additionalProperties", False)
        props = schema.get("properties", {})
        if props:
            schema["required"] = list(props.keys())
        for sub in props.values():
            _make_strict(sub)

    for sub in schema.get("allOf", []):
        _make_strict(sub)
    for sub in schema.get("anyOf", []):
        _make_strict(sub)
    for sub in schema.get("oneOf", []):
        _make_strict(sub)

    for sub in (schema.get("$defs") or {}).values():
        _make_strict(sub)

    if "items" in schema:
        _make_strict(schema["items"])


def get_json_schema() -> dict:
    """Return an OpenAI-compatible strict JSON schema dict for PropertyFeatures."""
    raw = PropertyFeatures.model_json_schema()
    schema = copy.deepcopy(raw)
    _make_strict(schema)
    return {
        "name":   "PropertyFeatures",
        "strict": True,
        "schema": schema,
    }
