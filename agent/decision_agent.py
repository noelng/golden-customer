"""
agent/decision_agent.py — ReAct-style Agentic AI Decision Layer.

Resolves uncertain customer record pairs that the rules-based ETL engine
flagged (name similarity 60–84%, or missing DOB in one record).

Runs in two modes:
  - LIVE:  Uses OpenAI gpt-4o-mini via OPENAI_API_KEY env var
  - MOCK:  Falls back to a data-driven smart-mock that uses pre-computed
           evidence scores (name, email, DOB, phone) to produce a realistic
           MERGE or SPLIT decision — no API key required for the demo.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from .prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from .tools import (
    check_address_match,
    compare_email_handles,
    compare_names,
    compare_phone_numbers,
    normalize_dob,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool registry — maps tool name → callable
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "compare_names":          compare_names,
    "normalize_dob":          normalize_dob,
    "compare_email_handles":  compare_email_handles,
    "check_address_match":    check_address_match,
    "compare_phone_numbers":  compare_phone_numbers,
}

# OpenAI function-calling schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "compare_names",
            "description": "Compare two customer name strings using fuzzy matching. Strips honorifics before comparing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_a": {"type": "string", "description": "First name to compare"},
                    "name_b": {"type": "string", "description": "Second name to compare"},
                },
                "required": ["name_a", "name_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_dob",
            "description": "Parse a date string in any known format and return YYYY-MM-DD, or null.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {"type": "string", "description": "Date string to normalize"},
                },
                "required": ["date_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_email_handles",
            "description": "Compare the local parts (before @) of two email addresses for similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_a": {"type": "string", "description": "First email address"},
                    "email_b": {"type": "string", "description": "Second email address"},
                },
                "required": ["email_a", "email_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_address_match",
            "description": "Compare two address strings after normalization for similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "addr_a": {"type": "string", "description": "First address"},
                    "addr_b": {"type": "string", "description": "Second address"},
                },
                "required": ["addr_a", "addr_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_phone_numbers",
            "description": (
                "Compare two phone number strings after stripping formatting and normalising "
                "country-code prefixes (+60 / 0060 / 60 → local 0xxx). "
                "Returns exact match flag and digit-level similarity score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_a": {"type": "string", "description": "First phone number"},
                    "phone_b": {"type": "string", "description": "Second phone number"},
                },
                "required": ["phone_a", "phone_b"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Risk flag helpers
# ---------------------------------------------------------------------------
def _compute_risk_flags(
    name_score: int,
    dob_match: Optional[bool],
    email_sim: float,
    phone_match: bool,
    addr_sim: float,
    decision: str,
) -> list[str]:
    """
    Return a list of compliance risk flag strings for the audit log.

    These let downstream KYC/AML teams filter decisions that need human review.
    """
    flags = []
    if name_score < 70:
        flags.append("LOW_NAME_SCORE")
    if dob_match is False:
        flags.append("DOB_MISMATCH")
    if dob_match is None:
        flags.append("DOB_MISSING")
    if email_sim < 0.3 and email_sim > 0.0:
        flags.append("LOW_EMAIL_SIMILARITY")
    if phone_match is False and addr_sim < 0.5:
        flags.append("PHONE_AND_ADDRESS_MISMATCH")
    if decision == "MERGE" and name_score < 75:
        flags.append("MERGE_LOW_CONFIDENCE_NAME")
    if decision == "SPLIT" and name_score >= 80:
        flags.append("SPLIT_HIGH_NAME_SCORE")   # worth a human look
    return flags


# ---------------------------------------------------------------------------
# Smart mock — data-driven fallback (no API key required)
# ---------------------------------------------------------------------------
def _smart_mock_decide(
    rec_a: dict,
    rec_b: dict,
    name_scores: dict,
    email_scores: dict,
    phone_result: dict,
    dob_match: Optional[bool],
    addr_scores: dict,
) -> dict:
    """
    Evidence-driven mock decision used when no OpenAI API key is set.

    Scoring rubric (max 100 points):
      +40  name best_score / 100 * 40
      +20  exact phone match
      +20  email similarity * 20
      +10  DOB match (both present and equal)
      +10  address similarity * 10

    Thresholds:
      >= 60 → MERGE
       < 60 → SPLIT
    """
    name_score  = name_scores.get("best_score", 0)
    email_sim   = email_scores.get("similarity", 0.0)
    phone_exact = phone_result.get("match", False)
    addr_sim    = addr_scores.get("similarity", 0.0)

    score = 0.0
    score += (name_score / 100) * 40
    score += 20 if phone_exact else phone_result.get("similarity", 0.0) * 10
    score += email_sim * 20
    score += 10 if dob_match is True else 0
    score += addr_sim * 10

    decision   = "MERGE" if score >= 60 else "SPLIT"
    confidence = round(min(score / 100, 0.99), 2)

    # Build a human-readable rationale from the evidence
    evidence_parts = []
    if name_score >= 90:
        evidence_parts.append(f"strong name match ({name_score}%)")
    elif name_score >= 70:
        evidence_parts.append(f"moderate name match ({name_score}%)")
    else:
        evidence_parts.append(f"weak name match ({name_score}%)")

    if phone_exact:
        evidence_parts.append("identical phone number")
    elif phone_result.get("similarity", 0) > 0.8:
        evidence_parts.append("similar phone number")
    else:
        evidence_parts.append("different phone numbers")

    if email_sim >= 0.9:
        evidence_parts.append("near-identical email handles")
    elif email_sim >= 0.4:
        evidence_parts.append("similar email handles")
    elif email_sim > 0:
        evidence_parts.append("dissimilar email handles")

    if dob_match is True:
        evidence_parts.append("DOB confirmed match")
    elif dob_match is False:
        evidence_parts.append("DOB mismatch")
    else:
        evidence_parts.append("DOB unavailable for comparison")

    rationale = (
        f"{'MERGE' if decision == 'MERGE' else 'SPLIT'} based on evidence score {score:.0f}/100: "
        + "; ".join(evidence_parts) + "."
    )

    return {
        "decision":   decision,
        "confidence": confidence,
        "rationale":  rationale,
        "steps":      1,
    }


# ---------------------------------------------------------------------------
# Live ReAct agent (OpenAI function-calling loop)
# ---------------------------------------------------------------------------
def _live_decide(rec_a: dict, rec_b: dict, client) -> dict:
    """Run a ReAct loop via OpenAI tool calls and return the final decision."""
    pair_description = (
        f"Uncertain pair to resolve:\n"
        f"Record A ({rec_a.get('source', 'unknown')}): "
        f"name=\"{rec_a.get('name','')}\", "
        f"dob={rec_a.get('dob', 'None')}, "
        f"email=\"{rec_a.get('email','')}\", "
        f"phone=\"{rec_a.get('phone','')}\", "
        f"address=\"{rec_a.get('address','')}\"\n"
        f"Record B ({rec_b.get('source', 'unknown')}): "
        f"name=\"{rec_b.get('name','')}\", "
        f"dob={rec_b.get('dob', 'None')}, "
        f"email=\"{rec_b.get('email','')}\", "
        f"phone=\"{rec_b.get('phone','')}\", "
        f"address=\"{rec_b.get('address','')}\"\n"
        f"Resolve this pair."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": pair_description},
    ]

    steps = 0
    max_steps = 8

    while steps < max_steps:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = response.choices[0].message
        steps += 1

        # Agent called a tool
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                fn = TOOL_REGISTRY.get(fn_name)
                if fn:
                    try:
                        result = fn(**fn_args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}
                logger.debug("  Tool %s(%s) -> %s", fn_name, fn_args, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
            continue

        # Agent produced final text — parse the JSON decision
        content = msg.content or ""
        json_matches = re.findall(r'\{[^{}]*"decision"[^{}]*\}', content, re.DOTALL)
        if json_matches:
            try:
                parsed = json.loads(json_matches[-1])
                return {
                    "decision":   parsed.get("decision", "SPLIT"),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "rationale":  parsed.get("rationale", content.strip()),
                    "steps":      steps,
                }
            except json.JSONDecodeError:
                pass
        # Fallback: couldn't parse structured output
        return {
            "decision":   "SPLIT",
            "confidence": 0.5,
            "rationale":  f"Agent response unparseable — flagged for manual review. Raw: {content[:200]}",
            "steps":      steps,
        }

    return {
        "decision":   "SPLIT",
        "confidence": 0.4,
        "rationale":  "Agent reached max reasoning steps without conclusion — flagged for manual review.",
        "steps":      steps,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
class DecisionAgent:
    """
    ReAct-style agent that resolves uncertain customer record pairs.

    Usage:
        agent = DecisionAgent()
        result = agent.decide(rec_a, rec_b)
        # result: {"decision": "MERGE"|"SPLIT", "confidence": float,
        #          "rationale": str, "risk_flags": list[str], "log_entry": dict}
    """

    def __init__(self, log_path: str = "agent_decisions.jsonl"):
        self.log_path = log_path
        self._client = None
        self._mode = "mock"

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                self._mode = "live"
                logger.info("DecisionAgent: running in LIVE mode (gpt-4o-mini)")
            except ImportError:
                logger.warning("openai package not installed — falling back to mock mode")
        else:
            logger.info("DecisionAgent: OPENAI_API_KEY not set — running in MOCK mode")

    @property
    def mode(self) -> str:
        return self._mode

    def decide(
        self,
        rec_a: dict,
        rec_b: dict,
        golden_id: Optional[str] = None,
    ) -> dict:
        """
        Resolve one uncertain pair.

        Args:
            rec_a:     Dict with keys: source, id, name, dob, email, phone, address
            rec_b:     Dict with keys: source, id, name, dob, email, phone, address
            golden_id: If MERGE, the golden_id this will be assigned to (optional)

        Returns:
            {
                "decision":    "MERGE" | "SPLIT",
                "confidence":  float,
                "rationale":   str,
                "risk_flags":  list[str],   # compliance flags for human review queue
                "log_entry":   dict,        # persisted to agent_decisions.jsonl
            }
        """
        # Pre-compute all evidence scores (used by both mock and live modes for audit log)
        name_scores  = compare_names(rec_a.get("name", ""), rec_b.get("name", ""))
        email_scores = compare_email_handles(rec_a.get("email", ""), rec_b.get("email", ""))
        addr_scores  = check_address_match(rec_a.get("address", ""), rec_b.get("address", ""))
        phone_result = compare_phone_numbers(rec_a.get("phone", ""), rec_b.get("phone", ""))
        dob_a        = normalize_dob(rec_a.get("dob"))
        dob_b        = normalize_dob(rec_b.get("dob"))
        dob_match    = (dob_a == dob_b) if (dob_a and dob_b) else None

        # Get AI decision
        if self._mode == "live":
            result = _live_decide(rec_a, rec_b, self._client)
        else:
            result = _smart_mock_decide(
                rec_a, rec_b,
                name_scores, email_scores, phone_result, dob_match, addr_scores,
            )

        decision   = result["decision"]
        confidence = result["confidence"]
        rationale  = result["rationale"]
        steps      = result.get("steps", 1)

        # Compute compliance risk flags
        risk_flags = _compute_risk_flags(
            name_score  = name_scores.get("best_score", 0),
            dob_match   = dob_match,
            email_sim   = email_scores.get("similarity", 0.0),
            phone_match = phone_result.get("match", False),
            addr_sim    = addr_scores.get("similarity", 0.0),
            decision    = decision,
        )

        # Build audit log entry
        log_entry = {
            "decision_id":    str(uuid.uuid4()),
            "golden_id":      golden_id,
            "record_a_id":    rec_a.get("id", ""),
            "record_b_id":    rec_b.get("id", ""),
            "record_a_name":  rec_a.get("name", ""),
            "record_b_name":  rec_b.get("name", ""),
            "record_a_source": rec_a.get("source", ""),
            "record_b_source": rec_b.get("source", ""),
            "decision":       decision,
            "confidence":     confidence,
            "rationale":      rationale,
            "name_score":     name_scores.get("best_score", 0),
            "dob_match":      (1 if dob_match else 0) if dob_match is not None else None,
            "email_score":    email_scores.get("similarity", 0.0),
            "phone_match":    phone_result.get("match", False),
            "phone_similarity": phone_result.get("similarity", 0.0),
            "address_score":  addr_scores.get("similarity", 0.0),
            "risk_flags":     risk_flags,
            "agent_steps":    steps,
            "agent_mode":     self._mode,
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }

        # Append to JSONL log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        logger.info(
            "Agent [%s] %s <-> %s -> %s (confidence=%.2f, flags=%s)",
            self._mode, rec_a.get("name"), rec_b.get("name"),
            decision, confidence, risk_flags or "none",
        )

        return {
            "decision":   decision,
            "confidence": confidence,
            "rationale":  rationale,
            "risk_flags": risk_flags,
            "log_entry":  log_entry,
        }
