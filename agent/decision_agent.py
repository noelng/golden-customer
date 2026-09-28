"""
agent/decision_agent.py — ReAct-style Agentic AI Decision Layer.

Resolves uncertain customer record pairs that the rules-based ETL engine
flagged (name similarity 60–84%, or missing DOB in one record).

Runs in two modes:
  - LIVE:  Uses OpenAI gpt-4o-mini via OPENAI_API_KEY env var
  - MOCK:  Falls back to deterministic pre-scripted decisions when no API key
           is set — ensures the demo works without any API dependency.
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
]


# ---------------------------------------------------------------------------
# Mock decisions — deterministic fallback for demo without API key
# ---------------------------------------------------------------------------
MOCK_DECISIONS = {
    # Key pattern: frozenset of cleaned names
    frozenset(["jon smith", "john smith"]): {
        "decision": "MERGE", "confidence": 0.97,
        "rationale": "Identical email and phone; 'Jon' is a well-known nickname variant of 'John' — all corroborating fields match.",
        "steps": 3,
    },
    frozenset(["aisha binte rahman", "aisha rahman"]): {
        "decision": "MERGE", "confidence": 0.88,
        "rationale": "Clean names match exactly after removing 'binte' honorific; email handles are complementary abbreviations; identical phone confirms same person.",
        "steps": 4,
    },
    frozenset(["aisha rahman", "aisha rahman"]): {
        "decision": "MERGE", "confidence": 0.88,
        "rationale": "Names match exactly; email handles are complementary abbreviations of the same name; identical phone and address confirm same person.",
        "steps": 4,
    },
    frozenset(["jonathan smith", "jon smith"]): {
        "decision": "MERGE", "confidence": 0.93,
        "rationale": "'Jonathan' and 'Jon' are the same name (full vs. short form); email and phone are identical.",
        "steps": 3,
    },
    frozenset(["jonathan smith", "john smith"]): {
        "decision": "MERGE", "confidence": 0.91,
        "rationale": "'Jonathan' and 'John' share the same root name; same email domain and phone support this merge.",
        "steps": 3,
    },
    frozenset(["maria garcia", "maría garcía"]): {
        "decision": "MERGE", "confidence": 0.99,
        "rationale": "Records differ only by Unicode accent marks on the same Latin characters; identical email and phone confirm same person.",
        "steps": 2,
    },
    frozenset(["james r. thornton", "james thornton"]): {
        "decision": "MERGE", "confidence": 0.96,
        "rationale": "Middle initial 'R.' in mortgage record matches core banking; identical email, phone, and address confirm same person.",
        "steps": 3,
    },
}

_DEFAULT_MOCK = {
    "decision": "MERGE", "confidence": 0.78,
    "rationale": "Insufficient distinguishing evidence to confirm separate identities; moderate name similarity with matching contact details suggests same person — flagged for human review.",
    "steps": 3,
}


def _mock_decide(rec_a: dict, rec_b: dict) -> dict:
    """Return a deterministic mock decision for demo mode."""
    import unicodedata

    def clean(n):
        if not n:
            return ""
        n = "".join(
            c for c in unicodedata.normalize("NFD", n.lower())
            if unicodedata.category(c) != "Mn"
        )
        import re as _re
        for p in [r"\bbin\b", r"\bbinti\b", r"\bbinte\b", r"\bmr\.?\b", r"\bms\.?\b",
                  r"\bdr\.?\b", r"\bmdm\.?\b", r"\bmohd\b"]:
            n = _re.sub(p, "", n)
        n = _re.sub(r"\b[a-z]\.\s*", "", n)
        return _re.sub(r"\s+", " ", n).strip()

    key = frozenset([clean(rec_a.get("name", "")), clean(rec_b.get("name", ""))])
    result = MOCK_DECISIONS.get(key, _DEFAULT_MOCK).copy()
    return result


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
                logger.debug("  Tool %s(%s) → %s", fn_name, fn_args, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
            continue

        # Agent produced final text — parse the JSON decision
        content = msg.content or ""
        # Extract the last JSON object from the response
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
        #          "rationale": str, "log_entry": dict}
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
                "decision":  "MERGE" | "SPLIT",
                "confidence": float,
                "rationale":  str,
                "log_entry":  dict,   # persisted to agent_decisions.jsonl
            }
        """
        # Compute evidence scores regardless of mode (for the audit log)
        name_scores  = compare_names(rec_a.get("name", ""), rec_b.get("name", ""))
        email_scores = compare_email_handles(rec_a.get("email", ""), rec_b.get("email", ""))
        addr_scores  = check_address_match(rec_a.get("address", ""), rec_b.get("address", ""))
        dob_a        = normalize_dob(rec_a.get("dob"))
        dob_b        = normalize_dob(rec_b.get("dob"))
        dob_match    = (dob_a == dob_b) if (dob_a and dob_b) else None

        # Get AI decision
        if self._mode == "live":
            result = _live_decide(rec_a, rec_b, self._client)
        else:
            result = _mock_decide(rec_a, rec_b)

        decision   = result["decision"]
        confidence = result["confidence"]
        rationale  = result["rationale"]
        steps      = result.get("steps", 1)

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
            "address_score":  addr_scores.get("similarity", 0.0),
            "agent_steps":    steps,
            "agent_mode":     self._mode,
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }

        # Append to JSONL log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        logger.info(
            "Agent [%s] %s ↔ %s → %s (confidence=%.2f)",
            self._mode, rec_a.get("name"), rec_b.get("name"), decision, confidence
        )

        return {
            "decision":   decision,
            "confidence": confidence,
            "rationale":  rationale,
            "log_entry":  log_entry,
        }
