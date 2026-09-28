"""
agent/tools.py — Tool functions available to the ReAct Decision Agent.

Each tool is a pure Python function that the agent can call during its
reasoning loop to gather evidence before making a MERGE/SPLIT decision.
"""

import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher

from thefuzz import fuzz

# ---------------------------------------------------------------------------
# Date formats the normalizer understands
# ---------------------------------------------------------------------------
_DATE_FORMATS = [
    "%Y-%m-%d",       # 1985-03-08       (ISO — core_banking)
    "%d/%m/%Y",       # 08/03/1985       (credit_card)
    "%B %d %Y",       # March 8 1985     (mortgage, with day-no-zero)
    "%B %d, %Y",      # March 8, 1985
    "%d-%m-%Y",       # 08-03-1985
    "%m/%d/%Y",       # 03/08/1985  (US format)
    "%Y/%m/%d",
    "%b %d %Y",       # Mar 8 1985
    "%b %d, %Y",
]


def _strip_accents(s: str) -> str:
    """Normalize unicode characters to their ASCII base form."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _clean_name(name: str) -> str:
    """Lowercase, strip accents, remove titles/honorifics, collapse spaces."""
    if not name:
        return ""
    name = _strip_accents(name.lower().strip())
    # Remove common honorifics / particles that differ across systems
    for particle in [r"\bbin\b", r"\bbinti\b", r"\bbinte\b", r"\bbinti\b",
                     r"\bbin\b", r"\bmr\.?\b", r"\bms\.?\b", r"\bdr\.?\b",
                     r"\bmdm\.?\b", r"\bmohd\b", r"\bmohammad\b", r"\bmuhamad\b"]:
        name = re.sub(particle, "", name)
    # Remove initials like "R." "A." that expand differently across systems
    name = re.sub(r"\b[a-z]\.\s*", "", name)
    return re.sub(r"\s+", " ", name).strip()


def compare_names(name_a: str, name_b: str) -> dict:
    """
    Compare two name strings and return fuzzy similarity scores.

    Returns:
        {
            "token_sort_ratio":  int,   # order-independent token comparison
            "partial_ratio":     int,   # substring match (handles initials)
            "token_set_ratio":   int,   # best of token permutations
            "best_score":        int,   # maximum of all three
            "clean_a":           str,
            "clean_b":           str,
        }
    """
    if not name_a or not name_b:
        return {"token_sort_ratio": 0, "partial_ratio": 0,
                "token_set_ratio": 0, "best_score": 0,
                "clean_a": name_a or "", "clean_b": name_b or ""}

    ca, cb = _clean_name(name_a), _clean_name(name_b)
    tsr = fuzz.token_sort_ratio(ca, cb)
    pr  = fuzz.partial_ratio(ca, cb)
    tset = fuzz.token_set_ratio(ca, cb)
    best = max(tsr, pr, tset)
    return {
        "token_sort_ratio": tsr,
        "partial_ratio":    pr,
        "token_set_ratio":  tset,
        "best_score":       best,
        "clean_a":          ca,
        "clean_b":          cb,
    }


def normalize_dob(date_str) -> str | None:
    """
    Parse a date string in any known format and return YYYY-MM-DD, or None.

    Handles:
        "1985-03-08"      → "1985-03-08"
        "08/03/1985"      → "1985-03-08"
        "March 8 1985"    → "1985-03-08"
        "March 15 1978"   → "1978-03-15"  (mortgage uses day without zero-pad)
    """
    if not date_str or str(date_str).strip() in ("", "nan", "NaT", "None"):
        return None
    s = str(date_str).strip()
    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # Try each known format
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Last-ditch: try to extract YYYY-MM-DD pattern from free text
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def compare_email_handles(email_a: str, email_b: str) -> dict:
    """
    Compare the local parts (before @) of two email addresses.

    Returns:
        {
            "handle_a":    str,
            "handle_b":    str,
            "similarity":  float,   # 0.0–1.0
            "same_domain": bool,
        }
    """
    def _parse(e):
        if not e or "@" not in str(e):
            return "", ""
        parts = str(e).strip().lower().split("@", 1)
        return parts[0], parts[1]

    ha, da = _parse(email_a)
    hb, db = _parse(email_b)

    if not ha or not hb:
        return {"handle_a": ha, "handle_b": hb, "similarity": 0.0, "same_domain": da == db}

    sim = SequenceMatcher(None, ha, hb).ratio()
    return {
        "handle_a":    ha,
        "handle_b":    hb,
        "similarity":  round(sim, 3),
        "same_domain": da == db,
    }


def check_address_match(addr_a: str, addr_b: str) -> dict:
    """
    Compare two address strings after normalization.

    Returns:
        {
            "similarity": float,   # 0.0–1.0
            "clean_a":    str,
            "clean_b":    str,
        }
    """
    def _clean_addr(a):
        if not a:
            return ""
        a = _strip_accents(str(a).lower())
        # Normalize common abbreviations
        replacements = {
            r"\bjln\b":    "jalan",
            r"\bpj\b":     "petaling jaya",
            r"\bkl\b":     "kuala lumpur",
            r"\btaman\b":  "taman",
            r"\bss\b":     "ss",
        }
        for pat, repl in replacements.items():
            a = re.sub(pat, repl, a)
        return re.sub(r"\s+", " ", a).strip()

    ca, cb = _clean_addr(addr_a), _clean_addr(addr_b)
    if not ca or not cb:
        return {"similarity": 0.0, "clean_a": ca, "clean_b": cb}
    sim = fuzz.token_sort_ratio(ca, cb) / 100.0
    return {"similarity": round(sim, 3), "clean_a": ca, "clean_b": cb}
