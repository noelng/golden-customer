"""
etl_pipeline.py — Layer 1: Rules-Based ETL Engine

Ingests three source CSVs → normalizes → matches/deduplicates →
loads into SQLite golden_customers.db.

Match strategy:
  AUTO-MERGE:  email exact match  OR  (name ≥ 85% AND DOB match)
  UNCERTAIN:   name 60–84%  OR  (name ≥ 60% AND DOB missing in either record)
  NEW RECORD:  name < 60% with no email match

Uncertain records are yielded for the agent layer to resolve.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pandas as pd
from thefuzz import fuzz

from agent.tools import normalize_dob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH     = Path(__file__).parent / "golden_customers.db"
SCHEMA_PATH = Path(__file__).parent / "schema" / "golden_customer.sql"

AUTO_MERGE_THRESHOLD   = 85   # name similarity ≥ this + DOB match → auto merge
UNCERTAIN_LOW          = 60   # name similarity ≥ this → flag as uncertain
UNCERTAIN_HIGH         = 84   # name similarity ≤ this (with 85+ being auto-merge)


# ---------------------------------------------------------------------------
# Source-system field mappings
# ---------------------------------------------------------------------------
SOURCE_CONFIGS = {
    "core_banking": {
        "id":      "cust_id",
        "name":    "full_name",
        "dob":     "date_of_birth",
        "email":   "email",
        "phone":   "phone",
        "address": "address",
    },
    "credit_card": {
        "id":      "client_ref",
        "name":    "name",
        "dob":     "dob",
        "email":   "contact_email",
        "phone":   "mobile",
        "address": "billing_address",
    },
    "mortgage": {
        "id":      "applicant_id",
        "name":    "applicant_name",
        "dob":     "birth_date",
        "email":   "email_address",
        "phone":   "contact_number",
        "address": "property_address",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name_similarity(a: str, b: str) -> int:
    """Return best fuzzy score between two name strings."""
    if not a or not b:
        return 0
    return max(
        fuzz.token_sort_ratio(a.lower(), b.lower()),
        fuzz.token_set_ratio(a.lower(), b.lower()),
        fuzz.partial_ratio(a.lower(), b.lower()),
    )


def _safe_str(val) -> str | None:
    """Convert a pandas value to a clean string, or None if empty/NaN."""
    if val is None:
        return None
    s = str(val).strip()
    return None if s.lower() in ("", "nan", "nat", "none") else s


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def ingest_source(csv_path: str, source_name: str) -> list[dict]:
    """
    Load a CSV and normalize it to a canonical dict list.

    Each record will have keys:
        source, id, name, dob (normalized YYYY-MM-DD), email, phone, address
    """
    mapping = SOURCE_CONFIGS[source_name]
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # Replace empty strings with None equivalent
    df = df.replace({"": None, "nan": None, "NaN": None})

    records = []
    seen_ids = set()

    for _, row in df.iterrows():
        rec_id = _safe_str(row.get(mapping["id"]))
        if rec_id in seen_ids:
            logger.warning("Duplicate source ID %s in %s — skipping", rec_id, source_name)
            continue
        seen_ids.add(rec_id)

        raw_dob = _safe_str(row.get(mapping["dob"]))
        norm_dob = normalize_dob(raw_dob) if raw_dob else None

        records.append({
            "source":  source_name,
            "id":      rec_id,
            "name":    _safe_str(row.get(mapping["name"])),
            "dob":     norm_dob,
            "email":   _safe_str(row.get(mapping["email"])),
            "phone":   _safe_str(row.get(mapping["phone"])),
            "address": _safe_str(row.get(mapping["address"])),
        })

    logger.info("Ingested %d records from %s", len(records), source_name)
    return records


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create (or connect to) the SQLite database and apply the schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(ddl)
    conn.commit()
    logger.info("Database ready at %s", db_path)
    return conn


def _get_all_golden(conn: sqlite3.Connection) -> list[dict]:
    """Load all current golden records as dicts."""
    rows = conn.execute("SELECT * FROM golden_customers").fetchall()
    return [dict(r) for r in rows]


def _upsert_golden(conn: sqlite3.Connection, record: dict):
    """Insert or update a golden record."""
    conn.execute(
        """
        INSERT INTO golden_customers
            (golden_id, canonical_name, dob, email, phone, address,
             source_ids, products, confidence, decision_source,
             created_at, last_updated)
        VALUES
            (:golden_id, :canonical_name, :dob, :email, :phone, :address,
             :source_ids, :products, :confidence, :decision_source,
             :created_at, :last_updated)
        ON CONFLICT(golden_id) DO UPDATE SET
            canonical_name  = excluded.canonical_name,
            dob             = COALESCE(excluded.dob, golden_customers.dob),
            email           = COALESCE(excluded.email, golden_customers.email),
            phone           = COALESCE(excluded.phone, golden_customers.phone),
            address         = COALESCE(excluded.address, golden_customers.address),
            source_ids      = excluded.source_ids,
            products        = excluded.products,
            confidence      = excluded.confidence,
            decision_source = excluded.decision_source,
            last_updated    = excluded.last_updated
        """,
        record,
    )


def _persist_agent_log(conn: sqlite3.Connection, log_entry: dict):
    """Write an agent decision log entry to the DB."""
    conn.execute(
        """
        INSERT OR IGNORE INTO agent_decision_log
            (decision_id, golden_id, record_a_id, record_b_id,
             record_a_name, record_b_name, record_a_source, record_b_source,
             decision, confidence, rationale, name_score, dob_match,
             email_score, address_score, agent_steps, created_at)
        VALUES
            (:decision_id, :golden_id, :record_a_id, :record_b_id,
             :record_a_name, :record_b_name, :record_a_source, :record_b_source,
             :decision, :confidence, :rationale, :name_score, :dob_match,
             :email_score, :address_score, :agent_steps, :created_at)
        """,
        log_entry,
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _find_match(
    incoming: dict,
    golden_records: list[dict],
) -> tuple[dict | None, str, float]:
    """
    Find the best matching golden record for an incoming source record.

    Returns:
        (matched_golden_record | None, match_type, confidence)
        match_type: "email_match" | "auto_merge" | "uncertain" | "new"
    """
    email = (incoming.get("email") or "").lower().strip()
    name  = incoming.get("name") or ""
    dob   = incoming.get("dob")

    best_match       = None
    best_name_score  = 0
    best_match_type  = "new"
    best_confidence  = 1.0

    for golden in golden_records:
        g_email = (golden.get("email") or "").lower().strip()
        g_name  = golden.get("canonical_name") or ""
        g_dob   = golden.get("dob")

        # --- Tier 1: Exact email match ---
        if email and g_email and email == g_email:
            return golden, "email_match", 1.0

        # --- Tier 2 & 3: Name-based matching ---
        name_score = _name_similarity(name, g_name)
        if name_score < UNCERTAIN_LOW:
            continue

        if name_score >= AUTO_MERGE_THRESHOLD:
            # Need DOB to confirm auto-merge
            if dob and g_dob:
                if dob == g_dob:
                    if name_score > best_name_score:
                        best_match      = golden
                        best_name_score = name_score
                        best_match_type = "auto_merge"
                        best_confidence = round(name_score / 100, 2)
                else:
                    # Same name, different DOB → uncertain (possible relatives)
                    if name_score > best_name_score:
                        best_match      = golden
                        best_name_score = name_score
                        best_match_type = "uncertain"
                        best_confidence = round(name_score / 100 * 0.7, 2)
            else:
                # High name match but DOB missing in one → uncertain
                if name_score > best_name_score:
                    best_match      = golden
                    best_name_score = name_score
                    best_match_type = "uncertain"
                    best_confidence = round(name_score / 100 * 0.8, 2)

        elif UNCERTAIN_LOW <= name_score <= UNCERTAIN_HIGH:
            # Mid-range name match → always uncertain
            if name_score > best_name_score:
                best_match      = golden
                best_name_score = name_score
                best_match_type = "uncertain"
                best_confidence = round(name_score / 100 * 0.75, 2)

    return best_match, best_match_type, best_confidence


def _merge_into_golden(
    golden: dict,
    incoming: dict,
    confidence: float,
    decision_source: str,
) -> dict:
    """Merge incoming record fields into a golden record (incoming fills gaps)."""
    source_ids = json.loads(golden.get("source_ids") or "{}")
    source_ids[incoming["source"]] = incoming["id"]

    products = json.loads(golden.get("products") or "[]")
    if incoming["source"] not in products:
        products.append(incoming["source"])

    return {
        **golden,
        "dob":             golden["dob"] or incoming.get("dob"),
        "email":           golden["email"] or incoming.get("email"),
        "phone":           golden["phone"] or incoming.get("phone"),
        "address":         golden["address"] or incoming.get("address"),
        "source_ids":      json.dumps(source_ids),
        "products":        json.dumps(sorted(products)),
        "confidence":      min(golden.get("confidence", 1.0), confidence),
        "decision_source": decision_source if confidence < 1.0 else golden.get("decision_source", "auto"),
        "last_updated":    _now(),
    }


def _new_golden(incoming: dict, confidence: float = 1.0, decision_source: str = "auto") -> dict:
    """Create a new golden record from a source record."""
    return {
        "golden_id":       str(uuid.uuid4()),
        "canonical_name":  incoming["name"] or "Unknown",
        "dob":             incoming.get("dob"),
        "email":           incoming.get("email"),
        "phone":           incoming.get("phone"),
        "address":         incoming.get("address"),
        "source_ids":      json.dumps({incoming["source"]: incoming["id"]}),
        "products":        json.dumps([incoming["source"]]),
        "confidence":      confidence,
        "decision_source": decision_source,
        "created_at":      _now(),
        "last_updated":    _now(),
    }


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def run_etl(
    data_dir: str | Path = "data",
    db_path: Path = DB_PATH,
    agent=None,
) -> dict:
    """
    Run the full ETL pipeline.

    Args:
        data_dir: Directory containing the three source CSVs
        db_path:  SQLite output path
        agent:    DecisionAgent instance (or None to skip agent layer)

    Returns:
        stats dict with counts for reporting
    """
    data_dir = Path(data_dir)
    conn = init_db(db_path)

    stats = {
        "total_raw":         0,
        "auto_merge_email":  0,
        "auto_merge_name":   0,
        "agent_merge":       0,
        "agent_split":       0,
        "new_records":       0,
        "true_duplicates":   0,
        "uncertain_sent":    0,
        "golden_total":      0,
        "sources":           {},
    }

    # Ingest all sources
    all_records: list[dict] = []
    for source_name, config in SOURCE_CONFIGS.items():
        csv_file = data_dir / f"{source_name}.csv"
        if not csv_file.exists():
            logger.warning("CSV not found: %s — skipping", csv_file)
            continue
        records = ingest_source(str(csv_file), source_name)
        stats["sources"][source_name] = len(records)
        stats["total_raw"] += len(records)
        all_records.extend(records)

    logger.info("Total records ingested: %d", stats["total_raw"])

    # Deduplicate within each source first (true duplicates like CB028 = CB001)
    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for rec in all_records:
        h = f"{(rec.get('name') or '').lower()}|{rec.get('dob') or ''}|{(rec.get('email') or '').lower()}"
        if h in seen_hashes:
            stats["true_duplicates"] += 1
            logger.info("True duplicate skipped: %s (%s)", rec["name"], rec["source"])
            continue
        seen_hashes.add(h)
        deduped.append(rec)

    logger.info("After intra-source dedup: %d records", len(deduped))

    # Process each record against the growing golden set
    for rec in deduped:
        golden_records = _get_all_golden(conn)
        matched, match_type, confidence = _find_match(rec, golden_records)

        if match_type == "email_match":
            # Auto-merge on exact email
            updated = _merge_into_golden(matched, rec, confidence=1.0, decision_source="auto")
            _upsert_golden(conn, updated)
            stats["auto_merge_email"] += 1
            logger.debug("Email match: %s → golden %s", rec["name"], matched["golden_id"])

        elif match_type == "auto_merge":
            # Auto-merge on name + DOB
            updated = _merge_into_golden(matched, rec, confidence=confidence, decision_source="auto")
            _upsert_golden(conn, updated)
            stats["auto_merge_name"] += 1
            logger.debug("Name+DOB match: %s → golden %s", rec["name"], matched["golden_id"])

        elif match_type == "uncertain" and agent is not None:
            # Send to AI agent
            stats["uncertain_sent"] += 1
            rec_a = {
                "source":  matched["decision_source"],
                "id":      matched["golden_id"],
                "name":    matched["canonical_name"],
                "dob":     matched.get("dob"),
                "email":   matched.get("email"),
                "phone":   matched.get("phone"),
                "address": matched.get("address"),
            }
            # Recover source from source_ids
            src_ids = json.loads(matched.get("source_ids") or "{}")
            if src_ids:
                rec_a["source"] = list(src_ids.keys())[0]

            rec_b = {
                "source":  rec["source"],
                "id":      rec["id"],
                "name":    rec["name"],
                "dob":     rec.get("dob"),
                "email":   rec.get("email"),
                "phone":   rec.get("phone"),
                "address": rec.get("address"),
            }

            result = agent.decide(rec_a, rec_b, golden_id=matched["golden_id"])

            if result["decision"] == "MERGE":
                updated = _merge_into_golden(
                    matched, rec,
                    confidence=result["confidence"],
                    decision_source="agent",
                )
                _upsert_golden(conn, updated)
                stats["agent_merge"] += 1
            else:
                new_rec = _new_golden(rec, confidence=result["confidence"], decision_source="agent")
                _upsert_golden(conn, new_rec)
                stats["agent_split"] += 1

            _persist_agent_log(conn, result["log_entry"])

        elif match_type == "uncertain" and agent is None:
            # No agent — treat as new record conservatively
            new_rec = _new_golden(rec, confidence=confidence, decision_source="auto")
            _upsert_golden(conn, new_rec)
            stats["new_records"] += 1

        else:
            # No match → new golden record
            new_rec = _new_golden(rec)
            _upsert_golden(conn, new_rec)
            stats["new_records"] += 1

    conn.commit()
    stats["golden_total"] = conn.execute("SELECT COUNT(*) FROM golden_customers").fetchone()[0]
    conn.close()

    logger.info(
        "ETL complete — golden records: %d | auto-merged: %d | agent-resolved: %d | new: %d",
        stats["golden_total"],
        stats["auto_merge_email"] + stats["auto_merge_name"],
        stats["agent_merge"] + stats["agent_split"],
        stats["new_records"],
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from agent.decision_agent import DecisionAgent
    agent = DecisionAgent()
    stats = run_etl(data_dir=Path(__file__).parent / "data", agent=agent)
    print(json.dumps(stats, indent=2))
