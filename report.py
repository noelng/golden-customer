"""
report.py — Formatted terminal summary for the demo presentation.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "golden_customers.db"
AGENT_LOG = Path(__file__).parent / "agent_decisions.jsonl"


def _bar(value: int, max_val: int, width: int = 20) -> str:
    filled = int((value / max_val) * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_report(stats: dict | None = None):
    """Print a formatted terminal summary of the pipeline results."""

    # Load stats from DB if not provided
    if stats is None:
        stats = {}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total_golden = conn.execute("SELECT COUNT(*) FROM golden_customers").fetchone()[0]
    auto_records = conn.execute(
        "SELECT COUNT(*) FROM golden_customers WHERE decision_source='auto'"
    ).fetchone()[0]
    agent_records = conn.execute(
        "SELECT COUNT(*) FROM golden_customers WHERE decision_source='agent'"
    ).fetchone()[0]
    avg_confidence = conn.execute(
        "SELECT AVG(confidence) FROM golden_customers"
    ).fetchone()[0] or 0

    # Multi-product customers
    multi_product = conn.execute(
        "SELECT COUNT(*) FROM golden_customers WHERE json_array_length(products) > 1"
    ).fetchone()[0]

    # Agent log
    agent_merges = conn.execute(
        "SELECT COUNT(*) FROM agent_decision_log WHERE decision='MERGE'"
    ).fetchone()[0]
    agent_splits = conn.execute(
        "SELECT COUNT(*) FROM agent_decision_log WHERE decision='SPLIT'"
    ).fetchone()[0]
    conn.close()

    # Load agent log for detail
    agent_decisions = []
    if AGENT_LOG.exists():
        with open(AGENT_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        agent_decisions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    total_raw    = stats.get("total_raw", 90)
    sources      = stats.get("sources", {})
    true_dups    = stats.get("true_duplicates", 0)
    auto_email   = stats.get("auto_merge_email", 0)
    auto_name    = stats.get("auto_merge_name", 0)
    uncertain    = stats.get("uncertain_sent", 0)
    new_recs     = stats.get("new_records", 0)
    agent_mode   = stats.get("agent_mode", "mock")

    W = 62  # box width

    def box_line(text="", pad=True):
        content = f" {text}" if pad else text
        print(f"║{content:<{W}}║")

    def divider():
        print(f"╠{'═' * W}╣")

    print(f"\n╔{'═' * W}╗")
    box_line("      GOLDEN CUSTOMER RECORD  ·  PIPELINE REPORT")
    box_line(f"      {'LIVE MODE (OpenAI gpt-4o-mini)' if agent_mode == 'live' else 'DEMO MODE (Mock AI · set OPENAI_API_KEY for live)'}")
    divider()

    box_line("  SOURCE RECORDS INGESTED")
    for src, count in sources.items():
        label = src.replace("_", " ").title()
        box_line(f"    {label:<25} →  {count:>3} records")
    box_line()
    box_line(f"    {'True duplicates removed':<25} →  {true_dups:>3} records")
    box_line(f"    {'TOTAL RAW (unique)':<25} →  {total_raw - true_dups:>3} records")

    divider()
    box_line("  DEDUPLICATION RESULTS")
    box_line(f"    {'Email exact-match merges':<32} →  {auto_email:>2}")
    box_line(f"    {'Name+DOB auto-merges':<32} →  {auto_name:>2}")
    box_line(f"    {'Sent to AI Agent':<32} →  {uncertain:>2}")
    box_line(f"      ↳ Agent MERGE decisions        →  {agent_merges:>2}")
    box_line(f"      ↳ Agent SPLIT decisions         →  {agent_splits:>2}")
    box_line(f"    {'New golden records created':<32} →  {new_recs:>2}")

    divider()
    box_line("  GOLDEN RECORD SUMMARY")
    box_line(f"    {'Total golden records':<32} →  {total_golden:>2}")
    box_line(f"    {'Multi-product customers':<32} →  {multi_product:>2}")
    box_line(f"    {'Avg match confidence':<32}   {avg_confidence:.0%}")
    quality_bar = _bar(int(avg_confidence * 100), 100, 28)
    box_line(f"    {quality_bar}  {avg_confidence:.0%}")

    if agent_decisions:
        divider()
        box_line("  AI AGENT DECISIONS (Audit Trail)")
        for d in agent_decisions[:6]:  # Show first 6 for terminal
            icon = "✓ MERGE" if d["decision"] == "MERGE" else "✗ SPLIT"
            box_line(f"    [{icon}] conf={d['confidence']:.2f}")
            name_a = (d.get("record_a_name") or "")[:22]
            name_b = (d.get("record_b_name") or "")[:22]
            box_line(f"      {name_a} ↔ {name_b}")
            rationale = (d.get("rationale") or "")[:56]
            box_line(f"      → {rationale}")
            box_line()

    divider()
    box_line("  COMPLIANCE IMPACT  (vs. manual reconciliation)")
    kpi_cols = int((1 - total_golden / max(total_raw - true_dups, 1)) * 100)
    box_line(f"    Duplicate noise eliminated   {_bar(kpi_cols, 100, 20)}  {kpi_cols}%")
    box_line(f"    KYC false positives reduced  →  {auto_email + auto_name + agent_merges:>2} merged records")
    box_line(f"    Cross-product risk coverage  →  {multi_product:>2} customers across ≥2 products")
    box_line(f"    Every AI decision auditable  →  agent_decisions.jsonl")
    box_line()
    box_line("  ✓ Dashboard:  golden-customer/dashboard.html")
    box_line("  ✓ Database:   golden-customer/golden_customers.db")
    print(f"╚{'═' * W}╝\n")


if __name__ == "__main__":
    print_report()
