# 🏦 Golden Customer Record Pipeline

> A Master Data Management (MDM) / KYC deduplication pipeline that unifies fragmented
> customer data across three banking source systems into a single, authoritative
> **Golden Customer Record** — built with IBM Bob.

---

## Problem Statement

A bank's customer data lives in three siloed systems — **Core Banking**, **Credit Card**, and **Mortgage** — each with different field names, ID formats, date conventions, and record structures. This fragmentation causes:

- Duplicate customer outreach from marketing
- KYC/AML compliance false positives and false negatives
- Inaccurate cross-product risk aggregation
- Weeks of manual reconciliation by data engineering teams

---

## Solution

This pipeline ingests all three source extracts, normalises them into a canonical schema, matches and deduplicates records using a **two-layer strategy**, and loads the results into a unified SQLite **golden record** store — with a live HTML dashboard and full audit trail.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Source Systems (CSV)                    │
│   core_banking.csv   credit_card.csv   mortgage.csv        │
└────────────────────────┬────────────────────────────────────┘
                         │ ingest + normalise
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Layer 1 — Rules-Based ETL Engine                 │
│                    etl_pipeline.py                          │
│                                                             │
│  Exact email match          → AUTO MERGE  (confidence 1.0)  │
│  Name ≥ 85% + DOB match     → AUTO MERGE  (confidence 0.85+)│
│  Name 60–84% or DOB missing → UNCERTAIN  → send to agent   │
│  Name < 60%, no email match → NEW RECORD                   │
└────────────────────────┬────────────────────────────────────┘
                         │ uncertain pairs
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Layer 2 — ReAct AI Decision Agent                  │
│               agent/decision_agent.py                       │
│                                                             │
│  LIVE mode:  OpenAI gpt-4o-mini (set OPENAI_API_KEY)       │
│  MOCK mode:  deterministic fallback — no API key needed     │
│                                                             │
│  Tools available to the agent:                              │
│    compare_names          fuzzy name similarity             │
│    normalize_dob          multi-format date parsing         │
│    compare_email_handles  email local-part similarity       │
│    check_address_match    normalised address comparison     │
└────────────────────────┬────────────────────────────────────┘
                         │ MERGE / SPLIT decision + audit log
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              golden_customers.db  (SQLite)                  │
│   golden_customers table   +   agent_decision_log table     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              dashboard.html  +  terminal report
```

---

## How IBM Bob Was Used

This entire solution was **designed, scaffolded, and built using IBM Bob** — an AI software engineering assistant. Here is exactly how Bob contributed at each stage:

### 1. Problem Scoping & Architecture Design
Bob was given the raw problem statement (fragmented bank customer data, KYC/MDM requirements) and produced the two-layer architecture — rules-first ETL followed by an AI agent for ambiguous cases — reasoning that deterministic rules should handle clear-cut matches while the agent layer handles the fuzzy middle ground that rules can't reliably resolve.

### 2. Schema Design (`schema/golden_customer.sql`)
Bob designed the canonical `golden_customers` table schema, choosing:
- UUID primary keys to decouple from any source system's ID format
- JSON columns (`source_ids`, `products`) to track multi-system provenance without extra join tables
- A separate `agent_decision_log` table for full AI decision auditability, required for KYC/AML compliance

### 3. Mock Data Generation (`data/*.csv`)
Bob generated three realistic mock CSV files — 30 rows each — with:
- Deliberately inconsistent field names across systems (`full_name` vs `name` vs `applicant_name`)
- Mixed date formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `Month DD YYYY`)
- Deliberate duplicates: exact copies, nickname variants (`Jon` / `John`), honorific differences (`Aisha binte Rahman` / `Aisha Rahman`), Unicode accent variants (`María García` / `Maria Garcia`)

### 4. ETL Pipeline (`etl_pipeline.py`)
Bob wrote the full rules-based ETL engine including:
- Source field mapping configuration (`SOURCE_CONFIGS`)
- Multi-strategy name similarity using `thefuzz` (token sort, token set, partial ratio)
- Three-tier match logic (email exact → name+DOB auto-merge → uncertain)
- Hash-based true-duplicate detection across sources
- SQLite upsert with gap-fill merge semantics (incoming data fills `NULL` fields in existing golden records)

### 5. AI Agent Layer (`agent/`)
Bob designed and implemented the ReAct-style agent:
- **Tool functions** (`agent/tools.py`) — pure Python callables with typed return dicts that the agent uses as evidence-gathering instruments
- **OpenAI function-calling loop** (`agent/decision_agent.py`) — a `while` loop that handles tool calls, appends observations to the message chain, and parses the final structured JSON decision
- **System prompt + few-shot examples** (`agent/prompts.py`) — a KYC-domain system prompt with two worked examples (Jon/John, Aisha binte Rahman) that teach the agent the reasoning pattern
- **Mock mode** — a deterministic fallback keyed on cleaned name pairs so the demo runs without any API dependency

### 6. Orchestrator & Dashboard (`run_pipeline.py`)
Bob built the end-to-end orchestrator and generated a self-contained HTML dashboard with:
- KPI cards (golden records, auto-merges, AI decisions, multi-product customers)
- Pipeline flow bar charts
- Searchable golden records table with confidence bars
- AI agent audit trail table with rationale column

### 7. Test Suite (`tests/`)
Bob wrote 69 tests covering:
- Unit tests for every tool function
- Match logic unit tests (email match, auto-merge, uncertain, new record)
- Full pipeline integration tests (dedup count, cross-system merge, schema validation)
- Agent mock mode tests (decision schema, log writing, evidence fields)

### Bob's Impact
| Without Bob | With Bob |
|-------------|----------|
| Days of schema design and field mapping | Minutes — Bob generated the schema from the problem description |
| Manual ETL scripting | Full ETL engine with three-tier match strategy, generated and explained |
| Separate AI integration effort | ReAct agent with tools, prompt engineering, and mock fallback — all in one pass |
| Ad-hoc testing | 69 structured tests across both layers |
| Static report | Live HTML dashboard with charts, search, and audit trail |

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python run_pipeline.py
```

This will:
1. Initialise the SQLite database
2. Ingest and normalise all three source CSVs
3. Run the rules-based ETL (auto-merges)
4. Run the AI agent on uncertain pairs (mock mode by default)
5. Generate `dashboard.html` and open it in your browser
6. Print a formatted terminal report

### Use live AI (optional)
```bash
set OPENAI_API_KEY=sk-...      # Windows
export OPENAI_API_KEY=sk-...   # Mac/Linux
python run_pipeline.py
```

### Run tests
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
golden-customer/
├── data/
│   ├── core_banking.csv       # Source 1: 30 rows, YYYY-MM-DD dates
│   ├── credit_card.csv        # Source 2: 30 rows, DD/MM/YYYY dates
│   └── mortgage.csv           # Source 3: 30 rows, "Month DD YYYY" dates
├── schema/
│   └── golden_customer.sql    # SQLite DDL — golden_customers + agent_decision_log
├── agent/
│   ├── decision_agent.py      # ReAct agent — live (gpt-4o-mini) + mock mode
│   ├── tools.py               # compare_names, normalize_dob, compare_email_handles, check_address_match
│   ├── prompts.py             # System prompt + few-shot examples
│   └── __init__.py
├── tests/
│   ├── test_etl.py            # 34 ETL tests
│   └── test_agent.py          # 35 agent + tool tests
├── etl_pipeline.py            # Layer 1 — rules-based ETL engine
├── run_pipeline.py            # Orchestrator + dashboard generator
├── report.py                  # Terminal summary report
├── requirements.txt
└── README.md
```

---

## Match Strategy Reference

| Condition | Action | Confidence |
|-----------|--------|------------|
| Exact email match | AUTO MERGE | 1.0 |
| Name similarity ≥ 85% **and** DOB match | AUTO MERGE | name_score / 100 |
| Name similarity ≥ 85%, DOB missing in one | UNCERTAIN → Agent | name_score / 100 × 0.8 |
| Name similarity ≥ 85%, DOBs differ | UNCERTAIN → Agent | name_score / 100 × 0.7 |
| Name similarity 60–84% | UNCERTAIN → Agent | name_score / 100 × 0.75 |
| Name similarity < 60%, no email match | NEW RECORD | 1.0 |

---

## Output Files

| File | Description |
|------|-------------|
| `golden_customers.db` | SQLite database — golden records + agent decision log |
| `dashboard.html` | Self-contained HTML dashboard (open in any browser) |
| `agent_decisions.jsonl` | JSONL audit log — one entry per AI decision |

---

## Why It Matters

MDM projects like this typically take **weeks** of manual field mapping, reconciliation scripting, and QA. This demo shows how the engineering heavy-lifting — schema design, ETL logic, AI integration, testing, and reporting — can be generated rather than hand-built, leaving teams free to focus on the domain rules and compliance requirements that actually require human judgment.

---

*Built with [IBM Bob](https://www.ibm.com/products/bob) · Golden Customer Record Demo*
