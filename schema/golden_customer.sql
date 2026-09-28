-- =============================================================
--  Golden Customer Record — Canonical Schema
--  Master Data Management / KYC Golden Record Pipeline
-- =============================================================

-- Core golden record table
CREATE TABLE IF NOT EXISTS golden_customers (
    golden_id       TEXT PRIMARY KEY,           -- UUID v4
    canonical_name  TEXT NOT NULL,              -- Normalised display name
    dob             TEXT,                       -- ISO-8601: YYYY-MM-DD
    email           TEXT,                       -- Deduplicated primary email
    phone           TEXT,                       -- Primary phone number
    address         TEXT,                       -- Best-quality address on file
    source_ids      TEXT NOT NULL DEFAULT '{}', -- JSON: {"core_banking":"CB001","credit_card":"CC101"}
    products        TEXT NOT NULL DEFAULT '[]', -- JSON array: ["core_banking","credit_card","mortgage"]
    confidence      REAL NOT NULL DEFAULT 1.0,  -- Match confidence: 1.0=exact, 0.85+=auto, <0.85=agent
    decision_source TEXT NOT NULL DEFAULT 'auto', -- "auto" | "agent"
    created_at      TEXT NOT NULL,              -- ISO-8601 timestamp
    last_updated    TEXT NOT NULL               -- ISO-8601 timestamp
);

-- Agent decision audit log — every AI decision is persisted here
CREATE TABLE IF NOT EXISTS agent_decision_log (
    decision_id     TEXT PRIMARY KEY,           -- UUID v4
    golden_id       TEXT,                       -- FK → golden_customers (nullable if SPLIT)
    record_a_id     TEXT NOT NULL,              -- Source record A identifier
    record_b_id     TEXT NOT NULL,              -- Source record B identifier
    record_a_name   TEXT,
    record_b_name   TEXT,
    record_a_source TEXT,                       -- "core_banking" | "credit_card" | "mortgage"
    record_b_source TEXT,
    decision        TEXT NOT NULL,              -- "MERGE" | "SPLIT"
    confidence      REAL NOT NULL,
    rationale       TEXT NOT NULL,              -- Human-readable AI reasoning
    name_score      REAL,                       -- Fuzzy name similarity score
    dob_match       INTEGER,                    -- 1=match, 0=no match, NULL=unknown
    email_score     REAL,                       -- Email handle similarity
    address_score   REAL,                       -- Address similarity
    agent_steps     INTEGER DEFAULT 0,          -- Number of ReAct reasoning steps taken
    created_at      TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_golden_email    ON golden_customers(email);
CREATE INDEX IF NOT EXISTS idx_golden_dob      ON golden_customers(dob);
CREATE INDEX IF NOT EXISTS idx_golden_name     ON golden_customers(canonical_name);
CREATE INDEX IF NOT EXISTS idx_agent_golden_id ON agent_decision_log(golden_id);
CREATE INDEX IF NOT EXISTS idx_agent_decision  ON agent_decision_log(decision);
