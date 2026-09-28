"""
tests/test_etl.py — Tests for the ETL pipeline (Layer 1).
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools import normalize_dob
from etl_pipeline import (
    _find_match,
    _merge_into_golden,
    _name_similarity,
    _new_golden,
    ingest_source,
    init_db,
    run_etl,
)


# ---------------------------------------------------------------------------
# normalize_dob
# ---------------------------------------------------------------------------
class TestNormalizeDob:
    def test_iso_format_passthrough(self):
        assert normalize_dob("1985-03-08") == "1985-03-08"

    def test_slash_format_dd_mm_yyyy(self):
        assert normalize_dob("08/03/1985") == "1985-03-08"

    def test_long_month_name(self):
        assert normalize_dob("March 8 1985") == "1985-03-08"

    def test_long_month_name_with_day_zero(self):
        assert normalize_dob("November 22 1990") == "1990-11-22"

    def test_long_date_mortgage_format(self):
        assert normalize_dob("March 15 1978") == "1978-03-15"

    def test_none_input(self):
        assert normalize_dob(None) is None

    def test_empty_string(self):
        assert normalize_dob("") is None

    def test_nan_string(self):
        assert normalize_dob("nan") is None

    def test_already_normalized_returns_same(self):
        result = normalize_dob("1992-07-30")
        assert result == "1992-07-30"

    def test_different_formats_same_result(self):
        """The three DOB formats used across source systems must normalize identically."""
        dob_core_banking  = normalize_dob("1978-06-15")    # core_banking format
        dob_credit_card   = normalize_dob("15/06/1978")    # credit_card format
        dob_mortgage      = normalize_dob("June 15 1978")  # mortgage format
        assert dob_core_banking == dob_credit_card == dob_mortgage == "1978-06-15"


# ---------------------------------------------------------------------------
# Name similarity
# ---------------------------------------------------------------------------
class TestNameSimilarity:
    def test_exact_match_is_100(self):
        assert _name_similarity("James Thornton", "James Thornton") == 100

    def test_nickname_jon_john(self):
        score = _name_similarity("Jon Smith", "John Smith")
        assert score >= 85, f"Jon/John should score ≥85, got {score}"

    def test_honorific_stripped_aisha(self):
        from agent.tools import compare_names
        result = compare_names("Aisha binte Rahman", "Aisha Rahman")
        assert result["best_score"] >= 90

    def test_unicode_accent_variant(self):
        from agent.tools import compare_names
        result = compare_names("Maria Garcia", "María García")
        assert result["best_score"] >= 95

    def test_completely_different_names(self):
        score = _name_similarity("James Thornton", "Priya Nair")
        assert score < 40

    def test_partial_name_initials(self):
        from agent.tools import compare_names
        result = compare_names("P. Nair", "Priya Nair")
        assert result["best_score"] >= 60


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
class TestIngest:
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"

    def test_core_banking_row_count(self, data_dir):
        records = ingest_source(str(data_dir / "core_banking.csv"), "core_banking")
        # 30 rows, all with unique cust_id — ingest returns all 30
        # CB028 shares name+DOB with CB001 but is caught as a true duplicate
        # later in run_etl (hash-based dedup), not at ingest time
        assert len(records) == 30

    def test_credit_card_row_count(self, data_dir):
        records = ingest_source(str(data_dir / "credit_card.csv"), "credit_card")
        assert len(records) == 30

    def test_mortgage_row_count(self, data_dir):
        records = ingest_source(str(data_dir / "mortgage.csv"), "mortgage")
        assert len(records) == 30

    def test_dob_normalized_on_ingest(self, data_dir):
        """Credit card uses DD/MM/YYYY — must normalize to YYYY-MM-DD."""
        records = ingest_source(str(data_dir / "credit_card.csv"), "credit_card")
        for r in records:
            if r["dob"] is not None:
                assert len(r["dob"]) == 10
                assert r["dob"][4] == "-" and r["dob"][7] == "-"

    def test_mortgage_dob_normalized(self, data_dir):
        """Mortgage uses 'Month DD YYYY' — must normalize to YYYY-MM-DD."""
        records = ingest_source(str(data_dir / "mortgage.csv"), "mortgage")
        for r in records:
            if r["dob"] is not None:
                assert len(r["dob"]) == 10, f"Bad DOB format: {r['dob']}"

    def test_source_field_mapped(self, data_dir):
        records = ingest_source(str(data_dir / "core_banking.csv"), "core_banking")
        assert all("name" in r for r in records)
        assert all("email" in r for r in records)
        assert all(r["source"] == "core_banking" for r in records)

    def test_null_email_allowed(self, data_dir):
        """Some records have missing emails — should ingest as None, not error."""
        records = ingest_source(str(data_dir / "core_banking.csv"), "core_banking")
        null_email = [r for r in records if r["email"] is None]
        assert len(null_email) >= 1, "Expected at least one record with null email"


# ---------------------------------------------------------------------------
# Match logic
# ---------------------------------------------------------------------------
class TestMatchLogic:
    def _make_golden(self, **kwargs):
        defaults = {
            "golden_id":       "g-001",
            "canonical_name":  "James Thornton",
            "dob":             "1978-06-15",
            "email":           "james.thornton@email.com",
            "phone":           "+60123456789",
            "address":         "12 Jalan Bukit Bintang KL",
            "source_ids":      '{"core_banking":"CB001"}',
            "products":        '["core_banking"]',
            "confidence":      1.0,
            "decision_source": "auto",
            "created_at":      "2024-01-01T00:00:00+00:00",
            "last_updated":    "2024-01-01T00:00:00+00:00",
        }
        defaults.update(kwargs)
        return defaults

    def test_email_exact_match(self):
        incoming = {"name": "J. Thornton", "dob": None, "email": "james.thornton@email.com",
                    "source": "credit_card", "id": "CC101"}
        golden = [self._make_golden()]
        match, match_type, conf = _find_match(incoming, golden)
        assert match_type == "email_match"
        assert conf == 1.0

    def test_auto_merge_name_dob(self):
        incoming = {"name": "James Thornton", "dob": "1978-06-15",
                    "email": "jt@other.com", "source": "mortgage", "id": "MG001"}
        golden = [self._make_golden(email="james.thornton@email.com")]
        match, match_type, conf = _find_match(incoming, golden)
        assert match_type == "auto_merge"
        assert conf >= 0.85

    def test_uncertain_high_name_missing_dob(self):
        incoming = {"name": "Jon Smith", "dob": None,
                    "email": "jon.smith@email.com", "source": "credit_card", "id": "CC107"}
        golden = [self._make_golden(
            golden_id="g-007",
            canonical_name="John Smith",
            dob="1983-04-25",
            email="jon.smith.other@email.com",
        )]
        match, match_type, conf = _find_match(incoming, golden)
        assert match_type == "uncertain"

    def test_no_match_different_person(self):
        incoming = {"name": "Priya Nair", "dob": "1992-07-30",
                    "email": "priya.nair@hotmail.com", "source": "credit_card", "id": "CC104"}
        golden = [self._make_golden()]
        match, match_type, conf = _find_match(incoming, golden)
        assert match_type == "new"

    def test_merge_preserves_source_ids(self):
        incoming = {"name": "James Thornton", "dob": "1978-06-15", "email": None,
                    "source": "mortgage", "id": "MG001", "phone": None, "address": None}
        golden = self._make_golden()
        merged = _merge_into_golden(golden, incoming, confidence=0.95, decision_source="auto")
        source_ids = json.loads(merged["source_ids"])
        assert "core_banking" in source_ids
        assert "mortgage" in source_ids
        assert source_ids["mortgage"] == "MG001"

    def test_merge_fills_null_fields(self):
        golden = self._make_golden(dob=None, phone=None)
        incoming = {"name": "James Thornton", "dob": "1978-06-15",
                    "phone": "+60123456789", "email": "james.thornton@email.com",
                    "source": "mortgage", "id": "MG001", "address": None}
        merged = _merge_into_golden(golden, incoming, confidence=0.95, decision_source="auto")
        assert merged["dob"] == "1978-06-15"
        assert merged["phone"] == "+60123456789"


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_dedup_reduces_record_count(self):
        """Golden record count must be significantly less than raw input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent.parent / "data"
            stats = run_etl(data_dir=data_dir, db_path=db, agent=None)
            raw = stats["total_raw"]
            golden = stats["golden_total"]
            assert golden < raw, f"Expected dedup: golden ({golden}) < raw ({raw})"

    def test_true_duplicate_detected(self):
        """CB028 is an exact copy of CB001 — must be caught as true duplicate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent.parent / "data"
            stats = run_etl(data_dir=data_dir, db_path=db, agent=None)
            assert stats["true_duplicates"] >= 1

    def test_cross_system_customers_merged(self):
        """Same customers appear in all 3 systems — must resolve to 1 golden record each."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent.parent / "data"
            run_etl(data_dir=data_dir, db_path=db, agent=None)
            conn = sqlite3.connect(db)
            multi = conn.execute(
                "SELECT COUNT(*) FROM golden_customers WHERE json_array_length(products) >= 2"
            ).fetchone()[0]
            conn.close()
            assert multi >= 10, f"Expected ≥10 multi-product records, got {multi}"

    def test_all_source_ids_tracked(self):
        """source_ids JSON must contain the originating system ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent.parent / "data"
            run_etl(data_dir=data_dir, db_path=db, agent=None)
            conn = sqlite3.connect(db)
            rows = conn.execute("SELECT source_ids FROM golden_customers").fetchall()
            conn.close()
            for row in rows:
                ids = json.loads(row[0])
                assert isinstance(ids, dict)
                assert len(ids) >= 1

    def test_golden_record_schema(self):
        """Every golden record must have required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent.parent / "data"
            run_etl(data_dir=data_dir, db_path=db, agent=None)
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM golden_customers").fetchall()
            conn.close()
            for row in rows:
                r = dict(row)
                assert r["golden_id"], "golden_id must not be empty"
                assert r["canonical_name"], "canonical_name must not be empty"
                assert r["confidence"] is not None
                assert r["decision_source"] in ("auto", "agent")
                assert r["created_at"]
