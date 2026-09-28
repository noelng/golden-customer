"""
tests/test_agent.py — Tests for the Agentic AI layer (Layer 2).
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools import (
    check_address_match,
    compare_email_handles,
    compare_names,
    compare_phone_numbers,
    normalize_dob,
)
from agent.decision_agent import DecisionAgent, _compute_risk_flags, _smart_mock_decide


# ---------------------------------------------------------------------------
# Tool tests: compare_names
# ---------------------------------------------------------------------------
class TestCompareNames:
    def test_identical_names_score_100(self):
        result = compare_names("James Thornton", "James Thornton")
        assert result["best_score"] == 100

    def test_nickname_variant_jon_john(self):
        result = compare_names("Jon Smith", "John Smith")
        assert result["best_score"] >= 85

    def test_honorific_binte_stripped(self):
        result = compare_names("Aisha binte Rahman", "Aisha Rahman")
        assert result["best_score"] >= 90

    def test_honorific_bin_stripped(self):
        result = compare_names("Ahmad Farid bin Hassan", "Ahmad Farid Hassan")
        assert result["best_score"] >= 90

    def test_unicode_accent_normalization(self):
        result = compare_names("María García", "Maria Garcia")
        assert result["best_score"] >= 95

    def test_completely_different_names_low_score(self):
        result = compare_names("James Thornton", "Priya Nair")
        assert result["best_score"] < 40

    def test_empty_name_returns_zero(self):
        result = compare_names("", "James Thornton")
        assert result["best_score"] == 0

    def test_none_name_returns_zero(self):
        result = compare_names(None, "James Thornton")
        assert result["best_score"] == 0

    def test_middle_initial_variant(self):
        result = compare_names("James R. Thornton", "James Thornton")
        assert result["best_score"] >= 85

    def test_result_has_required_keys(self):
        result = compare_names("Alice", "Alice")
        assert "token_sort_ratio" in result
        assert "partial_ratio" in result
        assert "token_set_ratio" in result
        assert "best_score" in result
        assert "clean_a" in result
        assert "clean_b" in result

    def test_full_vs_abbreviated_name(self):
        result = compare_names("Jonathan Smith", "Jon Smith")
        assert result["best_score"] >= 80


# ---------------------------------------------------------------------------
# Tool tests: normalize_dob
# ---------------------------------------------------------------------------
class TestNormalizeDobTool:
    def test_three_source_formats_identical(self):
        cb  = normalize_dob("1978-06-15")
        cc  = normalize_dob("15/06/1978")
        mg  = normalize_dob("June 15 1978")
        assert cb == cc == mg == "1978-06-15"

    def test_returns_none_for_empty(self):
        assert normalize_dob("") is None
        assert normalize_dob(None) is None

    def test_returns_none_for_nan(self):
        assert normalize_dob("nan") is None

    def test_iso_format_preserved(self):
        assert normalize_dob("1990-11-22") == "1990-11-22"

    def test_march_format(self):
        assert normalize_dob("March 8 1985") == "1985-03-08"

    def test_slash_dd_mm_yyyy(self):
        assert normalize_dob("30/07/1992") == "1992-07-30"


# ---------------------------------------------------------------------------
# Tool tests: compare_email_handles
# ---------------------------------------------------------------------------
class TestCompareEmailHandles:
    def test_identical_emails_score_1(self):
        result = compare_email_handles("jon.smith@email.com", "jon.smith@email.com")
        assert result["similarity"] == 1.0
        assert result["same_domain"] is True

    def test_different_handles_same_domain(self):
        result = compare_email_handles("a.rahman@gmail.com", "aisha.r@gmail.com")
        assert result["same_domain"] is True
        assert 0 < result["similarity"] < 1.0

    def test_different_domains(self):
        result = compare_email_handles("user@gmail.com", "user@yahoo.com")
        assert result["same_domain"] is False

    def test_empty_email(self):
        result = compare_email_handles("", "user@email.com")
        assert result["similarity"] == 0.0

    def test_none_email(self):
        result = compare_email_handles(None, "user@email.com")
        assert result["similarity"] == 0.0

    def test_result_has_required_keys(self):
        result = compare_email_handles("a@b.com", "c@d.com")
        assert "handle_a" in result
        assert "handle_b" in result
        assert "similarity" in result
        assert "same_domain" in result


# ---------------------------------------------------------------------------
# Tool tests: check_address_match
# ---------------------------------------------------------------------------
class TestCheckAddressMatch:
    def test_identical_addresses_score_1(self):
        result = check_address_match(
            "12 Jalan Bukit Bintang KL",
            "12 Jalan Bukit Bintang KL"
        )
        assert result["similarity"] == 1.0

    def test_abbreviated_vs_full(self):
        result = check_address_match(
            "45 Taman Desa PJ",
            "45 Taman Desa Petaling Jaya"
        )
        assert result["similarity"] > 0.5

    def test_completely_different_addresses(self):
        result = check_address_match(
            "12 Jalan Bukit Bintang KL",
            "99 Jalan Ipoh Penang"
        )
        assert result["similarity"] < 0.7

    def test_empty_address(self):
        result = check_address_match("", "12 Jalan Bukit Bintang KL")
        assert result["similarity"] == 0.0

    def test_result_has_required_keys(self):
        result = check_address_match("A", "B")
        assert "similarity" in result
        assert "clean_a" in result
        assert "clean_b" in result


# ---------------------------------------------------------------------------
# DecisionAgent: mock mode
# ---------------------------------------------------------------------------
class TestDecisionAgentMock:
    @pytest.fixture
    def agent(self, tmp_path):
        log = tmp_path / "decisions.jsonl"
        return DecisionAgent(log_path=str(log))

    def test_agent_starts_in_mock_mode_without_key(self, agent):
        """Without OPENAI_API_KEY, agent must run in mock mode."""
        import os
        if "OPENAI_API_KEY" in os.environ:
            pytest.skip("OPENAI_API_KEY is set — mock mode test skipped")
        assert agent.mode == "mock"

    def test_jon_john_mock_merge(self, agent):
        rec_a = {"source": "core_banking",  "id": "CB007", "name": "Jon Smith",
                 "dob": None, "email": "jon.smith@email.com", "phone": "+60156677889",
                 "address": "34 Bangsar South KL"}
        rec_b = {"source": "credit_card",   "id": "CC107", "name": "John Smith",
                 "dob": "25/04/1983", "email": "jon.smith@email.com", "phone": "+60156677889",
                 "address": "34 Bangsar South KL"}
        result = agent.decide(rec_a, rec_b)
        assert result["decision"] == "MERGE"
        assert result["confidence"] > 0.8
        assert isinstance(result["rationale"], str) and len(result["rationale"]) > 10

    def test_aisha_honorific_mock_merge(self, agent):
        rec_a = {"source": "core_banking", "id": "CB008", "name": "Aisha binte Rahman",
                 "dob": None, "email": "a.rahman@gmail.com", "phone": "+60189900112",
                 "address": "19 Chow Kit KL"}
        rec_b = {"source": "credit_card",  "id": "CC108", "name": "Aisha Rahman",
                 "dob": "14/07/1990", "email": "aisha.r@gmail.com", "phone": "+60189900112",
                 "address": "19 Chow Kit KL"}
        result = agent.decide(rec_a, rec_b)
        assert result["decision"] == "MERGE"
        assert result["confidence"] >= 0.7

    def test_decision_log_written(self, agent, tmp_path):
        log_path = tmp_path / "decisions.jsonl"
        agent_with_log = DecisionAgent(log_path=str(log_path))
        rec_a = {"source": "core_banking",  "id": "CB007", "name": "Jon Smith",
                 "dob": None, "email": "jon.smith@email.com", "phone": "+1", "address": ""}
        rec_b = {"source": "credit_card",   "id": "CC107", "name": "John Smith",
                 "dob": "1983-04-25", "email": "jon.smith@email.com", "phone": "+1", "address": ""}
        agent_with_log.decide(rec_a, rec_b)
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["decision"] in ("MERGE", "SPLIT")
        assert "rationale" in entry
        assert "confidence" in entry
        assert "decision_id" in entry
        assert "created_at" in entry

    def test_log_entry_has_evidence_fields(self, agent, tmp_path):
        log_path = tmp_path / "decisions.jsonl"
        agent_with_log = DecisionAgent(log_path=str(log_path))
        rec_a = {"source": "core_banking", "id": "CB001", "name": "James Thornton",
                 "dob": "1978-06-15", "email": "james.thornton@email.com",
                 "phone": "+60123456789", "address": "12 Jalan Bukit Bintang KL"}
        rec_b = {"source": "mortgage",     "id": "MG001", "name": "James R. Thornton",
                 "dob": "March 15 1978", "email": "james.thornton@email.com",
                 "phone": "+60123456789", "address": "12 Jalan Bukit Bintang KL"}
        agent_with_log.decide(rec_a, rec_b)
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert entry["name_score"] is not None
        assert entry["email_score"] is not None
        assert entry["record_a_source"] == "core_banking"
        assert entry["record_b_source"] == "mortgage"

    def test_unicode_names_decide_without_error(self, agent):
        rec_a = {"source": "core_banking", "id": "CB023", "name": "Maria Garcia",
                 "dob": None, "email": "maria.garcia@email.com", "phone": "+60154567890",
                 "address": "25 KLCC KL"}
        rec_b = {"source": "mortgage",     "id": "MG023", "name": "María García",
                 "dob": "April 12 1988", "email": "maria.garcia@email.com",
                 "phone": "+60154567890", "address": "25 KLCC KL"}
        result = agent.decide(rec_a, rec_b)
        assert result["decision"] in ("MERGE", "SPLIT")
        assert isinstance(result["confidence"], float)

    def test_result_schema(self, agent):
        rec_a = {"source": "core_banking", "id": "CB001", "name": "Alice Test",
                 "dob": None, "email": "alice@test.com", "phone": "", "address": ""}
        rec_b = {"source": "credit_card",  "id": "CC001", "name": "Alyse Test",
                 "dob": "1990-01-01", "email": "alyse@test.com", "phone": "", "address": ""}
        result = agent.decide(rec_a, rec_b)
        assert "decision" in result
        assert "confidence" in result
        assert "rationale" in result
        assert "risk_flags" in result
        assert "log_entry" in result
        assert result["decision"] in ("MERGE", "SPLIT")
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["risk_flags"], list)


# ---------------------------------------------------------------------------
# Tool tests: compare_phone_numbers
# ---------------------------------------------------------------------------
class TestComparePhoneNumbers:
    def test_identical_phones_match(self):
        result = compare_phone_numbers("+60123456789", "+60123456789")
        assert result["match"] is True
        assert result["similarity"] == 1.0

    def test_plus60_vs_local_normalised_match(self):
        """'+60123456789' and '0123456789' should normalise to the same digits."""
        result = compare_phone_numbers("+60123456789", "0123456789")
        assert result["match"] is True

    def test_different_phones_no_match(self):
        result = compare_phone_numbers("+60112233001", "+60198877665")
        assert result["match"] is False

    def test_empty_phone_returns_zero(self):
        result = compare_phone_numbers("", "+60123456789")
        assert result["match"] is False
        assert result["similarity"] == 0.0

    def test_none_phone_returns_zero(self):
        result = compare_phone_numbers(None, "+60123456789")
        assert result["match"] is False
        assert result["similarity"] == 0.0

    def test_result_has_required_keys(self):
        result = compare_phone_numbers("+601", "+602")
        assert "normalised_a" in result
        assert "normalised_b" in result
        assert "match" in result
        assert "similarity" in result

    def test_similar_but_not_identical_phones(self):
        """One digit difference should give high but not 1.0 similarity."""
        result = compare_phone_numbers("0123456789", "0123456780")
        assert result["match"] is False
        assert result["similarity"] > 0.8


# ---------------------------------------------------------------------------
# Smart mock decision logic
# ---------------------------------------------------------------------------
class TestSmartMockDecide:
    """Tests for the data-driven _smart_mock_decide function."""

    def _name(self, a, b):
        from agent.tools import compare_names
        return compare_names(a, b)

    def _email(self, a, b):
        from agent.tools import compare_email_handles
        return compare_email_handles(a, b)

    def _phone(self, a, b):
        return compare_phone_numbers(a, b)

    def _addr(self, a, b):
        from agent.tools import check_address_match
        return check_address_match(a, b)

    def test_identical_phone_and_high_name_yields_merge(self):
        result = _smart_mock_decide(
            {"name": "Jon Smith"}, {"name": "John Smith"},
            self._name("Jon Smith", "John Smith"),
            self._email("jon.smith@email.com", "jon.smith@email.com"),
            self._phone("+60156677889", "+60156677889"),
            None,  # dob_match unknown
            self._addr("34 Bangsar KL", "34 Bangsar KL"),
        )
        assert result["decision"] == "MERGE"
        assert result["confidence"] > 0.6

    def test_different_phone_and_low_name_yields_split(self):
        result = _smart_mock_decide(
            {"name": "James Lim"}, {"name": "James Tan"},
            self._name("James Lim", "James Tan"),
            self._email("james.lim@gmail.com", "james.tan@yahoo.com"),
            self._phone("+60112233001", "+60198877665"),
            False,  # dob mismatch
            self._addr("10 Ampang KL", "52 Kepong KL"),
        )
        assert result["decision"] == "SPLIT"
        assert result["confidence"] > 0.3

    def test_dob_match_boosts_score(self):
        """Adding a DOB match should raise the score and tip marginal cases toward MERGE."""
        base = _smart_mock_decide(
            {"name": "Wei Lin"}, {"name": "Wei Ling"},
            self._name("Wei Lin", "Wei Ling"),
            self._email("w.lin@gmail.com", "w.ling@gmail.com"),
            self._phone("+60111111111", "+60111111111"),
            None,   # no DOB
            self._addr("", ""),
        )
        with_dob = _smart_mock_decide(
            {"name": "Wei Lin"}, {"name": "Wei Ling"},
            self._name("Wei Lin", "Wei Ling"),
            self._email("w.lin@gmail.com", "w.ling@gmail.com"),
            self._phone("+60111111111", "+60111111111"),
            True,   # DOB confirms same person
            self._addr("", ""),
        )
        assert with_dob["confidence"] >= base["confidence"]

    def test_result_schema(self):
        result = _smart_mock_decide(
            {"name": "Alice"}, {"name": "Alicia"},
            self._name("Alice", "Alicia"),
            self._email("", ""),
            self._phone("", ""),
            None,
            self._addr("", ""),
        )
        assert result["decision"] in ("MERGE", "SPLIT")
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["rationale"], str) and len(result["rationale"]) > 10
        assert "steps" in result

    def test_rationale_contains_evidence_summary(self):
        result = _smart_mock_decide(
            {"name": "Jon Smith"}, {"name": "John Smith"},
            self._name("Jon Smith", "John Smith"),
            self._email("jon.smith@email.com", "jon.smith@email.com"),
            self._phone("+60156677889", "+60156677889"),
            None,
            self._addr("34 Bangsar KL", "34 Bangsar KL"),
        )
        # Rationale should mention the score and key evidence
        assert "score" in result["rationale"].lower() or "%" in result["rationale"]


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------
class TestRiskFlags:
    def test_no_flags_for_clean_merge(self):
        flags = _compute_risk_flags(
            name_score=95, dob_match=True, email_sim=0.95,
            phone_match=True, addr_sim=0.9, decision="MERGE",
        )
        assert flags == []

    def test_low_name_score_flagged(self):
        flags = _compute_risk_flags(
            name_score=60, dob_match=None, email_sim=0.5,
            phone_match=True, addr_sim=0.5, decision="MERGE",
        )
        assert "LOW_NAME_SCORE" in flags

    def test_dob_mismatch_flagged(self):
        flags = _compute_risk_flags(
            name_score=85, dob_match=False, email_sim=0.5,
            phone_match=True, addr_sim=0.8, decision="SPLIT",
        )
        assert "DOB_MISMATCH" in flags

    def test_dob_missing_flagged(self):
        flags = _compute_risk_flags(
            name_score=85, dob_match=None, email_sim=0.9,
            phone_match=True, addr_sim=0.9, decision="MERGE",
        )
        assert "DOB_MISSING" in flags

    def test_phone_and_address_mismatch_flagged(self):
        flags = _compute_risk_flags(
            name_score=75, dob_match=None, email_sim=0.5,
            phone_match=False, addr_sim=0.2, decision="MERGE",
        )
        assert "PHONE_AND_ADDRESS_MISMATCH" in flags

    def test_merge_with_low_name_score_flagged(self):
        flags = _compute_risk_flags(
            name_score=70, dob_match=None, email_sim=0.9,
            phone_match=True, addr_sim=0.9, decision="MERGE",
        )
        assert "MERGE_LOW_CONFIDENCE_NAME" in flags

    def test_split_with_high_name_score_flagged(self):
        """A SPLIT on a high name score is suspicious and worth human review."""
        flags = _compute_risk_flags(
            name_score=88, dob_match=False, email_sim=0.1,
            phone_match=False, addr_sim=0.1, decision="SPLIT",
        )
        assert "SPLIT_HIGH_NAME_SCORE" in flags

    def test_multiple_flags_returned(self):
        flags = _compute_risk_flags(
            name_score=60, dob_match=False, email_sim=0.1,
            phone_match=False, addr_sim=0.1, decision="MERGE",
        )
        assert len(flags) >= 3


# ---------------------------------------------------------------------------
# DecisionAgent: result schema now includes risk_flags
# ---------------------------------------------------------------------------
class TestDecisionAgentRiskFlags:
    @pytest.fixture
    def agent(self, tmp_path):
        return DecisionAgent(log_path=str(tmp_path / "decisions.jsonl"))

    def test_result_includes_risk_flags(self, agent):
        rec_a = {"source": "core_banking", "id": "CB001", "name": "Alice Test",
                 "dob": None, "email": "alice@test.com", "phone": "", "address": ""}
        rec_b = {"source": "credit_card",  "id": "CC001", "name": "Alyse Test",
                 "dob": "1990-01-01", "email": "alyse@test.com", "phone": "", "address": ""}
        result = agent.decide(rec_a, rec_b)
        assert "risk_flags" in result
        assert isinstance(result["risk_flags"], list)

    def test_log_entry_includes_risk_flags(self, agent):
        rec_a = {"source": "core_banking", "id": "CB001", "name": "James Thornton",
                 "dob": "1978-06-15", "email": "james.thornton@email.com",
                 "phone": "+60123456789", "address": "12 Jalan Bukit Bintang KL"}
        rec_b = {"source": "mortgage", "id": "MG001", "name": "James R. Thornton",
                 "dob": "1978-06-15", "email": "james.thornton@email.com",
                 "phone": "+60123456789", "address": "12 Jalan Bukit Bintang KL"}
        result = agent.decide(rec_a, rec_b)
        assert "risk_flags" in result["log_entry"]
        assert isinstance(result["log_entry"]["risk_flags"], list)

    def test_log_entry_includes_phone_fields(self, agent):
        rec_a = {"source": "core_banking", "id": "CB001", "name": "Jon Smith",
                 "dob": None, "email": "jon.smith@email.com",
                 "phone": "+60156677889", "address": "34 Bangsar South KL"}
        rec_b = {"source": "credit_card", "id": "CC107", "name": "John Smith",
                 "dob": "1983-04-25", "email": "jon.smith@email.com",
                 "phone": "+60156677889", "address": "34 Bangsar South KL"}
        result = agent.decide(rec_a, rec_b)
        log = result["log_entry"]
        assert "phone_match" in log
        assert "phone_similarity" in log
        assert log["phone_match"] is True

    def test_clearly_different_people_yields_split(self, agent):
        """Records with different surnames, DOBs, phones and emails must SPLIT."""
        rec_a = {"source": "core_banking", "id": "CB099", "name": "James Lim",
                 "dob": "1985-07-22", "email": "james.lim@gmail.com",
                 "phone": "+60112233001", "address": "10 Ampang KL"}
        rec_b = {"source": "credit_card",  "id": "CC099", "name": "James Tan",
                 "dob": "1991-03-14", "email": "james.tan@yahoo.com",
                 "phone": "+60198877665", "address": "52 Kepong KL"}
        result = agent.decide(rec_a, rec_b)
        assert result["decision"] == "SPLIT"

    def test_same_person_different_format_yields_merge(self, agent):
        """Same person, phone with and without country code, must MERGE."""
        rec_a = {"source": "core_banking", "id": "CB001", "name": "James Thornton",
                 "dob": "1978-06-15", "email": "james.thornton@email.com",
                 "phone": "+60123456789", "address": "12 Jalan Bukit Bintang KL"}
        rec_b = {"source": "mortgage",     "id": "MG001", "name": "James R. Thornton",
                 "dob": "March 15 1978", "email": "james.thornton@email.com",
                 "phone": "0123456789", "address": "12 Jalan Bukit Bintang KL"}
        result = agent.decide(rec_a, rec_b)
        assert result["decision"] == "MERGE"
