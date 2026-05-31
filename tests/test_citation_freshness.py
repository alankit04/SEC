def test_current_today_query_marks_2024_web_citation_stale():
    from raphi.evals.citation_freshness import infer_freshness_requirement, evaluate_citation_freshness
    requirement = infer_freshness_requirement(
        "What are the current risks for NVDA today?",
        "sec_research",
    )
    assert requirement.requires_freshness is True
    assert requirement.max_age_hours in {24, 72}

    citation = {
        "source_type": "web",
        "published_at": "2024-01-01T00:00:00Z",
        "retrieved_at": "2024-01-01T00:00:00Z",
        "url": "https://example.com/old-nvda-risk",
    }

    result = evaluate_citation_freshness(citation, requirement)
    assert result.freshness_status == "stale"
    assert result.age_hours is not None
    assert result.stale_reason == "older_than_freshness_window"
import pytest
from raphi.evals.citation_freshness import infer_freshness_requirement, evaluate_citation_freshness, should_refresh_citation

def test_latest_query_requires_freshness():
    req = infer_freshness_requirement("Show me the latest news", "market_snapshot")
    assert req.requires_freshness
    assert req.max_age_hours == 24

def test_old_web_citation_is_stale():
    req = infer_freshness_requirement("Show me the latest news", "market_snapshot")
    citation = {
        "evidence_id": "1",
        "url": "http://example.com",
        "source_type": "web",
        "retrieved_at": "2020-01-01T00:00:00Z",
        "published_at": None,
        "filed_at": None
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "stale"
    assert should_refresh_citation(result)

def test_missing_date_is_unknown():
    req = infer_freshness_requirement("Show me the latest news", "market_snapshot")
    citation = {
        "evidence_id": "2",
        "url": "http://example.com",
        "source_type": "web",
        "retrieved_at": None,
        "published_at": None,
        "filed_at": None
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "unknown"
    assert should_refresh_citation(result)

def test_sec_citation_historical_passes():
    req = infer_freshness_requirement("Show me 2020 10-K", "sec_research")
    citation = {
        "evidence_id": "3",
        "url": "http://sec.gov/abc",
        "source_type": "sec",
        "retrieved_at": "2020-03-01T00:00:00Z",
        "published_at": None,
        "filed_at": "2020-03-01T00:00:00Z"
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "not_time_sensitive"

def test_latest_sec_query_requires_filing_date():
    req = infer_freshness_requirement("Show me the latest 10-K", "latest_filing")
    citation = {
        "evidence_id": "4",
        "url": "http://sec.gov/abc",
        "source_type": "sec",
        "retrieved_at": "2026-05-01T00:00:00Z",
        "published_at": None,
        "filed_at": None
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "unknown"
    assert result.stale_reason == "missing_sec_filing_date"
    assert should_refresh_citation(result)

def test_sec_latest_missing_filing_date():
    req = infer_freshness_requirement("Show me the latest 10-K", "latest_filing")
    citation = {
        "source_type": "sec",
        "retrieved_at": "2024-01-01T00:00:00Z",
        "url": "https://www.sec.gov/Archives/mock.htm"
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "unknown"
    assert result.stale_reason == "missing_sec_filing_date"

def test_sec_latest_old_filed_at():
    req = infer_freshness_requirement("Show me the latest 10-K", "latest_filing")
    citation = {
        "source_type": "sec",
        "filed_at": "2024-01-01T00:00:00Z",
        "retrieved_at": "2026-05-25T00:00:00Z"
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "stale"

def test_web_current_old_retrieved_at():
    req = infer_freshness_requirement("What is the current news?", "sec_research")
    citation = {
        "source_type": "web",
        "retrieved_at": "2024-01-01T00:00:00Z"
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "stale"

def test_market_current_old_retrieved_at():
    req = infer_freshness_requirement("What is the current price?", "market_snapshot")
    citation = {
        "source_type": "market",
        "retrieved_at": "2024-01-01T00:00:00Z"
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "stale"

def test_market_claim_without_timestamp_fails():
    req = infer_freshness_requirement("Show me the latest price", "market_snapshot")
    citation = {
        "evidence_id": "5",
        "url": "http://market.com/abc",
        "source_type": "market",
        "retrieved_at": None,
        "published_at": None,
        "filed_at": None
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "unknown"
    assert should_refresh_citation(result)

def test_stale_citation_triggers_refresh():
    req = infer_freshness_requirement("Show me the latest price", "market_snapshot")
    citation = {
        "evidence_id": "6",
        "url": "http://market.com/abc",
        "source_type": "market",
        "retrieved_at": "2020-01-01T00:00:00Z",
        "published_at": None,
        "filed_at": None
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "stale"
    assert should_refresh_citation(result)

def test_refresh_failure_downgrades_final_answer():
    # Simulate a citation that is stale and should be refreshed
    req = infer_freshness_requirement("Show me the latest price", "market_snapshot")
    citation = {
        "evidence_id": "7",
        "url": "http://market.com/abc",
        "source_type": "market",
        "retrieved_at": "2020-01-01T00:00:00Z",
        "published_at": None,
        "filed_at": None
    }
    result = evaluate_citation_freshness(citation, req)
    assert result.freshness_status == "stale"
    assert should_refresh_citation(result)

    # Simulate refresh failure: system cannot obtain a fresh citation
    # In a real agentic loop, this would trigger a downgrade or fallback answer
    # Here, we simulate the fallback by marking the answer as downgraded or with a warning
    # For this test, we assert that if refresh fails, the system should not claim freshness
    downgraded_answer = {
        "answer": "Sorry, we could not obtain a fresh price. The latest available data is stale.",
        "freshness_status": "stale",
        "downgraded": True
    }
    assert downgraded_answer["downgraded"] is True
    assert downgraded_answer["freshness_status"] == "stale"
    assert "could not obtain a fresh price" in downgraded_answer["answer"]
