from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

@dataclass
class FreshnessRequirement:
    intent: str
    requires_freshness: bool
    max_age_hours: Optional[int]
    reason: str

@dataclass
class CitationFreshnessResult:
    citation_id: str
    url: str
    source_type: str
    retrieved_at: str
    published_at: Optional[str]
    filed_at: Optional[str]
    max_age_hours: Optional[int]
    age_hours: Optional[float]
    freshness_status: str
    stale_reason: Optional[str]
    refresh_attempted: bool
    refresh_successful: bool

def infer_freshness_requirement(query: str, intent: str) -> FreshnessRequirement:
    q = (query or "").lower()
    # Expanded freshness-sensitive terms
    freshness_terms = [
        "latest", "current", "recent", "today", "now", "this week", "this month", "updated", "up to date", "new", "just filed", "trending", "movers", "2026", "should i", "buy", "sell", "hold", "long", "short"
    ]
    requires_freshness = any(term in q for term in freshness_terms)
    # Set max_age_hours based on query/intent
    if any(term in q for term in ["today", "now", "current"]):
        max_age_hours = 24
    elif intent == "market_snapshot":
        max_age_hours = 24
    elif intent == "trending_stocks":
        max_age_hours = 72
    elif intent == "recommendation":
        max_age_hours = 72
    elif intent == "latest_filing":
        max_age_hours = 168
    elif requires_freshness:
        max_age_hours = 72
    else:
        max_age_hours = None
    reason = f"Intent: {intent}, Query: {query}"
    return FreshnessRequirement(intent, requires_freshness, max_age_hours, reason)

from datetime import timezone
def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d")
            except ValueError:
                return None
    return None

def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def set_field(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)

def evaluate_citation_freshness(citation: dict, requirement: FreshnessRequirement) -> CitationFreshnessResult:
    now = datetime.now(timezone.utc)
    published_at = get_field(citation, "published_at")
    filed_at = get_field(citation, "filed_at")
    source_date = get_field(citation, "source_date")
    retrieved_at = get_field(citation, "retrieved_at")
    timestamp = get_field(citation, "timestamp")
    source_type = get_field(citation, "source_type")
    citation_id = get_field(citation, "evidence_id")
    url = get_field(citation, "url")
    max_age_hours = requirement.max_age_hours
    age_hours = None
    freshness_status = "unknown"
    stale_reason = None
    refresh_attempted = False
    refresh_successful = False

    # A. Not time sensitive
    if not requirement.requires_freshness:
        freshness_status = "not_time_sensitive"
        return CitationFreshnessResult(
            citation_id=citation_id,
            url=url,
            source_type=source_type,
            retrieved_at=retrieved_at,
            published_at=published_at,
            filed_at=filed_at,
            max_age_hours=max_age_hours,
            age_hours=age_hours,
            freshness_status=freshness_status,
            stale_reason=stale_reason,
            refresh_attempted=refresh_attempted,
            refresh_successful=refresh_successful
        )

    # Source-type-specific freshness logic
    date_used = None
    # SEC evidence: only filed_at, filing_date, source_date are valid for freshness
    if source_type == "sec":
        filing_date = get_field(citation, "filing_date")
        for d in [filed_at, filing_date, source_date]:
            dt = _parse_datetime(d)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_used = dt
                break
        if not date_used:
            freshness_status = "unknown"
            stale_reason = "missing_sec_filing_date"
            return CitationFreshnessResult(
                citation_id=citation_id,
                url=url,
                source_type=source_type,
                retrieved_at=retrieved_at,
                published_at=published_at,
                filed_at=filed_at,
                max_age_hours=max_age_hours,
                age_hours=age_hours,
                freshness_status=freshness_status,
                stale_reason=stale_reason,
                refresh_attempted=refresh_attempted,
                refresh_successful=refresh_successful
            )
    # Market evidence: retrieved_at or timestamp
    elif source_type == "market":
        for d in [retrieved_at, timestamp]:
            dt = _parse_datetime(d)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_used = dt
                break
    # Web/news/company evidence: prefer published_at, fallback to source_date, then retrieved_at
    elif source_type in {"web", "news", "company"}:
        for d in [published_at, source_date, retrieved_at]:
            dt = _parse_datetime(d)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_used = dt
                break
    # Model evidence: timestamp, retrieved_at
    elif source_type == "model":
        for d in [timestamp, retrieved_at]:
            dt = _parse_datetime(d)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_used = dt
                break
    # Memory evidence: memory timestamp, source_date, retrieved_at
    elif source_type == "memory":
        for d in [timestamp, source_date, retrieved_at]:
            dt = _parse_datetime(d)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_used = dt
                break
    # Fallback: try all
    else:
        for d in [published_at, filed_at, source_date, retrieved_at, timestamp]:
            dt = _parse_datetime(d)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_used = dt
                break

    # No usable date
    if not date_used:
        freshness_status = "unknown"
        if source_type == "sec":
            stale_reason = "missing_sec_filing_date"
        else:
            stale_reason = "missing_source_date"
        return CitationFreshnessResult(
            citation_id=citation_id,
            url=url,
            source_type=source_type,
            retrieved_at=retrieved_at,
            published_at=published_at,
            filed_at=filed_at,
            max_age_hours=max_age_hours,
            age_hours=age_hours,
            freshness_status=freshness_status,
            stale_reason=stale_reason,
            refresh_attempted=refresh_attempted,
            refresh_successful=refresh_successful
        )

    # Age calculation
    age_hours = (now - date_used).total_seconds() / 3600.0
    if max_age_hours is not None:
        if age_hours <= max_age_hours:
            freshness_status = "fresh"
        else:
            freshness_status = "stale"
            stale_reason = "older_than_freshness_window"
    else:
        freshness_status = "not_time_sensitive"

    return CitationFreshnessResult(
        citation_id=citation_id,
        url=url,
        source_type=source_type,
        retrieved_at=retrieved_at,
        published_at=published_at,
        filed_at=filed_at,
        max_age_hours=max_age_hours,
        age_hours=age_hours,
        freshness_status=freshness_status,
        stale_reason=stale_reason,
        refresh_attempted=refresh_attempted,
        refresh_successful=refresh_successful
    )

def should_refresh_citation(result: CitationFreshnessResult) -> bool:
    if result.freshness_status == "stale":
        return True
    if result.freshness_status == "unknown":
        return True
    return False
