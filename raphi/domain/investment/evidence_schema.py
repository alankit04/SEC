from dataclasses import dataclass
from typing import Optional, List, Union, Dict

@dataclass
class EvidencePacket:
    evidence_id: str
    claim_id: str
    claim: str
    ticker: str
    source_type: str  # sec | market | web | model | portfolio | memory
    source_name: str
    provider: str
    url: str
    canonical_url: str
    accession: Optional[str]
    form: Optional[str]
    timestamp: str
    retrieved_at: str
    published_at: Optional[str]
    filed_at: Optional[str]
    source_date: Optional[str]
    value: Optional[Union[str, float, int]]
    confidence: str  # high | medium | low
    supports_claim: bool
    raw_excerpt: str
    query_used: str
    snapshot_hash: Optional[str]
    freshness_required: bool
    freshness_window_hours: Optional[int]
    freshness_status: str  # fresh | stale | unknown | not_time_sensitive
    refresh_attempted: bool
    refresh_successful: bool
    stale_reason: Optional[str]

@dataclass
class ClaimCitationMapEntry:
    claim_id: str
    claim: str
    evidence_ids: List[str]
    citation_urls: List[str]
    support_status: str  # supported | partially_supported | unsupported
    freshness_status: str  # fresh | stale | unknown | not_time_sensitive
    citation_age_hours: Optional[float]
    stale_reason: Optional[str]
    notes: str

@dataclass
class ModelSignal:
    ticker: str
    model_name: str
    model_version: str
    signal: str  # bullish | bearish | neutral | unavailable
    confidence: Optional[float]
    source: str  # ml_model | gnn_model | deterministic_score
    features: Dict
    timestamp: str
    limitations: List[str]

@dataclass
class Recommendation:
    decision: str  # research_only | bullish_watch | bearish_watch | buy | sell | hold | long | short | none
    confidence: Optional[float]
    confidence_source: str
    rationale: List[str]
    risk_framing: List[str]
    allowed: bool
    downgrade_reason: Optional[str]
