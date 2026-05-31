from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
import uuid
import datetime

@dataclass
class ToolPlanStep:
    id: str
    tool_name: str
    purpose: str
    required: bool
    args: dict
    expected_output: str

@dataclass
class ToolTrace:
    tool_name: str
    args: dict
    started_at: str
    ended_at: str
    latency_ms: int
    ok: bool
    error: Optional[str]
    output_summary: str
    status: str = ""

@dataclass
class EvidencePacket:
    evidence_id: str
    claim_id: str
    claim: str
    ticker: str
    source_type: str
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
    confidence: str
    supports_claim: bool
    raw_excerpt: str
    query_used: str
    snapshot_hash: Optional[str]
    freshness_required: bool
    freshness_window_hours: Optional[int]
    freshness_status: str
    refresh_attempted: bool
    refresh_successful: bool
    stale_reason: Optional[str]

@dataclass
class ClaimCitationMapEntry:
    claim_id: str
    claim: str
    evidence_ids: List[str]
    citation_urls: List[str]
    support_status: str
    freshness_status: str
    citation_age_hours: Optional[float]
    stale_reason: Optional[str]
    notes: str

@dataclass
class ModelSignal:
    ticker: str
    model_name: str
    model_version: str
    signal: str
    confidence: Optional[float]
    source: str
    features: dict
    timestamp: str
    limitations: List[str]

@dataclass
class Recommendation:
    decision: str
    confidence: Optional[float]
    confidence_source: str
    rationale: List[str]
    risk_framing: List[str]
    allowed: bool
    downgrade_reason: Optional[str]

@dataclass
class ReflectionResult:
    passed: bool
    issues: List[dict]
    retryable: bool
    suggested_plan_changes: List[dict]

@dataclass
class WorkflowState:
    run_id: str
    user_query: str
    intent: str
    risk_class: str
    entities: List[str]
    tickers: List[str]
    unknown_tickers: List[str] = field(default_factory=list)
    validated_tickers: List[str] = field(default_factory=list)
    invalid_tickers: List[str] = field(default_factory=list)
    newly_registered_tickers: List[str] = field(default_factory=list)
    ticker_registration_status: Dict[str, Any] = field(default_factory=dict)
    gnn_registration_status: Dict[str, Any] = field(default_factory=dict)
    memory_updates: List[dict] = field(default_factory=list)
    universe: List[str] = field(default_factory=list)
    time_window: str = ""
    workflow_name: str = ""
    discovery_candidates: List[dict] = field(default_factory=list)
    discovery_queries_used: List[str] = field(default_factory=list)
    perception: dict = field(default_factory=dict)
    tool_plan: List[ToolPlanStep] = field(default_factory=list)
    tool_trace: List[ToolTrace] = field(default_factory=list)
    retrieval_results: dict = field(default_factory=dict)
    ranking_table: List[dict] = field(default_factory=list)
    evidence_packets: List[EvidencePacket] = field(default_factory=list)
    claim_citation_map: List[ClaimCitationMapEntry] = field(default_factory=list)
    model_signals: List[ModelSignal] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    recommendation: Optional[Recommendation] = None
    reflection: Optional[ReflectionResult] = None
    governance_status: dict = field(default_factory=dict)
    eval_status: dict = field(default_factory=dict)
    citation_freshness_status: dict = field(default_factory=dict)
    final_answer: str = ""
    errors: List[dict] = field(default_factory=list)

    def to_dict(self):
        for trace in self.tool_trace:
            if isinstance(trace, dict):
                trace.setdefault("status", "success" if trace.get("ok") else "failed")
            elif not getattr(trace, "status", ""):
                trace.status = "success" if trace.ok else "failed"
        data = asdict(self)
        # Always expose detected_tickers in the API contract
        detected = self.perception.get("detected_tickers") if self.perception else None
        data["detected_tickers"] = detected if detected is not None else self.tickers
        # Always include ticker status fields, even if empty
        for k in [
            "invalid_tickers",
            "validated_tickers",
            "newly_registered_tickers",
            "ticker_registration_status",
            "uncertainty_flags",
            "errors",
        ]:
            if k not in data:
                if k.endswith("_tickers") or k == "uncertainty_flags" or k == "errors":
                    data[k] = []
                else:
                    data[k] = {}
        return data

    def add_trace(self, trace: ToolTrace):
        if isinstance(trace, dict):
            trace.setdefault("status", "success" if trace.get("ok") else "failed")
        elif not getattr(trace, "status", ""):
            trace.status = "success" if trace.ok else "failed"
        self.tool_trace.append(trace)

    def add_error(self, error: dict):
        self.errors.append(error)

    def add_uncertainty_flag(self, flag: str):
        self.uncertainty_flags.append(flag)

    def has_model_provenance(self):
        return bool(self.model_signals)

    def has_evidence_for_claims(self):
        return bool(self.evidence_packets)

    def requires_governance(self):
        return self.intent in ["recommendation", "investment_memo"]

    def has_fresh_citations_for_current_claims(self):
        for entry in self.claim_citation_map:
            if entry.freshness_status not in ("fresh", "not_time_sensitive"):
                return False
        return True
