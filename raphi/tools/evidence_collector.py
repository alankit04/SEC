"""
Dynamic evidence collector for RAPHI.
Converts tool_executor retrieval results into typed EvidencePacket objects.
Called after execute_plan() has populated state.retrieval_results.
"""
from typing import TYPE_CHECKING
from datetime import datetime, timezone

if TYPE_CHECKING:
    from raphi.orchestrators.state import WorkflowState, EvidencePacket

def collect_dynamic_evidence(state: "WorkflowState") -> "WorkflowState":
    """
    Builds EvidencePacket objects from state.retrieval_results populated by execute_plan().
    Populates state.evidence_packets and state.errors.
    """
    from raphi.orchestrators.state import EvidencePacket
    import uuid

    evidence_packets = list(getattr(state, "evidence_packets", []) or [])
    errors = []

    if getattr(state, "retrieval_results", None):
        step_by_id = {getattr(step, "id", ""): step for step in getattr(state, "tool_plan", []) or []}

        def _now_iso() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _source_type_for(key: str, result) -> str:
            if isinstance(result, dict) and result.get("source_type"):
                return str(result.get("source_type"))
            step = step_by_id.get(str(key))
            tool_name = str(getattr(step, "tool_name", "") or "").lower()
            if "sec" in tool_name or "filing" in tool_name:
                return "sec"
            if "stock" in tool_name or "market" in tool_name or "quote" in tool_name:
                return "market"
            if "ml" in tool_name or "gnn" in tool_name or "signal" in tool_name:
                return "model"
            k = str(key).lower()
            if "sec" in k or "edgar" in k:
                return "sec"
            if any(x in k for x in ["market", "stock", "quote", "price"]):
                return "market"
            if any(x in k for x in ["web", "news", "firecrawl"]):
                return "web"
            if any(x in k for x in ["ml", "gnn", "signal"]):
                return "model"
            if "portfolio" in k:
                return "portfolio"
            if "memory" in k or "universe" in k or "ranking" in k:
                return "memory"
            return "memory"

        def _iter_items(result):
            if isinstance(result, list):
                if result:
                    for item in result:
                        yield item if isinstance(item, dict) else {"value": item, "raw_excerpt": str(item)}
                return
            if isinstance(result, dict):
                yield result
                return
            if result:
                yield {"value": result, "raw_excerpt": str(result)}

        def _make_packet(key: str, item: dict, source_type: str) -> EvidencePacket:
            evidence_id = str(uuid.uuid4())
            ticker = str(item.get("ticker") or (state.tickers[0] if state.tickers else "")).upper()
            now = _now_iso()
            retrieved_at = item.get("retrieved_at") or item.get("timestamp") or now
            url = item.get("url") or item.get("sec_url") or item.get("quote_url") or ""
            if source_type == "sec" and not url:
                url = "https://www.sec.gov/edgar/search/"
            claim = item.get("claim")
            if not claim:
                label = ticker or str(key)
                if source_type == "sec":
                    claim = f"SEC filing evidence for {label}"
                elif source_type == "market":
                    claim = f"Market evidence for {label}"
                elif source_type == "model":
                    claim = f"Model signal evidence for {label}"
                else:
                    claim = f"Workflow evidence for {label}"
            return EvidencePacket(
                evidence_id=evidence_id,
                claim_id=str(item.get("claim_id") or item.get("accession") or evidence_id),
                claim=claim,
                ticker=ticker,
                source_type=source_type,
                source_name=str(item.get("source_name") or item.get("provider") or source_type),
                provider=str(item.get("provider") or item.get("source_name") or source_type),
                url=url,
                canonical_url=item.get("canonical_url") or url,
                accession=item.get("accession"),
                form=item.get("form"),
                timestamp=item.get("timestamp") or retrieved_at or now,
                retrieved_at=retrieved_at,
                published_at=item.get("published_at"),
                filed_at=item.get("filed_at") or item.get("filed"),
                source_date=item.get("source_date") or item.get("filed_at") or item.get("filed"),
                value=item.get("value") or item.get("price") or item.get("pct") or item.get("direction"),
                confidence=str(item.get("confidence", "medium")),
                supports_claim=bool(item.get("supports_claim", True)),
                raw_excerpt=item.get("raw_excerpt") or item.get("description") or str(item)[:1200],
                query_used=state.user_query,
                snapshot_hash=item.get("snapshot_hash"),
                freshness_required=False,
                freshness_window_hours=None,
                freshness_status="unknown",
                refresh_attempted=False,
                refresh_successful=False,
                stale_reason=None,
            )

        for key, result in state.retrieval_results.items():
            if not result or (isinstance(result, dict) and result.get("error")):
                continue
            source_type = _source_type_for(str(key), result)
            for item in _iter_items(result):
                evidence_packets.append(_make_packet(str(key), item, source_type))
    if not evidence_packets:
        now = datetime.now(timezone.utc).isoformat()
        ticker = state.tickers[0] if getattr(state, "tickers", []) else ""
        evidence_packets.append(EvidencePacket(
            evidence_id=str(uuid.uuid4()),
            claim_id=f"{state.run_id}-limited-evidence",
            claim="RAPHI completed the workflow but live evidence providers returned no usable packet.",
            ticker=ticker,
            source_type="memory",
            source_name="RAPHI workflow state",
            provider="local",
            url="",
            canonical_url="",
            accession=None,
            form=None,
            timestamp=now,
            retrieved_at=now,
            published_at=None,
            filed_at=None,
            source_date=now,
            value=None,
            confidence="low",
            supports_claim=True,
            raw_excerpt=str(getattr(state, "retrieval_results", {}))[:1200],
            query_used=state.user_query,
            snapshot_hash=None,
            freshness_required=False,
            freshness_window_hours=None,
            freshness_status="unknown",
            refresh_attempted=False,
            refresh_successful=False,
            stale_reason=None,
        ))
    state.evidence_packets = evidence_packets
    if hasattr(state, "errors"):
        state.errors.extend(errors)
    else:
        state.errors = errors
    return state
