"""Neo4j-backed permanent graph memory for RAPHI.

The memory layer stores durable facts as graph nodes, reinforces repeated facts
by increasing frequency/importance, and retrieves a compact context slice for
future AI calls. It uses Neo4j's HTTP transactional endpoint so the project does
not need an extra Python driver dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

try:
    from paths import PROJECT_ROOT
except ImportError:  # pragma: no cover - package import path
    from backend.paths import PROJECT_ROOT


DEFAULT_PROJECT_ID = os.environ.get("RAPHI_MEMORY_PROJECT_ID", PROJECT_ROOT.name)
DEFAULT_USER_ID = os.environ.get("RAPHI_MEMORY_USER_ID", "local-user")
LOCAL_MEMORY_DIR = PROJECT_ROOT / ".raphi_memory"
LOCAL_MEMORY_FILE = LOCAL_MEMORY_DIR / "memory.json"

MEMORY_KEYWORDS = {
    "neo4j": "Technology",
    "memgraph": "Technology",
    "gnn": "Technology",
    "graph neural network": "Technology",
    "graph database": "Technology",
    "memory": "Concept",
    "permanent memory": "Concept",
    "sec": "Domain",
    "edgar": "Domain",
    "xbrl": "Domain",
    "fastapi": "Technology",
    "a2a": "Technology",
    "mcp": "Technology",
    "raphi": "Project",
    "ticker": "Domain",
    "conviction ledger": "Feature",
    "portfolio": "Feature",
    "user preference": "Concept",
}

STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "before",
    "being", "between", "but", "can", "does", "done", "for", "from",
    "have", "into", "just", "like", "more", "need", "not", "that",
    "the", "then", "this", "through", "time", "want", "what", "when",
    "where", "with", "would", "you", "your",
}


class GraphMemoryError(RuntimeError):
    """Raised when Neo4j is configured but unavailable or returns an error."""


@dataclass
class MemoryCandidate:
    kind: str
    text: str
    importance: float = 0.55
    confidence: float = 0.72
    entities: list[dict[str, str]] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"[^\w\s./:-]", "", text)
    return text[:700]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_.:/-]{3,}", text.lower())
    return list(dict.fromkeys(t for t in tokens if t not in STOP_WORDS))[:24]


def memory_id(kind: str, text: str) -> str:
    return f"mem-{kind}-{stable_hash(normalize_text(text))}"


def entity_id(entity_type: str, name: str) -> str:
    return f"ent-{entity_type.lower()}-{stable_hash(name.lower())}"


def clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def metadata_to_json(metadata: Optional[dict[str, Any]]) -> str:
    """Serialize arbitrary request metadata into a Neo4j-safe node property."""
    if not metadata:
        return "{}"
    return json.dumps(metadata, sort_keys=True, default=str)


def extract_entities(text: str) -> list[dict[str, str]]:
    entities: dict[str, dict[str, str]] = {}

    for match in re.findall(r"(?:[\w.-]+/)*[\w.-]+\.(?:py|js|json|md|html|txt|tsx|ts)", text):
        name = match.strip()
        entities[entity_id("File", name)] = {"id": entity_id("File", name), "type": "File", "name": name}

    for match in re.findall(r"\b20\d{2}q[1-4]\b", text, flags=re.IGNORECASE):
        name = match.lower()
        entities[entity_id("Quarter", name)] = {
            "id": entity_id("Quarter", name),
            "type": "Quarter",
            "name": name,
        }

    for token in re.findall(r"\b[A-Z]{2,5}\b", text):
        if token not in {"HTTP", "JSON", "HTML", "API", "FAST", "SEC", "GNN"}:
            entities[entity_id("Ticker", token)] = {
                "id": entity_id("Ticker", token),
                "type": "Ticker",
                "name": token,
            }

    lowered = text.lower()
    for keyword, entity_type in MEMORY_KEYWORDS.items():
        if keyword in lowered:
            display_names = {
                "neo4j": "Neo4j",
                "memgraph": "Memgraph",
                "fastapi": "FastAPI",
                "gnn": "GNN",
                "sec": "SEC",
                "mcp": "MCP",
                "a2a": "A2A",
            }
            display = display_names.get(keyword, keyword)
            entities[entity_id(entity_type, display)] = {
                "id": entity_id(entity_type, display),
                "type": entity_type,
                "name": display,
            }

    return list(entities.values())[:24]


def classify_sentence(sentence: str) -> Optional[tuple[str, float, float]]:
    lowered = sentence.lower()
    if re.search(r"\bi\s+(want|need|prefer|like|dont|don't|do not)\b", lowered):
        return "user_preference", 0.82, 0.86
    if "no scaffolding" in lowered or "actual implementation" in lowered:
        return "user_preference", 0.9, 0.9
    if "main" in lowered and "purpose" in lowered:
        return "user_goal", 0.86, 0.84
    if any(term in lowered for term in ("must", "should", "before you", "do it end to end")):
        return "task_requirement", 0.76, 0.78
    if any(term in lowered for term in ("neo4j", "memgraph", "gnn", "graph database", "permanent memory")):
        return "architecture_decision", 0.78, 0.8
    if re.search(r"\b[\w.-]+\.(py|js|json|md|html|txt|tsx|ts)\b", sentence):
        return "project_fact", 0.64, 0.74
    return None


def extract_memories(
    user_text: str,
    assistant_text: str = "",
    source: str = "interaction",
    base_importance: float = 0.55,
) -> list[MemoryCandidate]:
    """Extract durable memory candidates from an interaction."""
    candidates: list[MemoryCandidate] = []
    seen: set[str] = set()

    user_chunks = re.split(r"[\n.!?]+", user_text)
    for raw in user_chunks:
        sentence = clip(raw, 420)
        if not sentence:
            continue
        classified = classify_sentence(sentence)
        if not classified:
            continue
        kind, importance, confidence = classified
        key = f"{kind}:{normalize_text(sentence)}"
        if key in seen:
            continue
        seen.add(key)
        candidates.append(MemoryCandidate(
            kind=kind,
            text=sentence,
            importance=max(base_importance, importance),
            confidence=confidence,
            entities=extract_entities(sentence),
        ))
        if kind != "architecture_decision" and any(
            term in sentence.lower()
            for term in ("neo4j", "memgraph", "gnn", "graph database", "permanent memory")
        ):
            arch_key = f"architecture_decision:{normalize_text(sentence)}"
            if arch_key not in seen:
                seen.add(arch_key)
                candidates.append(MemoryCandidate(
                    kind="architecture_decision",
                    text=sentence,
                    importance=max(base_importance, 0.78),
                    confidence=0.8,
                    entities=extract_entities(sentence),
                ))

    combined = f"{user_text}\n{assistant_text}"
    entities = extract_entities(combined)
    if entities:
        entity_text = ", ".join(f"{e['type']}:{e['name']}" for e in entities[:12])
        candidates.append(MemoryCandidate(
            kind="interaction_entities",
            text=f"Interaction referenced {entity_text}",
            importance=max(base_importance, 0.58),
            confidence=0.7,
            entities=entities,
        ))

    if not candidates and user_text.strip():
        candidates.append(MemoryCandidate(
            kind="interaction_summary",
            text=clip(user_text, 360),
            importance=base_importance,
            confidence=0.62,
            entities=entities,
        ))

    return candidates[:12]


class Neo4jGraphMemory:
    """Permanent project memory backed by Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: float = 8.0,
    ) -> None:
        self.uri = (uri or os.environ.get("NEO4J_URI") or "http://localhost:7474").rstrip("/")
        self.user = user or os.environ.get("NEO4J_USER") or "neo4j"
        self.password = password if password is not None else os.environ.get("NEO4J_PASSWORD", "")
        self.database = database or os.environ.get("NEO4J_DATABASE") or "neo4j"
        self.project_id = project_id or DEFAULT_PROJECT_ID
        self.user_id = user_id or DEFAULT_USER_ID
        self.timeout = timeout
        self._schema_ready = False

    @property
    def configured(self) -> bool:
        return bool(self.password)

    def status(self) -> dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "available": False,
                "project_id": self.project_id,
                "message": "Set NEO4J_PASSWORD to enable permanent graph memory.",
            }
        try:
            self._commit([{"statement": "RETURN 1 AS ok"}])
            return {
                "configured": True,
                "available": True,
                "project_id": self.project_id,
                "database": self.database,
                "uri": self.uri,
            }
        except Exception as exc:
            return {
                "configured": True,
                "available": False,
                "project_id": self.project_id,
                "database": self.database,
                "uri": self.uri,
                "error": str(exc),
            }

    def setup_schema(self) -> None:
        if not self.configured or self._schema_ready:
            return
        now = utc_now()
        schema_statements = [
            "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT interaction_id IF NOT EXISTS FOR (i:Interaction) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX memory_kind IF NOT EXISTS FOR (m:Memory) ON (m.kind)",
            "CREATE INDEX memory_last_seen IF NOT EXISTS FOR (m:Memory) ON (m.last_seen)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]
        for statement in schema_statements:
            self._commit([{"statement": statement}])

        self._commit([
            {
                "statement": """
                MERGE (p:Project {id: $project_id})
                SET p.name = $project_id,
                    p.root = $root,
                    p.updated_at = $now
                """,
                "parameters": {
                    "project_id": self.project_id,
                    "root": str(PROJECT_ROOT),
                    "now": now,
                },
            },
            {
                "statement": """
                MERGE (u:User {id: $user_id})
                SET u.updated_at = $now
                """,
                "parameters": {"user_id": self.user_id, "now": now},
            },
        ])
        self._schema_ready = True

    def remember_interaction(
        self,
        user_text: str,
        assistant_text: str = "",
        source: str = "interaction",
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 0.55,
    ) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "stored": 0, "reason": "neo4j_not_configured"}
        self.setup_schema()

        now = utc_now()
        interaction_id = f"int-{uuid.uuid4().hex}"
        candidates = extract_memories(user_text, assistant_text, source, importance)
        statements: list[dict[str, Any]] = [
            {
                "statement": """
                MATCH (p:Project {id: $project_id})
                MATCH (u:User {id: $user_id})
                CREATE (i:Interaction {
                    id: $interaction_id,
                    source: $source,
                    user_text: $user_text,
                    assistant_text: $assistant_text,
                    summary: $summary,
                    metadata_json: $metadata_json,
                    created_at: $now
                })
                MERGE (u)-[:SAID]->(i)
                MERGE (i)-[:IN_PROJECT]->(p)
                """,
                "parameters": {
                    "project_id": self.project_id,
                    "user_id": self.user_id,
                    "interaction_id": interaction_id,
                    "source": source,
                    "user_text": clip(user_text, 6000),
                    "assistant_text": clip(assistant_text, 6000),
                    "summary": clip(user_text, 240),
                    "metadata_json": metadata_to_json(metadata),
                    "now": now,
                },
            }
        ]

        stored = []
        for candidate in candidates:
            mid = memory_id(candidate.kind, candidate.text)
            stored.append({"id": mid, "kind": candidate.kind, "text": candidate.text})
            statements.append({
                "statement": """
                MATCH (p:Project {id: $project_id})
                MATCH (i:Interaction {id: $interaction_id})
                MERGE (m:Memory {id: $memory_id})
                ON CREATE SET
                    m.kind = $kind,
                    m.text = $text,
                    m.summary = $summary,
                    m.normalized_text = $normalized_text,
                    m.importance = $importance,
                    m.confidence = $confidence,
                    m.frequency = 1,
                    m.first_seen = $now,
                    m.last_seen = $now
                ON MATCH SET
                    m.frequency = coalesce(m.frequency, 0) + 1,
                    m.last_seen = $now,
                    m.importance = CASE
                        WHEN coalesce(m.importance, 0.0) > $importance
                        THEN coalesce(m.importance, 0.0)
                        ELSE $importance
                    END,
                    m.confidence = CASE
                        WHEN coalesce(m.confidence, 0.0) > $confidence
                        THEN coalesce(m.confidence, 0.0)
                        ELSE $confidence
                    END
                MERGE (i)-[:CREATED]->(m)
                MERGE (m)-[:IN_PROJECT]->(p)
                """,
                "parameters": {
                    "project_id": self.project_id,
                    "interaction_id": interaction_id,
                    "memory_id": mid,
                    "kind": candidate.kind,
                    "text": candidate.text,
                    "summary": clip(candidate.text, 180),
                    "normalized_text": normalize_text(candidate.text),
                    "importance": float(candidate.importance),
                    "confidence": float(candidate.confidence),
                    "now": now,
                },
            })
            for entity in candidate.entities:
                statements.append({
                    "statement": """
                    MATCH (m:Memory {id: $memory_id})
                    MATCH (p:Project {id: $project_id})
                    MERGE (e:Entity {id: $entity_id})
                    ON CREATE SET
                        e.type = $entity_type,
                        e.name = $entity_name,
                        e.created_at = $now
                    ON MATCH SET e.last_seen = $now
                    MERGE (m)-[r:MENTIONS]->(e)
                    ON CREATE SET r.weight = 1
                    ON MATCH SET r.weight = coalesce(r.weight, 0) + 1
                    MERGE (e)-[:IN_PROJECT]->(p)
                    """,
                    "parameters": {
                        "project_id": self.project_id,
                        "memory_id": mid,
                        "entity_id": entity["id"],
                        "entity_type": entity["type"],
                        "entity_name": entity["name"],
                        "now": now,
                    },
                })

        self._commit(statements)
        return {"ok": True, "interaction_id": interaction_id, "stored": len(stored), "memories": stored}

    def retrieve_context(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not self.configured or not query.strip():
            return []
        self.setup_schema()
        tokens = tokenize(query)
        entities = [e["name"].lower() for e in extract_entities(query)]
        if not tokens and not entities:
            return []

        result = self._commit([{
            "statement": """
            MATCH (m:Memory)-[:IN_PROJECT]->(:Project {id: $project_id})
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            WITH m, collect(e.name) AS entities, collect(toLower(e.name)) AS entity_names
            WITH m, entities,
                 reduce(text_score = 0.0, token IN $tokens |
                    text_score + CASE
                        WHEN toLower(coalesce(m.text, '') + ' ' + coalesce(m.summary, '') + ' ' + coalesce(m.kind, '')) CONTAINS token
                        THEN 1.0 ELSE 0.0 END
                 ) +
                 reduce(entity_score = 0.0, entity IN entity_names |
                    entity_score + CASE WHEN entity IN $entities THEN 2.0 ELSE 0.0 END
                 ) +
                 coalesce(m.importance, 0.0) +
                 (coalesce(m.frequency, 1) * 0.08) AS score
            WHERE score > 0.35
            RETURN m.id AS id,
                   m.kind AS kind,
                   m.text AS text,
                   m.summary AS summary,
                   m.importance AS importance,
                   m.confidence AS confidence,
                   m.frequency AS frequency,
                   m.last_seen AS last_seen,
                   entities AS entities,
                   score AS score
            ORDER BY score DESC, m.last_seen DESC
            LIMIT $limit
            """,
            "parameters": {
                "project_id": self.project_id,
                "tokens": tokens,
                "entities": entities,
                "limit": int(max(1, min(limit, 25))),
            },
        }])
        rows = []
        for record in self._records(result, 0):
            row = {k: v for k, v in record.items()}
            rows.append(row)
        return rows

    def format_context(self, memories: list[dict[str, Any]]) -> str:
        lines = []
        for item in memories:
            freq = item.get("frequency") or 1
            kind = item.get("kind") or "memory"
            text = item.get("text") or item.get("summary") or ""
            entities = ", ".join(item.get("entities") or [])
            suffix = f" | entities: {entities}" if entities else ""
            lines.append(f"- [{kind}, seen {freq}x] {clip(text, 260)}{suffix}")
        return "\n".join(lines)

    def reinforce(self, memory_id_value: str, delta: float = 0.1) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "reason": "neo4j_not_configured"}
        self.setup_schema()
        self._commit([{
            "statement": """
            MATCH (m:Memory {id: $memory_id})
            SET m.importance = coalesce(m.importance, 0.0) + $delta,
                m.last_seen = $now
            RETURN m.id AS id, m.importance AS importance
            """,
            "parameters": {"memory_id": memory_id_value, "delta": float(delta), "now": utc_now()},
        }])
        return {"ok": True, "memory_id": memory_id_value}

    def _commit(self, statements: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.configured:
            raise GraphMemoryError("Neo4j is not configured; set NEO4J_PASSWORD.")
        endpoint = f"{self.uri}/db/{self.database}/tx/commit"
        payload = {"statements": statements}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload, auth=(self.user, self.password))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GraphMemoryError(f"Neo4j HTTP request failed: {exc}") from exc

        data = response.json()
        errors = data.get("errors") or []
        if errors:
            message = "; ".join(e.get("message", str(e)) for e in errors)
            raise GraphMemoryError(message)
        return data

    @staticmethod
    def _records(result: dict[str, Any], result_index: int) -> list[dict[str, Any]]:
        results = result.get("results") or []
        if result_index >= len(results):
            return []
        columns = results[result_index].get("columns") or []
        rows = []
        for row in results[result_index].get("data") or []:
            values = row.get("row") or []
            rows.append(dict(zip(columns, values)))
        return rows


class LocalGraphMemory:
    """Durable local graph-memory fallback used when Neo4j is unavailable.

    It preserves the same public methods as Neo4jGraphMemory so chat/A2A memory
    stays real in local development. Nodes are represented as JSON records with
    extracted entities, frequency, importance, confidence, and timestamps.
    """

    def __init__(
        self,
        path: Optional[os.PathLike[str] | str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        self.path = Path(path) if path is not None else LOCAL_MEMORY_FILE
        self.project_id = project_id or DEFAULT_PROJECT_ID
        self.user_id = user_id or DEFAULT_USER_ID

    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "available": True,
            "backend": "local_json",
            "project_id": self.project_id,
            "path": str(self.path),
            "memory_count": len(self._load().get("memories", {})),
        }

    def remember_interaction(
        self,
        user_text: str,
        assistant_text: str = "",
        source: str = "interaction",
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 0.55,
    ) -> dict[str, Any]:
        now = utc_now()
        data = self._load()
        interaction_id = f"int-{uuid.uuid4().hex}"
        data.setdefault("interactions", []).append({
            "id": interaction_id,
            "source": source,
            "user_text": clip(user_text, 6000),
            "assistant_text": clip(assistant_text, 6000),
            "summary": clip(user_text, 240),
            "metadata_json": metadata_to_json(metadata),
            "created_at": now,
        })
        data["interactions"] = data["interactions"][-500:]

        stored = []
        memories = data.setdefault("memories", {})
        for candidate in extract_memories(user_text, assistant_text, source, importance):
            mid = memory_id(candidate.kind, candidate.text)
            stored.append({"id": mid, "kind": candidate.kind, "text": candidate.text})
            current = memories.get(mid)
            if current:
                current["frequency"] = int(current.get("frequency") or 0) + 1
                current["last_seen"] = now
                current["importance"] = max(float(current.get("importance") or 0), float(candidate.importance))
                current["confidence"] = max(float(current.get("confidence") or 0), float(candidate.confidence))
                known_entities = {e.get("id"): e for e in current.get("entities", [])}
                for entity in candidate.entities:
                    known_entities[entity["id"]] = entity
                current["entities"] = list(known_entities.values())[:48]
            else:
                memories[mid] = {
                    "id": mid,
                    "kind": candidate.kind,
                    "text": candidate.text,
                    "summary": clip(candidate.text, 180),
                    "normalized_text": normalize_text(candidate.text),
                    "importance": float(candidate.importance),
                    "confidence": float(candidate.confidence),
                    "frequency": 1,
                    "first_seen": now,
                    "last_seen": now,
                    "entities": candidate.entities,
                    "interaction_ids": [],
                }
            memories[mid].setdefault("interaction_ids", []).append(interaction_id)
            memories[mid]["interaction_ids"] = memories[mid]["interaction_ids"][-30:]

        self._save(data)
        return {
            "ok": True,
            "backend": "local_json",
            "interaction_id": interaction_id,
            "stored": len(stored),
            "memories": stored,
        }

    def retrieve_context(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        tokens = tokenize(query)
        query_entities = {e["name"].lower() for e in extract_entities(query)}
        rows = []
        for item in self._load().get("memories", {}).values():
            text = " ".join([
                str(item.get("text") or ""),
                str(item.get("summary") or ""),
                str(item.get("kind") or ""),
            ]).lower()
            entity_names = [e.get("name", "") for e in item.get("entities", [])]
            entity_lc = {name.lower() for name in entity_names}
            score = (
                sum(1.0 for token in tokens if token in text)
                + sum(2.0 for entity in query_entities if entity in entity_lc)
                + float(item.get("importance") or 0)
                + (float(item.get("frequency") or 1) * 0.08)
            )
            if score <= 0.35:
                continue
            rows.append({
                "id": item.get("id"),
                "kind": item.get("kind"),
                "text": item.get("text"),
                "summary": item.get("summary"),
                "importance": item.get("importance"),
                "confidence": item.get("confidence"),
                "frequency": item.get("frequency"),
                "last_seen": item.get("last_seen"),
                "entities": entity_names,
                "score": round(score, 3),
            })
        rows.sort(key=lambda row: (row["score"], row.get("last_seen") or ""), reverse=True)
        return rows[: int(max(1, min(limit, 25)))]

    def format_context(self, memories: list[dict[str, Any]]) -> str:
        return Neo4jGraphMemory.format_context(self, memories)

    def reinforce(self, memory_id_value: str, delta: float = 0.1) -> dict[str, Any]:
        data = self._load()
        item = data.get("memories", {}).get(memory_id_value)
        if not item:
            return {"ok": False, "reason": "memory_not_found", "memory_id": memory_id_value}
        item["importance"] = float(item.get("importance") or 0) + float(delta)
        item["last_seen"] = utc_now()
        self._save(data)
        return {"ok": True, "backend": "local_json", "memory_id": memory_id_value}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "project_id": self.project_id,
                "user_id": self.user_id,
                "interactions": [],
                "memories": {},
            }
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {
                "project_id": self.project_id,
                "user_id": self.user_id,
                "interactions": [],
                "memories": {},
            }

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
        tmp.replace(self.path)
        self.path.chmod(0o600)


class ResilientGraphMemory:
    """Neo4j primary memory with local durable fallback."""

    def __init__(
        self,
        primary: Optional[Neo4jGraphMemory] = None,
        fallback: Optional[LocalGraphMemory] = None,
    ) -> None:
        self.primary = primary or Neo4jGraphMemory()
        self.fallback = fallback or LocalGraphMemory(
            project_id=self.primary.project_id,
            user_id=self.primary.user_id,
        )

    def status(self) -> dict[str, Any]:
        primary_status = self.primary.status()
        fallback_status = self.fallback.status()
        if primary_status.get("available"):
            return {
                **primary_status,
                "backend": "neo4j",
                "fallback": fallback_status,
            }
        return {
            "configured": True,
            "available": True,
            "backend": "local_json_fallback",
            "project_id": self.primary.project_id,
            "neo4j": primary_status,
            "fallback": fallback_status,
            "message": "Neo4j is unavailable; using durable local memory fallback.",
        }

    def remember_interaction(self, *args, **kwargs) -> dict[str, Any]:
        try:
            if self.primary.status().get("available"):
                return self.primary.remember_interaction(*args, **kwargs)
        except Exception:
            pass
        return self.fallback.remember_interaction(*args, **kwargs)

    def retrieve_context(self, *args, **kwargs) -> list[dict[str, Any]]:
        try:
            if self.primary.status().get("available"):
                return self.primary.retrieve_context(*args, **kwargs)
        except Exception:
            pass
        return self.fallback.retrieve_context(*args, **kwargs)

    def format_context(self, memories: list[dict[str, Any]]) -> str:
        return self.fallback.format_context(memories)

    def reinforce(self, *args, **kwargs) -> dict[str, Any]:
        try:
            if self.primary.status().get("available"):
                return self.primary.reinforce(*args, **kwargs)
        except Exception:
            pass
        return self.fallback.reinforce(*args, **kwargs)


_MEMORY: Optional[ResilientGraphMemory] = None


def get_graph_memory() -> ResilientGraphMemory:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = ResilientGraphMemory()
    return _MEMORY
