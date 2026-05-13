import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from graph_memory import LocalGraphMemory, Neo4jGraphMemory, ResilientGraphMemory, extract_memories


class RecordingMemory(Neo4jGraphMemory):
    def __init__(self):
        super().__init__(uri="http://neo4j.test:7474", user="neo4j", password="pw", project_id="test-project")
        self.calls = []

    def _commit(self, statements):
        self.calls.append(statements)
        statement = statements[0]["statement"]
        if "RETURN m.id AS id" in statement:
            return {
                "results": [{
                    "columns": ["id", "kind", "text", "summary", "importance", "confidence", "frequency", "last_seen", "entities", "score"],
                    "data": [{
                        "row": [
                            "mem-user_preference-abc",
                            "user_preference",
                            "User wants Neo4j graph memory.",
                            "User wants Neo4j graph memory.",
                            0.9,
                            0.86,
                            3,
                            "2026-05-05T00:00:00+00:00",
                            ["Neo4j", "GNN"],
                            4.2,
                        ]
                    }],
                }],
                "errors": [],
            }
        return {"results": [{"columns": ["ok"], "data": [{"row": [1]}]}], "errors": []}


def test_extract_memories_captures_user_goal_and_entities():
    memories = extract_memories(
        "I want permanent memory with Neo4j and a GNN so the AI does not reread everything.",
        "",
    )

    kinds = {m.kind for m in memories}
    entity_names = {e["name"] for m in memories for e in m.entities}

    assert "user_preference" in kinds
    assert "architecture_decision" in kinds
    assert "Neo4j" in entity_names
    assert "GNN" in entity_names


def test_remember_interaction_writes_project_memory_graph():
    memory = RecordingMemory()

    result = memory.remember_interaction(
        user_text="I want actual implementation with Neo4j permanent memory, no scaffolding.",
        assistant_text="Implemented graph memory.",
        source="test",
        metadata={"case": "unit"},
        importance=0.8,
    )

    assert result["ok"] is True
    assert result["stored"] >= 1
    flattened = "\n".join(stmt["statement"] for call in memory.calls for stmt in call)
    assert "MERGE (m:Memory" in flattened
    assert "MERGE (m)-[r:MENTIONS]->(e)" in flattened
    interaction_params = next(
        stmt["parameters"]
        for call in memory.calls
        for stmt in call
        if "CREATE (i:Interaction" in stmt["statement"]
    )
    assert interaction_params["metadata_json"] == '{"case": "unit"}'
    assert "metadata" not in interaction_params


def test_retrieve_context_formats_memory_rows():
    memory = RecordingMemory()

    rows = memory.retrieve_context("Neo4j GNN permanent memory", limit=3)
    context = memory.format_context(rows)

    assert rows[0]["kind"] == "user_preference"
    assert "User wants Neo4j graph memory" in context
    assert "seen 3x" in context


def test_unconfigured_memory_is_non_blocking():
    memory = Neo4jGraphMemory(password="")

    result = memory.remember_interaction("I want graph memory.", "ok")

    assert result["ok"] is False
    assert result["reason"] == "neo4j_not_configured"


def test_resilient_memory_uses_local_fallback_when_neo4j_is_down(tmp_path):
    class DownMemory(Neo4jGraphMemory):
        def __init__(self):
            super().__init__(uri="http://neo4j.test:7474", user="neo4j", password="pw", project_id="test-project")

        def status(self):
            return {"configured": True, "available": False, "error": "connection refused", "project_id": "test-project"}

    memory = ResilientGraphMemory(
        primary=DownMemory(),
        fallback=LocalGraphMemory(path=tmp_path / "memory.json", project_id="test-project"),
    )

    result = memory.remember_interaction("I want actual implementation with permanent memory.", "Done.")
    rows = memory.retrieve_context("permanent memory", limit=3)
    status = memory.status()

    assert result["ok"] is True
    assert result["backend"] == "local_json"
    assert rows
    assert status["available"] is True
    assert status["backend"] == "local_json_fallback"
