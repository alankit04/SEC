"""
citation_index.py — durable citation memory for RAPHI.

Postgres is the primary backend when CITATION_DATABASE_URL is configured.
SQLite FTS5 is the local fallback so development and tests remain fully local.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import firecrawl_client
except ImportError:  # pragma: no cover
    from backend import firecrawl_client

try:  # optional in local/test mode
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    psycopg = None
    dict_row = None


BASE_DIR = Path(__file__).parent.parent
DEFAULT_SQLITE_PATH = BASE_DIR / "data" / "citation_index.sqlite"
DEFAULT_DATABASE_URL = os.environ.get("CITATION_DATABASE_URL", "").strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", str(text or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _snippet(text: str, query: str, *, max_chars: int = 320) -> str:
    clean = " ".join(_clean_text(text).split())
    if not clean:
        return ""
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9$.-]{3,}", query or "")]
    lower = clean.lower()
    pos = min((lower.find(t) for t in terms if lower.find(t) >= 0), default=0)
    start = max(pos - 90, 0)
    end = min(start + max_chars, len(clean))
    return clean[start:end].strip()


def _stable_id(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def _content_hash(text: str) -> str:
    return hashlib.sha256(_clean_text(text).encode("utf-8", errors="ignore")).hexdigest()


def chunk_text(text: str, *, max_words: int = 180, overlap: int = 30) -> list[str]:
    words = _clean_text(text).split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks


@dataclass
class CitationDocument:
    ticker: str = ""
    source_type: str = "web"
    title: str = ""
    url: str = ""
    text: str = ""
    published_at: str = ""
    retrieved_at: str = ""
    metadata: dict[str, Any] | None = None


class CitationIndex:
    def __init__(
        self,
        *,
        database_url: str | None = None,
        sqlite_path: Path | None = None,
    ):
        self.database_url = database_url if database_url is not None else DEFAULT_DATABASE_URL
        self.sqlite_path = Path(sqlite_path or DEFAULT_SQLITE_PATH)
        self.backend = "postgres" if self.database_url and psycopg is not None else "sqlite"
        self.init_db()

    def _pg_conn(self):
        if not self.database_url or psycopg is None:
            raise RuntimeError("Postgres backend is not available")
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _sqlite_conn(self) -> sqlite3.Connection:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        if self.backend == "postgres":
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_postgres(self) -> None:
        with self._pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS citation_sources (
                        id TEXT PRIMARY KEY,
                        user_scope TEXT NOT NULL DEFAULT 'global',
                        ticker TEXT NOT NULL DEFAULT '',
                        source_type TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        url TEXT NOT NULL DEFAULT '',
                        domain TEXT NOT NULL DEFAULT '',
                        published_at TEXT NOT NULL DEFAULT '',
                        retrieved_at TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS citation_chunks (
                        id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL REFERENCES citation_sources(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        token_count INTEGER NOT NULL DEFAULT 0,
                        content_hash TEXT NOT NULL UNIQUE
                    )
                """)
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_citation_sources_ticker ON citation_sources(ticker)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_citation_sources_scope ON citation_sources(user_scope)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_citation_chunks_source ON citation_chunks(source_id)"
                )
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_citation_chunks_fts
                    ON citation_chunks
                    USING GIN (to_tsvector('english', text))
                """)
                cur.execute("ALTER TABLE citation_sources ADD COLUMN IF NOT EXISTS user_scope TEXT NOT NULL DEFAULT 'global'")
            conn.commit()

    def _init_sqlite(self) -> None:
        with self._sqlite_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS citation_sources (
                    id TEXT PRIMARY KEY,
                    user_scope TEXT NOT NULL DEFAULT 'global',
                    ticker TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL DEFAULT '',
                    domain TEXT NOT NULL DEFAULT '',
                    published_at TEXT NOT NULL DEFAULT '',
                    retrieved_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS citation_chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(source_id) REFERENCES citation_sources(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS citation_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    source_id UNINDEXED,
                    user_scope,
                    ticker,
                    source_type,
                    title,
                    url UNINDEXED,
                    text
                )
            """)
            cols = [row[1] for row in conn.execute("PRAGMA table_info(citation_sources)").fetchall()]
            if "user_scope" not in cols:
                conn.execute("ALTER TABLE citation_sources ADD COLUMN user_scope TEXT NOT NULL DEFAULT 'global'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_citation_sources_ticker ON citation_sources(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_citation_sources_scope ON citation_sources(user_scope)")
            conn.commit()

    def add_document(self, doc: CitationDocument | dict, *, user_scope: str = "global") -> dict:
        if isinstance(doc, dict):
            doc = CitationDocument(**doc)
        text = _clean_text(doc.text)
        if not text:
            return {"inserted_chunks": 0, "skipped": True, "reason": "empty text"}

        ticker = str(doc.ticker or "").upper().strip()
        source_type = str(doc.source_type or "web").strip()[:64]
        title = str(doc.title or "").strip()[:500]
        url = str(doc.url or "").strip()
        retrieved_at = doc.retrieved_at or _now_iso()
        metadata = doc.metadata or {}
        scope = str(user_scope or "global").strip()[:128] or "global"
        source_id = _stable_id(scope, ticker, source_type, url, title)
        chunks = chunk_text(text)
        if not chunks:
            return {"inserted_chunks": 0, "skipped": True, "reason": "no chunks"}

        if self.backend == "postgres":
            return self._add_document_postgres(
                source_id, scope, ticker, source_type, title, url, doc.published_at,
                retrieved_at, metadata, chunks
            )
        return self._add_document_sqlite(
            source_id, scope, ticker, source_type, title, url, doc.published_at,
            retrieved_at, metadata, chunks
        )

    def _add_document_postgres(
        self,
        source_id: str,
        user_scope: str,
        ticker: str,
        source_type: str,
        title: str,
        url: str,
        published_at: str,
        retrieved_at: str,
        metadata: dict,
        chunks: list[str],
    ) -> dict:
        import json

        inserted = 0
        with self._pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO citation_sources
                    (id, user_scope, ticker, source_type, title, url, domain, published_at, retrieved_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        user_scope = EXCLUDED.user_scope,
                        ticker = EXCLUDED.ticker,
                        source_type = EXCLUDED.source_type,
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        domain = EXCLUDED.domain,
                        published_at = EXCLUDED.published_at,
                        retrieved_at = EXCLUDED.retrieved_at,
                        metadata = EXCLUDED.metadata
                """, (source_id, user_scope, ticker, source_type, title, url, _domain(url), published_at or "", retrieved_at, json.dumps(metadata)))
                for idx, chunk in enumerate(chunks):
                    chunk_id = _stable_id(source_id, idx, chunk)
                    cur.execute("""
                        INSERT INTO citation_chunks
                        (id, source_id, chunk_index, text, token_count, content_hash)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_hash) DO NOTHING
                    """, (chunk_id, source_id, idx, chunk, len(chunk.split()), _content_hash(chunk)))
                    inserted += int(cur.rowcount or 0)
            conn.commit()
        return {"source_id": source_id, "inserted_chunks": inserted, "backend": self.backend}

    def _add_document_sqlite(
        self,
        source_id: str,
        user_scope: str,
        ticker: str,
        source_type: str,
        title: str,
        url: str,
        published_at: str,
        retrieved_at: str,
        metadata: dict,
        chunks: list[str],
    ) -> dict:
        import json

        inserted = 0
        with self._sqlite_conn() as conn:
            conn.execute("""
                INSERT INTO citation_sources
                (id, user_scope, ticker, source_type, title, url, domain, published_at, retrieved_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_scope=excluded.user_scope,
                    ticker=excluded.ticker,
                    source_type=excluded.source_type,
                    title=excluded.title,
                    url=excluded.url,
                    domain=excluded.domain,
                    published_at=excluded.published_at,
                    retrieved_at=excluded.retrieved_at,
                    metadata=excluded.metadata
            """, (source_id, user_scope, ticker, source_type, title, url, _domain(url), published_at or "", retrieved_at, json.dumps(metadata)))
            for idx, chunk in enumerate(chunks):
                chunk_id = _stable_id(source_id, idx, chunk)
                try:
                    conn.execute("""
                        INSERT INTO citation_chunks
                        (id, source_id, chunk_index, text, token_count, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (chunk_id, source_id, idx, chunk, len(chunk.split()), _content_hash(chunk)))
                    conn.execute("""
                        INSERT INTO citation_fts
                        (chunk_id, source_id, user_scope, ticker, source_type, title, url, text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (chunk_id, source_id, user_scope, ticker, source_type, title, url, chunk))
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
            conn.commit()
        return {"source_id": source_id, "inserted_chunks": inserted, "backend": self.backend}

    def search(
        self,
        query: str,
        *,
        user_scope: str = "global",
        ticker: str = "",
        source_type: str = "",
        limit: int = 5,
    ) -> dict:
        clean_query = " ".join(str(query or "").split())[:500]
        ticker = str(ticker or "").upper().strip()
        source_type = str(source_type or "").strip()
        scope = str(user_scope or "global").strip()[:128] or "global"
        limit = min(max(int(limit or 5), 1), 20)
        if not clean_query:
            return {"query": clean_query, "ticker": ticker, "results": [], "count": 0, "backend": self.backend}
        if self.backend == "postgres":
            rows = self._search_postgres(clean_query, user_scope=scope, ticker=ticker, source_type=source_type, limit=limit)
        else:
            rows = self._search_sqlite(clean_query, user_scope=scope, ticker=ticker, source_type=source_type, limit=limit)
        results = []
        for idx, row in enumerate(rows, start=1):
            text = row.get("text", "")
            results.append({
                "id": idx,
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "domain": _domain(row.get("url", "")),
                "snippet": _snippet(text, clean_query),
                "ticker": row.get("ticker", ""),
                "source_type": row.get("source_type", ""),
                "provider": "RAPHI Citation Index",
                "retrieved_at": row.get("retrieved_at", ""),
                "score": row.get("score"),
            })
        return {
            "provider": "local_citation_index",
            "query": clean_query,
            "user_scope": scope,
            "ticker": ticker,
            "backend": self.backend,
            "results": results,
            "count": len(results),
            "retrieved_at": _now_iso(),
        }

    def _search_postgres(self, query: str, *, user_scope: str, ticker: str, source_type: str, limit: int) -> list[dict]:
        clauses = ["to_tsvector('english', c.text) @@ plainto_tsquery('english', %s)"]
        where_params: list[Any] = [query]
        clauses.append("s.user_scope = %s")
        where_params.append(user_scope)
        if ticker:
            clauses.append("s.ticker = %s")
            where_params.append(ticker)
        if source_type:
            clauses.append("s.source_type = %s")
            where_params.append(source_type)
        params = [query, *where_params, limit]
        sql = f"""
            SELECT
                c.text,
                s.ticker,
                s.source_type,
                s.title,
                s.url,
                s.retrieved_at,
                ts_rank(to_tsvector('english', c.text), plainto_tsquery('english', %s)) AS score
            FROM citation_chunks c
            JOIN citation_sources s ON s.id = c.source_id
            WHERE {' AND '.join(clauses)}
            ORDER BY score DESC, s.retrieved_at DESC
            LIMIT %s
        """
        with self._pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    def _search_sqlite(self, query: str, *, user_scope: str, ticker: str, source_type: str, limit: int) -> list[dict]:
        terms = " OR ".join(re.findall(r"[A-Za-z0-9]{3,}", query)) or query
        clauses = ["citation_fts MATCH ?"]
        params: list[Any] = [terms]
        clauses.append("f.user_scope = ?")
        params.append(user_scope)
        if ticker:
            clauses.append("f.ticker = ?")
            params.append(ticker)
        if source_type:
            clauses.append("f.source_type = ?")
            params.append(source_type)
        params.append(limit)
        sql = f"""
            SELECT
                f.text,
                f.ticker,
                f.source_type,
                f.title,
                f.url,
                s.retrieved_at,
                bm25(citation_fts) AS score
            FROM citation_fts f
            JOIN citation_sources s ON s.id = f.source_id
            WHERE {' AND '.join(clauses)}
            ORDER BY score ASC
            LIMIT ?
        """
        with self._sqlite_conn() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def ingest_sec_ticker(self, sec_reader, ticker: str, *, user_scope: str = "global", limit_filings: int = 8) -> dict:
        ticker = str(ticker or "").upper().strip()
        inserted_docs = 0
        inserted_chunks = 0

        filings = sec_reader.ticker_filings(ticker, limit=limit_filings)
        for filing in filings:
            citation = filing.get("citation", {}) or {}
            accession = filing.get("accession") or filing.get("adsh") or citation.get("accession", "")
            form = filing.get("form") or citation.get("form") or "SEC filing"
            title = f"{ticker} {form} {accession}".strip()
            text = (
                f"{ticker} SEC filing {form}. Accession {accession}. "
                f"Filed {filing.get('filed') or citation.get('filed')}. "
                f"Period {filing.get('period') or citation.get('period')}. "
                f"Quarter {filing.get('quarter') or citation.get('quarter')}. "
                f"SEC URL {filing.get('sec_url') or citation.get('sec_url')}."
            )
            added = self.add_document(CitationDocument(
                ticker=ticker,
                source_type="sec_filing",
                title=title,
                url=filing.get("sec_url") or citation.get("sec_url") or "",
                text=text,
                published_at=filing.get("filed") or citation.get("filed") or "",
                metadata={"citation": citation, "accession": accession, "form": form},
            ), user_scope=user_scope)
            inserted_docs += 1
            inserted_chunks += int(added.get("inserted_chunks", 0))

        try:
            entries = sec_reader.company_financial_entries(ticker, limit_filings=limit_filings)
        except Exception:
            entries = []
        for entry in entries:
            citation = entry.get("citation", {}) or {}
            metric = entry.get("metric", "financial_metric")
            accession = entry.get("accession") or citation.get("accession", "")
            value = entry.get("val")
            title = f"{ticker} {metric} {entry.get('form', '')} {accession}".strip()
            text = (
                f"{ticker} XBRL financial metric {metric}. "
                f"Tag {entry.get('tag')}. Value {value} {entry.get('uom', '')}. "
                f"Form {entry.get('form')}. Accession {accession}. "
                f"Filed {entry.get('filed')}. Period {entry.get('period')}. "
                f"SEC URL {entry.get('sec_url') or citation.get('sec_url')}."
            )
            added = self.add_document(CitationDocument(
                ticker=ticker,
                source_type="sec_xbrl",
                title=title,
                url=entry.get("sec_url") or citation.get("sec_url") or "",
                text=text,
                published_at=entry.get("filed") or citation.get("filed") or "",
                metadata={"citation": citation, "metric": metric, "tag": entry.get("tag")},
            ), user_scope=user_scope)
            inserted_docs += 1
            inserted_chunks += int(added.get("inserted_chunks", 0))

        return {
            "ticker": ticker,
            "indexed_documents": inserted_docs,
            "inserted_chunks": inserted_chunks,
            "backend": self.backend,
        }

    def refresh_from_firecrawl(self, query: str, *, user_scope: str = "global", ticker: str = "", limit: int = 3) -> dict:
        if not firecrawl_client.is_available():
            return {
                "available": False,
                "ingested": 0,
                "error": "FIRECRAWL_API_KEY not configured",
            }
        results = firecrawl_client.search_web(query, limit=min(max(limit, 1), 5), scrape_results=True, max_chars_per_result=5000)
        ingested = 0
        inserted_chunks = 0
        errors: list[str] = []
        for item in results:
            if not item.get("success"):
                if item.get("error"):
                    errors.append(str(item["error"]))
                continue
            text = item.get("markdown") or item.get("description") or ""
            if not text:
                continue
            added = self.add_document(CitationDocument(
                ticker=ticker,
                source_type="firecrawl_web",
                title=item.get("title", ""),
                url=item.get("url", ""),
                text=text,
                metadata={"query": query, "description": item.get("description", "")},
            ), user_scope=user_scope)
            ingested += 1
            inserted_chunks += int(added.get("inserted_chunks", 0))
        return {
            "available": True,
            "ingested": ingested,
            "inserted_chunks": inserted_chunks,
            "errors": errors[:3],
        }

    def search_with_refresh(
        self,
        query: str,
        *,
        user_scope: str = "global",
        ticker: str = "",
        limit: int = 5,
        refresh_if_missing: bool = False,
        min_results: int = 2,
    ) -> dict:
        first = self.search(query, user_scope=user_scope, ticker=ticker, limit=limit)
        if not refresh_if_missing or first.get("count", 0) >= min_results:
            first["refresh"] = {"attempted": False}
            return first
        refresh = self.refresh_from_firecrawl(query, user_scope=user_scope, ticker=ticker, limit=max(min_results, 3))
        second = self.search(query, user_scope=user_scope, ticker=ticker, limit=limit)
        second["refresh"] = {"attempted": True, **refresh}
        return second

    def export_user_data(self, user_scope: str, *, limit: int = 500) -> dict:
        scope = str(user_scope or "global").strip()[:128] or "global"
        limit = min(max(int(limit), 1), 5000)
        if self.backend == "postgres":
            with self._pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT s.id, s.user_scope, s.ticker, s.source_type, s.title, s.url, s.retrieved_at, c.text
                        FROM citation_sources s
                        JOIN citation_chunks c ON c.source_id = s.id
                        WHERE s.user_scope = %s
                        ORDER BY s.retrieved_at DESC
                        LIMIT %s
                        """,
                        (scope, limit),
                    )
                    rows = [dict(r) for r in cur.fetchall()]
        else:
            with self._sqlite_conn() as conn:
                rows = [
                    dict(r)
                    for r in conn.execute(
                        """
                        SELECT s.id, s.user_scope, s.ticker, s.source_type, s.title, s.url, s.retrieved_at, c.text
                        FROM citation_sources s
                        JOIN citation_chunks c ON c.source_id = s.id
                        WHERE s.user_scope = ?
                        ORDER BY s.retrieved_at DESC
                        LIMIT ?
                        """,
                        (scope, limit),
                    ).fetchall()
                ]
        return {"user_scope": scope, "records": rows, "count": len(rows)}

    def delete_user_data(self, user_scope: str) -> dict:
        scope = str(user_scope or "global").strip()[:128] or "global"
        if self.backend == "postgres":
            with self._pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM citation_sources WHERE user_scope = %s", (scope,))
                    ids = [row[0] for row in cur.fetchall()]
                    if ids:
                        cur.execute("DELETE FROM citation_chunks WHERE source_id = ANY(%s)", (ids,))
                        cur.execute("DELETE FROM citation_sources WHERE id = ANY(%s)", (ids,))
                    deleted = len(ids)
                conn.commit()
        else:
            with self._sqlite_conn() as conn:
                ids = [r[0] for r in conn.execute("SELECT id FROM citation_sources WHERE user_scope = ?", (scope,)).fetchall()]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    conn.execute(f"DELETE FROM citation_chunks WHERE source_id IN ({placeholders})", ids)
                    conn.execute(f"DELETE FROM citation_fts WHERE source_id IN ({placeholders})", ids)
                    conn.execute(f"DELETE FROM citation_sources WHERE id IN ({placeholders})", ids)
                conn.commit()
                deleted = len(ids)
        return {"user_scope": scope, "deleted_sources": deleted}

    def status(self) -> dict:
        if self.backend == "postgres":
            with self._pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) AS n FROM citation_sources")
                    sources = int(cur.fetchone()["n"])
                    cur.execute("SELECT COUNT(*) AS n FROM citation_chunks")
                    chunks = int(cur.fetchone()["n"])
            return {
                "backend": self.backend,
                "postgres_configured": bool(self.database_url),
                "sources": sources,
                "chunks": chunks,
            }
        with self._sqlite_conn() as conn:
            sources = int(conn.execute("SELECT COUNT(*) FROM citation_sources").fetchone()[0])
            chunks = int(conn.execute("SELECT COUNT(*) FROM citation_chunks").fetchone()[0])
        return {
            "backend": self.backend,
            "postgres_configured": bool(self.database_url),
            "sqlite_path": str(self.sqlite_path.relative_to(BASE_DIR)),
            "sources": sources,
            "chunks": chunks,
        }


def get_citation_index() -> CitationIndex:
    return CitationIndex()
