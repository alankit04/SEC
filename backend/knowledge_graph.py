"""
knowledge_graph.py — Financial knowledge graph backed by Neo4j.

Schema
------
Nodes:
  User    {email}
  Company {ticker, name, cik, sic_2}
  Sector  {sic_2, name}

Relationships:
  (User)-[:WATCHES]->(Company)                      from settings.json watchlist
  (User)-[:QUERIED {ts}]->(Company)                 written on every MCP tool call
  (Company)-[:IN_SECTOR]->(Sector)                  from SEC sub.txt SIC code
  (Company)-[:CORRELATED_WITH {rho: float}]->(Company)  from GNN output (Phase 3)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("raphi.knowledge_graph")

_DEFAULT_USER_ID = os.environ.get("RAPHI_MEMORY_USER_ID", "local-user")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lookup_ticker_info(ticker: str) -> dict[str, str | None]:
    """Return {name, cik, sic_2} from local SEC data. Returns partial on failure."""
    result: dict[str, str | None] = {"name": ticker, "cik": None, "sic_2": None}
    try:
        from backend.paths import PROJECT_ROOT
        from backend.sec_data import SECData
        import pandas as pd

        sd = SECData(PROJECT_ROOT / "data")
        cik = sd.cik_for_ticker(ticker)
        if not cik:
            return result
        result["cik"] = cik

        cik_int = int(cik)
        for quarter in ("2025q4", "2025q3", "2025q2", "2025q1"):
            df = sd._load_sub(quarter)
            if df.empty:
                continue
            rows = df[df["cik"] == cik_int]
            if rows.empty:
                continue
            row = rows.iloc[0]
            result["name"] = str(row["name"])
            sic_raw = row.get("sic")
            if pd.notna(sic_raw):
                result["sic_2"] = str(int(float(sic_raw))).zfill(4)[:2]
            break
    except Exception as exc:
        logger.debug("SIC lookup failed for %s: %s", ticker, exc)
    return result


class KnowledgeGraph:
    """Financial knowledge graph. Fails silently if Neo4j is unreachable."""

    _instance: "KnowledgeGraph | None" = None

    @classmethod
    def get(cls) -> "KnowledgeGraph":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.uri      = (uri      or os.environ.get("NEO4J_URI")      or "http://localhost:7474").rstrip("/")
        self.user     = user      or os.environ.get("NEO4J_USER")     or "neo4j"
        self.password = password  if password is not None else os.environ.get("NEO4J_PASSWORD", "")
        self.database = database  or os.environ.get("NEO4J_DATABASE") or "neo4j"
        self.timeout  = timeout
        self._schema_ready = False

    @property
    def configured(self) -> bool:
        return bool(self.password)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        if not self.configured or self._schema_ready:
            return
        stmts = [
            "CREATE CONSTRAINT kg_user_email    IF NOT EXISTS FOR (u:User)    REQUIRE u.email   IS UNIQUE",
            "CREATE CONSTRAINT kg_company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker  IS UNIQUE",
            "CREATE CONSTRAINT kg_sector_sic2    IF NOT EXISTS FOR (s:Sector)  REQUIRE s.sic_2   IS UNIQUE",
            "CREATE INDEX kg_company_cik IF NOT EXISTS FOR (c:Company) ON (c.cik)",
        ]
        try:
            for stmt in stmts:
                self._commit([{"statement": stmt}])
            self._schema_ready = True
            logger.info("KnowledgeGraph: schema ready")
        except Exception as exc:
            logger.warning("KnowledgeGraph: schema setup failed: %s", exc)

    # ------------------------------------------------------------------
    # Phase 1 — seed from watchlist
    # ------------------------------------------------------------------

    def seed_watchlist(self, user_email: str, watchlist: list[str]) -> None:
        """Create User + Company + Sector nodes and WATCHES + IN_SECTOR edges."""
        if not self.configured:
            return
        self.ensure_schema()

        try:
            from backend.sec_data import SIC_INDUSTRIES
        except ImportError:
            SIC_INDUSTRIES: dict[str, str] = {}

        try:
            self._commit([{
                "statement": "MERGE (u:User {email: $email}) SET u.updated_at = $now",
                "parameters": {"email": user_email, "now": _utc_now()},
            }])
        except Exception as exc:
            logger.warning("KnowledgeGraph: failed to upsert user %s: %s", user_email, exc)
            return

        for ticker in watchlist:
            try:
                info   = _lookup_ticker_info(ticker)
                sic_2  = info["sic_2"]
                sector = SIC_INDUSTRIES.get(sic_2, "Unknown") if sic_2 else "Unknown"

                stmts: list[dict[str, Any]] = [
                    {
                        "statement": """
                            MERGE (c:Company {ticker: $ticker})
                            SET c.name       = $name,
                                c.cik        = $cik,
                                c.sic_2      = $sic_2,
                                c.updated_at = $now
                        """,
                        "parameters": {
                            "ticker": ticker, "name": info["name"],
                            "cik": info["cik"] or "", "sic_2": sic_2 or "",
                            "now": _utc_now(),
                        },
                    },
                    {
                        "statement": """
                            MATCH (u:User {email: $email})
                            MATCH (c:Company {ticker: $ticker})
                            MERGE (u)-[:WATCHES]->(c)
                        """,
                        "parameters": {"email": user_email, "ticker": ticker},
                    },
                ]

                if sic_2:
                    stmts += [
                        {
                            "statement": """
                                MERGE (s:Sector {sic_2: $sic_2})
                                SET s.name = $sector_name
                            """,
                            "parameters": {"sic_2": sic_2, "sector_name": sector},
                        },
                        {
                            "statement": """
                                MATCH (c:Company {ticker: $ticker})
                                MATCH (s:Sector {sic_2: $sic_2})
                                MERGE (c)-[:IN_SECTOR]->(s)
                            """,
                            "parameters": {"ticker": ticker, "sic_2": sic_2},
                        },
                    ]

                self._commit(stmts)
                logger.debug("KnowledgeGraph: seeded %s (sic_2=%s, sector=%s)", ticker, sic_2, sector)

            except Exception as exc:
                logger.warning("KnowledgeGraph: seed failed for %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Phase 2 — record live queries
    # ------------------------------------------------------------------

    def record_query(self, user_email: str, ticker: str) -> None:
        """Write a QUERIED edge from User to Company. Fire-and-forget — never raises."""
        if not self.configured:
            return
        try:
            self.ensure_schema()
            self._commit([
                {
                    "statement": """
                        MERGE (u:User {email: $email})
                        MERGE (c:Company {ticker: $ticker})
                        ON CREATE SET c.name = $ticker, c.cik = '', c.sic_2 = ''
                        CREATE (u)-[:QUERIED {ts: $ts}]->(c)
                    """,
                    "parameters": {"email": user_email, "ticker": ticker, "ts": _utc_now()},
                }
            ])
        except Exception as exc:
            logger.debug("KnowledgeGraph: record_query failed (%s %s): %s", user_email, ticker, exc)

    # ------------------------------------------------------------------
    # Phase 3 — correlation edges from GNN
    # ------------------------------------------------------------------

    def seed_correlations(
        self,
        tickers: list[str],
        corr_mat: "Any",  # np.ndarray (N, N)
        sic_map:  dict[str, str],
    ) -> None:
        """
        Export GNN correlation matrix to Neo4j as CORRELATED_WITH edges.
        Only writes pairs where |rho| >= 0.65 and the edge is cross-sector
        (same-sector pairs are already linked via IN_SECTOR).
        """
        if not self.configured:
            return
        self.ensure_schema()

        from backend.gnn_model import CORR_THRESHOLD
        n = len(tickers)
        batch: list[dict[str, Any]] = []

        for i in range(n):
            for j in range(i + 1, n):
                rho = float(corr_mat[i, j])
                if abs(rho) < CORR_THRESHOLD:
                    continue
                ti, tj = tickers[i], tickers[j]
                # skip same-sector pairs — they're already connected through Sector node
                if sic_map.get(ti, "")[:2] == sic_map.get(tj, "")[:2] != "00":
                    continue
                batch.append({
                    "statement": """
                        MATCH (a:Company {ticker: $ti})
                        MATCH (b:Company {ticker: $tj})
                        MERGE (a)-[r:CORRELATED_WITH]->(b)
                        SET r.rho = $rho, r.updated_at = $now
                    """,
                    "parameters": {"ti": ti, "tj": tj, "rho": round(rho, 4), "now": _utc_now()},
                })

        if not batch:
            logger.info("KnowledgeGraph: no cross-sector correlation edges to write")
            return

        try:
            self._commit(batch)
            logger.info("KnowledgeGraph: wrote %d CORRELATED_WITH edges", len(batch))
        except Exception as exc:
            logger.warning("KnowledgeGraph: seed_correlations failed: %s", exc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def unwatched_peers(self, user_email: str) -> list[dict[str, str]]:
        """Sector peers of the watchlist that the user is NOT watching."""
        if not self.configured:
            return []
        try:
            result = self._commit([{
                "statement": """
                    MATCH (u:User {email: $email})-[:WATCHES]->(watched:Company)
                          -[:IN_SECTOR]->(s:Sector)
                          <-[:IN_SECTOR]-(peer:Company)
                    WHERE NOT (u)-[:WATCHES]->(peer)
                      AND peer.ticker <> watched.ticker
                    RETURN DISTINCT peer.ticker AS ticker,
                                    peer.name   AS name,
                                    s.name      AS sector
                    ORDER BY sector, ticker
                    LIMIT 50
                """,
                "parameters": {"email": user_email},
            }])
            return self._rows(result)
        except Exception as exc:
            logger.debug("KnowledgeGraph: unwatched_peers failed: %s", exc)
            return []

    def correlated_with(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
        """Companies with a CORRELATED_WITH edge to the given ticker, sorted by |rho|."""
        if not self.configured:
            return []
        try:
            result = self._commit([{
                "statement": """
                    MATCH (a:Company {ticker: $ticker})-[r:CORRELATED_WITH]->(b:Company)
                    RETURN b.ticker AS ticker, b.name AS name, r.rho AS rho
                    ORDER BY abs(r.rho) DESC
                    LIMIT $limit
                """,
                "parameters": {"ticker": ticker.upper(), "limit": limit},
            }])
            return self._rows(result)
        except Exception as exc:
            logger.debug("KnowledgeGraph: correlated_with failed: %s", exc)
            return []

    def query_history(self, user_email: str, limit: int = 20) -> list[dict[str, Any]]:
        """Most-queried companies by this user, with query count."""
        if not self.configured:
            return []
        try:
            result = self._commit([{
                "statement": """
                    MATCH (u:User {email: $email})-[q:QUERIED]->(c:Company)
                    RETURN c.ticker AS ticker, c.name AS name,
                           count(q) AS queries,
                           max(q.ts) AS last_queried
                    ORDER BY queries DESC
                    LIMIT $limit
                """,
                "parameters": {"email": user_email, "limit": limit},
            }])
            return self._rows(result)
        except Exception as exc:
            logger.debug("KnowledgeGraph: query_history failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _commit(self, statements: list[dict[str, Any]]) -> dict[str, Any]:
        endpoint = f"{self.uri}/db/{self.database}/tx/commit"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json={"statements": statements},
                                   auth=(self.user, self.password))
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Neo4j HTTP error: {exc}") from exc

        data   = resp.json()
        errors = data.get("errors") or []
        if errors:
            raise RuntimeError("; ".join(e.get("message", str(e)) for e in errors))
        return data

    @staticmethod
    def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for res in result.get("results") or []:
            cols = res.get("columns", [])
            for datum in res.get("data", []):
                out.append(dict(zip(cols, datum.get("row", []))))
        return out
