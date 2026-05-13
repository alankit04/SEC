# RAPHI Graph Memory

RAPHI now has a Neo4j-backed permanent graph memory layer. It stores durable
interaction facts as graph nodes and retrieves only the relevant context for
future AI calls, so the AI does not need to reread the full conversation.

## Run Neo4j

Docker Desktop must be installed and running. Neo4j itself is isolated to this
project through `docker-compose.neo4j.yml` and stores its database files under
`.docker/neo4j/`, similar to how `.venv/` isolates Python packages.

```bash
export NEO4J_PASSWORD='choose-a-local-password'
docker-compose -f docker-compose.neo4j.yml up -d
```

Then start RAPHI with the same password in the environment:

```bash
export NEO4J_URI='http://localhost:7474'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='choose-a-local-password'
npm start
```

Open Neo4j Browser at `http://localhost:7474` and log in with:

```text
username: neo4j
password: the value of NEO4J_PASSWORD
```

## API

- `GET /api/memory/status` checks Neo4j connectivity.
- `POST /api/memory/remember` writes a memory interaction.
- `GET /api/memory/retrieve?q=...` returns graph-ranked memory context.

The `/api/chat`, `/api/memo/{ticker}`, and A2A executor paths automatically
retrieve memory before generating and write the completed interaction afterward.

## Graph Shape

- `Project`, `User`, `Interaction`, `Memory`, and `Entity` nodes.
- `SAID`, `CREATED`, `MENTIONS`, and `IN_PROJECT` relationships.
- Repeated facts merge into the same `Memory` node and increase `frequency`.
- Retrieval ranks by text match, entity match, importance, and repetition.
