# connector-neo4j

A [Neo4j](https://neo4j.com) graph **data connector** for the
[Spice.ai](https://github.com/spiceai/spiceai) runtime. It exposes a Neo4j node
label as a SQL table, so nodes can be queried in plain SQL and federated/joined
alongside other Spice datasets.

> **Status: experimental.** Two modes implemented and verified (unit +
> mock-network + live end-to-end): **node-label tables** and **Cypher-passthrough**
> (relationship traversals). Not part of upstream Spice. Built against Spice
> **v2.0.1** (DataFusion 52).

## What it does

Register a Neo4j node label as a dataset. Connection params only — the **schema
is discovered by introspecting the label** (`db.schema.nodeTypeProperties()`), so
no per-label field configuration is needed:

```yaml
# spicepod.yaml
datasets:
  - from: neo4j:Person          # neo4j:<NodeLabel>
    name: people
    acceleration: { enabled: false }  # REQUIRED — see Limitations
    params:
      neo4j_host: localhost
      neo4j_port: "7474"
      neo4j_database: neo4j
      neo4j_username: neo4j
      neo4j_password: ${secrets:neo4j_password}
```

Query it as a normal table. Comparison/`IN` predicates and the projection push
down into a single Cypher `MATCH`:

```sql
SELECT name, age FROM people WHERE age > 30 LIMIT 10;
-- → MATCH (n:`Person`) WHERE n.`age` > 30 RETURN n.`name` AS name, n.`age` AS age LIMIT 10
```

One `scan()` = one Cypher query. Consumed predicates are reported `Exact`.

## Dynamic schema

At dataset registration the connector introspects the label's property keys/types
and builds the Arrow schema:

- one column per **property**, typed from Neo4j: `Long`/`Integer`→`Int64`,
  `Double`/`Float`→`Float64`, `Boolean`→`Boolean`, `String`→`Utf8`; other types
  (arrays, temporal, spatial) → `Utf8` (stringified).

This works for **any** label — nothing is hardcoded.

## Cypher passthrough (relationship traversals)

For traversals or arbitrary read queries, give the dataset a `cypher` param
instead of using the path as a label. The connector samples the query
(`CALL { <cypher> } RETURN * LIMIT 1`) to infer the schema and exposes the RETURN
columns as a table:

```yaml
  - from: neo4j:knows               # path is just a logical name in this mode
    name: knows
    acceleration: { enabled: false }
    params:
      neo4j_host: localhost
      neo4j_cypher: "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS person, b.name AS friend"
```

```sql
SELECT person, friend FROM knows WHERE person = 'Alice';
```

The Cypher is opaque, so there is **no predicate/LIMIT pushdown** — DataFusion
applies `WHERE`/`LIMIT` on top; projection is still honored (only projected
columns are built). The statement must be a single **read** query.

## Configuration

Params are prefixed with the connector name (`neo4j_…`) in `spicepod.yaml`; Spice
strips the prefix. Secrets should come from a Spice secret store.

| Param | Default | Notes |
|---|---|---|
| `host` | `localhost` | Neo4j host |
| `port` | `7474` | Neo4j HTTP port |
| `secure` | `false` | `true` → TLS (https) |
| `database` | `neo4j` | Database name |
| `username` | — | HTTP Basic username (secret) |
| `password` | — | HTTP Basic password (secret) |
| `timeout_ms` | `10000` | Per-request timeout |
| `connect_timeout_ms` | `3000` | Connection timeout |
| `max_retries` | `2` | Retries on transient failures (timeouts/connection/5xx) |
| `cypher` | — | Read Cypher defining the dataset → Cypher-passthrough mode (else path = node label) |

## Design notes

- **Transport:** Neo4j HTTP transactional Cypher API
  (`POST /db/<database>/tx/commit`) over a single pooled, timeout-bounded
  `reqwest` client (HTTP Basic auth + TLS) — no Bolt driver.
- **Resilience:** transient failures (timeouts, connection resets, **5xx**)
  retried with **jittered** exponential backoff; Cypher/API errors and 4xx are
  returned immediately. `describe` at registration is a startup
  reachability/existence check.
- **Pushdown:** comparison (`=`,`<>`,`<`,`<=`,`>`,`>=`) / `IN` predicates on
  property columns, column **projection** (RETURN only projected props), and
  `LIMIT`.
- **Observability:** `tracing` spans + OpenTelemetry metrics under the
  `connector_neo4j` meter (`neo4j.query.requests` / `.errors` / `.retries` /
  `.duration_seconds`).

## Build & test

```bash
# unit + mock-network tests (wiremock; no live Neo4j needed)
cargo test -p connector-neo4j

# build into spiced (from the spiceai workspace root)
cargo build --release -p spiced --features neo4j
```

Tests cover: column→row mapping, HTTP Basic auth header, Cypher-error vs 5xx retry
behavior, malformed-JSON handling, label introspection parsing, and predicate
pushdown / Cypher literal escaping.

## Limitations

- **Cypher-passthrough mode has no pushdown** — the query is opaque, so `WHERE`,
  `LIMIT`, and joins are applied by DataFusion on top (the full result is fetched
  first). Use a bounded/efficient query. Label mode *does* push down predicates.
- **Acceleration must be disabled** (`acceleration.enabled: false`) — else Spice
  materializes the rows and re-queries go to the accelerator, not Neo4j.
- **A `LIMIT`-less scan returns all matching nodes** (standard table semantics);
  add a `LIMIT` for large labels.
- **Introspection** relies on the built-in `db.schema.nodeTypeProperties()`
  procedure being available and the label having sampled property metadata.
- **Fork connector** — pinned to Spice v2.0.1; carries rebase maintenance until
  upstreamed.

## License

Apache-2.0, matching the Spice.ai project.
