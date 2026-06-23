# connector-milvus

A [Milvus](https://milvus.io) vector-search **data connector** for the
[Spice.ai](https://github.com/spiceai/spiceai) runtime. It exposes a Milvus
collection as a SQL table, so an ANN (nearest-neighbor) search can be expressed
in plain SQL and federated alongside other Spice datasets.

> **Status: experimental.** This connector is functional and verified
> end-to-end, but it is **not part of upstream Spice** and has **not** been
> through upstream review or production qualification. See **Limitations** below
> before relying on it. Built against the DataFusion version used by Spice
> `trunk` (DataFusion 54).

## What it does

Register a Milvus collection as a dataset:

```yaml
# spicepod.yaml
datasets:
  - from: milvus:documents          # milvus:<collection>
    name: documents
    acceleration: { enabled: false } # REQUIRED — see Limitations
    params:
      milvus_host: localhost
      milvus_port: "19530"
      milvus_vector_field: embedding
      milvus_metric: COSINE
      milvus_output_fields: doc_type,source_id,product_id,title,text
```

Then query it. The query embedding is passed via a `query_vector = '[...]'`
predicate; `LIMIT` becomes the top-k; `product_id` / `doc_type` predicates are
pushed down into Milvus as a boolean filter:

```sql
SELECT doc_type, source_id, title, score
FROM documents
WHERE query_vector = '[0.013, -0.021, ...]'   -- the query embedding (JSON array)
  AND doc_type IN ('ticket')                   -- optional → Milvus filter
LIMIT 10;
```

All consumed predicates are reported `Exact`, so DataFusion does not re-apply
them. One `scan()` maps to exactly one Milvus `search()` call.

## Configuration

Params are prefixed with the connector name (`milvus_…`) in `spicepod.yaml`;
Spice strips the prefix. Secrets should come from a Spice secret store.

| Param | Default | Notes |
|---|---|---|
| `host` | `localhost` | Milvus host |
| `port` | `19530` | Milvus HTTP/gRPC port (multiplexed) |
| `secure` | `false` | `true` → use TLS (https) |
| `token` | — | Bearer token / API key (secret) |
| `username` / `password` | — | Combined into `username:password` if `token` unset (secret) |
| `timeout_ms` | `10000` | Per-request timeout |
| `connect_timeout_ms` | `3000` | Connection timeout |
| `max_retries` | `2` | Retries on transient transport failures |
| `vector_field` | `embedding` | Float-vector field to search |
| `metric` | `COSINE` | `COSINE` \| `L2` \| `IP` |
| `output_fields` | `doc_type,source_id,product_id,title,text` | Scalar fields to return |

## Design notes

- **Transport:** Milvus 2.4+ REST v2 (`POST /v2/vectordb/entities/search`) over a
  single pooled `reqwest` client — no protobuf/tonic vendoring.
- **Resilience:** request + connect timeouts; transient failures (timeouts,
  connection resets) are retried with exponential backoff. API errors (e.g.
  "collection not found") are returned immediately.
- **Observability:** `tracing` spans around each search (`target =
  connector_milvus`).
- **Pushdown:** `query_vector`, `product_id =`, and `doc_type =`/`IN` only.

## Layout

| File | Role |
|---|---|
| `src/milvus.rs` | Pooled async Milvus REST client (auth, TLS, timeouts, retries) |
| `src/exec.rs` | `MilvusExec` — `ExecutionPlan`; one search → one Arrow `RecordBatch` |
| `src/table_provider.rs` | `MilvusTableProvider` — schema, pushdown, `scan()` (+ unit tests) |
| `src/lib.rs` | `DataConnectorFactory` / `DataConnector` + config parsing |

## Build & test

```bash
# unit tests (no network needed)
cargo test -p connector-milvus

# build it into spiced (from the spiceai workspace root)
make -C bin/spiced SPICED_CUSTOM_FEATURES="postgres milvus"
```

## Limitations (read before production use)

- **Acceleration must be disabled** (`acceleration.enabled: false`) — Spice's
  accelerator would materialize the rows into DuckDB/SQLite and lose the ANN
  index.
- **Query interface is a convention, not standard SQL** — the search vector
  travels as a JSON string in a `query_vector = '...'` predicate. This works but
  is a pragmatic interface (length/parse limits); a `milvus_search(...)` UDTF or
  proper array binding would be cleaner.
- **Schema is fixed** to `(doc_type, source_id, product_id, title, text, score)`.
  Reusing this connector for a differently-shaped collection requires editing the
  schema in `exec.rs`.
- **No integration test** against a live Milvus in CI yet (unit tests cover the
  pure predicate logic only).
- **Built against Spice `trunk`** (unstable). Pin to a released Spice tag before
  production.

## License

Apache-2.0, matching the Spice.ai project.
