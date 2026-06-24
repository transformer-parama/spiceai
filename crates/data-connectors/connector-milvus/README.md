# connector-milvus

A [Milvus](https://milvus.io) vector-search **data connector** for the
[Spice.ai](https://github.com/spiceai/spiceai) runtime. It exposes a Milvus
collection as a SQL table, so an ANN (nearest-neighbor) search can be expressed
in plain SQL and federated/joined alongside other Spice datasets.

> **Status: experimental.** Functional and verified end-to-end (unit + mock
> network tests; live hybrid queries), but **not part of upstream Spice** and not
> through upstream review. Built against Spice **v2.0.1** (DataFusion 52).

## What it does

Register a Milvus collection as a dataset. Connection params only — the
**schema is discovered by introspecting the collection** (`collections/describe`),
so no per-collection field configuration is needed:

```yaml
# spicepod.yaml
datasets:
  - from: milvus:documents          # milvus:<collection>
    name: documents
    acceleration: { enabled: false } # REQUIRED — see Limitations
    params:
      milvus_host: localhost
      milvus_port: "19530"
      milvus_metric: COSINE          # COSINE | L2 | IP
```

The connector auto-detects the vector field and exposes every scalar column the
collection has (plus the synthetic `query_vector` input and `score` output).
Query it with the query embedding in a `query_vector = '[...]'` predicate;
`LIMIT` is the top-k; equality/`IN` predicates on scalar columns push down into
Milvus as a boolean filter:

```sql
SELECT doc_type, source_id, title, score
FROM documents
WHERE query_vector = '[0.013, -0.021, ...]'   -- the query embedding (JSON array)
  AND doc_type IN ('ticket')                   -- optional → Milvus filter
LIMIT 10;
```

One `scan()` = one Milvus `search()`. Consumed predicates are reported `Exact`.

## Dynamic schema

At dataset registration the connector calls `describe_collection` and builds the
Arrow schema from the result:

- `query_vector` — `Utf8`, input-only (carries the embedding; always NULL in output)
- one column per **scalar** field, typed from Milvus: `Int8/16/32`→`Int32`,
  `Int64`→`Int64`, `Float`→`Float32`, `Double`→`Float64`, `Bool`→`Boolean`,
  `VarChar`/`JSON`/other→`Utf8`. Vector fields are excluded from output.
- `score` — `Float32`, the similarity, **normalized so higher = more relevant**
  for every metric (L2 distances are negated; COSINE/IP pass through).

This works for **any** collection layout — nothing is hardcoded.

## The `query_vector` contract (important)

The query embedding is passed as a **JSON array string** in a
`query_vector = '[...]'` equality predicate. This is a pragmatic interface, not
standard SQL — callers must:

- send a JSON array of floats matching the collection's **vector dimension**;
- expect a clear error if it's missing ("a query embedding is required") or
  malformed ("query_vector must be a JSON array of floats …").

A `milvus_search(...)` UDTF or a typed vector column would be a cleaner
interface; this approach trades elegance for working today.

## Configuration

Params are prefixed with the connector name (`milvus_…`) in `spicepod.yaml`;
Spice strips the prefix. Secrets should come from a Spice secret store.

| Param | Default | Notes |
|---|---|---|
| `host` | `localhost` | Milvus host |
| `port` | `19530` | Milvus HTTP/gRPC port (multiplexed) |
| `secure` | `false` | `true` → TLS (https) |
| `token` | — | Bearer token / API key (secret) |
| `username` / `password` | — | Combined into `username:password` if `token` unset (secret) |
| `timeout_ms` | `10000` | Per-request timeout |
| `connect_timeout_ms` | `3000` | Connection timeout |
| `max_retries` | `2` | Retries on transient failures (timeouts/connection/5xx) |
| `tls_ca_cert` | — | Path to a PEM CA cert to trust for TLS (internal CA / self-signed) |
| `tls_skip_verify` | `false` | Skip TLS cert verification (DANGER; dev / self-signed only) |
| `metric` | `COSINE` | `COSINE` \| `L2` \| `IP` |
| `vector_field` | *(auto-detected)* | Override the float-vector field to search |
| `output_fields` | *(all scalar fields)* | Restrict the returned scalar columns |

## Design notes

- **Transport:** Milvus REST v2 over a single pooled, timeout-bounded `reqwest`
  client (auth + TLS) — no protobuf/tonic.
- **Resilience:** transient failures (timeouts, connection resets, **5xx**)
  retried with **jittered** exponential backoff; API errors (`code != 0`) and
  4xx are returned immediately. `describe` at registration is a startup
  reachability/existence check.
- **Pushdown:** `query_vector`, `product_id =`, `doc_type =`/`IN`, and column
  **projection** (only the projected scalar fields are fetched from Milvus).
- **Observability:** `tracing` spans + OpenTelemetry metrics under the
  `connector_milvus` meter (`milvus_search_requests` / `_errors` / `_retries` /
  `_duration_ms`).

## Build & test

```bash
# unit + mock-network tests (wiremock; no live Milvus needed)
cargo test -p connector-milvus

# build into spiced (from the spiceai workspace root)
make -C bin/spiced SPICED_CUSTOM_FEATURES="postgres milvus"
```

Tests cover: predicate extraction & Milvus-filter building, distance→score
mapping (incl. L2 sign flip), retry-on-5xx but not on `code != 0`, bearer-auth
header, malformed-JSON handling, and `describe` parsing/vector detection.

## Limitations

- **Acceleration must be disabled** (`acceleration.enabled: false`) — else Spice
  materializes the rows into DuckDB/SQLite and the ANN index is lost.
- **Query interface is a convention** (`query_vector = '[...]'`), not standard
  SQL — see the contract section above.
- **No live-Milvus integration test in this crate's CI** — covered by the app's
  `scripts/integration_test.py` against a deployed stack.
- **Fork connector** — pinned to Spice v2.0.1; carries rebase maintenance until
  upstreamed.

## License

Apache-2.0, matching the Spice.ai project.
