/*
Copyright 2026 The Spice.ai OSS Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//! Async Neo4j client over the HTTP transactional Cypher API.
//!
//! Neo4j exposes `POST /db/<database>/tx/commit` (default HTTP port 7474) which
//! takes Cypher statements as JSON and returns column-oriented rows -- so no Bolt
//! driver is needed.
//!
//! [`Neo4jConnection`] owns a pooled, timeout-bounded `reqwest::Client` shared
//! across datasets/queries. Transient transport failures (timeouts, connection
//! resets, 5xx) are retried with jittered exponential backoff; Cypher/API errors
//! are returned immediately. Auth via HTTP Basic (username/password); TLS via the
//! `secure` flag. Labels are introspected via [`Neo4jConnection::describe_label`]
//! so the connector builds its Arrow schema dynamically. OpenTelemetry metrics
//! are emitted under the `connector_neo4j` meter.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Duration;

use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::{global, KeyValue};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use snafu::Snafu;

// ---- OpenTelemetry metrics (flow into Spice's global meter provider) ----
static METER: LazyLock<Meter> = LazyLock::new(|| global::meter("connector_neo4j"));
static QUERY_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("neo4j_query_requests").with_description("Neo4j Cypher queries").build()
});
static QUERY_ERRORS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("neo4j_query_errors").with_description("Failed Neo4j queries").build()
});
static QUERY_RETRIES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("neo4j_query_retries").with_description("Neo4j query retries").build()
});
static QUERY_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    METER
        .f64_histogram("neo4j_query_duration_ms")
        .with_description("Neo4j query wall-clock duration (ms)")
        .build()
});

/// How to *reach* Neo4j. Built once from `spicepod.yaml` params and shared.
pub struct ConnectionConfig {
    pub host: String,
    pub port: u16,
    pub secure: bool,
    pub database: String,
    /// HTTP Basic credentials. None = no auth (e.g. auth disabled).
    pub username: Option<String>,
    pub password: Option<String>,
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub max_retries: u32,
    /// Skip TLS certificate verification (DANGER; dev / self-signed only).
    pub tls_skip_verify: bool,
    /// Path to a PEM CA certificate to trust (internal CA / self-signed server).
    pub tls_ca_cert_path: Option<String>,
}

/// A property discovered by label introspection.
#[derive(Clone, Debug)]
pub struct PropertyField {
    pub name: String,
    pub type_name: String, // Neo4j type, e.g. "String", "Long", "Double", "Boolean"
}

/// A pooled connection to a Neo4j deployment (cheap to clone via `Arc` inside).
#[derive(Clone)]
pub struct Neo4jConnection {
    http: reqwest::Client,
    base_url: String,
    database: String,
    auth: Option<(String, String)>,
    max_retries: u32,
    retry_base: Duration,
}

impl std::fmt::Debug for Neo4jConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neo4jConnection")
            .field("base_url", &self.base_url)
            .field("database", &self.database)
            .field("authenticated", &self.auth.is_some())
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

#[derive(Debug, Snafu)]
pub enum Neo4jError {
    #[snafu(display("neo4j client build error: {message}"))]
    Build { message: String },
    #[snafu(display("neo4j http error: {source}"), context(false))]
    Http { source: reqwest::Error },
    /// A Cypher/server error (Neo4j error codes are strings, e.g.
    /// `Neo.ClientError.Statement.SyntaxError`).
    #[snafu(display("neo4j api error {code}: {message}"))]
    Api { code: String, message: String },
    #[snafu(display("neo4j decode error: {message}"))]
    Decode { message: String },
}

impl Neo4jError {
    /// Retry transport failures and 5xx; never retry Cypher/API errors or 4xx --
    /// those fail identically every time.
    fn is_retryable(&self) -> bool {
        match self {
            Neo4jError::Http { source } => {
                source.is_timeout()
                    || source.is_connect()
                    || source.is_request()
                    || source.status().is_some_and(|s| s.is_server_error())
            }
            _ => false,
        }
    }
}

// ---- HTTP tx/commit response shapes ----
#[derive(Deserialize)]
struct TxResponse {
    #[serde(default)]
    results: Vec<TxResult>,
    #[serde(default)]
    errors: Vec<TxError>,
}

#[derive(Deserialize)]
struct TxResult {
    #[serde(default)]
    columns: Vec<String>,
    #[serde(default)]
    data: Vec<TxRow>,
}

#[derive(Deserialize)]
struct TxRow {
    #[serde(default)]
    row: Vec<Value>,
}

#[derive(Deserialize)]
struct TxError {
    #[serde(default)]
    code: String,
    #[serde(default)]
    message: String,
}

impl Neo4jConnection {
    pub fn new(cfg: ConnectionConfig) -> Result<Self, Neo4jError> {
        let mut builder = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .connect_timeout(cfg.connect_timeout);
        if cfg.tls_skip_verify {
            builder = builder.danger_accept_invalid_certs(true);
        }
        if let Some(path) = &cfg.tls_ca_cert_path {
            let pem = std::fs::read(path)
                .map_err(|e| BuildSnafu { message: format!("read tls_ca_cert '{path}': {e}") }.build())?;
            let cert = reqwest::Certificate::from_pem(&pem)
                .map_err(|e| BuildSnafu { message: format!("parse tls_ca_cert '{path}': {e}") }.build())?;
            builder = builder.add_root_certificate(cert);
        }
        let http = builder.build().map_err(|e| BuildSnafu { message: e.to_string() }.build())?;
        let scheme = if cfg.secure { "https" } else { "http" };
        let auth = match (cfg.username, cfg.password) {
            (Some(u), Some(p)) => Some((u, p)),
            (Some(u), None) => Some((u, String::new())),
            _ => None,
        };
        Ok(Self {
            http,
            base_url: format!("{scheme}://{}:{}", cfg.host, cfg.port),
            database: cfg.database,
            auth,
            max_retries: cfg.max_retries,
            retry_base: Duration::from_millis(200),
        })
    }

    fn authed(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth {
            Some((u, p)) => req.basic_auth(u, Some(p)),
            None => req,
        }
    }

    /// Introspect a node label's properties so the connector can build its Arrow
    /// schema dynamically. Doubles as a reachability/existence check at dataset
    /// registration. Uses the built-in `db.schema.nodeTypeProperties()`.
    ///
    /// `db.schema.nodeTypeProperties()` reports one row per *node-label combination*,
    /// not per label. A node that carries several labels (e.g. GraphRAG's
    /// `:Entity:asst_xyz:performance_metric`) makes the same property (`id`, `name`,
    /// …) appear once for EVERY combination the target label participates in. So the
    /// results are de-duplicated by property name here — otherwise the Arrow schema
    /// would hold duplicate fields and every generated `RETURN n.id AS id, …, n.id AS
    /// id` would be rejected by Neo4j ("Multiple result columns with the same name").
    /// When a property is seen with more than one type (across combinations, or a
    /// multi-type `propertyTypes` array), the types are MERGED the same way sampled
    /// values are in [`describe_cypher`]: Int+Float widen to Double, anything else
    /// falls back to String (lossless, since the row builder stringifies). Insertion
    /// order of first appearance is preserved for a stable column order.
    pub async fn describe_label(&self, label: &str) -> Result<Vec<PropertyField>, Neo4jError> {
        let cypher = "CALL db.schema.nodeTypeProperties() \
                      YIELD nodeLabels, propertyName, propertyTypes \
                      WHERE $label IN nodeLabels \
                      RETURN propertyName, propertyTypes";
        let rows = self.run_query(cypher, json!({ "label": label })).await?;
        let mut order: Vec<String> = Vec::new();
        let mut merged: HashMap<String, InferredType> = HashMap::new();
        for row in rows {
            let Some(name) = row.get("propertyName").and_then(Value::as_str) else {
                continue;
            };
            // Merge across every type listed for this row (a property may be reported
            // with more than one concrete type), defaulting to String when absent.
            let row_type = row
                .get("propertyTypes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(InferredType::from_neo4j_name)
                .reduce(InferredType::merge)
                .unwrap_or(InferredType::Str);
            merged
                .entry(name.to_string())
                .and_modify(|t| *t = t.merge(row_type))
                .or_insert_with(|| {
                    order.push(name.to_string());
                    row_type
                });
        }
        Ok(order
            .into_iter()
            .map(|name| {
                let type_name = merged[&name].neo4j_name().to_string();
                PropertyField { name, type_name }
            })
            .collect())
    }

    /// Run a Cypher statement and return the rows as name->value maps.
    pub async fn run_query(
        &self,
        cypher: &str,
        params: Value,
    ) -> Result<Vec<Map<String, Value>>, Neo4jError> {
        Ok(self.run_query_cols(cypher, params).await?.1)
    }

    /// Like [`run_query`] but also returns the result column names (in order,
    /// present even when there are zero rows). Retries transient failures with
    /// jittered backoff and emits OpenTelemetry metrics.
    pub async fn run_query_cols(
        &self,
        cypher: &str,
        params: Value,
    ) -> Result<(Vec<String>, Vec<Map<String, Value>>), Neo4jError> {
        let attrs = [KeyValue::new("database", self.database.clone())];
        QUERY_REQUESTS.add(1, &attrs);
        let t0 = std::time::Instant::now();
        let out = self.run_retrying(cypher, params).await;
        QUERY_DURATION.record(t0.elapsed().as_secs_f64() * 1000.0, &attrs);
        if out.is_err() {
            QUERY_ERRORS.add(1, &attrs);
        }
        out
    }

    /// Infer the schema of an arbitrary read Cypher query by sampling up to
    /// [`SCHEMA_SAMPLE_ROWS`] rows (`CALL { <cypher> } RETURN * LIMIT N`) and
    /// MERGING the JSON types observed for each column across all sampled rows.
    ///
    /// Merging (rather than reading a single row) removes two correctness bugs:
    ///   * the column type no longer depends on which row Neo4j returned first
    ///     (mixed Int/Double now widen to Double instead of silently nulling the
    ///     rows that don't fit the first row's guess), and
    ///   * a single NULL/absent value in the sampled row no longer collapses a
    ///     whole numeric column to String.
    ///
    /// A column with no non-null sample (empty result, or all-null) defaults to
    /// String — the lossless fallback, since the row builder stringifies anything.
    pub async fn describe_cypher(&self, cypher: &str) -> Result<Vec<PropertyField>, Neo4jError> {
        // Strip a trailing statement separator: it is legal in a bare statement but
        // becomes a syntax error once the Cypher is embedded inside `CALL { ... }`.
        let inner = cypher.trim().trim_end_matches(';').trim();
        let probe = format!("CALL {{ {inner} }} RETURN * LIMIT {SCHEMA_SAMPLE_ROWS}");
        let (cols, rows) = self.run_query_cols(&probe, Value::Null).await?;
        Ok(cols
            .into_iter()
            .map(|name| {
                let mut merged: Option<InferredType> = None;
                for r in &rows {
                    match r.get(&name) {
                        Some(v) if !v.is_null() => {
                            let t = InferredType::from_value(v);
                            merged = Some(merged.map_or(t, |m| m.merge(t)));
                        }
                        _ => {}
                    }
                }
                let type_name = merged.map_or("String", InferredType::neo4j_name).to_string();
                PropertyField { name, type_name }
            })
            .collect())
    }

    async fn run_retrying(
        &self,
        cypher: &str,
        params: Value,
    ) -> Result<(Vec<String>, Vec<Map<String, Value>>), Neo4jError> {
        let url = format!("{}/db/{}/tx/commit", self.base_url, self.database);
        let body = json!({
            "statements": [{ "statement": cypher, "parameters": params }]
        });

        let mut attempt: u32 = 0;
        loop {
            match self.try_run(&url, &body).await {
                Ok(out) => return Ok(out),
                Err(e) if attempt < self.max_retries && e.is_retryable() => {
                    QUERY_RETRIES.add(1, &[KeyValue::new("database", self.database.clone())]);
                    let backoff = self.retry_base * 2u32.saturating_pow(attempt);
                    let jitter = Duration::from_millis((rand::random::<f64>() * 200.0) as u64);
                    tracing::warn!(
                        target: "connector_neo4j",
                        attempt, error = %e,
                        "neo4j query failed; retrying after {:?}", backoff + jitter
                    );
                    tokio::time::sleep(backoff + jitter).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn try_run(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<(Vec<String>, Vec<Map<String, Value>>), Neo4jError> {
        let resp = self.authed(self.http.post(url).json(body)).send().await?.error_for_status()?;
        let parsed: TxResponse =
            resp.json().await.map_err(|e| DecodeSnafu { message: e.to_string() }.build())?;
        if let Some(err) = parsed.errors.into_iter().next() {
            return ApiSnafu { code: err.code, message: err.message }.fail();
        }
        let Some(result) = parsed.results.into_iter().next() else {
            return Ok((vec![], vec![]));
        };
        let cols = result.columns;
        let rows: Vec<Map<String, Value>> = result
            .data
            .into_iter()
            .map(|d| {
                let mut map = Map::with_capacity(cols.len());
                for (i, c) in cols.iter().enumerate() {
                    if let Some(v) = d.row.get(i) {
                        map.insert(c.clone(), v.clone());
                    }
                }
                map
            })
            .collect();
        Ok((cols, rows))
    }
}

/// Number of rows sampled to infer a Cypher-passthrough schema. Enough to observe
/// mixed Int/Double columns and skip leading NULLs, cheap enough for a probe.
const SCHEMA_SAMPLE_ROWS: usize = 200;

/// A merge-able type inferred from sampled JSON values, so a column's type is a
/// function of ALL sampled rows, not just the first one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum InferredType {
    Bool,
    Int,
    Float,
    Str,
}

impl InferredType {
    /// Classify one non-null JSON value. Strings, arrays and objects (nodes,
    /// relationships, lists, maps, temporals rendered as text) all map to `Str`,
    /// which the row builder stringifies losslessly.
    fn from_value(v: &Value) -> Self {
        match v {
            Value::Bool(_) => Self::Bool,
            Value::Number(n) if n.is_i64() || n.is_u64() => Self::Int,
            Value::Number(_) => Self::Float,
            _ => Self::Str,
        }
    }

    /// Classify a Neo4j *type name* (from `db.schema.nodeTypeProperties`) into the
    /// same buckets as sampled values, so label-mode schema inference merges types
    /// identically to Cypher-mode. Kept in lock-step with `exec::arrow_type_for`:
    /// only scalar numeric/boolean types get a native bucket; everything else
    /// (String, DateTime, Point, lists, …) is `Str` and rendered as text.
    fn from_neo4j_name(name: &str) -> Self {
        match name {
            "Long" | "Integer" => Self::Int,
            "Double" | "Float" => Self::Float,
            "Boolean" => Self::Bool,
            _ => Self::Str,
        }
    }

    /// Combine two observed types for the same column. Identical types stay put;
    /// Int+Float widen to Float; anything else falls back to Str (never loses data,
    /// since Str stringifies). This is what makes the inferred schema stable across
    /// rows and runs.
    fn merge(self, other: Self) -> Self {
        use InferredType::{Float, Int};
        match (self, other) {
            (a, b) if a == b => a,
            (Int, Float) | (Float, Int) => Float,
            _ => Self::Str,
        }
    }

    fn neo4j_name(self) -> &'static str {
        match self {
            Self::Bool => "Boolean",
            Self::Int => "Long",
            Self::Float => "Double",
            Self::Str => "String",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn inferred_type_merge_is_order_independent_and_widens() {
        use serde_json::json;
        let it = InferredType::from_value;
        // Int then Float, and Float then Int, both widen to Double (no data lost).
        assert_eq!(it(&json!(1)).merge(it(&json!(2.5))).neo4j_name(), "Double");
        assert_eq!(it(&json!(2.5)).merge(it(&json!(1))).neo4j_name(), "Double");
        // Same type is stable.
        assert_eq!(it(&json!(1)).merge(it(&json!(2))).neo4j_name(), "Long");
        assert_eq!(it(&json!(true)).merge(it(&json!(false))).neo4j_name(), "Boolean");
        // Incompatible mix (number + string, or a node/list) collapses to String.
        assert_eq!(it(&json!(1)).merge(it(&json!("x"))).neo4j_name(), "String");
        assert_eq!(it(&json!({"a":1})).neo4j_name(), "String");
        assert_eq!(it(&json!([1, 2])).neo4j_name(), "String");
    }

    fn conn(uri: &str, auth: Option<(&str, &str)>, max_retries: u32) -> Neo4jConnection {
        let hostport = uri.strip_prefix("http://").unwrap();
        let (host, port) = hostport.split_once(':').unwrap();
        Neo4jConnection::new(ConnectionConfig {
            host: host.to_string(),
            port: port.parse().unwrap(),
            secure: false,
            database: "neo4j".to_string(),
            username: auth.map(|(u, _)| u.to_string()),
            password: auth.map(|(_, p)| p.to_string()),
            timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(2),
            max_retries,
            tls_skip_verify: false,
            tls_ca_cert_path: None,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn maps_columns_to_named_rows() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{
                    "columns": ["name", "age"],
                    "data": [{"row": ["Alice", 30]}, {"row": ["Bob", 25]}]
                }],
                "errors": []
            })))
            .mount(&s)
            .await;
        let rows = conn(&s.uri(), None, 0).run_query("MATCH (n) RETURN n", Value::Null).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name").and_then(Value::as_str), Some("Alice"));
        assert_eq!(rows[0].get("age").and_then(Value::as_i64), Some(30));
        assert_eq!(rows[1].get("name").and_then(Value::as_str), Some("Bob"));
    }

    #[tokio::test]
    async fn basic_auth_is_sent() {
        let s = MockServer::start().await;
        // base64("neo4j:secret") == "bmVvNGo6c2VjcmV0"
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .and(header("authorization", "Basic bmVvNGo6c2VjcmV0"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results":[],"errors":[]})))
            .expect(1)
            .mount(&s)
            .await;
        conn(&s.uri(), Some(("neo4j", "secret")), 0).run_query("RETURN 1", Value::Null).await.unwrap();
    }

    #[tokio::test]
    async fn cypher_error_is_not_retried() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [],
                "errors": [{"code":"Neo.ClientError.Statement.SyntaxError","message":"bad cypher"}]
            })))
            .expect(1) // verified on drop: exactly one call, no retry
            .mount(&s)
            .await;
        let err = conn(&s.uri(), None, 3).run_query("MATCH", Value::Null).await.unwrap_err();
        assert!(matches!(err, Neo4jError::Api { .. }));
    }

    #[tokio::test]
    async fn transient_5xx_is_retried_then_succeeds() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&s)
            .await;
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results":[],"errors":[]})))
            .mount(&s)
            .await;
        let rows = conn(&s.uri(), None, 3).run_query("RETURN 1", Value::Null).await.unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn malformed_json_is_decode_error() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&s)
            .await;
        let err = conn(&s.uri(), None, 0).run_query("RETURN 1", Value::Null).await.unwrap_err();
        assert!(matches!(err, Neo4jError::Decode { .. }));
    }

    #[tokio::test]
    async fn describe_label_parses_properties() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{
                    "columns": ["propertyName", "propertyTypes"],
                    "data": [
                        {"row": ["name", ["String"]]},
                        {"row": ["age", ["Long"]]}
                    ]
                }],
                "errors": []
            })))
            .mount(&s)
            .await;
        let props = conn(&s.uri(), None, 0).describe_label("Person").await.unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(props[0].name, "name");
        assert_eq!(props[0].type_name, "String");
        assert_eq!(props[1].name, "age");
        assert_eq!(props[1].type_name, "Long");
    }

    #[tokio::test]
    async fn describe_label_dedupes_multi_label_properties_and_merges_types() {
        // Real GraphRAG shape: `Entity` nodes also carry a per-tenant label and a
        // per-type label, so db.schema.nodeTypeProperties() reports the SAME property
        // once per label-combination. describe_label must collapse those to one field
        // each (else the generated RETURN has duplicate aliases and Neo4j rejects it),
        // and merge a property seen as both Long and Double into Double.
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{
                    "columns": ["propertyName", "propertyTypes"],
                    "data": [
                        {"row": ["id", ["String"]]},
                        {"row": ["name", ["String"]]},
                        {"row": ["frequency", ["Long"]]},
                        // second label-combination: same props repeated ...
                        {"row": ["id", ["String"]]},
                        {"row": ["name", ["String"]]},
                        // ... but `frequency` observed as Double here -> widen
                        {"row": ["frequency", ["Double"]]},
                        {"row": ["created_at", ["DateTime"]]}
                    ]
                }],
                "errors": []
            })))
            .mount(&s)
            .await;
        let props = conn(&s.uri(), None, 0).describe_label("Entity").await.unwrap();
        // Four distinct columns, first-seen order preserved.
        let names: Vec<&str> = props.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["id", "name", "frequency", "created_at"]);
        // Long + Double widened to Double; DateTime falls back to String.
        assert_eq!(props[2].type_name, "Double");
        assert_eq!(props[3].type_name, "String");
    }

    #[tokio::test]
    async fn describe_cypher_infers_columns_and_types() {
        let s = MockServer::start().await;
        // the probe (CALL { <cypher> } RETURN * LIMIT 1) returns columns + 1 row
        Mock::given(method("POST"))
            .and(path("/db/neo4j/tx/commit"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{
                    "columns": ["person", "friends"],
                    "data": [{"row": ["Alice", 3]}]
                }],
                "errors": []
            })))
            .mount(&s)
            .await;
        let props = conn(&s.uri(), None, 0)
            .describe_cypher("MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS person, count(b) AS friends")
            .await
            .unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(props[0].name, "person");
        assert_eq!(props[0].type_name, "String");
        assert_eq!(props[1].name, "friends");
        assert_eq!(props[1].type_name, "Long"); // inferred from the sampled integer
    }
}
