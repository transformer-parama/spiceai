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

use std::sync::LazyLock;
use std::time::Duration;

use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::{global, KeyValue};
use serde::Deserialize;
use serde_json::{json, Map, Value};

// ---- OpenTelemetry metrics (flow into Spice's global meter provider) ----
static METER: LazyLock<Meter> = LazyLock::new(|| global::meter("connector_neo4j"));
static QUERY_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("neo4j.query.requests").with_description("Neo4j Cypher queries").build()
});
static QUERY_ERRORS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("neo4j.query.errors").with_description("Failed Neo4j queries").build()
});
static QUERY_RETRIES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("neo4j.query.retries").with_description("Neo4j query retries").build()
});
static QUERY_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    METER
        .f64_histogram("neo4j.query.duration_seconds")
        .with_description("Neo4j query wall-clock duration")
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

#[derive(Debug)]
pub enum Neo4jError {
    Build(String),
    Http(reqwest::Error),
    /// A Cypher/server error (Neo4j error codes are strings, e.g.
    /// `Neo.ClientError.Statement.SyntaxError`).
    Api { code: String, message: String },
    Decode(String),
}

impl std::fmt::Display for Neo4jError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Neo4jError::Build(m) => write!(f, "neo4j client build error: {m}"),
            Neo4jError::Http(e) => write!(f, "neo4j http error: {e}"),
            Neo4jError::Api { code, message } => write!(f, "neo4j api error {code}: {message}"),
            Neo4jError::Decode(m) => write!(f, "neo4j decode error: {m}"),
        }
    }
}
impl std::error::Error for Neo4jError {}
impl From<reqwest::Error> for Neo4jError {
    fn from(e: reqwest::Error) -> Self {
        Neo4jError::Http(e)
    }
}

impl Neo4jError {
    /// Retry transport failures and 5xx; never retry Cypher/API errors or 4xx --
    /// those fail identically every time.
    fn is_retryable(&self) -> bool {
        match self {
            Neo4jError::Http(e) => {
                e.is_timeout()
                    || e.is_connect()
                    || e.is_request()
                    || e.status().is_some_and(|s| s.is_server_error())
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
                .map_err(|e| Neo4jError::Build(format!("read tls_ca_cert '{path}': {e}")))?;
            let cert = reqwest::Certificate::from_pem(&pem)
                .map_err(|e| Neo4jError::Build(format!("parse tls_ca_cert '{path}': {e}")))?;
            builder = builder.add_root_certificate(cert);
        }
        let http = builder.build().map_err(|e| Neo4jError::Build(e.to_string()))?;
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
    pub async fn describe_label(&self, label: &str) -> Result<Vec<PropertyField>, Neo4jError> {
        let cypher = "CALL db.schema.nodeTypeProperties() \
                      YIELD nodeLabels, propertyName, propertyTypes \
                      WHERE $label IN nodeLabels \
                      RETURN propertyName, propertyTypes";
        let rows = self.run_query(cypher, json!({ "label": label })).await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let Some(name) = row.get("propertyName").and_then(Value::as_str) else {
                continue;
            };
            let type_name = row
                .get("propertyTypes")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(Value::as_str)
                .unwrap_or("String")
                .to_string();
            out.push(PropertyField { name: name.to_string(), type_name });
        }
        Ok(out)
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
        QUERY_DURATION.record(t0.elapsed().as_secs_f64(), &attrs);
        if out.is_err() {
            QUERY_ERRORS.add(1, &attrs);
        }
        out
    }

    /// Infer the schema of an arbitrary read Cypher query by sampling one row
    /// (`CALL { <cypher> } RETURN * LIMIT 1`). Column names come from the result
    /// columns (present even with no rows); types are inferred from the sample
    /// value, defaulting to String.
    pub async fn describe_cypher(&self, cypher: &str) -> Result<Vec<PropertyField>, Neo4jError> {
        let probe = format!("CALL {{ {cypher} }} RETURN * LIMIT 1");
        let (cols, rows) = self.run_query_cols(&probe, Value::Null).await?;
        Ok(cols
            .into_iter()
            .map(|name| {
                let type_name = rows
                    .first()
                    .and_then(|r| r.get(&name))
                    .map(neo4j_type_of)
                    .unwrap_or("String")
                    .to_string();
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
            resp.json().await.map_err(|e| Neo4jError::Decode(e.to_string()))?;
        if let Some(err) = parsed.errors.into_iter().next() {
            return Err(Neo4jError::Api { code: err.code, message: err.message });
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

/// Infer a Neo4j-ish type name from a sampled JSON value (for Cypher-passthrough
/// schema inference). Maps onward via `exec::arrow_type_for`.
fn neo4j_type_of(v: &Value) -> &'static str {
    match v {
        Value::Bool(_) => "Boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "Long"
            } else {
                "Double"
            }
        }
        _ => "String",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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
        assert!(matches!(err, Neo4jError::Decode(_)));
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
