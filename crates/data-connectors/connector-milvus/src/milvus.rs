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

//! Async Milvus client over the RESTful v2 API.
//!
//! Milvus 2.4+ multiplexes HTTP and gRPC on the same port (default 19530), so
//! ANN search and schema introspection are JSON POSTs -- no protobuf/tonic.
//!
//! [`MilvusConnection`] owns a pooled, timeout-bounded `reqwest::Client` shared
//! across datasets/queries. Transient transport failures (timeouts, connection
//! resets, 5xx) are retried with jittered exponential backoff; API errors are
//! returned immediately. Auth via bearer token; TLS via the `secure` flag. The
//! connector is collection-agnostic via [`MilvusConnection::describe_collection`].
//! OpenTelemetry metrics are emitted under the `connector_milvus` meter.

use std::sync::LazyLock;
use std::time::Duration;

use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::{global, KeyValue};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ---- OpenTelemetry metrics (flow into Spice's global meter provider) ----
static METER: LazyLock<Meter> = LazyLock::new(|| global::meter("connector_milvus"));
static SEARCH_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("milvus_search_requests").with_description("Milvus ANN searches").build()
});
static SEARCH_ERRORS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("milvus_search_errors").with_description("Failed Milvus searches").build()
});
static SEARCH_RETRIES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER.u64_counter("milvus_search_retries").with_description("Milvus search retries").build()
});
static SEARCH_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    METER
        .f64_histogram("milvus_search_duration_ms")
        .with_description("Milvus search wall-clock duration (ms)")
        .build()
});

/// How to *reach* Milvus. Built once from `spicepod.yaml` params and shared.
pub struct ConnectionConfig {
    pub host: String,
    pub port: u16,
    pub secure: bool,
    /// Bearer token (`username:password`, or an API key). None = no auth.
    pub token: Option<String>,
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub max_retries: u32,
    /// Skip TLS certificate verification (DANGER; dev / self-signed only).
    pub tls_skip_verify: bool,
    /// Path to a PEM CA certificate to trust (internal CA / self-signed server).
    pub tls_ca_cert_path: Option<String>,
}

/// What to *query* -- one Milvus collection.
#[derive(Clone, Debug)]
pub struct MilvusCollection {
    pub collection: String,
    pub vector_field: String,
    pub metric: String, // COSINE | L2 | IP
    pub output_fields: Vec<String>,
}

impl MilvusCollection {
    /// Milvus L2 is a distance (lower = closer); COSINE/IP are similarities
    /// (higher = closer). We normalize so `score` is ALWAYS higher-is-better.
    fn is_distance_metric(&self) -> bool {
        self.metric.eq_ignore_ascii_case("L2")
    }
}

/// A field discovered by collection introspection.
#[derive(Clone, Debug)]
pub struct CollectionField {
    pub name: String,
    pub type_name: String, // Milvus type, e.g. "Int64", "VarChar", "FloatVector"
    pub is_vector: bool,
}

/// A pooled connection to a Milvus deployment (cheap to clone via `Arc` inside).
#[derive(Clone)]
pub struct MilvusConnection {
    http: reqwest::Client,
    base_url: String,
    token: Option<String>,
    max_retries: u32,
    retry_base: Duration,
}

impl std::fmt::Debug for MilvusConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MilvusConnection")
            .field("base_url", &self.base_url)
            .field("authenticated", &self.token.is_some())
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

#[derive(Debug)]
pub enum MilvusError {
    Build(String),
    Http(reqwest::Error),
    Api { code: i64, message: String },
    Decode(String),
}

impl std::fmt::Display for MilvusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MilvusError::Build(m) => write!(f, "milvus client build error: {m}"),
            MilvusError::Http(e) => write!(f, "milvus http error: {e}"),
            MilvusError::Api { code, message } => write!(f, "milvus api error {code}: {message}"),
            MilvusError::Decode(m) => write!(f, "milvus decode error: {m}"),
        }
    }
}
impl std::error::Error for MilvusError {}
impl From<reqwest::Error> for MilvusError {
    fn from(e: reqwest::Error) -> Self {
        MilvusError::Http(e)
    }
}

impl MilvusError {
    /// Retry transport failures and 5xx; never retry API errors (code != 0) or
    /// 4xx -- those fail identically every time.
    fn is_retryable(&self) -> bool {
        match self {
            MilvusError::Http(e) => {
                e.is_timeout()
                    || e.is_connect()
                    || e.is_request()
                    || e.status().is_some_and(|s| s.is_server_error())
            }
            _ => false,
        }
    }
}

/// One ANN hit: the output-field values plus the (normalized) similarity score.
#[derive(Debug, Clone)]
pub struct Hit {
    pub fields: serde_json::Map<String, Value>,
    pub score: f32,
}

#[derive(Serialize)]
struct SearchBody<'a> {
    #[serde(rename = "collectionName")]
    collection_name: &'a str,
    data: Vec<Vec<f32>>,
    #[serde(rename = "annsField")]
    anns_field: &'a str,
    limit: usize,
    #[serde(rename = "outputFields")]
    output_fields: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<String>,
    #[serde(rename = "searchParams")]
    search_params: Value,
}

#[derive(Deserialize)]
struct SearchResponse {
    code: i64,
    #[serde(default)]
    message: String,
    #[serde(default)]
    data: Vec<serde_json::Map<String, Value>>,
}

#[derive(Deserialize, Default)]
struct DescribeData {
    #[serde(default)]
    fields: Vec<DescribeField>,
}

#[derive(Deserialize)]
struct DescribeField {
    name: String,
    #[serde(rename = "type", default)]
    type_name: String,
}

#[derive(Deserialize)]
struct DescribeResponse {
    code: i64,
    #[serde(default)]
    message: String,
    #[serde(default)]
    data: DescribeData,
}

impl MilvusConnection {
    pub fn new(cfg: ConnectionConfig) -> Result<Self, MilvusError> {
        let mut builder = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .connect_timeout(cfg.connect_timeout);
        if cfg.tls_skip_verify {
            builder = builder.danger_accept_invalid_certs(true);
        }
        if let Some(path) = &cfg.tls_ca_cert_path {
            let pem = std::fs::read(path)
                .map_err(|e| MilvusError::Build(format!("read tls_ca_cert '{path}': {e}")))?;
            let cert = reqwest::Certificate::from_pem(&pem)
                .map_err(|e| MilvusError::Build(format!("parse tls_ca_cert '{path}': {e}")))?;
            builder = builder.add_root_certificate(cert);
        }
        let http = builder.build().map_err(|e| MilvusError::Build(e.to_string()))?;
        let scheme = if cfg.secure { "https" } else { "http" };
        Ok(Self {
            http,
            base_url: format!("{scheme}://{}:{}", cfg.host, cfg.port),
            token: cfg.token,
            max_retries: cfg.max_retries,
            retry_base: Duration::from_millis(200),
        })
    }

    fn authed(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(t) => req.bearer_auth(t),
            None => req,
        }
    }

    /// Introspect a collection's fields so the connector can build its Arrow
    /// schema dynamically (works for ANY collection layout). Doubles as a
    /// reachability/existence check at dataset registration.
    pub async fn describe_collection(
        &self,
        collection: &str,
    ) -> Result<Vec<CollectionField>, MilvusError> {
        let url = format!("{}/v2/vectordb/collections/describe", self.base_url);
        let resp = self
            .authed(self.http.post(url).json(&json!({ "collectionName": collection })))
            .send()
            .await?
            .error_for_status()?;
        let parsed: DescribeResponse = resp
            .json()
            .await
            .map_err(|e| MilvusError::Decode(e.to_string()))?;
        if parsed.code != 0 {
            return Err(MilvusError::Api { code: parsed.code, message: parsed.message });
        }
        Ok(parsed
            .data
            .fields
            .into_iter()
            .map(|f| CollectionField {
                is_vector: f.type_name.to_lowercase().contains("vector"),
                name: f.name,
                type_name: f.type_name,
            })
            .collect())
    }

    /// Run a single ANN search (metric-aware score), retrying transient failures
    /// with jittered backoff. Emits OpenTelemetry metrics.
    pub async fn search(
        &self,
        coll: &MilvusCollection,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<String>,
    ) -> Result<Vec<Hit>, MilvusError> {
        let attrs = [KeyValue::new("collection", coll.collection.clone())];
        SEARCH_REQUESTS.add(1, &attrs);
        let t0 = std::time::Instant::now();
        let out = self.search_retrying(coll, vector, limit, filter).await;
        SEARCH_DURATION.record(t0.elapsed().as_secs_f64() * 1000.0, &attrs);
        if out.is_err() {
            SEARCH_ERRORS.add(1, &attrs);
        }
        out
    }

    async fn search_retrying(
        &self,
        coll: &MilvusCollection,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<String>,
    ) -> Result<Vec<Hit>, MilvusError> {
        let url = format!("{}/v2/vectordb/entities/search", self.base_url);
        let body = SearchBody {
            collection_name: &coll.collection,
            data: vec![vector],
            anns_field: &coll.vector_field,
            limit,
            output_fields: &coll.output_fields,
            filter,
            search_params: json!({ "metricType": coll.metric }),
        };
        let flip = coll.is_distance_metric();

        let mut attempt: u32 = 0;
        loop {
            match self.try_search(&url, &body, flip).await {
                Ok(hits) => return Ok(hits),
                Err(e) if attempt < self.max_retries && e.is_retryable() => {
                    SEARCH_RETRIES.add(1, &[KeyValue::new("collection", coll.collection.clone())]);
                    let backoff = self.retry_base * 2u32.saturating_pow(attempt);
                    let jitter = Duration::from_millis((rand::random::<f64>() * 200.0) as u64);
                    tracing::warn!(
                        target: "connector_milvus",
                        attempt, collection = %coll.collection, error = %e,
                        "milvus search failed; retrying after {:?}", backoff + jitter
                    );
                    tokio::time::sleep(backoff + jitter).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn try_search(
        &self,
        url: &str,
        body: &SearchBody<'_>,
        flip_score: bool,
    ) -> Result<Vec<Hit>, MilvusError> {
        let resp = self.authed(self.http.post(url).json(body)).send().await?.error_for_status()?;
        let parsed: SearchResponse = resp
            .json()
            .await
            .map_err(|e| MilvusError::Decode(e.to_string()))?;
        if parsed.code != 0 {
            return Err(MilvusError::Api { code: parsed.code, message: parsed.message });
        }
        Ok(parsed
            .data
            .into_iter()
            .map(|mut row| {
                let raw = row
                    .remove("distance")
                    .or_else(|| row.remove("score"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                // normalize so higher = more relevant for every metric
                let score = if flip_score { -raw } else { raw };
                Hit { fields: row, score }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn conn(uri: &str, token: Option<&str>, max_retries: u32) -> MilvusConnection {
        let hostport = uri.strip_prefix("http://").unwrap();
        let (host, port) = hostport.split_once(':').unwrap();
        MilvusConnection::new(ConnectionConfig {
            host: host.to_string(),
            port: port.parse().unwrap(),
            secure: false,
            token: token.map(str::to_string),
            timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(2),
            max_retries,
            tls_skip_verify: false,
            tls_ca_cert_path: None,
        })
        .unwrap()
    }

    fn coll(metric: &str) -> MilvusCollection {
        MilvusCollection {
            collection: "c".to_string(),
            vector_field: "embedding".to_string(),
            metric: metric.to_string(),
            output_fields: vec!["title".to_string()],
        }
    }

    #[tokio::test]
    async fn cosine_score_is_passed_through() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/entities/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"code":0, "data":[{"title":"x","distance":0.9}]}),
            ))
            .mount(&s)
            .await;
        let hits = conn(&s.uri(), None, 0).search(&coll("COSINE"), vec![0.0; 4], 1, None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 0.9).abs() < 1e-6);
        assert_eq!(hits[0].fields.get("title").and_then(Value::as_str), Some("x"));
    }

    #[tokio::test]
    async fn l2_score_is_negated_higher_is_better() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/entities/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"code":0,"data":[{"distance":4.0}]})))
            .mount(&s)
            .await;
        let hits = conn(&s.uri(), None, 0).search(&coll("L2"), vec![0.0; 4], 1, None).await.unwrap();
        assert!((hits[0].score - (-4.0)).abs() < 1e-6);
    }

    #[tokio::test]
    async fn api_error_code_is_not_retried() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/entities/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"code":100, "message":"collection not found"}),
            ))
            .expect(1) // verified on drop: exactly one call, no retry
            .mount(&s)
            .await;
        let err = conn(&s.uri(), None, 3).search(&coll("COSINE"), vec![0.0; 4], 1, None).await.unwrap_err();
        assert!(matches!(err, MilvusError::Api { code: 100, .. }));
    }

    #[tokio::test]
    async fn transient_5xx_is_retried_then_succeeds() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/entities/search"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&s)
            .await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/entities/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"code":0,"data":[]})))
            .mount(&s)
            .await;
        let hits = conn(&s.uri(), None, 3).search(&coll("COSINE"), vec![0.0; 4], 1, None).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn bearer_token_is_sent() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/entities/search"))
            .and(header("authorization", "Bearer secret-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"code":0,"data":[]})))
            .expect(1)
            .mount(&s)
            .await;
        conn(&s.uri(), Some("secret-token"), 0).search(&coll("COSINE"), vec![0.0; 4], 1, None).await.unwrap();
    }

    #[tokio::test]
    async fn malformed_json_is_decode_error() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/entities/search"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&s)
            .await;
        let err = conn(&s.uri(), None, 0).search(&coll("COSINE"), vec![0.0; 4], 1, None).await.unwrap_err();
        assert!(matches!(err, MilvusError::Decode(_)));
    }

    #[tokio::test]
    async fn describe_parses_fields_and_flags_vector() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/vectordb/collections/describe"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "code":0,
                "data":{"fields":[
                    {"name":"id","type":"Int64"},
                    {"name":"title","type":"VarChar"},
                    {"name":"embedding","type":"FloatVector"}
                ]}
            })))
            .mount(&s)
            .await;
        let fields = conn(&s.uri(), None, 0).describe_collection("c").await.unwrap();
        assert_eq!(fields.len(), 3);
        assert!(fields.iter().find(|f| f.name == "embedding").unwrap().is_vector);
        assert!(!fields.iter().find(|f| f.name == "id").unwrap().is_vector);
    }
}
