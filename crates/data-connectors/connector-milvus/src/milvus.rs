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
//! across datasets/queries. Transient transport failures are retried with
//! jittered exponential backoff; API errors are returned immediately. Auth via
//! bearer token; TLS via the `secure` flag. The connector is collection-agnostic:
//! [`MilvusConnection::describe_collection`] builds the table schema dynamically.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
    /// Only transport-level failures are worth retrying; a bad collection or
    /// malformed request will fail identically every time.
    fn is_retryable(&self) -> bool {
        match self {
            MilvusError::Http(e) => e.is_timeout() || e.is_connect() || e.is_request(),
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
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .connect_timeout(cfg.connect_timeout)
            .build()
            .map_err(|e| MilvusError::Build(e.to_string()))?;
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

    /// Run a single ANN search, retrying transient failures with jittered
    /// backoff. `filter` is a Milvus boolean expression or None.
    pub async fn search(
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
