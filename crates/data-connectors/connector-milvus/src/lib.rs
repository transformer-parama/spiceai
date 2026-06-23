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

//! Milvus data connector for the Spice.ai runtime.
//!
//! Exposes a Milvus collection as a SQL table; an ANN search is expressed as
//! `... WHERE query_vector = '[...]' [AND product_id = N] [AND doc_type IN (...)]
//! LIMIT k`. The query vector, filters, and limit are pushed into a single
//! Milvus `search()` call. Keep `acceleration.enabled: false` for the dataset.
//!
//! A single pooled, timeout-bounded connection (with optional bearer auth + TLS
//! and transient-failure retries) is built once and shared across all datasets
//! and queries served by this connector.

mod exec;
mod milvus;
mod table_provider;

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::datasource::TableProvider;
use runtime::component::dataset::Dataset;
use runtime::dataconnector::{
    ConnectorParams, DataConnector, DataConnectorError, DataConnectorFactory, DataConnectorResult,
};
use runtime::parameters::ParameterSpec;
use secrecy::ExposeSecret;

use crate::milvus::{ConnectionConfig, MilvusCollection, MilvusConnection};
use crate::table_provider::MilvusTableProvider;

/// Milvus data connector. Holds a shared pooled connection plus the collection
/// field-mapping; the collection name is resolved per dataset in `read_provider`.
pub struct MilvusConnector {
    conn: Arc<MilvusConnection>,
    vector_field: String,
    metric: String,
    output_fields: Vec<String>,
}

impl std::fmt::Debug for MilvusConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MilvusConnector")
            .field("conn", &self.conn)
            .finish_non_exhaustive()
    }
}

#[derive(Default, Copy, Clone)]
pub struct MilvusFactory {}

impl MilvusFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    #[must_use]
    pub fn new_arc() -> Arc<dyn DataConnectorFactory> {
        Arc::new(Self {}) as Arc<dyn DataConnectorFactory>
    }
}

const PARAMETERS: &[ParameterSpec] = &[
    ParameterSpec::component("host")
        .description("Milvus host.")
        .default("localhost"),
    ParameterSpec::component("port")
        .description("Milvus HTTP/gRPC port (multiplexed).")
        .default("19530"),
    ParameterSpec::component("secure")
        .description("Use TLS (https) to reach Milvus.")
        .default("false"),
    ParameterSpec::component("token")
        .description("Bearer token (or 'username:password' / API key) for auth.")
        .secret(),
    ParameterSpec::component("username")
        .description("Username (combined with 'password' into a token if 'token' is unset).")
        .secret(),
    ParameterSpec::component("password")
        .description("Password (used with 'username').")
        .secret(),
    ParameterSpec::component("timeout_ms")
        .description("Per-request timeout in milliseconds.")
        .default("10000"),
    ParameterSpec::component("connect_timeout_ms")
        .description("Connection timeout in milliseconds.")
        .default("3000"),
    ParameterSpec::component("max_retries")
        .description("Retries for transient transport failures (timeouts/connection errors).")
        .default("2"),
    ParameterSpec::component("vector_field")
        .description("Name of the float-vector field to search.")
        .default("embedding"),
    ParameterSpec::component("metric")
        .description("Distance metric: COSINE | L2 | IP.")
        .default("COSINE"),
    ParameterSpec::component("output_fields")
        .description("Comma-separated scalar fields to return.")
        .default("doc_type,source_id,product_id,title,text"),
];

impl DataConnectorFactory for MilvusFactory {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn create(
        &self,
        params: ConnectorParams,
    ) -> Pin<Box<dyn Future<Output = runtime::dataconnector::NewDataConnectorResult> + Send>> {
        Box::pin(async move {
            // Non-empty component param, as a plain String, or None.
            let p = |key: &str| -> Option<String> {
                params
                    .parameters
                    .get(key)
                    .ok()
                    .map(|s| s.expose_secret().to_string())
                    .filter(|s| !s.is_empty())
            };
            let connect_err = |msg: String| DataConnectorError::UnableToConnectInternal {
                dataconnector: "milvus".to_string(),
                connector_component: params.component.clone(),
                source: Box::<dyn std::error::Error + Send + Sync>::from(msg),
            };

            let port: u16 = p("port")
                .unwrap_or_else(|| "19530".to_string())
                .parse()
                .map_err(|e| connect_err(format!("invalid port: {e}")))?;

            // Prefer an explicit token; otherwise derive one from username:password.
            let token = p("token").or_else(|| match (p("username"), p("password")) {
                (Some(u), Some(pw)) => Some(format!("{u}:{pw}")),
                _ => None,
            });

            let u64_or = |key: &str, default: u64| -> u64 {
                p(key).and_then(|v| v.parse().ok()).unwrap_or(default)
            };
            let bool_flag = |key: &str| -> bool {
                p(key).map(|v| matches!(v.as_str(), "true" | "1" | "yes")).unwrap_or(false)
            };

            let cfg = ConnectionConfig {
                host: p("host").unwrap_or_else(|| "localhost".to_string()),
                port,
                secure: bool_flag("secure"),
                token,
                timeout: Duration::from_millis(u64_or("timeout_ms", 10_000)),
                connect_timeout: Duration::from_millis(u64_or("connect_timeout_ms", 3_000)),
                max_retries: u64_or("max_retries", 2) as u32,
            };
            let conn = Arc::new(MilvusConnection::new(cfg).map_err(|e| connect_err(e.to_string()))?);

            let output_fields = p("output_fields")
                .unwrap_or_else(|| "doc_type,source_id,product_id,title,text".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            Ok(Arc::new(MilvusConnector {
                conn,
                vector_field: p("vector_field").unwrap_or_else(|| "embedding".to_string()),
                metric: p("metric").unwrap_or_else(|| "COSINE".to_string()),
                output_fields,
            }) as Arc<dyn DataConnector>)
        })
    }

    fn prefix(&self) -> &'static str {
        "milvus"
    }

    fn parameters(&self) -> &'static [ParameterSpec] {
        PARAMETERS
    }
}

#[async_trait]
impl DataConnector for MilvusConnector {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn read_provider(
        &self,
        dataset: &Dataset,
    ) -> DataConnectorResult<Arc<dyn TableProvider>> {
        // `from: milvus:<collection>` -> dataset.path() is the collection name.
        let coll = MilvusCollection {
            collection: dataset.path().to_string(),
            vector_field: self.vector_field.clone(),
            metric: self.metric.clone(),
            output_fields: self.output_fields.clone(),
        };
        Ok(Arc::new(MilvusTableProvider::new(Arc::clone(&self.conn), coll))
            as Arc<dyn TableProvider>)
    }
}

pub const CONNECTOR_NAME: &str = "milvus";

#[must_use]
pub fn factory() -> Arc<dyn DataConnectorFactory> {
    MilvusFactory::new_arc()
}
