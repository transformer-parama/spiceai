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

//! Neo4j data connector for the Spice.ai runtime.
//!
//! Exposes a Neo4j node label as a SQL table: `from: neo4j:<Label>`. The
//! connector introspects the label's properties (`db.schema.nodeTypeProperties`)
//! and builds the Arrow schema dynamically -- one typed column per property.
//! `SELECT ... WHERE ... LIMIT k` compiles to a single Cypher
//! `MATCH (n:<Label>) WHERE ... RETURN ... LIMIT k`; comparison/`IN` predicates
//! and the projection push down.
//!
//! A single pooled, timeout-bounded connection (with optional HTTP Basic auth +
//! TLS and transient-failure retries) is built once and shared across all
//! datasets and queries served by this connector.

mod exec;
mod neo4j;
mod table_provider;

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::datasource::TableProvider;
use runtime::component::dataset::Dataset;
use runtime::dataconnector::{
    ConnectorComponent, ConnectorParams, DataConnector, DataConnectorError, DataConnectorFactory,
    DataConnectorResult,
};
use runtime::parameters::ParameterSpec;
use secrecy::ExposeSecret;

use crate::exec::arrow_type_for;
use crate::neo4j::{ConnectionConfig, Neo4jConnection};
use crate::table_provider::Neo4jTableProvider;

/// Neo4j data connector. Holds a shared pooled connection; the node label is
/// resolved per dataset in `read_provider`.
pub struct Neo4jConnector {
    conn: Arc<Neo4jConnection>,
}

impl std::fmt::Debug for Neo4jConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neo4jConnector").field("conn", &self.conn).finish()
    }
}

#[derive(Default, Copy, Clone)]
pub struct Neo4jFactory {}

impl Neo4jFactory {
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
        .description("Neo4j host.")
        .default("localhost"),
    ParameterSpec::component("port")
        .description("Neo4j HTTP port.")
        .default("7474"),
    ParameterSpec::component("secure")
        .description("Use TLS (https) to reach Neo4j.")
        .default("false"),
    ParameterSpec::component("database")
        .description("Neo4j database name.")
        .default("neo4j"),
    ParameterSpec::component("username")
        .description("Username for HTTP Basic auth.")
        .secret(),
    ParameterSpec::component("password")
        .description("Password for HTTP Basic auth.")
        .secret(),
    ParameterSpec::component("timeout_ms")
        .description("Per-request timeout in milliseconds.")
        .default("10000"),
    ParameterSpec::component("connect_timeout_ms")
        .description("Connection timeout in milliseconds.")
        .default("3000"),
    ParameterSpec::component("max_retries")
        .description("Retries for transient transport failures (timeouts/connection/5xx).")
        .default("2"),
];

impl DataConnectorFactory for Neo4jFactory {
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
                dataconnector: "neo4j".to_string(),
                connector_component: params.component.clone(),
                source: Box::<dyn std::error::Error + Send + Sync>::from(msg),
            };

            let port: u16 = p("port")
                .unwrap_or_else(|| "7474".to_string())
                .parse()
                .map_err(|e| connect_err(format!("invalid port: {e}")))?;

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
                database: p("database").unwrap_or_else(|| "neo4j".to_string()),
                username: p("username"),
                password: p("password"),
                timeout: Duration::from_millis(u64_or("timeout_ms", 10_000)),
                connect_timeout: Duration::from_millis(u64_or("connect_timeout_ms", 3_000)),
                max_retries: u64_or("max_retries", 2) as u32,
            };
            let conn = Arc::new(Neo4jConnection::new(cfg).map_err(|e| connect_err(e.to_string()))?);

            Ok(Arc::new(Neo4jConnector { conn }) as Arc<dyn DataConnector>)
        })
    }

    fn prefix(&self) -> &'static str {
        "neo4j"
    }

    fn parameters(&self) -> &'static [ParameterSpec] {
        PARAMETERS
    }
}

#[async_trait]
impl DataConnector for Neo4jConnector {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn read_provider(
        &self,
        dataset: &Dataset,
    ) -> DataConnectorResult<Arc<dyn TableProvider>> {
        // `from: neo4j:<Label>` -> dataset.path() is the node label.
        let label = dataset.path().to_string();

        let err = |msg: String| DataConnectorError::UnableToConnectInternal {
            dataconnector: "neo4j".to_string(),
            connector_component: ConnectorComponent::from(dataset),
            source: Box::<dyn std::error::Error + Send + Sync>::from(msg),
        };

        // Introspect the label's properties (also a reachability/existence check)
        // and build the Arrow schema dynamically -- works for ANY label.
        let props = self
            .conn
            .describe_label(&label)
            .await
            .map_err(|e| err(format!("describe label '{label}': {e}")))?;
        if props.is_empty() {
            return Err(err(format!(
                "label '{label}' has no introspectable properties \
                 (unknown label, or the database has no nodes with it?)"
            )));
        }

        let arrow_fields: Vec<Field> = props
            .iter()
            .map(|p| Field::new(&p.name, arrow_type_for(&p.type_name), true))
            .collect();
        let schema: SchemaRef = Arc::new(Schema::new(arrow_fields));

        Ok(Arc::new(Neo4jTableProvider::new(Arc::clone(&self.conn), label, schema))
            as Arc<dyn TableProvider>)
    }
}

pub const CONNECTOR_NAME: &str = "neo4j";

#[must_use]
pub fn factory() -> Arc<dyn DataConnectorFactory> {
    Neo4jFactory::new_arc()
}
