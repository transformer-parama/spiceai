/*
Copyright 2024-2025 The Spice.ai OSS Authors

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

//! Construction of a Milvus [`MilvusVector`] index from a dataset's `vectors:`
//! block + column embedding config. The collection is introspected
//! (`describe_collection`) to derive the vector field; the dimension comes from
//! the configured embedding model (or `vector_size`); the primary key comes
//! from the base table. The index returns primary key(s) + score, which the
//! search layer joins back to the base table.

use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::sql::TableReference;
use llms::embeddings::get_or_infer_size;
use milvus_client::{ConnectionConfig, MilvusCollection, MilvusConnection, arrow_type_for};
use search::index::milvus::MilvusVector;
use spicepod::{param::Params, semantic::ColumnLevelEmbeddingConfig, vector::VectorStore};
use tokio::sync::RwLock;

use crate::model::EmbeddingModelStore;
use crate::parameters::{ParameterSpec, Parameters};
use runtime_secrets::{Secrets, get_params_with_secrets};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub(crate) const PARAMETERS: &[ParameterSpec] = &[
    ParameterSpec::component("host").description("Milvus host (default localhost)."),
    ParameterSpec::component("port").description("Milvus port (default 19530)."),
    ParameterSpec::component("collection").description("Milvus collection name (required)."),
    ParameterSpec::component("vector_field")
        .description("Vector field name (introspected from the collection if unset)."),
    ParameterSpec::component("metric").description("Distance metric: COSINE | L2 | IP."),
    ParameterSpec::component("partition")
        .description("Optional Milvus partition to scope searches to (per-tenant isolation)."),
    ParameterSpec::component("token")
        .description("Bearer token (`username:password`, or an API key).")
        .secret(),
    ParameterSpec::component("username")
        .description("Username; combined with password into a token.")
        .secret(),
    ParameterSpec::component("password")
        .description("Password.")
        .secret(),
    ParameterSpec::component("secure").description("Use TLS for the connection."),
    ParameterSpec::component("timeout_ms").description("Request timeout in milliseconds."),
    ParameterSpec::component("connect_timeout_ms")
        .description("Connect timeout in milliseconds."),
    ParameterSpec::component("max_retries")
        .description("Max retries on transient transport failures."),
    ParameterSpec::component("tls_skip_verify")
        .description("Skip TLS certificate verification (dev/self-signed only)."),
    ParameterSpec::component("tls_ca_cert").description("Path to a PEM CA certificate to trust."),
];

fn string_from_params<'a>(p: &'a Parameters, key: &str) -> Option<&'a str> {
    p.get(key).expose().ok()
}

async fn get_store_params(
    vector_store_config: &VectorStore,
    secrets: Arc<RwLock<Secrets>>,
) -> Result<Parameters, BoxError> {
    let params = vector_store_config
        .params
        .as_ref()
        .map(Params::as_string_map)
        .unwrap_or_default();
    let params_with_secrets = get_params_with_secrets(Arc::clone(&secrets), &params).await;
    let params = Parameters::try_new(
        "Milvus vector store",
        params_with_secrets.into_iter().collect(),
        "milvus",
        Arc::clone(&secrets),
        PARAMETERS,
    )
    .await?;
    Ok(params)
}

/// Build a [`MilvusVector`] index for `column` of dataset `ds_name`.
#[expect(clippy::too_many_arguments)]
pub async fn try_from_table(
    ds_name: &TableReference,
    column: String,
    config: ColumnLevelEmbeddingConfig,
    vector_store_config: &VectorStore,
    primary_keys: Vec<String>,
    inner_schema: SchemaRef,
    embedding_models: Arc<RwLock<EmbeddingModelStore>>,
    secrets: Arc<RwLock<Secrets>>,
) -> Result<MilvusVector, BoxError> {
    // Primary key: spicepod `row_ids` override, else the base table's primary key.
    let primary_key: Vec<Field> = config
        .row_ids
        .clone()
        .unwrap_or(primary_keys)
        .into_iter()
        .filter_map(|c| {
            inner_schema
                .column_with_name(c.as_str())
                .map(|(_, f)| f.clone())
        })
        .collect();
    if primary_key.is_empty() {
        return Err(Box::from(format!(
            "Cannot make a Milvus vector index for table '{ds_name}' column '{column}': no primary key. Set `row_ids` on the embedding, or a primary key on the dataset."
        )));
    }

    // The embedding model (embeds queries; same model the vectors were built with).
    let model = {
        let model_read = embedding_models.read().await;
        let Some(model) = model_read.get(&config.model) else {
            return Err(Box::from(format!(
                "Cannot make a Milvus vector index for table '{ds_name}' column '{column}': embedding model '{}' is not defined in the Spicepod or failed to load.",
                config.model
            )));
        };
        Arc::clone(model)
    };

    let params = get_store_params(vector_store_config, Arc::clone(&secrets)).await?;

    // Connection.
    let port: u16 = string_from_params(&params, "port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(19530);
    let token = string_from_params(&params, "token")
        .map(str::to_string)
        .or_else(|| {
            match (
                string_from_params(&params, "username"),
                string_from_params(&params, "password"),
            ) {
                (Some(u), Some(pw)) => Some(format!("{u}:{pw}")),
                _ => None,
            }
        });
    let bool_flag = |key: &str| -> bool {
        string_from_params(&params, key)
            .map(|v| matches!(v, "true" | "1" | "yes"))
            .unwrap_or(false)
    };
    let u64_or = |key: &str, default: u64| -> u64 {
        string_from_params(&params, key)
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let cfg = ConnectionConfig {
        host: string_from_params(&params, "host")
            .unwrap_or("localhost")
            .to_string(),
        port,
        secure: bool_flag("secure"),
        token,
        timeout: Duration::from_millis(u64_or("timeout_ms", 10_000)),
        connect_timeout: Duration::from_millis(u64_or("connect_timeout_ms", 3_000)),
        max_retries: u64_or("max_retries", 2) as u32,
        tls_skip_verify: bool_flag("tls_skip_verify"),
        tls_ca_cert_path: string_from_params(&params, "tls_ca_cert").map(str::to_string),
    };
    let conn = Arc::new(MilvusConnection::new(cfg).map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?);

    // Collection + vector field (introspected if not given).
    let collection = string_from_params(&params, "collection")
        .ok_or_else(|| {
            Box::<dyn std::error::Error + Send + Sync>::from(format!(
                "Milvus vector index for table '{ds_name}' requires `milvus_collection`."
            ))
        })?
        .to_string();
    let fields = conn
        .describe_collection(&collection)
        .await
        .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;
    let vector_field = match string_from_params(&params, "vector_field") {
        Some(v) => v.to_string(),
        None => fields
            .iter()
            .find(|f| f.is_vector)
            .map(|f| f.name.clone())
            .ok_or_else(|| {
                Box::<dyn std::error::Error + Send + Sync>::from(format!(
                    "No vector field found in Milvus collection '{collection}'. Set `milvus_vector_field`."
                ))
            })?,
    };
    let metric = string_from_params(&params, "metric")
        .unwrap_or("COSINE")
        .to_string();
    // Optional per-tenant partition: scope every ANN search to this Milvus partition
    // (isolation for multi-assistant collections partitioned by assistant_id).
    let partition = string_from_params(&params, "partition").map(str::to_string);

    // Dimension: configured `vector_size`, else inferred from the model.
    let dimension: i32 = match config.vector_size {
        Some(s) => i32::try_from(s).unwrap_or(i32::MAX),
        None => get_or_infer_size(&model)
            .await
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?,
    };

    // Milvus-only mode: when the base table IS the Milvus collection itself (the
    // `from: milvus:<collection>` connector, detectable by its `query_vector`
    // input column), the collection holds ALL scalar columns (text, etc.), not
    // just the vectors. Return every scalar field so the search layer serves the
    // row data directly from Milvus — no join-back to a separate base table.
    // Otherwise (Postgres/Iceberg base) keep the classic primary-key + score and
    // let `SearchQueryProvider` join back to that base for the row data.
    let milvus_only = inner_schema.column_with_name("query_vector").is_some();

    // The index returns these data columns + a `score` column. In Milvus-only
    // mode that's every scalar (non-vector) field; otherwise just the primary key.
    let data_fields: Vec<Field> = if milvus_only {
        fields
            .iter()
            .filter(|f| !f.is_vector)
            .map(|f| Field::new(&f.name, arrow_type_for(&f.type_name), true))
            .collect()
    } else {
        primary_key.clone()
    };
    let output_fields: Vec<String> = data_fields.iter().map(|f| f.name().clone()).collect();
    let coll = MilvusCollection {
        collection,
        vector_field,
        metric,
        output_fields,
        partition,
    };

    let mut schema_fields: Vec<Field> = data_fields;
    schema_fields.push(Field::new("score", DataType::Float32, false));
    let schema: SchemaRef = Arc::new(Schema::new(schema_fields));

    let table = data_components::milvus::MilvusVectorsTable::new(
        conn,
        coll,
        schema,
        i64::from(dimension),
    );

    Ok(MilvusVector::new(
        table,
        column,
        primary_key,
        model,
        i64::from(dimension),
    ))
}
