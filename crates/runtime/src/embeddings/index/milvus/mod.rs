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
use search::index::milvus::{MilvusTextIndex, MilvusVector};
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
    ParameterSpec::component("sparse_field")
        .description(
            "Sparse (BM25) vector field for full-text `text_search`. Auto-discovered from \
             the collection's BM25 function on the embedded text column if unset.",
        ),
    ParameterSpec::component("metric").description("Distance metric: COSINE | L2 | IP."),
    ParameterSpec::component("partition")
        .description("Optional Milvus partition to scope searches to (per-tenant isolation)."),
    ParameterSpec::component("require_partition")
        .description(
            "Fail closed: refuse to build this vector index unless `partition` is set. \
             Prevents a missing partition from silently searching the whole \
             (multi-tenant) collection. Also enabled by SPICE_MILVUS_REQUIRE_PARTITION=true.",
        ),
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

/// Build the Milvus [`ConnectionConfig`] shared by the vector and full-text
/// index paths from the resolved store parameters.
fn connection_config_from_params(params: &Parameters) -> ConnectionConfig {
    let port: u16 = string_from_params(params, "port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(19530);
    let token = string_from_params(params, "token")
        .map(str::to_string)
        .or_else(|| {
            match (
                string_from_params(params, "username"),
                string_from_params(params, "password"),
            ) {
                (Some(u), Some(pw)) => Some(format!("{u}:{pw}")),
                _ => None,
            }
        });
    let bool_flag = |key: &str| -> bool {
        string_from_params(params, key)
            .map(|v| matches!(v, "true" | "1" | "yes"))
            .unwrap_or(false)
    };
    let u64_or = |key: &str, default: u64| -> u64 {
        string_from_params(params, key)
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    ConnectionConfig {
        host: string_from_params(params, "host")
            .unwrap_or("localhost")
            .to_string(),
        port,
        secure: bool_flag("secure"),
        token,
        timeout: Duration::from_millis(u64_or("timeout_ms", 10_000)),
        connect_timeout: Duration::from_millis(u64_or("connect_timeout_ms", 3_000)),
        max_retries: u64_or("max_retries", 2) as u32,
        tls_skip_verify: bool_flag("tls_skip_verify"),
        tls_ca_cert_path: string_from_params(params, "tls_ca_cert").map(str::to_string),
    }
}

/// Whether per-tenant partition scoping is required (per-index param or the
/// process-wide `SPICE_MILVUS_REQUIRE_PARTITION` env). When required but the
/// partition is empty, both the vector and full-text index paths fail closed to
/// avoid serving an unscoped (whole-collection) search across tenants.
fn require_partition_enabled(params: &Parameters) -> bool {
    let flag = string_from_params(params, "require_partition")
        .map(|v| matches!(v, "true" | "1" | "yes"))
        .unwrap_or(false);
    flag || matches!(
        std::env::var("SPICE_MILVUS_REQUIRE_PARTITION").ok().as_deref(),
        Some("true" | "1" | "yes")
    )
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
    let cfg = connection_config_from_params(&params);
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

    // Fail closed: when partition is required (per-index param or the process-wide
    // SPICE_MILVUS_REQUIRE_PARTITION env) but unset, refuse to build the index — this
    // is the ONLY tenant boundary on the vector_search path (there is no scalar-filter
    // fallback), so an unscoped search would leak every tenant's vectors.
    let require_partition = require_partition_enabled(&params);
    if require_partition && partition.as_deref().unwrap_or("").is_empty() {
        return Err(Box::<dyn std::error::Error + Send + Sync>::from(format!(
            "Milvus vector index for table '{ds_name}' requires `milvus_partition` \
             (require_partition / SPICE_MILVUS_REQUIRE_PARTITION set) but it is empty — \
             refusing to serve an unscoped (whole-collection) vector_search"
        )));
    }

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

/// Build a [`MilvusTextIndex`] (BM25 full-text search) for `column` of dataset
/// `ds_name`, enabling the keyword half of hybrid search directly on Milvus.
///
/// Returns `Ok(None)` when the collection has no BM25 full-text function on the
/// embedded text column (and no explicit `sparse_field` override) — the dataset
/// simply stays vector-only. The sparse (BM25 output) field is auto-discovered
/// from the collection's function definitions, so a BM25-ready collection needs
/// zero extra configuration.
pub async fn try_text_index_from_table(
    ds_name: &TableReference,
    column: String,
    config: ColumnLevelEmbeddingConfig,
    vector_store_config: &VectorStore,
    primary_keys: Vec<String>,
    inner_schema: SchemaRef,
    secrets: Arc<RwLock<Secrets>>,
) -> Result<Option<MilvusTextIndex>, BoxError> {
    // Primary key: spicepod `row_ids` override, else the base table's primary key.
    // Required to fuse text-search hits back with vector-search hits (join key).
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
        // No join key — cannot fuse; skip the text index (vector-only).
        tracing::debug!(
            "Milvus full-text index for '{ds_name}' column '{column}' skipped: no primary key."
        );
        return Ok(None);
    }

    let params = get_store_params(vector_store_config, Arc::clone(&secrets)).await?;
    let cfg = connection_config_from_params(&params);
    let conn = Arc::new(
        MilvusConnection::new(cfg)
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?,
    );

    let collection = string_from_params(&params, "collection")
        .ok_or_else(|| {
            Box::<dyn std::error::Error + Send + Sync>::from(format!(
                "Milvus full-text index for table '{ds_name}' requires `milvus_collection`."
            ))
        })?
        .to_string();

    let info = conn
        .describe_collection_info(&collection)
        .await
        .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;

    // Sparse (BM25 output) field: explicit override, else auto-discovered from
    // the collection's BM25 function whose input is the embedded text column.
    let sparse_field = string_from_params(&params, "sparse_field")
        .map(str::to_string)
        .or_else(|| info.bm25_sparse_field_for(&column));
    let Some(sparse_field) = sparse_field else {
        tracing::info!(
            "Milvus dataset '{ds_name}' has no BM25 full-text function on column '{column}'; \
             text_search/hybrid is unavailable for it (vector_search still works). Add a BM25 \
             function + sparse field to the collection, or set `milvus_sparse_field`, to enable it."
        );
        return Ok(None);
    };

    // Optional per-tenant partition + fail-closed check (mirrors the vector path;
    // the same tenant boundary must apply to full-text search).
    let partition = string_from_params(&params, "partition").map(str::to_string);
    if require_partition_enabled(&params) && partition.as_deref().unwrap_or("").is_empty() {
        return Err(Box::<dyn std::error::Error + Send + Sync>::from(format!(
            "Milvus full-text index for table '{ds_name}' requires `milvus_partition` \
             (require_partition / SPICE_MILVUS_REQUIRE_PARTITION set) but it is empty — \
             refusing to serve an unscoped (whole-collection) text_search"
        )));
    }

    // Milvus-only mode (base IS the Milvus connector, detected by `query_vector`):
    // return every scalar field so the fused result carries the row data; else
    // return just the primary key and let the search layer join back.
    let milvus_only = inner_schema.column_with_name("query_vector").is_some();
    let data_fields: Vec<Field> = if milvus_only {
        info.fields
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
        vector_field: sparse_field, // annsField for the BM25 sparse search
        metric: "BM25".to_string(),
        output_fields,
        partition,
    };

    let mut schema_fields: Vec<Field> = data_fields;
    schema_fields.push(Field::new("score", DataType::Float32, false));
    let schema: SchemaRef = Arc::new(Schema::new(schema_fields));

    // Dimension is unused for sparse BM25 search; pass 0.
    let table = data_components::milvus::MilvusVectorsTable::new(conn, coll, schema, 0);

    Ok(Some(MilvusTextIndex::new(table, column, primary_key)))
}
