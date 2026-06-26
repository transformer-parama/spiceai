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

//! Milvus as a Spice [`VectorIndex`].
//!
//! An external-store vector index (like S3 Vectors): the embedding vectors live
//! in Milvus, keyed by the base table's primary key. `query_table_provider`
//! embeds the query text (engine-side) and runs the ANN search, returning the
//! primary key(s) + a `_score` column. The [`SearchQueryProvider`] then joins
//! that back to the base table (which holds the row data) on the primary key.
//!
//! [`SearchQueryProvider`]: crate::provider::SearchQueryProvider

use std::any::Any;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use data_components::milvus::{
    CachedQueryVector, ComputeQueryVector, MilvusQueryTable, MilvusVectorsTable,
};
use datafusion::catalog::TableProvider;
use datafusion::datasource::DefaultTableSource;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::LogicalPlan;
use datafusion_expr::{LogicalPlanBuilder, cast, col};
use llms::embeddings::Embed;
use runtime_datafusion_index::Index;

use crate::SEARCH_SCORE_COLUMN_NAME;
use crate::index::milvus::compute_query::EmbedQuery;
use crate::index::{SearchIndex, VectorIndex};

mod compute_query;

/// The Milvus exec produces a similarity column named `score`; the search layer
/// expects [`SEARCH_SCORE_COLUMN_NAME`]. We alias it in `query_table_provider`.
const MILVUS_SCORE_NAME: &str = "score";

/// A Milvus-backed vector index.
#[derive(Clone, Debug)]
pub struct MilvusVector {
    /// Connection + collection + the search output schema (primary key + score).
    pub table: MilvusVectorsTable,
    /// The base-table column whose values were embedded into Milvus.
    pub embedded_column: String,
    /// The primary key joining the Milvus results back to the base table.
    pub primary_key: Vec<Field>,
    /// The embedding model used to embed queries (same model that built the
    /// collection's vectors).
    pub compute_query: Arc<dyn Embed>,
    /// Embedding dimension.
    pub dimension: i64,
}

impl MilvusVector {
    #[must_use]
    pub fn new(
        table: MilvusVectorsTable,
        embedded_column: String,
        primary_key: Vec<Field>,
        compute_query: Arc<dyn Embed>,
        dimension: i64,
    ) -> Self {
        Self {
            table,
            embedded_column,
            primary_key,
            compute_query,
            dimension,
        }
    }
}

impl Index for MilvusVector {
    fn name(&self) -> &'static str {
        "milvus_vector_index"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn required_columns(&self) -> Vec<String> {
        // The columns needed to (re)build the index from the base table: the
        // primary key(s) plus the embedded text column.
        let mut cols: Vec<String> = self
            .primary_key
            .iter()
            .map(|f| f.name().clone())
            .collect();
        cols.push(self.embedded_column.clone());
        cols
    }
}

#[async_trait]
impl SearchIndex for MilvusVector {
    fn search_column(&self) -> String {
        self.embedded_column.clone()
    }

    fn primary_fields(&self) -> Vec<Field> {
        self.primary_key.clone()
    }

    async fn write(
        &self,
        _record: RecordBatch,
    ) -> Result<RecordBatch, Box<dyn std::error::Error + Send + Sync>> {
        // MVP: the Milvus collection is populated out-of-band (the vectors
        // already exist). Spice-managed ingestion is not implemented yet.
        Err(Box::from(
            "milvus_vector_index: write/ingestion is not implemented; populate the Milvus collection out-of-band".to_string(),
        ))
    }

    fn as_vector_index(self: Arc<Self>) -> Option<Arc<dyn VectorIndex>> {
        Some(Arc::clone(&self) as Arc<dyn VectorIndex>)
    }

    fn query_table_provider(&self, query: &str) -> Result<Arc<LogicalPlan>, DataFusionError> {
        // Lazily embed the query at scan time (cached for repeated scans).
        let compute_vector = Arc::new(CachedQueryVector::new(
            Arc::new(EmbedQuery(Arc::clone(&self.compute_query))),
            query.to_string(),
        )) as Arc<dyn ComputeQueryVector>;

        let table: Arc<dyn TableProvider> = Arc::new(MilvusQueryTable::new(
            self.table.clone(),
            compute_vector,
            query.to_string(),
        ));

        // Project the primary key(s) + the score column, aliased to the search
        // layer's internal score name. The SearchQueryProvider joins these back
        // to the base table on the primary key to materialize the row data.
        let mut projection: Vec<_> = self.primary_key.iter().map(|f| col(f.name())).collect();
        // /v1/search aggregation requires the score column to be Float64; the
        // Milvus exec produces Float32, so cast it here.
        projection.push(cast(col(MILVUS_SCORE_NAME), DataType::Float64).alias(SEARCH_SCORE_COLUMN_NAME));

        Ok(LogicalPlanBuilder::scan(
            "tbl",
            Arc::new(DefaultTableSource::new(table)),
            None,
        )?
        .project(projection)?
        .build()?
        .into())
    }
}

impl VectorIndex for MilvusVector {
    fn dimension(&self) -> i32 {
        i32::try_from(self.dimension).unwrap_or(i32::MAX)
    }

    fn list_table_provider(&self) -> Result<LogicalPlan, DataFusionError> {
        // Co-located/MVP: enumeration of the full index is not implemented.
        // Returning NotImplemented signals the search layer to not wrap this in
        // a VectorScanTableProvider (it uses query_table_provider directly).
        Err(DataFusionError::NotImplemented(
            "milvus_vector_index: list_table_provider is not implemented".to_string(),
        ))
    }

    fn derived_columns(&self) -> Vec<String> {
        // query_table_provider returns only primary key(s) + score; it does not
        // surface an embedding column, so nothing is derived.
        vec![]
    }
}
