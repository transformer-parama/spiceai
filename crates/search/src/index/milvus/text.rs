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

//! Milvus as a Spice full-text (BM25) [`SearchIndex`].
//!
//! The counterpart of [`super::MilvusVector`] for the keyword/lexical half of
//! hybrid search. Milvus 2.5+ can run native BM25 full-text search: a `VarChar`
//! field with an analyzer feeds a BM25 `Function` that produces a sparse vector
//! field, and searching that sparse field with raw query text ranks documents
//! by BM25. `query_table_provider` embeds NOTHING — it hands the query text to
//! Milvus and returns the primary key(s) (or, in Milvus-only mode, every scalar
//! field) plus a `_score` column.
//!
//! Registered alongside [`super::MilvusVector`] on the same dataset, this makes
//! `rrf(vector_search(t,'q'), text_search(t,'q'))` fuse dense + BM25 entirely on
//! Milvus — true hybrid search with no external FTS engine.

use std::any::Any;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use data_components::milvus::{MilvusTextSearchTable, MilvusVectorsTable};
use datafusion::catalog::TableProvider;
use datafusion::datasource::DefaultTableSource;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::LogicalPlan;
use datafusion_expr::{LogicalPlanBuilder, cast, col};
use runtime_datafusion_index::Index;

use crate::SEARCH_SCORE_COLUMN_NAME;
use crate::index::SearchIndex;

/// The Milvus exec produces a score column named `score`; the search layer
/// expects [`SEARCH_SCORE_COLUMN_NAME`]. We alias it in `query_table_provider`.
const MILVUS_SCORE_NAME: &str = "score";

/// A Milvus-backed BM25 full-text search index.
#[derive(Clone, Debug)]
pub struct MilvusTextIndex {
    /// Connection + collection descriptor (sparse `vector_field` + `BM25` metric)
    /// + the search output schema (data fields + score).
    pub table: MilvusVectorsTable,
    /// The base-table text column whose analyzed content Milvus BM25-indexed.
    pub text_column: String,
    /// The primary key joining the Milvus results back to the base table.
    pub primary_key: Vec<Field>,
}

impl MilvusTextIndex {
    #[must_use]
    pub fn new(
        table: MilvusVectorsTable,
        text_column: String,
        primary_key: Vec<Field>,
    ) -> Self {
        Self {
            table,
            text_column,
            primary_key,
        }
    }
}

impl Index for MilvusTextIndex {
    fn name(&self) -> &'static str {
        "milvus_text_index"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn required_columns(&self) -> Vec<String> {
        // The columns needed to (re)build the index from the base table: the
        // primary key(s) plus the full-text column.
        let mut cols: Vec<String> = self
            .primary_key
            .iter()
            .map(|f| f.name().clone())
            .collect();
        cols.push(self.text_column.clone());
        cols
    }
}

#[async_trait]
impl SearchIndex for MilvusTextIndex {
    fn search_column(&self) -> String {
        self.text_column.clone()
    }

    fn primary_fields(&self) -> Vec<Field> {
        self.primary_key.clone()
    }

    async fn write(
        &self,
        _record: RecordBatch,
    ) -> Result<RecordBatch, Box<dyn std::error::Error + Send + Sync>> {
        // The Milvus collection (and its BM25 function output) is populated
        // out-of-band; Spice-managed ingestion is not implemented.
        Err(Box::from(
            "milvus_text_index: write/ingestion is not implemented; populate the Milvus collection out-of-band".to_string(),
        ))
    }

    fn query_table_provider(&self, query: &str) -> Result<Arc<LogicalPlan>, DataFusionError> {
        let table: Arc<dyn TableProvider> =
            Arc::new(MilvusTextSearchTable::new(self.table.clone(), query.to_string()));

        // Project every column the index returns: pass the data columns through,
        // and cast the `score` column to Float64 aliased to the search layer's
        // score name (search aggregation requires Float64; the Milvus exec emits
        // Float32). For a classic base+index the table schema is primary_key +
        // score, so this yields primary_key + _score (SearchQueryProvider joins
        // back for the rest). For a Milvus-only index the schema is every scalar
        // column + score, so the row data comes straight from Milvus.
        let projection: Vec<_> = self
            .table
            .schema
            .fields()
            .iter()
            .map(|f| {
                if f.name() == MILVUS_SCORE_NAME {
                    cast(col(MILVUS_SCORE_NAME), DataType::Float64).alias(SEARCH_SCORE_COLUMN_NAME)
                } else {
                    col(f.name())
                }
            })
            .collect();

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
