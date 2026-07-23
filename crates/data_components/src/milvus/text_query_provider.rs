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

use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableProvider},
    common::Constraints,
    datasource::TableType,
    error::Result as DataFusionResult,
    logical_expr::TableProviderFilterPushDown,
    physical_plan::ExecutionPlan,
    prelude::Expr,
};
use milvus_client::MilvusTextExec;

use super::MilvusVectorsTable;

/// Default top-k when a query carries no LIMIT.
const DEFAULT_TOP_K: usize = 50;

/// A [`TableProvider`] that runs ONE Milvus BM25 full-text search for a query
/// string. Unlike [`super::MilvusQueryTable`], the query text is sent directly
/// to Milvus (no client-side embedding): Milvus applies the collection's BM25
/// function to score against the sparse field. This is the storage-layer
/// counterpart of the Milvus full-text `SearchIndex` in the `search` crate.
///
/// The wrapped [`MilvusVectorsTable`] must have `coll.vector_field` set to the
/// sparse (BM25 output) field and `coll.metric` set to `"BM25"`.
#[derive(Debug)]
pub struct MilvusTextSearchTable {
    table: MilvusVectorsTable,
    query: String,
}

impl MilvusTextSearchTable {
    #[must_use]
    pub fn new(table: MilvusVectorsTable, query: String) -> Self {
        Self { table, query }
    }
}

#[async_trait]
impl TableProvider for MilvusTextSearchTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.table.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn constraints(&self) -> Option<&Constraints> {
        Some(&self.table.constraints)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        // Scalar filters are not yet pushed into the Milvus search; let
        // DataFusion apply them after the scan.
        Ok(vec![TableProviderFilterPushDown::Unsupported; filters.len()])
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let exec = MilvusTextExec::new(
            Arc::clone(&self.table.conn),
            self.table.coll.clone(),
            Arc::clone(&self.table.schema),
            self.query.clone(),
            None, // no Milvus-side scalar filter yet
            limit.unwrap_or(DEFAULT_TOP_K),
            projection.cloned(),
        )?;

        Ok(Arc::new(exec))
    }
}
