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

//! `ExecutionPlan` that turns one Milvus ANN search into an Arrow RecordBatch.
//! Targets DataFusion 52 (Spice v2.0.1): `as_any` on ExecutionPlan,
//! `properties() -> &PlanProperties`, `PlanProperties::new` takes EmissionType
//! + Boundedness.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float32Builder, Int64Builder, StringBuilder};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};

use crate::milvus::{MilvusCollection, MilvusConnection};

/// Full table schema. `query_vector` is an input-only column (the JSON-encoded
/// query embedding arrives via a `query_vector = '[...]'` predicate); it is
/// always NULL in output rows.
pub fn table_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("query_vector", DataType::Utf8, true),
        Field::new("doc_type", DataType::Utf8, true),
        Field::new("source_id", DataType::Int64, true),
        Field::new("product_id", DataType::Int64, true),
        Field::new("title", DataType::Utf8, true),
        Field::new("text", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]))
}

#[derive(Debug)]
pub struct MilvusExec {
    conn: Arc<MilvusConnection>,
    coll: MilvusCollection,
    query_vector: Vec<f32>,
    filter: Option<String>,
    limit: usize,
    projected_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    props: PlanProperties,
}

impl MilvusExec {
    pub fn new(
        conn: Arc<MilvusConnection>,
        coll: MilvusCollection,
        query_vector: Vec<f32>,
        filter: Option<String>,
        limit: usize,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let full = table_schema();
        let projected_schema = match &projection {
            Some(idx) => Arc::new(full.project(idx).map_err(DataFusionError::from)?),
            None => full.clone(),
        };
        let props = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self { conn, coll, query_vector, filter, limit, projected_schema, projection, props })
    }

    fn build_batch(
        schema: SchemaRef,
        projection: &Option<Vec<usize>>,
        hits: Vec<crate::milvus::Hit>,
    ) -> Result<RecordBatch> {
        let mut query_vector = StringBuilder::new();
        let mut doc_type = StringBuilder::new();
        let mut source_id = Int64Builder::new();
        let mut product_id = Int64Builder::new();
        let mut title = StringBuilder::new();
        let mut text = StringBuilder::new();
        let mut score = Float32Builder::new();

        for h in &hits {
            query_vector.append_null();
            doc_type.append_option(h.fields.get("doc_type").and_then(|v| v.as_str()));
            source_id.append_option(h.fields.get("source_id").and_then(|v| v.as_i64()));
            product_id.append_option(h.fields.get("product_id").and_then(|v| v.as_i64()));
            title.append_option(h.fields.get("title").and_then(|v| v.as_str()));
            text.append_option(h.fields.get("text").and_then(|v| v.as_str()));
            score.append_value(h.score);
        }

        let all: Vec<ArrayRef> = vec![
            Arc::new(query_vector.finish()),
            Arc::new(doc_type.finish()),
            Arc::new(source_id.finish()),
            Arc::new(product_id.finish()),
            Arc::new(title.finish()),
            Arc::new(text.finish()),
            Arc::new(score.finish()),
        ];
        let cols = match projection {
            Some(idx) => idx.iter().map(|&i| all[i].clone()).collect(),
            None => all,
        };
        RecordBatch::try_new(schema, cols).map_err(DataFusionError::from)
    }
}

impl DisplayAs for MilvusExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "MilvusExec: collection={}, limit={}, filter={:?}",
            self.coll.collection, self.limit, self.filter
        )
    }
}

impl ExecutionPlan for MilvusExec {
    fn name(&self) -> &str {
        "MilvusExec"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn properties(&self) -> &PlanProperties {
        &self.props
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let conn = Arc::clone(&self.conn);
        let coll = self.coll.clone();
        let vector = self.query_vector.clone();
        let filter = self.filter.clone();
        let limit = self.limit;
        let schema = self.projected_schema.clone();
        let projection = self.projection.clone();

        let fut = async move {
            let span = tracing::debug_span!(
                target: "connector_milvus", "milvus_search",
                collection = %coll.collection, limit, dim = vector.len()
            );
            let _enter = span.enter();
            let hits = conn
                .search(&coll, vector, limit, filter)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            tracing::debug!(target: "connector_milvus", hits = hits.len(), "milvus search ok");
            Self::build_batch(schema, &projection, hits)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.projected_schema.clone(),
            stream,
        )))
    }
}
