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
//! Targets DataFusion 52 (Spice v2.0.1). The output schema is built DYNAMICALLY
//! from the collection's introspected fields (see lib.rs), so build_batch maps
//! every column by its Arrow `DataType` rather than by hardcoded names.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, BooleanBuilder, Float32Builder, Float64Builder, Int32Builder, Int64Builder,
    StringBuilder,
};
use datafusion::arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use serde_json::Value;

use crate::milvus::{Hit, MilvusCollection, MilvusConnection};

/// Map a Milvus field type name to an Arrow `DataType`. Vector fields are
/// excluded from output upstream; anything unmapped falls back to Utf8.
pub fn arrow_type_for(milvus_type: &str) -> DataType {
    match milvus_type {
        "Bool" => DataType::Boolean,
        "Int8" | "Int16" | "Int32" => DataType::Int32,
        "Int64" => DataType::Int64,
        "Float" => DataType::Float32,
        "Double" => DataType::Float64,
        _ => DataType::Utf8, // VarChar / String / JSON / Array / unknown
    }
}

#[derive(Debug)]
pub struct MilvusExec {
    conn: Arc<MilvusConnection>,
    coll: MilvusCollection,
    query_vector: Vec<f32>,
    filter: Option<String>,
    limit: usize,
    full_schema: SchemaRef,        // every column the table exposes
    projected_schema: SchemaRef,   // the columns this scan returns
    projection: Option<Vec<usize>>,
    props: PlanProperties,
}

impl MilvusExec {
    pub fn new(
        conn: Arc<MilvusConnection>,
        mut coll: MilvusCollection,
        full_schema: SchemaRef,
        query_vector: Vec<f32>,
        filter: Option<String>,
        limit: usize,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let projected_schema = match &projection {
            Some(idx) => Arc::new(full_schema.project(idx).map_err(DataFusionError::from)?),
            None => full_schema.clone(),
        };
        // projection pushdown: ask Milvus only for the scalar fields this scan
        // actually returns (query_vector is input-only; score is synthetic).
        coll.output_fields = projected_schema
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .filter(|n| n != "query_vector" && n != "score")
            .collect();
        let props = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            conn,
            coll,
            query_vector,
            filter,
            limit,
            full_schema,
            projected_schema,
            projection,
            props,
        })
    }

    /// Build one Arrow column for `field` from the JSON hit rows, dispatching on
    /// the field's Arrow type. `query_vector` is input-only (always null);
    /// `score` comes from the (normalized) similarity.
    pub(crate) fn build_column(field: &Field, hits: &[Hit]) -> ArrayRef {
        let name = field.name().as_str();
        if name == "query_vector" {
            let mut b = StringBuilder::new();
            for _ in hits {
                b.append_null();
            }
            return Arc::new(b.finish());
        }
        if name == "score" {
            let mut b = Float32Builder::new();
            for h in hits {
                b.append_value(h.score);
            }
            return Arc::new(b.finish());
        }
        match field.data_type() {
            DataType::Int64 => {
                let mut b = Int64Builder::new();
                for h in hits {
                    b.append_option(h.fields.get(name).and_then(Value::as_i64));
                }
                Arc::new(b.finish())
            }
            DataType::Int32 => {
                let mut b = Int32Builder::new();
                for h in hits {
                    b.append_option(h.fields.get(name).and_then(Value::as_i64).map(|v| v as i32));
                }
                Arc::new(b.finish())
            }
            DataType::Float32 => {
                let mut b = Float32Builder::new();
                for h in hits {
                    b.append_option(h.fields.get(name).and_then(Value::as_f64).map(|v| v as f32));
                }
                Arc::new(b.finish())
            }
            DataType::Float64 => {
                let mut b = Float64Builder::new();
                for h in hits {
                    b.append_option(h.fields.get(name).and_then(Value::as_f64));
                }
                Arc::new(b.finish())
            }
            DataType::Boolean => {
                let mut b = BooleanBuilder::new();
                for h in hits {
                    b.append_option(h.fields.get(name).and_then(Value::as_bool));
                }
                Arc::new(b.finish())
            }
            _ => {
                // Utf8 + fallback: pass strings through, stringify other JSON.
                let mut b = StringBuilder::new();
                for h in hits {
                    match h.fields.get(name) {
                        Some(Value::String(s)) => b.append_value(s),
                        Some(v) if !v.is_null() => b.append_value(v.to_string()),
                        _ => b.append_null(),
                    }
                }
                Arc::new(b.finish())
            }
        }
    }

    pub(crate) fn build_batch(
        full_schema: &SchemaRef,
        projection: &Option<Vec<usize>>,
        hits: &[Hit],
    ) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = full_schema
            .fields()
            .iter()
            .map(|f| Self::build_column(f, hits))
            .collect();
        let batch =
            RecordBatch::try_new(full_schema.clone(), cols).map_err(DataFusionError::from)?;
        match projection {
            Some(idx) => batch.project(idx).map_err(DataFusionError::from),
            None => Ok(batch),
        }
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
        let full_schema = self.full_schema.clone();
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
            Self::build_batch(&full_schema, &projection, &hits)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.projected_schema.clone(),
            stream,
        )))
    }
}

/// `ExecutionPlan` that turns one Milvus BM25 full-text search into an Arrow
/// RecordBatch. Mirrors [`MilvusExec`] but sends the query TEXT (not a vector)
/// to Milvus's sparse BM25 search — the counterpart used by `text_search`.
#[derive(Debug)]
pub struct MilvusTextExec {
    conn: Arc<MilvusConnection>,
    coll: MilvusCollection,
    query: String,
    filter: Option<String>,
    limit: usize,
    full_schema: SchemaRef,
    projected_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    props: PlanProperties,
}

impl MilvusTextExec {
    pub fn new(
        conn: Arc<MilvusConnection>,
        mut coll: MilvusCollection,
        full_schema: SchemaRef,
        query: String,
        filter: Option<String>,
        limit: usize,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let projected_schema = match &projection {
            Some(idx) => Arc::new(full_schema.project(idx).map_err(DataFusionError::from)?),
            None => full_schema.clone(),
        };
        // projection pushdown: ask Milvus only for the scalar fields this scan
        // returns (score is synthetic; there is no query_vector input column).
        coll.output_fields = projected_schema
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .filter(|n| n != "score")
            .collect();
        let props = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            conn,
            coll,
            query,
            filter,
            limit,
            full_schema,
            projected_schema,
            projection,
            props,
        })
    }
}

impl DisplayAs for MilvusTextExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "MilvusTextExec: collection={}, sparse_field={}, limit={}, filter={:?}",
            self.coll.collection, self.coll.vector_field, self.limit, self.filter
        )
    }
}

impl ExecutionPlan for MilvusTextExec {
    fn name(&self) -> &str {
        "MilvusTextExec"
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
        let query = self.query.clone();
        let filter = self.filter.clone();
        let limit = self.limit;
        let full_schema = self.full_schema.clone();
        let projection = self.projection.clone();

        let fut = async move {
            let span = tracing::debug_span!(
                target: "connector_milvus", "milvus_text_search",
                collection = %coll.collection, limit
            );
            let _enter = span.enter();
            let hits = conn
                .text_search(&coll, &query, limit, filter)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            tracing::debug!(target: "connector_milvus", hits = hits.len(), "milvus text_search ok");
            MilvusExec::build_batch(&full_schema, &projection, &hits)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.projected_schema.clone(),
            stream,
        )))
    }
}
