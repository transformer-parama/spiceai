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

//! `ExecutionPlan` that runs one Cypher query and turns the rows into an Arrow
//! RecordBatch. Targets DataFusion 52 (Spice v2.0.1). The schema is built
//! DYNAMICALLY from the label's introspected properties (see lib.rs); projection
//! is pushed into the Cypher RETURN, so the returned columns already match the
//! projected schema. Each column is built by its Arrow `DataType`.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder,
};
use datafusion::arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use serde_json::{Map, Value};

use crate::neo4j::Neo4jConnection;

/// Map a Neo4j property type name to an Arrow `DataType`. Anything unmapped
/// (arrays, temporal, spatial) falls back to Utf8 (stringified).
pub fn arrow_type_for(neo4j_type: &str) -> DataType {
    match neo4j_type {
        "Long" | "Integer" => DataType::Int64,
        "Double" | "Float" => DataType::Float64,
        "Boolean" => DataType::Boolean,
        "String" => DataType::Utf8,
        _ => DataType::Utf8,
    }
}

#[derive(Debug)]
pub struct Neo4jExec {
    conn: Arc<Neo4jConnection>,
    cypher: String,
    label: String,
    projected_schema: SchemaRef,
    props: Arc<PlanProperties>,
}

impl Neo4jExec {
    pub fn new(
        conn: Arc<Neo4jConnection>,
        cypher: String,
        label: String,
        projected_schema: SchemaRef,
    ) -> Self {
        let props = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self { conn, cypher, label, projected_schema, props }
    }

    /// Build one Arrow column for `field` from the row maps, dispatching on the
    /// field's Arrow type (column name == RETURN alias == property name).
    fn build_column(field: &Field, rows: &[Map<String, Value>]) -> ArrayRef {
        let name = field.name().as_str();
        match field.data_type() {
            DataType::Int64 => {
                let mut b = Int64Builder::new();
                for r in rows {
                    b.append_option(r.get(name).and_then(Value::as_i64));
                }
                Arc::new(b.finish())
            }
            DataType::Float64 => {
                let mut b = Float64Builder::new();
                for r in rows {
                    b.append_option(r.get(name).and_then(Value::as_f64));
                }
                Arc::new(b.finish())
            }
            DataType::Boolean => {
                let mut b = BooleanBuilder::new();
                for r in rows {
                    b.append_option(r.get(name).and_then(Value::as_bool));
                }
                Arc::new(b.finish())
            }
            _ => {
                // Utf8 + fallback: pass strings through, stringify other JSON.
                let mut b = StringBuilder::new();
                for r in rows {
                    match r.get(name) {
                        Some(Value::String(s)) => b.append_value(s),
                        Some(v) if !v.is_null() => b.append_value(v.to_string()),
                        _ => b.append_null(),
                    }
                }
                Arc::new(b.finish())
            }
        }
    }

    fn build_batch(schema: &SchemaRef, rows: &[Map<String, Value>]) -> Result<RecordBatch> {
        // Empty projection (e.g. COUNT(*)): a 0-column batch carrying the row count.
        if schema.fields().is_empty() {
            let opts = RecordBatchOptions::new().with_row_count(Some(rows.len()));
            return RecordBatch::try_new_with_options(schema.clone(), vec![], &opts)
                .map_err(DataFusionError::from);
        }
        let cols: Vec<ArrayRef> =
            schema.fields().iter().map(|f| Self::build_column(f, rows)).collect();
        RecordBatch::try_new(schema.clone(), cols).map_err(DataFusionError::from)
    }
}

impl DisplayAs for Neo4jExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Neo4jExec: label={}, cypher={:?}", self.label, self.cypher)
    }
}

impl ExecutionPlan for Neo4jExec {
    fn name(&self) -> &str {
        "Neo4jExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
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
        let cypher = self.cypher.clone();
        let label = self.label.clone();
        let schema = self.projected_schema.clone();

        let fut = async move {
            let span = tracing::debug_span!(
                target: "connector_neo4j", "neo4j_query", label = %label
            );
            let _enter = span.enter();
            let rows = conn
                .run_query(&cypher, Value::Null)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            tracing::debug!(target: "connector_neo4j", rows = rows.len(), "neo4j query ok");
            Self::build_batch(&schema, &rows)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(self.projected_schema.clone(), stream)))
    }
}
