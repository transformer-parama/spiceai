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

//! `MilvusTableProvider` -- exposes a Milvus collection as a DataFusion table.
//!
//! ```text
//! SELECT <scalar columns>, score
//! FROM <collection>
//! WHERE query_vector = '[0.01, -0.02, ...]'   -- the query embedding (JSON)
//!   AND <scalar_col> = <value>                 -- optional -> Milvus filter
//!   AND <scalar_col> IN (...)                  -- optional -> Milvus filter
//! LIMIT 50                                       -- -> Milvus top-k
//! ```
//!
//! `query_vector = '...'` carries the search vector; equality/comparison/`IN`
//! predicates on **any** scalar column (introspected from the collection) are
//! translated into a Milvus boolean filter. All consumed predicates are reported
//! `Exact` so DataFusion does not re-apply them.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

use milvus_client::MilvusExec;
use milvus_client::{MilvusCollection, MilvusConnection};

const DEFAULT_TOP_K: usize = 50;

#[derive(Debug)]
pub struct MilvusTableProvider {
    conn: Arc<MilvusConnection>,
    coll: MilvusCollection,
    schema: SchemaRef,
}

impl MilvusTableProvider {
    /// `schema` is built dynamically from the collection's introspected fields
    /// (see lib.rs): query_vector (input) + the collection's scalar columns + score.
    pub fn new(conn: Arc<MilvusConnection>, coll: MilvusCollection, schema: SchemaRef) -> Self {
        Self { conn, coll, schema }
    }
}

/// The raw `query_vector = '[...]'` literal (the JSON-encoded embedding), if the
/// predicate is present. Parsing/validation happens in `scan` so a malformed
/// vector yields a clear error distinct from "no embedding given".
fn query_vector_literal(filters: &[Expr]) -> Option<&str> {
    for f in filters {
        if let Expr::BinaryExpr(b) = f {
            if b.op == Operator::Eq {
                if let (Expr::Column(c), Expr::Literal(ScalarValue::Utf8(Some(s)), _)) =
                    (b.left.as_ref(), b.right.as_ref())
                {
                    if c.name == "query_vector" {
                        return Some(s.as_str());
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
fn extract_query_vector(filters: &[Expr]) -> Option<Vec<f32>> {
    query_vector_literal(filters).and_then(|s| serde_json::from_str(s).ok())
}

/// Escape a string value for a Milvus boolean-expression double-quoted literal.
fn milvus_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Render a scalar literal as a Milvus boolean-expression value, or None if the
/// type isn't pushable.
fn milvus_value(v: &ScalarValue) -> Option<String> {
    match v {
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Some(milvus_str(s)),
        ScalarValue::Int64(Some(n)) => Some(n.to_string()),
        ScalarValue::Int32(Some(n)) => Some(n.to_string()),
        ScalarValue::Float64(Some(n)) => Some(n.to_string()),
        ScalarValue::Float32(Some(n)) => Some(n.to_string()),
        ScalarValue::Boolean(Some(b)) => Some(b.to_string()),
        _ => None,
    }
}

fn milvus_op(op: Operator) -> Option<&'static str> {
    match op {
        Operator::Eq => Some("=="),
        Operator::NotEq => Some("!="),
        Operator::Lt => Some("<"),
        Operator::LtEq => Some("<="),
        Operator::Gt => Some(">"),
        Operator::GtEq => Some(">="),
        _ => None,
    }
}

/// A scalar column we can filter on: present in the (introspected) schema and not
/// one of the synthetic columns (`query_vector` is the search input; `score` is output).
fn is_filter_col(name: &str, schema: &SchemaRef) -> bool {
    name != "query_vector" && name != "score" && schema.field_with_name(name).is_ok()
}

/// Translate one predicate into a Milvus boolean-filter fragment, for ANY scalar
/// column the collection has (comparison / `IN`). Nothing schema-specific is hardcoded.
fn predicate_milvus(expr: &Expr, schema: &SchemaRef) -> Option<String> {
    match expr {
        Expr::BinaryExpr(b) => {
            let op = milvus_op(b.op)?;
            if let (Expr::Column(c), Expr::Literal(v, _)) = (b.left.as_ref(), b.right.as_ref()) {
                if is_filter_col(&c.name, schema) {
                    return Some(format!("{} {} {}", c.name, op, milvus_value(v)?));
                }
            }
            None
        }
        Expr::InList(il) if !il.negated => {
            if let Expr::Column(c) = il.expr.as_ref() {
                if is_filter_col(&c.name, schema) {
                    let mut vals = Vec::with_capacity(il.list.len());
                    for e in &il.list {
                        if let Expr::Literal(v, _) = e {
                            vals.push(milvus_value(v)?);
                        } else {
                            return None;
                        }
                    }
                    if !vals.is_empty() {
                        return Some(format!("{} in [{}]", c.name, vals.join(", ")));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Build a Milvus boolean filter from all pushable predicates (excludes the
/// `query_vector` search predicate, which is consumed as the search vector).
fn extract_milvus_filter(filters: &[Expr], schema: &SchemaRef) -> Option<String> {
    let parts: Vec<String> = filters.iter().filter_map(|f| predicate_milvus(f, schema)).collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" and "))
    }
}

fn is_pushable(expr: &Expr, schema: &SchemaRef) -> bool {
    // the query_vector = '[...]' predicate is the search input (consumed in scan)
    if let Expr::BinaryExpr(b) = expr {
        if b.op == Operator::Eq {
            if let Expr::Column(c) = b.left.as_ref() {
                if c.name == "query_vector" {
                    return true;
                }
            }
        }
    }
    predicate_milvus(expr, schema).is_some()
}

#[async_trait]
impl TableProvider for MilvusTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|f| {
                if is_pushable(f, &self.schema) {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let qv_literal = query_vector_literal(filters).ok_or_else(|| {
            DataFusionError::Plan(
                "milvus: a query embedding is required, e.g. \
                 WHERE query_vector = '[0.1, 0.2, ...]'"
                    .to_string(),
            )
        })?;
        let query_vector: Vec<f32> = serde_json::from_str(qv_literal).map_err(|e| {
            DataFusionError::Plan(format!(
                "milvus: query_vector must be a JSON array of floats matching the \
                 collection's vector dimension (e.g. '[0.1, 0.2]'): {e}"
            ))
        })?;
        let filter = extract_milvus_filter(filters, &self.schema);
        let top_k = limit.unwrap_or(DEFAULT_TOP_K);

        Ok(Arc::new(MilvusExec::new(
            Arc::clone(&self.conn),
            self.coll.clone(),
            self.schema.clone(),
            query_vector,
            filter,
            top_k,
            projection.cloned(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};

    // a representative introspected schema (generic columns — nothing hardcoded)
    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("query_vector", DataType::Utf8, true),
            Field::new("category", DataType::Utf8, true),
            Field::new("year", DataType::Int64, true),
            Field::new("score", DataType::Float32, true),
        ]))
    }

    #[test]
    fn extracts_query_vector_from_predicate() {
        let filters = vec![col("query_vector").eq(lit("[0.1,0.2,0.3]"))];
        assert_eq!(extract_query_vector(&filters), Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn no_query_vector_returns_none() {
        let filters = vec![col("year").eq(lit(3i64))];
        assert_eq!(extract_query_vector(&filters), None);
    }

    #[test]
    fn builds_filter_from_eq_predicates_on_any_scalar_column() {
        let filters = vec![
            col("query_vector").eq(lit("[0.1]")), // the search vector -- not a filter
            col("year").eq(lit(3i64)),
            col("category").eq(lit("news")),
        ];
        assert_eq!(
            extract_milvus_filter(&filters, &schema()),
            Some(r#"year == 3 and category == "news""#.to_string())
        );
    }

    #[test]
    fn comparison_operators_push_down() {
        assert_eq!(
            extract_milvus_filter(&[col("year").gt(lit(2000i64))], &schema()),
            Some("year > 2000".to_string())
        );
    }

    #[test]
    fn builds_in_list_filter() {
        let filters = vec![col("category").in_list(vec![lit("news"), lit("blog")], false)];
        assert_eq!(
            extract_milvus_filter(&filters, &schema()),
            Some(r#"category in ["news", "blog"]"#.to_string())
        );
    }

    #[test]
    fn pushdown_classification() {
        let s = schema();
        assert!(is_pushable(&col("query_vector").eq(lit("[0.1]")), &s));
        assert!(is_pushable(&col("year").eq(lit(3i64)), &s)); // any scalar column
        assert!(is_pushable(&col("category").in_list(vec![lit("news")], false), &s));
        assert!(!is_pushable(&col("missing").eq(lit("x")), &s)); // not in the schema
        assert!(!is_pushable(&col("score").eq(lit(1.0f32)), &s)); // synthetic column
    }
}
