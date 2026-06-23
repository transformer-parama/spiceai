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
//! SELECT doc_type, source_id, product_id, title, text, score
//! FROM documents
//! WHERE query_vector = '[0.01, -0.02, ...]'   -- the query embedding (JSON)
//!   AND product_id = 3                          -- optional -> Milvus filter
//!   AND doc_type IN ('ticket')                  -- optional -> Milvus filter
//! LIMIT 50                                       -- -> Milvus top-k
//! ```
//!
//! `query_vector = '...'` carries the search vector; `product_id` / `doc_type`
//! predicates are translated into a Milvus boolean filter. All consumed
//! predicates are reported `Exact` so DataFusion does not re-apply them.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

use crate::exec::MilvusExec;
use crate::milvus::{MilvusCollection, MilvusConnection};

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

/// Pull the query embedding out of a `query_vector = '[...]'` predicate.
fn extract_query_vector(filters: &[Expr]) -> Option<Vec<f32>> {
    for f in filters {
        if let Expr::BinaryExpr(b) = f {
            if b.op == Operator::Eq {
                if let (Expr::Column(c), Expr::Literal(ScalarValue::Utf8(Some(s)), _)) =
                    (b.left.as_ref(), b.right.as_ref())
                {
                    if c.name == "query_vector" {
                        return serde_json::from_str::<Vec<f32>>(s).ok();
                    }
                }
            }
        }
    }
    None
}

/// Escape a string value for a Milvus boolean-expression double-quoted literal.
fn milvus_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Translate `product_id`/`doc_type` predicates into a Milvus boolean filter.
fn extract_milvus_filter(filters: &[Expr]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for f in filters {
        if let Expr::BinaryExpr(b) = f {
            if b.op == Operator::Eq {
                if let Expr::Column(c) = b.left.as_ref() {
                    match (c.name.as_str(), b.right.as_ref()) {
                        ("product_id", Expr::Literal(ScalarValue::Int64(Some(v)), _)) => {
                            parts.push(format!("product_id == {v}"));
                        }
                        ("doc_type", Expr::Literal(ScalarValue::Utf8(Some(s)), _)) => {
                            parts.push(format!("doc_type == {}", milvus_str(s)));
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Expr::InList(il) = f {
            if let Expr::Column(c) = il.expr.as_ref() {
                if c.name == "doc_type" && !il.negated {
                    let vals: Vec<String> = il
                        .list
                        .iter()
                        .filter_map(|e| match e {
                            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => Some(milvus_str(s)),
                            _ => None,
                        })
                        .collect();
                    if !vals.is_empty() {
                        parts.push(format!("doc_type in [{}]", vals.join(",")));
                    }
                }
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" and "))
    }
}

fn is_pushable(expr: &Expr) -> bool {
    match expr {
        Expr::BinaryExpr(b) if b.op == Operator::Eq => matches!(
            b.left.as_ref(),
            Expr::Column(c) if matches!(c.name.as_str(), "query_vector" | "product_id" | "doc_type")
        ),
        Expr::InList(il) => matches!(il.expr.as_ref(), Expr::Column(c) if c.name == "doc_type"),
        _ => false,
    }
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
                if is_pushable(f) {
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
        let query_vector = extract_query_vector(filters).ok_or_else(|| {
            DataFusionError::Plan(
                "milvus: a query embedding is required, e.g. WHERE query_vector = '[...]'"
                    .to_string(),
            )
        })?;
        let filter = extract_milvus_filter(filters);
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
    use datafusion::prelude::{col, lit};

    #[test]
    fn extracts_query_vector_from_predicate() {
        let filters = vec![col("query_vector").eq(lit("[0.1,0.2,0.3]"))];
        assert_eq!(extract_query_vector(&filters), Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn no_query_vector_returns_none() {
        let filters = vec![col("product_id").eq(lit(3i64))];
        assert_eq!(extract_query_vector(&filters), None);
    }

    #[test]
    fn builds_milvus_filter_from_eq_predicates() {
        let filters = vec![
            col("query_vector").eq(lit("[0.1]")),
            col("product_id").eq(lit(3i64)),
            col("doc_type").eq(lit("ticket")),
        ];
        assert_eq!(
            extract_milvus_filter(&filters),
            Some(r#"product_id == 3 and doc_type == "ticket""#.to_string())
        );
    }

    #[test]
    fn builds_in_list_filter() {
        let filters = vec![col("doc_type").in_list(vec![lit("ticket"), lit("article")], false)];
        assert_eq!(
            extract_milvus_filter(&filters),
            Some(r#"doc_type in ["ticket","article"]"#.to_string())
        );
    }

    #[test]
    fn pushdown_classification() {
        assert!(is_pushable(&col("query_vector").eq(lit("[0.1]"))));
        assert!(is_pushable(&col("product_id").eq(lit(3i64))));
        assert!(is_pushable(&col("doc_type").in_list(vec![lit("ticket")], false)));
        assert!(!is_pushable(&col("title").eq(lit("x")))); // not a Milvus-pushable column
    }
}
