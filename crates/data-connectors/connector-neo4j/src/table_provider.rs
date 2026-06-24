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

//! `Neo4jTableProvider` -- exposes a Neo4j node label as a DataFusion table.
//!
//! ```text
//! SELECT name, age FROM people WHERE age > 30 LIMIT 10
//!   -> MATCH (n:`Person`) WHERE n.`age` > 30 RETURN n.`name` AS name, n.`age` AS age LIMIT 10
//! ```
//!
//! Equality/comparison/`IN` predicates on property columns push down into the
//! Cypher `WHERE`; the projection pushes into the `RETURN`; `LIMIT` passes
//! through. Pushed-down predicates are reported `Exact` so DataFusion does not
//! re-apply them.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::{Result, ScalarValue};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

use crate::exec::Neo4jExec;
use crate::neo4j::Neo4jConnection;

#[derive(Debug)]
pub struct Neo4jTableProvider {
    conn: Arc<Neo4jConnection>,
    label: String,
    schema: SchemaRef,
}

impl Neo4jTableProvider {
    /// `schema` is built dynamically from the label's introspected properties
    /// (see lib.rs): one typed column per property.
    pub fn new(conn: Arc<Neo4jConnection>, label: String, schema: SchemaRef) -> Self {
        Self { conn, label, schema }
    }
}

/// Backtick-escape a Cypher identifier (label or property name).
fn cypher_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Single-quote + escape a Cypher string literal.
fn cypher_str(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Render a scalar literal as Cypher, or None if the type isn't pushable.
fn scalar_to_cypher(v: &ScalarValue) -> Option<String> {
    match v {
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Some(cypher_str(s)),
        ScalarValue::Int64(Some(n)) => Some(n.to_string()),
        ScalarValue::Int32(Some(n)) => Some(n.to_string()),
        ScalarValue::Float64(Some(n)) => Some(n.to_string()),
        ScalarValue::Float32(Some(n)) => Some(n.to_string()),
        ScalarValue::Boolean(Some(b)) => Some(b.to_string()),
        _ => None,
    }
}

fn op_to_cypher(op: Operator) -> Option<&'static str> {
    match op {
        Operator::Eq => Some("="),
        Operator::NotEq => Some("<>"),
        Operator::Lt => Some("<"),
        Operator::LtEq => Some("<="),
        Operator::Gt => Some(">"),
        Operator::GtEq => Some(">="),
        _ => None,
    }
}

/// Translate a single predicate into a Cypher `WHERE` fragment, if it's a
/// comparison / `IN` on a known property column with literal operands.
fn predicate_cypher(expr: &Expr, schema: &SchemaRef) -> Option<String> {
    match expr {
        Expr::BinaryExpr(b) => {
            let op = op_to_cypher(b.op)?;
            if let (Expr::Column(c), Expr::Literal(v, _)) = (b.left.as_ref(), b.right.as_ref()) {
                if schema.field_with_name(&c.name).is_ok() {
                    let lit = scalar_to_cypher(v)?;
                    return Some(format!("n.{} {} {}", cypher_ident(&c.name), op, lit));
                }
            }
            None
        }
        Expr::InList(il) if !il.negated => {
            if let Expr::Column(c) = il.expr.as_ref() {
                if schema.field_with_name(&c.name).is_ok() {
                    let mut vals = Vec::with_capacity(il.list.len());
                    for e in &il.list {
                        if let Expr::Literal(v, _) = e {
                            vals.push(scalar_to_cypher(v)?);
                        } else {
                            return None;
                        }
                    }
                    if !vals.is_empty() {
                        return Some(format!("n.{} IN [{}]", cypher_ident(&c.name), vals.join(", ")));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

#[async_trait]
impl TableProvider for Neo4jTableProvider {
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
                if predicate_cypher(f, &self.schema).is_some() {
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
        let projected_schema = match projection {
            Some(idx) => Arc::new(self.schema.project(idx)?),
            None => self.schema.clone(),
        };

        let preds: Vec<String> =
            filters.iter().filter_map(|f| predicate_cypher(f, &self.schema)).collect();
        let where_clause =
            if preds.is_empty() { String::new() } else { format!(" WHERE {}", preds.join(" AND ")) };

        // Projection pushdown: RETURN only the projected property columns. An
        // empty projection (e.g. COUNT(*)) still needs one row per match.
        let return_clause = if projected_schema.fields().is_empty() {
            "RETURN 1 AS _cnt".to_string()
        } else {
            let cols: Vec<String> = projected_schema
                .fields()
                .iter()
                .map(|f| format!("n.{0} AS {0}", cypher_ident(f.name())))
                .collect();
            format!("RETURN {}", cols.join(", "))
        };

        let limit_clause = match limit {
            Some(k) => format!(" LIMIT {k}"),
            None => String::new(),
        };

        let cypher = format!(
            "MATCH (n:{}){} {}{}",
            cypher_ident(&self.label),
            where_clause,
            return_clause,
            limit_clause
        );

        Ok(Arc::new(Neo4jExec::new(
            Arc::clone(&self.conn),
            cypher,
            self.label.clone(),
            projected_schema,
        )))
    }
}

/// A Cypher-defined table (Phase 2): runs a fixed read Cypher statement and
/// exposes its RETURN columns as a table -- supporting relationship traversals.
/// The Cypher is opaque, so there is no predicate/LIMIT pushdown; DataFusion
/// applies `WHERE`/`LIMIT` on top. Projection is honored (only projected columns
/// are built). Schema is inferred at registration via `describe_cypher`.
#[derive(Debug)]
pub struct CypherTableProvider {
    conn: Arc<Neo4jConnection>,
    cypher: String,
    schema: SchemaRef,
}

impl CypherTableProvider {
    pub fn new(conn: Arc<Neo4jConnection>, cypher: String, schema: SchemaRef) -> Self {
        Self { conn, cypher, schema }
    }
}

#[async_trait]
impl TableProvider for CypherTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    // No supports_filters_pushdown override -> all filters Unsupported (default):
    // the Cypher is opaque, so DataFusion applies WHERE/LIMIT itself.

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let projected_schema = match projection {
            Some(idx) => Arc::new(self.schema.project(idx)?),
            None => self.schema.clone(),
        };
        Ok(Arc::new(Neo4jExec::new(
            Arc::clone(&self.conn),
            self.cypher.clone(),
            "cypher".to_string(),
            projected_schema,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]))
    }

    #[test]
    fn eq_predicate_pushes_down() {
        assert_eq!(
            predicate_cypher(&col("name").eq(lit("Alice")), &schema()),
            Some("n.`name` = 'Alice'".to_string())
        );
    }

    #[test]
    fn comparison_predicate_pushes_down() {
        assert_eq!(
            predicate_cypher(&col("age").gt(lit(30i64)), &schema()),
            Some("n.`age` > 30".to_string())
        );
    }

    #[test]
    fn in_list_pushes_down() {
        assert_eq!(
            predicate_cypher(&col("name").in_list(vec![lit("A"), lit("B")], false), &schema()),
            Some("n.`name` IN ['A', 'B']".to_string())
        );
    }

    #[test]
    fn unknown_column_not_pushable() {
        assert_eq!(predicate_cypher(&col("missing").eq(lit("x")), &schema()), None);
    }

    #[test]
    fn string_literals_are_escaped() {
        assert_eq!(
            predicate_cypher(&col("name").eq(lit("O'Brien")), &schema()),
            Some("n.`name` = 'O\\'Brien'".to_string())
        );
    }
}
