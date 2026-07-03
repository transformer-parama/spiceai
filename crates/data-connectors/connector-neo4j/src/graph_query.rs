//! `graph_query('<neo4j_dataset>', '<cypher>')` — a DataFusion table function that
//! runs an ARBITRARY read Cypher statement at query time (dynamic per-query
//! traversal) and exposes its RETURN columns as a federatable table. It borrows
//! the Neo4j connection from an already-registered neo4j dataset (first arg), so
//! no separate connection config is needed. Schema is inferred via `describe_cypher`.
//!
//! Unlike a `neo4j_cypher` dataset (fixed Cypher at registration), this lets the
//! Cypher be built per request (e.g. from linked entities), which is what GraphRAG
//! needs. Registered from `bin/spiced` (which can see both `runtime` and this crate)
//! because `runtime` cannot depend on this crate (it would be a dependency cycle).

use std::sync::{Arc, Weak};

use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::catalog::TableFunctionImpl;
use datafusion::common::{DataFusionError, Result, ScalarValue, TableReference};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::Expr;
use runtime::datafusion::DataFusion;

use crate::exec::arrow_type_for;
use crate::table_provider::{CypherTableProvider, Neo4jTableProvider};

pub const GRAPH_QUERY_UDTF_NAME: &str = "graph_query";

#[derive(Debug)]
pub struct GraphQueryTableFunc {
    df: Weak<DataFusion>,
}

impl GraphQueryTableFunc {
    #[must_use]
    pub fn new(df: Weak<DataFusion>) -> Self {
        Self { df }
    }
}

fn string_arg(expr: &Expr, which: &str) -> Result<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => Ok(s.clone()),
        _ => Err(DataFusionError::Plan(format!(
            "{GRAPH_QUERY_UDTF_NAME}(): the {which} argument must be a string literal"
        ))),
    }
}

impl TableFunctionImpl for GraphQueryTableFunc {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if args.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "{GRAPH_QUERY_UDTF_NAME}(neo4j_dataset, cypher) expects 2 string arguments, got {}",
                args.len()
            )));
        }
        let dataset = string_arg(&args[0], "first (neo4j dataset name)")?;
        let cypher = string_arg(&args[1], "second (Cypher query)")?;

        let df = self.df.upgrade().ok_or_else(|| {
            DataFusionError::Plan(format!(
                "{GRAPH_QUERY_UDTF_NAME}(): DataFusion instance has been dropped"
            ))
        })?;

        let provider = df
            .get_table_sync(&TableReference::from(dataset.as_str()))
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "{GRAPH_QUERY_UDTF_NAME}(): dataset '{dataset}' not found; pass a registered neo4j dataset name"
                ))
            })?;

        // A registered dataset's provider is wrapped (metadata / federation adaptor),
        // so unwrap to the concrete Neo4jTableProvider to borrow its connection.
        let neo = runtime::search::util::find_concrete_table_provider::<Neo4jTableProvider>(
            &provider,
        )
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "{GRAPH_QUERY_UDTF_NAME}(): dataset '{dataset}' is not a neo4j dataset"
            ))
        })?;
        let conn = neo.connection();

        // Infer the output schema by describing the Cypher. `call()` is sync but the
        // connector is async; run it on the current multi-thread runtime.
        let cypher_for_probe = cypher.clone();
        let props = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(conn.describe_cypher(&cypher_for_probe))
        })
        .map_err(|e| {
            DataFusionError::Plan(format!("{GRAPH_QUERY_UDTF_NAME}(): failed to describe Cypher: {e}"))
        })?;

        if props.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "{GRAPH_QUERY_UDTF_NAME}(): the Cypher returned no columns to infer a schema"
            )));
        }

        let schema = Arc::new(Schema::new(
            props
                .iter()
                .map(|p| Field::new(&p.name, arrow_type_for(&p.type_name), true))
                .collect::<Vec<Field>>(),
        ));

        Ok(Arc::new(CypherTableProvider::new(conn, cypher, schema)))
    }
}
