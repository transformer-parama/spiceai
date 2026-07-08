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

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::catalog::TableFunctionImpl;
use datafusion::common::{DataFusionError, Result, ScalarValue, TableReference};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::Expr;
use runtime::datafusion::DataFusion;

use crate::exec::arrow_type_for;
use crate::table_provider::{CypherTableProvider, Neo4jTableProvider};

pub const GRAPH_QUERY_UDTF_NAME: &str = "graph_query";

/// Max distinct Cypher strings whose inferred schema we cache. Bounds memory when
/// callers issue many unique (per-anchor) queries; on overflow the cache is cleared.
const SCHEMA_CACHE_CAP: usize = 512;

#[derive(Debug)]
pub struct GraphQueryTableFunc {
    df: Weak<DataFusion>,
    /// Caches the inferred output schema per (dataset, wrapped-cypher) so a repeated
    /// `graph_query` call skips the describe-probe, which otherwise re-runs the
    /// traversal a second time on every single call.
    schema_cache: Mutex<HashMap<String, SchemaRef>>,
}

impl GraphQueryTableFunc {
    #[must_use]
    pub fn new(df: Weak<DataFusion>) -> Self {
        Self { df, schema_cache: Mutex::new(HashMap::new()) }
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

/// Default cap on rows a single `graph_query` call may return.
const DEFAULT_MAX_ROWS: usize = 10_000;

/// Row cap, overridable via `SPICE_GRAPH_QUERY_MAX_ROWS` (`0` disables the wrap/cap).
fn max_rows() -> usize {
    std::env::var("SPICE_GRAPH_QUERY_MAX_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_ROWS)
}

/// Cypher clauses that mutate the graph or load external data. `graph_query` is a
/// READ-ONLY retrieval primitive, so any of these is rejected — defence against
/// injection / accidental writes when the Cypher is built from untrusted input.
const WRITE_KEYWORDS: &[&str] = &[
    "CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "DETACH", "FOREACH", "LOAD",
];

/// Blank out the CONTENTS of single- and double-quoted string literals (keeping the
/// surrounding structure) so the read-only scan sees only Cypher SYNTAX, never user
/// text. Without this, a perfectly read-only query whose literal happens to contain a
/// write-keyword word — e.g. a fulltext phrase `'data set'` or an entity name
/// `"create solutions"` — would be wrongly rejected. Handles `\'`/`\"`/`\\` escapes.
/// Backtick-quoted identifiers are left as-is (they can't smuggle an executable clause).
fn blank_string_literals(cypher: &str) -> String {
    let mut out = String::with_capacity(cypher.len());
    let mut quote: Option<char> = None;
    let mut chars = cypher.chars();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                out.push(' ');
                if c == '\\' {
                    // escaped char inside the literal: consume and blank it too
                    if chars.next().is_some() {
                        out.push(' ');
                    }
                } else if c == q {
                    quote = None;
                }
            }
            None => {
                if c == '\'' || c == '"' {
                    quote = Some(c);
                    out.push(' ');
                } else {
                    out.push(c);
                }
            }
        }
    }
    out
}

/// Reject Cypher containing a write/DDL clause. Tokenises on non-identifier characters
/// so `set_value`/`created_at` are single tokens that do NOT match `SET`/`CREATE`, while
/// `apoc.create` -> [`apoc`,`create`] IS caught. String-literal contents are blanked
/// first (see [`blank_string_literals`]), so a write keyword appearing inside quoted
/// user text is NOT a false positive — only Cypher syntax is scanned.
fn ensure_read_only(cypher: &str) -> Result<()> {
    let scannable = blank_string_literals(cypher);
    for token in scannable.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if token.is_empty() {
            continue;
        }
        if WRITE_KEYWORDS.contains(&token.to_ascii_uppercase().as_str()) {
            return Err(DataFusionError::Plan(format!(
                "{GRAPH_QUERY_UDTF_NAME}(): only read-only Cypher is allowed; found write clause '{token}'"
            )));
        }
    }
    Ok(())
}

/// Wrap the (already read-only) Cypher so at most `cap` rows are returned, using the same
/// `CALL {{ ... }} RETURN *` shape as schema inference so RETURN columns are preserved.
fn with_row_cap(cypher: &str, cap: usize) -> String {
    let inner = cypher.trim().trim_end_matches(';').trim();
    if cap == 0 {
        inner.to_string()
    } else {
        format!("CALL {{ {inner} }} RETURN * LIMIT {cap}")
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

        // Safety: read-only only, and cap the row count.
        ensure_read_only(&cypher)?;
        let cypher = with_row_cap(&cypher, max_rows());

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

        // Cache key is per (dataset, wrapped-cypher): the same call reuses the schema
        // and skips the describe-probe (which re-runs the traversal). NUL separates the
        // two parts so they can't collide.
        let cache_key = format!("{dataset}\u{0}{cypher}");
        if let Some(schema) = self
            .schema_cache
            .lock()
            .ok()
            .and_then(|c| c.get(&cache_key).cloned())
        {
            return Ok(Arc::new(CypherTableProvider::new(conn, cypher, schema)));
        }

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

        let schema: SchemaRef = Arc::new(Schema::new(
            props
                .iter()
                .map(|p| Field::new(&p.name, arrow_type_for(&p.type_name), true))
                .collect::<Vec<Field>>(),
        ));

        if let Ok(mut cache) = self.schema_cache.lock() {
            if cache.len() >= SCHEMA_CACHE_CAP {
                cache.clear(); // simple bound: drop everything rather than grow unbounded
            }
            cache.insert(cache_key, Arc::clone(&schema));
        }

        Ok(Arc::new(CypherTableProvider::new(conn, cypher, schema)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_accepts_reads() {
        for q in [
            "MATCH (a:Chunk)-[:NEXT*1..3]->(b) RETURN a.idx, b.idx",
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
            "MATCH (n) WHERE n.created_at > 0 RETURN n.set_value AS v LIMIT 5",
            "MATCH (a)-[r]->(b) RETURN type(r), count(*)",
        ] {
            assert!(ensure_read_only(q).is_ok(), "should be read-only: {q}");
        }
    }

    #[test]
    fn read_only_accepts_write_words_inside_string_literals() {
        // These are all read-only queries whose STRING LITERALS contain a word that
        // matches a write keyword. They must NOT be rejected (the pre-fix bug).
        for q in [
            "CALL db.index.fulltext.queryNodes('entity_names_fulltext', 'data set migration') YIELD node RETURN node.name",
            "MATCH (e:Entity {canonical_name:\"create solutions pvt ltd\"})<-[:MENTIONS]-(c) RETURN c.content AS text",
            "MATCH (n) WHERE n.title = 'load balancing services' RETURN n.title",
            "MATCH (n) WHERE n.note = 'set forth herein and delete on merge' RETURN n.id",
            "MATCH (n {name: 'O\\'Brien DROP table'}) RETURN n.name",
        ] {
            assert!(ensure_read_only(q).is_ok(), "literal text must not trip the guard: {q}");
        }
    }

    #[test]
    fn read_only_still_rejects_writes_next_to_literals() {
        // A real write clause outside a literal is still caught even when a literal
        // elsewhere also contains a keyword word.
        assert!(ensure_read_only("MATCH (n {x:'a set of things'}) SET n.y = 1").is_err());
        assert!(ensure_read_only("MATCH (n) WHERE n.a = 'create' CREATE (m)").is_err());
    }

    #[test]
    fn blank_string_literals_preserves_structure_only() {
        let out = blank_string_literals("MATCH (n {a:'SET x'}) RETURN n");
        assert!(!out.contains("SET"), "literal content must be blanked: {out}");
        assert!(out.contains("MATCH (n {a:"), "syntax must be preserved: {out}");
        assert!(out.contains("RETURN n"));
    }

    #[test]
    fn read_only_rejects_writes() {
        for q in [
            "MATCH (a) CREATE (a)-[:R]->(b)",
            "MERGE (n:X {id: 1})",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) SET n.x = 1",
            "MATCH (n) REMOVE n.x",
            "CALL apoc.create.node(['X'], {})",
            "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
        ] {
            assert!(ensure_read_only(q).is_err(), "should be rejected: {q}");
        }
    }

    #[test]
    fn row_cap_wraps_and_strips_semicolon() {
        let out = with_row_cap("MATCH (n) RETURN n.id AS id ;", 100);
        assert_eq!(out, "CALL { MATCH (n) RETURN n.id AS id } RETURN * LIMIT 100");
    }

    #[test]
    fn row_cap_zero_disables_wrap() {
        assert_eq!(with_row_cap("MATCH (n) RETURN n", 0), "MATCH (n) RETURN n");
    }
}
