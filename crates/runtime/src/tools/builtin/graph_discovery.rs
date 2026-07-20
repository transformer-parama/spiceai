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

//! Live graph-ontology discovery for Neo4j-backed datasets, surfaced through
//! `list_datasets`.
//!
//! A Neo4j dataset exposes only ONE node label as a table, so the dataset listing
//! alone tells a client nothing about what the graph actually contains. The values
//! that matter are **unguessable** — labels and relationship types are extracted
//! per-corpus (`TRANSFERS_TO`, `hospital_department`, …), so a model writing Cypher
//! must be handed them or it invents names and returns zero rows. This module
//! discovers them and attaches them to the dataset's `metadata`:
//!
//! * `node_labels`        — every node label (minus the per-tenant scope label)
//! * `relationship_types` — every relationship type
//! * `ontology`           — `subject -[rel]-> object` triples over LABEL SETS, e.g.
//!   `["Entity","patient"] -[TRANSFERS_TO]-> ["Entity","hospital_department"]`
//!
//! The triples are the point: flat lists let a model pair any label with any
//! relationship, while the triples say which pairings actually exist. Entity *types*
//! need no special handling — they are themselves labels (`Entity:patient`), so they
//! appear in `node_labels` and the triples show which labels are entity subtypes.
//!
//! This is deliberately **convention-free**: nothing assumes an `Entity` label or an
//! `entity_type` property, so it works for any Neo4j graph with zero configuration.
//!
//! Everything runs through the `graph_query` UDTF, so per-tenant scoping applies
//! automatically (`SPICE_NEO4J_SCOPE_LABEL`) and the discovered ontology is exactly
//! the calling assistant's.
//!
//! The graph schema is STATIC after ingestion, so it is discovered exactly ONCE per
//! (dataset, scope) and kept for the process lifetime — no TTL, no re-query.
//! `try_get_with` gives single-flight so concurrent first-calls hit Neo4j once.
//! Discovery is best-effort: failures yield `None` and never break the listing.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, LazyLock};

use arrow::array::{Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use moka::future::Cache;
use serde::Serialize;

use crate::Runtime;

/// One `subject -[rel]-> object` edge shape, where subject/object are the node's
/// label set (the per-tenant scope label removed).
#[derive(Clone, Debug, Serialize)]
pub struct OntologyTriple {
    pub subject: Vec<String>,
    pub rel: String,
    pub object: Vec<String>,
}

/// The property keys seen on nodes carrying a given label. Labels and relationship
/// types are unguessable; so are PROPERTY keys — a model writing Cypher must be told
/// which properties exist or it invents `patient_id` / `name` and gets nulls.
#[derive(Clone, Debug, Serialize)]
pub struct LabelProperties {
    pub label: String,
    pub properties: Vec<String>,
}

/// A Neo4j dataset's discovered graph ontology (scoped to the calling tenant).
#[derive(Clone, Debug, Default)]
pub struct GraphSchema {
    pub node_labels: Vec<String>,
    pub relationship_types: Vec<String>,
    pub ontology: Vec<OntologyTriple>,
    /// Per-label property keys. Best-effort: empty if discovery failed.
    pub node_properties: Vec<LabelProperties>,
}

/// (dataset, scope-label) -> discovered ontology, memoized for the process lifetime.
/// Keyed by scope so a change of `SPICE_NEO4J_SCOPE_LABEL` never serves another
/// tenant's cached ontology. No TTL: the schema is static after ingestion.
static GRAPH_SCHEMA_CACHE: LazyLock<Cache<String, Arc<GraphSchema>>> =
    LazyLock::new(|| Cache::builder().max_capacity(1024).build());

/// True when a dataset's `from:` targets the Neo4j connector.
#[must_use]
pub fn is_neo4j_from(from: &str) -> bool {
    from.trim_start().to_ascii_lowercase().starts_with("neo4j:")
}

/// Discover the ontology of this dataset's graph, scoped to the calling tenant.
/// `dataset` is the SQL name used as the `graph_query` connection anchor. Returns
/// `None` for a non-Neo4j dataset or if discovery fails.
pub async fn neo4j_graph_schema(
    rt: &Arc<Runtime>,
    from: &str,
    dataset: &str,
) -> Option<Arc<GraphSchema>> {
    if !is_neo4j_from(from) {
        return None;
    }
    let scope = std::env::var("SPICE_NEO4J_SCOPE_LABEL")
        .unwrap_or_default()
        .trim()
        .to_string();
    let key = format!("{dataset}\u{0}{scope}");

    // Owned captures so the memoized future is `'static` (moka may hold it).
    let rt = Arc::clone(rt);
    let dataset = dataset.to_string();

    GRAPH_SCHEMA_CACHE
        .try_get_with(key, async move {
            // Scope-enforced by graph_query: `MATCH (n)` / `MATCH ()` become
            // `(n:<scope>)` / `(:<scope>)` when a scope label is configured.
            let mut node_labels = distinct_strings(
                &rt,
                &dataset,
                "MATCH (n) UNWIND labels(n) AS v RETURN DISTINCT v",
            )
            .await
            .ok_or("node-label discovery failed")?;
            strip_scope(&mut node_labels, &scope);

            let relationship_types =
                distinct_strings(&rt, &dataset, "MATCH ()-[r]->() RETURN DISTINCT type(r) AS v")
                    .await
                    .ok_or("relationship-type discovery failed")?;

            let ontology = triples(
                &rt,
                &dataset,
                "MATCH (a)-[r]->(b) RETURN DISTINCT labels(a) AS s, type(r) AS r, labels(b) AS o",
                &scope,
            )
            .await
            .ok_or("ontology discovery failed")?;

            // Best-effort — property discovery must never break the (more valuable)
            // labels/relationships/ontology above. A `None` just yields no properties.
            let node_properties = label_properties(&rt, &dataset, &scope)
                .await
                .unwrap_or_default();

            Ok::<_, &'static str>(Arc::new(GraphSchema {
                node_labels,
                relationship_types,
                ontology,
                node_properties,
            }))
        })
        .await
        .ok()
}

/// Drop the per-tenant scope label — it is on every node, so it is pure noise in a
/// schema listing.
fn strip_scope(labels: &mut Vec<String>, scope: &str) {
    if !scope.is_empty() {
        labels.retain(|l| l != scope);
    }
}

/// Run `cypher` through `graph_query`, selecting `select`. Best-effort.
async fn run_graph_query(
    rt: &Arc<Runtime>,
    dataset: &str,
    cypher: &str,
    select: &str,
) -> Option<Vec<RecordBatch>> {
    // Escape single quotes for the SQL string literals (our fixed discovery queries
    // contain none, but the dataset name is not ours).
    let ds = dataset.replace('\'', "''");
    let cy = cypher.replace('\'', "''");
    let sql = format!("SELECT {select} FROM graph_query('{ds}', '{cy}')");

    rt.datafusion()
        .query_builder(&sql)
        .read_only(true)
        .build()
        .run()
        .await
        .ok()?
        .data
        .try_collect::<Vec<RecordBatch>>()
        .await
        .ok()
}

/// Column `i` of `b` as a `StringArray`, if it is one.
fn str_col(b: &RecordBatch, i: usize) -> Option<&StringArray> {
    b.column(i).as_any().downcast_ref::<StringArray>()
}

/// Run a one-column (`v`) discovery Cypher and collect the distinct strings.
async fn distinct_strings(rt: &Arc<Runtime>, dataset: &str, cypher: &str) -> Option<Vec<String>> {
    let batches = run_graph_query(rt, dataset, cypher, "v").await?;
    let mut out: Vec<String> = Vec::new();
    for b in &batches {
        if b.num_columns() == 0 {
            continue;
        }
        if let Some(sa) = str_col(b, 0) {
            for i in 0..sa.len() {
                if !sa.is_null(i) {
                    out.push(sa.value(i).to_string());
                }
            }
        }
    }
    out.sort();
    out.dedup();
    Some(out)
}

/// The Neo4j connector renders a Cypher list (`labels(n)`) as its JSON text, so a
/// label-set column arrives as `["Entity","patient"]`. Parse it back, dropping the
/// tenant scope label.
fn parse_label_set(raw: &str, scope: &str) -> Vec<String> {
    let mut labels: Vec<String> = serde_json::from_str::<Vec<String>>(raw).unwrap_or_default();
    strip_scope(&mut labels, scope);
    labels
}

/// Run the three-column (`s`, `r`, `o`) ontology Cypher and collect the triples.
async fn triples(
    rt: &Arc<Runtime>,
    dataset: &str,
    cypher: &str,
    scope: &str,
) -> Option<Vec<OntologyTriple>> {
    let batches = run_graph_query(rt, dataset, cypher, "s, r, o").await?;
    let mut out: Vec<OntologyTriple> = Vec::new();
    for b in &batches {
        if b.num_columns() < 3 {
            continue;
        }
        let (Some(s), Some(r), Some(o)) = (str_col(b, 0), str_col(b, 1), str_col(b, 2)) else {
            continue;
        };
        for i in 0..b.num_rows() {
            if s.is_null(i) || r.is_null(i) || o.is_null(i) {
                continue;
            }
            out.push(OntologyTriple {
                subject: parse_label_set(s.value(i), scope),
                rel: r.value(i).to_string(),
                object: parse_label_set(o.value(i), scope),
            });
        }
    }
    out.sort_by(|a, b| (&a.subject, &a.rel, &a.object).cmp(&(&b.subject, &b.rel, &b.object)));
    out.dedup_by(|a, b| a.subject == b.subject && a.rel == b.rel && a.object == b.object);
    Some(out)
}

/// Discover the property keys present on nodes, grouped by label. Runs the scoped
/// `MATCH (n) UNWIND labels(n) … UNWIND keys(n) …` — same full-scan cost profile as
/// node-label discovery, and equally one-shot/cached. A key seen on a multi-label
/// node (`:Entity:patient`) is attributed to each of its labels; that is intended —
/// it tells the model which properties are readable on a node bearing that label.
async fn label_properties(
    rt: &Arc<Runtime>,
    dataset: &str,
    scope: &str,
) -> Option<Vec<LabelProperties>> {
    let batches = run_graph_query(
        rt,
        dataset,
        "MATCH (n) UNWIND labels(n) AS s UNWIND keys(n) AS o RETURN DISTINCT s AS s, o AS o",
        "s, o",
    )
    .await?;

    let mut by_label: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for b in &batches {
        if b.num_columns() < 2 {
            continue;
        }
        let (Some(s), Some(o)) = (str_col(b, 0), str_col(b, 1)) else {
            continue;
        };
        for i in 0..b.num_rows() {
            if s.is_null(i) || o.is_null(i) {
                continue;
            }
            let label = s.value(i);
            // The per-tenant scope label is on every node — pure noise here.
            if !scope.is_empty() && label == scope {
                continue;
            }
            by_label
                .entry(label.to_string())
                .or_default()
                .insert(o.value(i).to_string());
        }
    }

    Some(
        by_label
            .into_iter()
            .map(|(label, props)| LabelProperties {
                label,
                properties: props.into_iter().collect(),
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_neo4j_from_matches_only_the_neo4j_connector() {
        assert!(is_neo4j_from("neo4j:Chunk"));
        assert!(is_neo4j_from("  NEO4J:Entity"));
        assert!(!is_neo4j_from("milvus:pixer_lite"));
        assert!(!is_neo4j_from("postgres:public.foo"));
    }

    #[test]
    fn parse_label_set_drops_the_tenant_label() {
        assert_eq!(
            parse_label_set(r#"["Entity","patient","asst_42"]"#, "asst_42"),
            vec!["Entity".to_string(), "patient".to_string()]
        );
        // No scope configured -> nothing stripped.
        assert_eq!(
            parse_label_set(r#"["Chunk"]"#, ""),
            vec!["Chunk".to_string()]
        );
        // Malformed input is tolerated (best-effort discovery).
        assert!(parse_label_set("not json", "asst_42").is_empty());
    }

    #[test]
    fn strip_scope_is_a_noop_without_a_scope() {
        let mut labels = vec!["Entity".to_string(), "asst_42".to_string()];
        strip_scope(&mut labels, "");
        assert_eq!(labels.len(), 2);
        strip_scope(&mut labels, "asst_42");
        assert_eq!(labels, vec!["Entity".to_string()]);
    }
}
