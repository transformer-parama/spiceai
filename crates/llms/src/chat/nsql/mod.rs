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

use async_openai::{
    error::OpenAIError,
    types::chat::{CreateChatCompletionRequest, CreateChatCompletionResponse},
};

use std::fmt::Write;

pub mod default;
pub(crate) mod json;
pub(crate) mod structured_output;

#[derive(Default)]
pub struct QueryGenerationContext {
    pub failed_attempts: Vec<FailedAttempt>,
    /// Tables that have a vector/semantic search index. When non-empty, the
    /// prompt tells the model it may use `vector_search(<table>, '<text>', <k>)`
    /// for semantic-similarity questions instead of `LIKE`.
    pub semantic_search_tables: Vec<String>,
    /// Name of a registered reranker model, if any. When set alongside
    /// `semantic_search_tables`, the prompt tells the model it may wrap
    /// `vector_search` in `rerank(...)` for higher-precision passage ranking.
    pub reranker: Option<String>,
    /// Datasets backed by the Neo4j connector. When non-empty, the prompt tells
    /// the model it may retrieve graph rows with the `graph_query('<dataset>',
    /// '<cypher>')` table function (mirrors `semantic_search_tables`, one
    /// connector over). The engine auto-populates this from the registered
    /// datasets whose source is `neo4j:`.
    pub graph_datasets: Vec<String>,
    /// Optional deployment-provided description of the graph's shape (scope
    /// label, anchor/label/relationship conventions, fulltext index name).
    /// The engine itself is ontology-agnostic, so this domain knowledge is
    /// injected by the operator (env `SPICE_NSQL_GRAPH_HINT`) and appended
    /// verbatim to the knowledge-graph prompt block when `graph_datasets` is set.
    pub graph_hint: Option<String>,
    /// Datasets that are SEARCH-ONLY: a pure vector store (Milvus) that can serve
    /// `vector_search(...)` but cannot serve a plain scan — it has no rows to read
    /// without a query embedding. Such a table answers `SELECT * FROM t` / `COUNT(*)`
    /// with ZERO rows, which the model would otherwise report as a real answer ("0
    /// chunks"). When non-empty the prompt forbids scanning/counting them, so the
    /// only access path is `vector_search`. Auto-populated from datasets whose
    /// source is `milvus:`. NOTE: this is deliberately narrower than
    /// `semantic_search_tables` — a Postgres/Iceberg table that merely HAS an
    /// embedding column is still fully scannable and must NOT be listed here.
    pub search_only_tables: Vec<String>,
}

pub struct FailedAttempt {
    pub attempted_query: String,
    pub error_message: String,
}

impl FailedAttempt {
    #[must_use]
    pub fn new(attempted_query: String, error_message: String) -> Self {
        Self {
            attempted_query,
            error_message,
        }
    }
}

/// Additional methods (beyond [`super::Chat`]), whereby a model can provide improved results for SQL code generation.
pub trait SqlGeneration: Sync + Send {
    fn create_request_for_query(
        &self,
        model_id: &str,
        query: &str,
        context: &QueryGenerationContext,
    ) -> Result<CreateChatCompletionRequest, OpenAIError>;

    fn parse_response(
        &self,
        resp: CreateChatCompletionResponse,
    ) -> Result<Option<String>, OpenAIError>;
}

/// Default system prompt for SQL code generation.
#[must_use]
pub fn create_prompt(query: &str, ctx: &QueryGenerationContext) -> String {
    let mut prompt = format!(
        r#"Task: Write a SQL query to answer this question: _\"{query}\"_. Instruction: Return only valid SQL code, nothing additional, don't wrap it in ```. Columns with capitals must be quoted. Write each table name exactly as shown and UNQUOTED, including any schema/catalog prefix, e.g. spice.public.my_table. NEVER wrap a qualified (dotted) table name in a single pair of quotes: "spice.public.my_table" is WRONG and will not be found (it is read as one literal name). Quote each part separately ("catalog"."schema"."table") — NOT the whole dotted string — whenever a name part has capitals, special characters, OR STARTS WITH A DIGIT: e.g. a catalog table lake.94deff...hex.my_table MUST be written lake."94deff...hex"."my_table" (an unquoted part that starts with a digit is a PARSE ERROR: `found: .94`). Use ONLY the tables and columns shown in the schema and sample messages provided above; never invent, guess, abbreviate, or rename a table or column. If a table name will not parse or is reported not-found, FIX THE QUOTING and keep the SAME table — NEVER substitute a different table (e.g. do not answer a question about one table by querying another just because it parses); a wrong-table answer is worse than an error. If the question mentions something not present in the schema, map it to the closest existing table/column instead of inventing a new name. Booleans may be stored as integers 0/1: look at the sample values for the column and compare accordingly (WHERE flag = 1), not WHERE flag = true, unless the samples actually show true/false. When the question asks for several INDEPENDENT values that come from different tables (for example an average from one table and a count from another), compute EACH value as its own SCALAR SUBQUERY in the SELECT list, e.g. SELECT (SELECT avg(x) FROM spice.public.t1) AS a, (SELECT count(*) FROM spice.public.t2) AS b. NEVER combine unrelated tables with CROSS JOIN, comma-joins, or a JOIN without a real key relationship: that builds a cartesian product of every row times every row and the query will never finish. A scalar subquery used as a SELECT-list value must return exactly ONE row and ONE column: never wrap a multi-row or multi-column subquery (one that returns several rows or columns, e.g. a top-k result) inside a scalar subquery, and never pack multiple rows into a single value with row_to_json, array_agg, or json_agg (those functions do not exist here)."#
    );

    if !ctx.failed_attempts.is_empty() {
        let failed_atttempts_str = format!(
            "\nUse incorrectly written SQL queries and associated errors below to ensure that the new query avoids repeating the same mistakes or generating identical queries:\n\n{}",
            failed_attempts_formatted(&ctx.failed_attempts)
        );
        prompt.push_str(&failed_atttempts_str);
    }

    if !ctx.semantic_search_tables.is_empty() {
        let tables = ctx.semantic_search_tables.join(", ");
        let example = ctx
            .semantic_search_tables
            .first()
            .map_or("my_table", String::as_str);
        let _ = write!(
            prompt,
            "\n\nSemantic search: these tables support vector similarity search over text: {tables}. The ONLY way to do ANY semantic / similarity / relevance / \"about\" / \"related to\" / passage-retrieval search is the `vector_search` table function — NO other similarity, ranking, full-text, or vector function, operator, or type exists in this engine. Do NOT use LIKE. Do NOT invent or call ANY of the following (NONE exist here and the query WILL fail): vector_search look-alikes (pg_vector_search, semantic_search, similarity_search, similarity_to_query, match), similarity/ranking functions (SIMILARITY, similarity, cosine_similarity, l2_distance, bm25_rank, ts_rank, ts_rank_cd), full-text functions (to_tsvector, to_tsquery, plainto_tsquery, websearch_to_tsquery), the pgvector distance operators (`<->`, `<=>`, `<#>`), or a `vector` type / `::vector` cast / vector literal. For ANY semantically-similar / about / relevant-to / top-passages question, use ONLY the table function `vector_search`. CRITICAL RULES: (1) its FIRST argument MUST be EXACTLY ONE of these vector-search tables — {tables} — written as a BARE, UNQUOTED identifier, never a quoted string. NEVER pass any other table (a graph/Neo4j table or a relational table) as the first argument even if it also has a `content`/`text`/`body` column — only the tables listed here have an embedding index, and any other table fails with \"does not have an embedding index\". (2) its SECOND argument is a PLAIN string literal search phrase — never a subquery, column reference, or expression. (3) Use LIMIT for top-k, never TOP. Example: SELECT _score, text FROM vector_search({example}, 'the search phrase', 5) ORDER BY _score DESC LIMIT 5. It returns a `_score` column (higher = more relevant) plus the table's columns. When a question asks for the top-k passages AND single-value facts from other tables, put vector_search in the FROM clause as the MAIN rows and add each other fact as its own scalar-subquery column, e.g. SELECT _score, text, (SELECT avg(age) FROM spice.public.other_table) AS avg_age FROM vector_search({example}, 'the search phrase', 5) ORDER BY _score DESC LIMIT 5. Do NOT place vector_search inside a scalar subquery."
        );

        if let Some(reranker) = &ctx.reranker {
            let _ = write!(
                prompt,
                "\n\nReranking for higher precision: a reranker named `{reranker}` is registered. For \"top-k\", \"most relevant\", \"best\", or passage-retrieval questions, prefer WRAPPING the vector_search call in the `rerank` table function: a cross-encoder reorders the hits and gives noticeably better ordering than raw vector scores. Retrieve MORE candidates in the inner vector_search (about 20-50) and keep the final k with the outer `limit =>`. Example: SELECT text, rerank_score FROM rerank(vector_search({example}, 'the search phrase', 30), document => text, model => '{reranker}', limit => 5). RULES: (1) the FIRST argument is the vector_search(...) call itself, with its bare, unquoted table name; (2) `document =>` must be the table's main free-text column shown in the schema (the passage/body/text column), written as a bare identifier and NOT a quoted string; (3) write `model => '{reranker}'` exactly; (4) `limit =>` is how many final rows to keep; (5) the output is the table's columns (minus the raw score) plus a `rerank_score` column, ALREADY ordered best-first, so add NO ORDER BY; (6) never place rerank inside a scalar subquery. Plain vector_search without rerank is still acceptable when reranking is unnecessary."
            );
        }
    }

    // Search-only (Milvus) tables. These are vector indexes, not scannable tables:
    // a plain scan returns zero rows rather than failing, so without this the model
    // happily emits `SELECT count(*) FROM chunks` and reports the resulting 0 as a
    // real answer. Naming them explicitly keeps the ONLY access path vector_search.
    if !ctx.search_only_tables.is_empty() {
        let tables = ctx.search_only_tables.join(", ");
        let first = ctx
            .search_only_tables
            .first()
            .map_or("my_table", String::as_str);
        let _ = write!(
            prompt,
            "\n\nSearch-only tables: {tables} are vector indexes, NOT scannable tables. They hold no rows you can read without a search phrase, and a direct scan silently returns ZERO rows — so a count from one is meaningless and MUST NOT be reported as an answer. NEVER write `SELECT ... FROM {first}` directly, and never COUNT(*), SUM, AVG, GROUP BY, or otherwise aggregate/scan over them, or JOIN to them as a plain table. The ONLY valid way to read these tables is the `vector_search` table function (optionally wrapped in `rerank`), which requires a search phrase: SELECT text FROM vector_search({first}, 'the search phrase', 5). If the question asks how many rows/records one of these tables has, or asks for a total/aggregate over it, that CANNOT be answered from this table — answer using another table that has the facts, or return the relevant passages via vector_search instead of a count."
        );
    }

    // Knowledge-graph (Neo4j) injection — mirrors the vector_search block, one
    // connector over. Names the graph datasets and teaches the `graph_query`
    // table function so the model can traverse the graph in the SAME federated
    // SQL as vector_search. The ontology (labels, anchors, scope) is deployment-
    // specific, so it is supplied verbatim via `graph_hint` rather than baked in.
    if !ctx.graph_datasets.is_empty() {
        let datasets = ctx.graph_datasets.join(", ");
        let first = ctx
            .graph_datasets
            .first()
            .map_or("graph", String::as_str);
        let _ = write!(
            prompt,
            "\n\nKnowledge graph: these datasets are Neo4j property graphs, queryable ONLY through the `graph_query` table function: {datasets}. Use it to retrieve entity-anchored facts and the text of related nodes. RULES: (1) call it in the FROM clause as graph_query('<dataset>', '<cypher>') where the FIRST argument is the dataset name written EXACTLY as shown, as a quoted string; (2) the SECOND argument is ONE read-only Cypher statement — only MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT / CALL db.index.* are allowed; NEVER CREATE, MERGE, DELETE, SET, REMOVE, or DETACH; (3) EVERY expression in the Cypher RETURN MUST be aliased with `AS` (RETURN e.name AS name, count(c) AS mentions) — an unaliased RETURN term (e.g. RETURN count(*)) fails with \"Expression in CALL {{ RETURN ... }} must be aliased\"; the outer SQL then selects those ALIASES directly (SELECT name, mentions FROM graph_query(...)), and NEVER writes node.property like e.name at the SQL level (that fails \"No field named e.name\"); (4) to count graph rows, compute the count INSIDE the Cypher (RETURN count(c) AS n) and select n — do NOT wrap graph_query in an outer SELECT count(*) (a graph_query that already returns one aggregate row would just yield 1); anything used in Cypher ORDER BY that is an aggregate (ORDER BY count(c)) must ALSO appear in that RETURN; (5) graph_query only runs Cypher against a graph — it CANNOT read a relational/catalog table; for a plain table (columns like age, count, id) use ordinary SQL on that table, never graph_query and never property-graph syntax (no LATERAL {{...}}, no n.prop::int) on it; (6) to answer a question from BOTH the graph and the passages, UNION ALL a graph_query(...) with a rerank(vector_search(...)) and give each side a literal `source` column; (7) write the function name BARE — NEVER prefix it with a catalog/schema: `graph_query(...)` is correct, `spice.public.graph_query(...)` is WRONG and fails with \"table function 'spice' not found\". The schema/catalog-prefix instruction above applies to TABLE names only, never to a table FUNCTION. Minimal example: SELECT name, mentions FROM graph_query('{first}', 'MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN e.name AS name, count(c) AS mentions ORDER BY mentions DESC LIMIT 10')."
        );
        if let Some(hint) = &ctx.graph_hint {
            let _ = write!(prompt, " GRAPH SHAPE (use these exact conventions): {hint}");
        }
    }

    prompt
}

fn failed_attempts_formatted(attempts: &Vec<FailedAttempt>) -> String {
    let mut previous_attempts = String::new();
    for attempt in attempts {
        let _ = write!(
            previous_attempts,
            "sql: `{}`\nerror: `{}`\n\n",
            attempt.attempted_query, attempt.error_message
        );
    }
    previous_attempts
}

#[cfg(test)]
mod tests {

    use default::DefaultSqlGeneration;

    use super::*;
    static MODEL_ID: &str = "model_id";

    #[test]
    fn test_default_create_request_for_query() {
        let req = DefaultSqlGeneration {}
            .create_request_for_query(
                MODEL_ID,
                "SELECT * FROM table",
                &QueryGenerationContext::default(),
            )
            .expect("failed to create request");
        let req_str = serde_json::to_string_pretty(&req).expect("failed to serialize");

        insta::assert_snapshot!("sql_gen_default", req_str);
    }

    #[test]
    fn test_json_create_request_for_query() {
        let req = json::JsonSchemaSqlGeneration {}
            .create_request_for_query(
                MODEL_ID,
                "SELECT * FROM table",
                &QueryGenerationContext::default(),
            )
            .expect("failed to create request");
        let req_str = serde_json::to_string_pretty(&req).expect("failed to serialize");

        insta::assert_snapshot!("sql_gen_json", req_str);
    }

    #[test]
    fn test_structured_output_create_request_for_query() {
        let req = structured_output::StructuredOutputSqlGeneration {}
            .create_request_for_query(
                MODEL_ID,
                "SELECT * FROM table",
                &QueryGenerationContext::default(),
            )
            .expect("failed to create request");
        let req_str = serde_json::to_string_pretty(&req).expect("failed to serialize");

        insta::assert_snapshot!("sql_gen_structured", req_str);
    }

    // --- vector_search (semantic search) prompt injection --------------------
    // The engine populates `semantic_search_tables` from the datasets that carry
    // a vector index, so `/v1/nsql` instructs the model to emit `vector_search`
    // instead of LIKE / invented functions. These lock that contract in.

    #[test]
    fn prompt_omits_vector_search_without_semantic_tables() {
        let prompt = create_prompt("any question", &QueryGenerationContext::default());
        assert!(!prompt.contains("vector_search"));
        assert!(!prompt.contains("Semantic search"));
    }

    #[test]
    fn prompt_injects_vector_search_with_bare_identifier_guidance() {
        let ctx = QueryGenerationContext {
            semantic_search_tables: vec!["ihi_search".to_string()],
            ..Default::default()
        };
        let prompt = create_prompt("text about ED overcrowding", &ctx);

        // It names the searchable table and steers the model to vector_search.
        assert!(prompt.contains("Semantic search"));
        assert!(prompt.contains("ihi_search"));
        assert!(prompt.contains("vector_search"));
        // The two failure modes seen with smaller models are called out:
        // (1) the first arg must be a BARE identifier, not a quoted string;
        assert!(prompt.contains("BARE, UNQUOTED identifier"));
        // (2) use LIMIT for top-k, never TOP.
        assert!(prompt.contains("never TOP"));
        // And the result contract (a `_score` column) is documented.
        assert!(prompt.contains("_score"));
        // The worked example uses the table as a bare identifier.
        assert!(prompt.contains("vector_search(ihi_search, 'the search phrase', 5)"));
        // ...and must NOT pass it as a string literal.
        assert!(!prompt.contains("vector_search('ihi_search'"));
    }

    #[test]
    fn prompt_lists_all_semantic_tables_and_examples_with_the_first() {
        let ctx = QueryGenerationContext {
            semantic_search_tables: vec!["docs_hybrid".to_string(), "other_search".to_string()],
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(prompt.contains("docs_hybrid, other_search"));
        // the example uses the FIRST table as the bare identifier
        assert!(prompt.contains("vector_search(docs_hybrid,"));
    }

    #[test]
    fn prompt_combines_failed_attempts_and_semantic_tables() {
        let ctx = QueryGenerationContext {
            failed_attempts: vec![FailedAttempt::new(
                "SELECT TOP 5 * FROM t".to_string(),
                "syntax error near TOP".to_string(),
            )],
            semantic_search_tables: vec!["ihi_search".to_string()],
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(prompt.contains("syntax error near TOP")); // failed-attempt feedback
        assert!(prompt.contains("vector_search")); // and the semantic instruction
    }

    // --- rerank prompt injection ---------------------------------------------
    // When a reranker is registered AND there are semantic tables, `/v1/nsql`
    // steers the model to wrap vector_search in rerank(...) for better ranking.

    #[test]
    fn prompt_injects_rerank_when_reranker_and_semantic_tables_present() {
        let ctx = QueryGenerationContext {
            semantic_search_tables: vec!["ihi_search".to_string()],
            reranker: Some("reranker".to_string()),
            ..Default::default()
        };
        let prompt = create_prompt("best passages about ED crowding", &ctx);
        assert!(prompt.contains("Reranking for higher precision"));
        // the worked example wraps vector_search in rerank with the reranker name
        assert!(prompt.contains(
            "rerank(vector_search(ihi_search, 'the search phrase', 30), document => text, model => 'reranker', limit => 5)"
        ));
        // the rerank output contract is documented
        assert!(prompt.contains("rerank_score"));
    }

    #[test]
    fn prompt_omits_rerank_when_no_reranker() {
        // semantic tables but NO reranker -> vector_search yes, rerank no.
        let ctx = QueryGenerationContext {
            semantic_search_tables: vec!["ihi_search".to_string()],
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(prompt.contains("vector_search"));
        assert!(!prompt.contains("Reranking for higher precision"));
        assert!(!prompt.contains("rerank("));
    }

    #[test]
    fn prompt_omits_rerank_without_semantic_tables_even_if_reranker_set() {
        // a reranker with no semantic tables -> no vector_search, no rerank.
        let ctx = QueryGenerationContext {
            reranker: Some("reranker".to_string()),
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(!prompt.contains("Reranking for higher precision"));
        assert!(!prompt.contains("rerank("));
    }

    // --- search-only (Milvus) prompt injection -------------------------------
    #[test]
    fn prompt_omits_search_only_block_without_search_only_tables() {
        let prompt = create_prompt("q", &QueryGenerationContext::default());
        assert!(!prompt.contains("Search-only tables"));
    }

    #[test]
    fn prompt_forbids_scanning_search_only_tables() {
        let ctx = QueryGenerationContext {
            search_only_tables: vec!["chunks".to_string()],
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(prompt.contains("Search-only tables"));
        // names the table and steers to vector_search as the only access path
        assert!(prompt.contains("chunks"));
        assert!(prompt.contains("vector_search(chunks,"));
        // the actual failure this prevents: a silent 0 reported as a real count
        assert!(prompt.contains("never COUNT(*)"));
    }

    // --- graph_query (knowledge graph) prompt injection ----------------------
    // The engine populates `graph_datasets` from datasets whose source is
    // `neo4j:`, so `/v1/nsql` instructs the model to emit `graph_query(...)`.
    #[test]
    fn prompt_omits_graph_query_without_graph_datasets() {
        let prompt = create_prompt("q", &QueryGenerationContext::default());
        assert!(!prompt.contains("graph_query"));
        assert!(!prompt.contains("Knowledge graph"));
    }

    #[test]
    fn prompt_injects_graph_query_with_dataset_name_and_readonly_guidance() {
        let ctx = QueryGenerationContext {
            graph_datasets: vec!["graph".to_string()],
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(prompt.contains("Knowledge graph"));
        assert!(prompt.contains("graph_query"));
        // names the dataset and uses it in the worked example
        assert!(prompt.contains("graph_query('graph',"));
        // steers the model away from writes
        assert!(prompt.contains("NEVER CREATE"));
        // the base prompt tells the model to schema-qualify TABLE names; make sure
        // it is told not to apply that to the table FUNCTION (observed live as
        // `spice.public.graph_query(...)` -> "table function 'spice' not found").
        assert!(prompt.contains("spice.public.graph_query(...)` is WRONG"));
    }

    #[test]
    fn prompt_appends_graph_hint_when_present() {
        let ctx = QueryGenerationContext {
            graph_datasets: vec!["graph".to_string()],
            graph_hint: Some("nodes carry :asst_x; anchors are (:Entity {canonical_name})".to_string()),
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(prompt.contains("GRAPH SHAPE"));
        assert!(prompt.contains(":Entity {canonical_name}"));
    }

    #[test]
    fn prompt_omits_graph_hint_without_graph_datasets() {
        // a hint with no graph datasets -> no graph block at all.
        let ctx = QueryGenerationContext {
            graph_hint: Some("some shape".to_string()),
            ..Default::default()
        };
        let prompt = create_prompt("q", &ctx);
        assert!(!prompt.contains("Knowledge graph"));
        assert!(!prompt.contains("GRAPH SHAPE"));
    }

    #[test]
    fn test_default_create_request_for_query_with_failed() {
        let mut ctx = QueryGenerationContext::default();
        ctx.failed_attempts.push(FailedAttempt {
            attempted_query: r#"SELECT * FROM "spice.public.table" LIMIT 1"#.to_string(),
            error_message:
                "Error during planning: table 'spice.public.spice.public.table' not found"
                    .to_string(),
        });

        let req = DefaultSqlGeneration {}
            .create_request_for_query(MODEL_ID, "SELECT * FROM table", &ctx)
            .expect("failed to create request");
        let req_str = serde_json::to_string_pretty(&req).expect("failed to serialize");

        insta::assert_snapshot!("sql_gen_default_with_failed_attempt", req_str);
    }
}
