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
        r#"Task: Write a SQL query to answer this question: _\"{query}\"_. Instruction: Return only valid SQL code, nothing additional, don't wrap it in ```. Columns with capitals must be quoted. Write each table name exactly as shown and UNQUOTED, including any schema/catalog prefix, e.g. spice.public.my_table. NEVER wrap a qualified (dotted) table name in a single pair of quotes: "spice.public.my_table" is WRONG and will not be found (it is read as one literal name). Only if a name part has capitals or special characters, quote each part separately ("spice"."public"."My_Table"), never the whole dotted string. Use ONLY the tables and columns shown in the schema and sample messages provided above; never invent, guess, abbreviate, or rename a table or column. If the question mentions something not present in the schema, map it to the closest existing table/column instead of inventing a new name. When the question asks for several INDEPENDENT values that come from different tables (for example an average from one table and a count from another), compute EACH value as its own SCALAR SUBQUERY in the SELECT list, e.g. SELECT (SELECT avg(x) FROM spice.public.t1) AS a, (SELECT count(*) FROM spice.public.t2) AS b. NEVER combine unrelated tables with CROSS JOIN, comma-joins, or a JOIN without a real key relationship: that builds a cartesian product of every row times every row and the query will never finish. A scalar subquery used as a SELECT-list value must return exactly ONE row and ONE column: never wrap a multi-row or multi-column query (such as a vector_search top-k, which returns several rows and columns) inside a scalar subquery, and never pack multiple rows into a single value with row_to_json, array_agg, or json_agg (those functions do not exist here)."#
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
            "\n\nSemantic search: these tables support vector similarity search over text: {tables}. The ONLY way to do ANY semantic / similarity / relevance / \"about\" / \"related to\" / passage-retrieval search is the `vector_search` table function — NO other similarity, ranking, full-text, or vector function, operator, or type exists in this engine. Do NOT use LIKE. Do NOT invent or call ANY of the following (NONE exist here and the query WILL fail): vector_search look-alikes (pg_vector_search, semantic_search, similarity_search, similarity_to_query, match), similarity/ranking functions (SIMILARITY, similarity, cosine_similarity, l2_distance, bm25_rank, ts_rank, ts_rank_cd), full-text functions (to_tsvector, to_tsquery, plainto_tsquery, websearch_to_tsquery), the pgvector distance operators (`<->`, `<=>`, `<#>`), or a `vector` type / `::vector` cast / vector literal. For ANY semantically-similar / about / relevant-to / top-passages question, use ONLY the table function `vector_search`. CRITICAL RULES: (1) its FIRST argument is the table name written as a BARE, UNQUOTED identifier — never a quoted string. (2) Use LIMIT for top-k, never TOP. Example: SELECT _score, text FROM vector_search({example}, 'the search phrase', 5) ORDER BY _score DESC LIMIT 5. It returns a `_score` column (higher = more relevant) plus the table's columns. When a question asks for the top-k passages AND single-value facts from other tables, put vector_search in the FROM clause as the MAIN rows and add each other fact as its own scalar-subquery column, e.g. SELECT _score, text, (SELECT avg(age) FROM spice.public.other_table) AS avg_age FROM vector_search({example}, 'the search phrase', 5) ORDER BY _score DESC LIMIT 5. Do NOT place vector_search inside a scalar subquery."
        );
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
        };
        let prompt = create_prompt("q", &ctx);
        assert!(prompt.contains("syntax error near TOP")); // failed-attempt feedback
        assert!(prompt.contains("vector_search")); // and the semantic instruction
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
