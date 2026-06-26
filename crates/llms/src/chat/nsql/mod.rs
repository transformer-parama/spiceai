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
        r#"Task: Write a SQL query to answer this question: _\"{query}\"_. Instruction: Return only valid SQL code, nothing additional, don't wrap it in ```. Columns with capitals must be quoted. For tables with schemas and catalogs '"catalog"."schema"."table"' not '"catalog.schema.table"'."#
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
            "\n\nSemantic search: these tables support vector similarity search over text: {tables}. For a question asking for text semantically similar to / about / relevant to a topic, do NOT use LIKE and do NOT invent functions (no SIMILARITY()); use the table function `vector_search`. CRITICAL RULES: (1) its FIRST argument is the table name written as a BARE, UNQUOTED identifier — never a quoted string. (2) Use LIMIT for top-k, never TOP. Example: SELECT _score, text FROM vector_search({example}, 'the search phrase', 5) ORDER BY _score DESC LIMIT 5. It returns a `_score` column (higher = more relevant) plus the table's columns."
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
