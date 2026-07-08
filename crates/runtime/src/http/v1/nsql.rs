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
use crate::{
    Runtime,
    datafusion::{
        SPICE_DEFAULT_CATALOG, SPICE_DEFAULT_SCHEMA,
        request_context_extension::get_current_datafusion,
    },
    http::v1::{ResponseMetadata, ResponseMimeType, to_http_response},
    model::LLMChatCompletionsModelStore,
    tools::{
        builtin::{
            sample::{
                SampleTableMethod, SampleTableParams, distinct::DistinctColumnsParams,
                random::RandomSampleParams, tool::SampleDataTool,
            },
            table_schema::{TableSchemaTool, TableSchemaToolParams},
        },
        utils::create_tool_use_messages,
    },
};
use async_openai::types::chat::ChatCompletionRequestMessage;
use axum::{
    Extension, Json,
    http::StatusCode,
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
};
use axum_extra::TypedHeader;
use datafusion::sql::TableReference;
use futures::{StreamExt, TryStreamExt};
use headers_accept::Accept;
use http::HeaderMap;
use runtime_datafusion::allowlist::ResolvedTableAwareAllowlist;
use runtime_request_context::{AsyncMarker, RequestContext};

use arrow::array::RecordBatch;
use itertools::Itertools;
use llms::chat::nsql::{FailedAttempt, QueryGenerationContext, default::DefaultSqlGeneration};
use serde::{Deserialize, Serialize};
use spicepod::component::model::ModelType;
use std::{
    sync::{Arc, LazyLock},
    time::Duration,
};
use tokio::sync::{RwLock, Semaphore};
use tracing::Span;
use tracing_futures::Instrument;

use super::accept_header_types;
use crate::datafusion::query::QueryBuilder;

// Default number of retries for NSQL queries if the generated query fails to execute.
// Each retry re-invokes the model AND re-executes the (possibly heavy, federated)
// query, so a high cap turns a single slow/mis-generated request into a model +
// compute storm under sustained load. Override with `SPICE_NSQL_MAX_RETRIES`.
const DEFAULT_NSQL_RETRIES: u8 = 3;

// Default wall-clock budget for an entire NSQL request (all retries combined). When
// exceeded, the request's cancellation token is tripped so the in-flight model call
// AND query execution stop, rather than running until a far-off client/proxy timeout
// while pinning runtime worker threads. Override with `SPICE_NSQL_BUDGET_SECS`.
const DEFAULT_NSQL_BUDGET_SECS: u64 = 45;

// Maximum number of concurrent sampling tools executions for NSQL
const DATA_SAMPLING_MAX_CONCURRENT: usize = 10;

// NSQL streaming keep alive interval in seconds
const NSQL_STREAM_KEEP_ALIVE: u64 = 30;

/// Max retries before NSQL gives up, from `SPICE_NSQL_MAX_RETRIES` or the default.
fn nsql_max_retries() -> u8 {
    std::env::var("SPICE_NSQL_MAX_RETRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NSQL_RETRIES)
}

/// Per-request wall-clock budget, from `SPICE_NSQL_BUDGET_SECS` or the default.
fn nsql_budget() -> Duration {
    let secs = std::env::var("SPICE_NSQL_BUDGET_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_NSQL_BUDGET_SECS);
    Duration::from_secs(secs)
}

/// Admission control for the NSQL path. Each NSQL request is heavy server-side
/// (schema + data sampling across every dataset, model generation, then federated
/// execution — times the retry count). Without a bound, a burst of NL requests —
/// including ones whose clients have already timed out but whose server-side work
/// keeps running — saturates the runtime worker threads and stalls *all* query
/// execution (even trivial `SELECT 1`). This semaphore caps how many NSQL requests
/// execute concurrently so overload degrades gracefully (requests queue) instead of
/// wedging the engine. Override with `SPICE_NSQL_MAX_CONCURRENT`.
static NSQL_CONCURRENCY: LazyLock<Semaphore> = LazyLock::new(|| {
    let limit = std::env::var("SPICE_NSQL_MAX_CONCURRENT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            // Each NSQL request can consume several cores (data sampling fanned out
            // across every dataset + federated/semantic execution), so allow at most
            // ~half the cores to run NSQL concurrently. This keeps headroom for the
            // data plane and stops a small box from saturating under NL load.
            let cores = std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(4);
            std::cmp::max(2, cores / 2)
        });
    Semaphore::new(limit)
});

/// Aborts a spawned task when dropped, so the per-request budget timer never
/// outlives the request it guards.
struct AbortOnDrop(tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Status + message for an NSQL request whose cancellation token fired: a
/// `504` when the per-request budget elapsed, otherwise a `499` (client/admin
/// cancel). Distinguished by whether the deadline has passed.
fn nsql_cancel_outcome(deadline: tokio::time::Instant, budget: Duration) -> (StatusCode, String) {
    if tokio::time::Instant::now() >= deadline {
        (
            StatusCode::GATEWAY_TIMEOUT,
            format!(
                "NSQL request exceeded its {}s time budget (SPICE_NSQL_BUDGET_SECS)",
                budget.as_secs()
            ),
        )
    } else {
        (
            StatusCode::from_u16(499).unwrap_or(StatusCode::REQUEST_TIMEOUT),
            "NSQL request cancelled".to_string(),
        )
    }
}

fn clean_model_based_sql(input: &str) -> String {
    let no_dashes = match input.strip_prefix("--") {
        Some(rest) => rest.to_string(),
        None => input.to_string(),
    };

    // Only take the first query, if there are multiple.
    let one_query = no_dashes.split(';').next().unwrap_or(&no_dashes);

    // Models using a JSON / structured-output response format sometimes leak the
    // JSON envelope into the SQL value: a stray leading `{`, a wrapping ```sql
    // fence, or — the case we actually hit — a trailing `}` on its own line.
    // None of these are ever valid at the start/end of a SQL statement, so strip
    // them; this fixes `Expected: end of statement, found: }` parse errors
    // without touching any `{`/`}` inside the statement (e.g. string literals).
    one_query
        .trim()
        .trim_start_matches("```sql")
        .trim_start_matches("```SQL")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
        .trim_start_matches('{')
        .trim_end_matches(|c: char| c == '}' || c.is_whitespace())
        .trim()
        .to_string()
}

/// Create subsequent Assistant and Tool messages simulating a model requesting to use the `sample_data` tool, then receiving the result for the following sampling methods:
///  - Distinct columns
///  - Random sample
///
/// Convert the [`SampleTableParams`] into how an LLM would ask to use it (via a [`ChatCompletionRequestAssistantMessage`]).
/// Convert the result of a [`SampleDataTool`] call how we would return it to the LLM, (via a [`ChatCompletionRequestToolMessage`]).
async fn sample_messages(
    sample_from: &[TableReference],
    rt: Arc<Runtime>,
    table_allowlist: Option<ResolvedTableAwareAllowlist>,
) -> Result<Vec<ChatCompletionRequestMessage>, Box<dyn std::error::Error + Send + Sync>> {
    let message_futures = sample_from.iter().flat_map(|dataset| {
        [
            SampleTableParams::DistinctColumns(DistinctColumnsParams {
                tbl: dataset.to_string(),
                limit: 3,
                cols: None,
            }),
            SampleTableParams::RandomSample(RandomSampleParams {
                tbl: dataset.to_string(),
                limit: 3,
            }),
        ]
        .into_iter()
        .map(|params| {
            let rt = Arc::clone(&rt);
            let allowlist = table_allowlist.clone();
            async move {
                let method = SampleTableMethod::from(&params);
                create_tool_use_messages(
                    &SampleDataTool::new(rt.datafusion(), method.clone())
                        .with_table_allowlist(allowlist),
                    format!("sample-{method:?}").as_str(),
                    &params,
                )
                .instrument(Span::current())
                .await
            }
        })
    });

    let tool_call_messages = futures::stream::iter(message_futures)
        .boxed()
        .buffer_unordered(DATA_SAMPLING_MAX_CONCURRENT)
        .try_collect::<Vec<_>>()
        .await?;

    Ok(tool_call_messages.into_iter().flatten().collect())
}

// Default TTL for the NSQL schema/sample message cache. The schema (`table_schema`)
// and data-sample (`DistinctColumns` + `RandomSample`) tool-use message blocks are a
// pure function of the resolved table set, but were previously rebuilt on EVERY
// request — re-running ~1 `SELECT DISTINCT` per column per table (~100 federated
// queries for a wide pod) against the live stores each time. That per-request fan-out
// is what saturates the backing stores (Neo4j, Iceberg catalog) under concurrent NL
// load. Override with `SPICE_NSQL_SAMPLE_CACHE_TTL_SECS`; `0` disables the cache.
const DEFAULT_NSQL_SAMPLE_CACHE_TTL_SECS: u64 = 3600;

/// TTL for the schema/sample message cache, from `SPICE_NSQL_SAMPLE_CACHE_TTL_SECS`
/// or the default. A value of `0` disables caching (every request re-samples).
fn nsql_sample_cache_ttl() -> Duration {
    let secs = std::env::var("SPICE_NSQL_SAMPLE_CACHE_TTL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NSQL_SAMPLE_CACHE_TTL_SECS);
    Duration::from_secs(secs)
}

/// Process-wide cache of the (schema | sample) tool-use message blocks NSQL feeds the
/// model, keyed by (kind, model, sorted table set). `moka`'s `try_get_with` gives
/// single-flight semantics, so a concurrent cold-start burst computes the fan-out
/// once *total* (not once per request). Entries expire after the configured TTL, so a
/// table's schema/sample values are refreshed periodically; structural changes
/// (datasets added/removed) change the key and self-invalidate.
static NSQL_SAMPLE_CACHE: LazyLock<
    moka::future::Cache<String, Arc<Vec<ChatCompletionRequestMessage>>>,
> = LazyLock::new(|| {
    moka::future::Cache::builder()
        .max_capacity(256)
        .time_to_live(nsql_sample_cache_ttl())
        .build()
});

/// Stable cache key for a (kind, model, table-set). Order-independent in the tables so
/// equivalent requests share an entry.
fn nsql_cache_key(kind: &str, model: &str, tables: &[TableReference]) -> String {
    let mut names: Vec<String> = tables.iter().map(ToString::to_string).collect();
    names.sort();
    names.dedup();
    format!("{kind}\u{0}{model}\u{0}{}", names.join("\u{1}"))
}

/// Build, or reuse from cache, the `table_schema` tool-use messages for `tables`.
async fn cached_schema_messages(
    rt: &Arc<Runtime>,
    model: &str,
    tables: &[TableReference],
    allowlist: Option<&ResolvedTableAwareAllowlist>,
    span: &Span,
) -> Result<Arc<Vec<ChatCompletionRequestMessage>>, String> {
    let ttl = nsql_sample_cache_ttl();
    // Owned captures so the init future is `'static` (required by moka's get-with).
    let rt = Arc::clone(rt);
    let allowlist = allowlist.cloned();
    let table_names: Vec<String> = tables.iter().map(ToString::to_string).collect();
    let span = span.clone();
    let init = async move {
        create_tool_use_messages(
            &TableSchemaTool::new(rt, None, None).with_table_allowlist(allowlist),
            "schemas-nsql",
            &TableSchemaToolParams::new(table_names),
        )
        .instrument(span)
        .await
        .map(Arc::new)
    };
    if ttl.is_zero() {
        return init.await.map_err(|e| e.to_string());
    }
    NSQL_SAMPLE_CACHE
        .try_get_with(nsql_cache_key("schema", model, tables), init)
        .await
        .map_err(|e| e.to_string())
}

/// Build, or reuse from cache, the data-sampling tool-use messages for `tables`.
async fn cached_sample_messages(
    rt: &Arc<Runtime>,
    model: &str,
    tables: &[TableReference],
    allowlist: Option<&ResolvedTableAwareAllowlist>,
    span: &Span,
) -> Result<Arc<Vec<ChatCompletionRequestMessage>>, String> {
    let ttl = nsql_sample_cache_ttl();
    let rt = Arc::clone(rt);
    let allowlist = allowlist.cloned();
    let tables_owned: Vec<TableReference> = tables.to_vec();
    let span = span.clone();
    let init = async move {
        sample_messages(&tables_owned, rt, allowlist)
            .instrument(span)
            .await
            .map(Arc::new)
    };
    if ttl.is_zero() {
        return init.await.map_err(|e| e.to_string());
    }
    NSQL_SAMPLE_CACHE
        .try_get_with(nsql_cache_key("sample", model, tables), init)
        .await
        .map_err(|e| e.to_string())
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub struct Request {
    /// The natural language query to be converted into SQL
    pub query: String,

    /// The name of the model to use for SQL generation. If omitted, Spice defaults to the only compatible LLM model configured in the Spicepod.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// If true, streams the response instead of waiting for completion
    #[serde(default)]
    pub stream: bool,

    /// Whether sample data is included in the context for SQL generation. Default: false
    #[serde(default = "default_sample_data_enabled")]
    pub sample_data_enabled: bool,

    /// Names of datasets to sample from when constructing model context; this is a sampling hint and does not restrict which tables queries can target. If omitted, all datasets are used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datasets: Option<Vec<String>>,

    /// Stable prompt-cache key forwarded to the configured NSQL model for provider-specific cache handling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

fn default_sample_data_enabled() -> bool {
    false
}

/// Checks if the request is asking to only generate SQL.
fn return_sql_only(accept: Option<&TypedHeader<Accept>>) -> bool {
    accept.is_some_and(|a| accept_header_types(a).contains(&"application/sql".to_string()))
}

/// Text-to-SQL (NSQL)
///
/// Generate and optionally execute a natural-language text-to-SQL (NSQL) query.
///
/// This endpoint generates a SQL query using a natural language query (NSQL) and optionally executes it.
/// The SQL query is generated by the specified model and executed if the `Accept` header is not set to `application/sql`.
/// When `stream` is true, the response is streamed as Server-Sent Events (SSE).
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/v1/nsql",
    operation_id = "post_nsql",
    tag = "SQL",
    params(
        ("Accept" = String, Header, description = "The format of the response, one of 'application/json' (default), 'application/vnd.spiceai.nsql.v1+json', 'application/sql', 'text/csv' or 'text/plain'. 'application/sql' will only return the SQL query generated by the model."),
    ),
    request_body(
        description = "Request body to generate an NSQL query",
        content((
            Request = "application/json",
            example = json!({
                "query": "Get the top 5 customers by total sales",
                "stream": false,
                "sample_data_enabled": false,
                "datasets": ["sales_data"],
                "prompt_cache_key": "sales-dashboard"
            })
        ))
    ),
    responses(
        (status = 200, description = "SQL query executed successfully", content((
            Vec<serde_json::Value> = "application/json",
            example = json!([
                {
                    "customer_id": "12345",
                    "total_sales": 150_000
                },
                {
                    "customer_id": "67890",
                    "total_sales": 125_000
                }
            ])
        ),
        (
            String = "application/sql",
            example = "
            SELECT customer_id, SUM(total_sales)
            FROM sales_data
            GROUP BY customer_id
            ORDER BY SUM(total_sales) DESC
            LIMIT 5
            "
        ),
        (
            serde_json::Value = "application/vnd.spiceai.nsql.v1+json",
            example = json!({
                "row_count": 2,
                "schema": {
                    "fields": [
                    {
                        "name": "customer_id",
                        "data_type": "String",
                        "nullable": false,
                        "dict_id": 0,
                        "dict_is_ordered": false
                    },
                    {
                        "name": "total_sales",
                        "data_type": "Int64",
                        "nullable": false,
                        "dict_id": 0,
                        "dict_is_ordered": false
                    }
                    ]
                },
                "data": [
                    {
                    "customer_id": "12345",
                    "total_sales": 150_000
                    },
                    {
                    "customer_id": "67890",
                    "total_sales": 125_000
                    }
                ],
                "sql": "SELECT customer_id, SUM(total_sales) AS total_sales\nFROM sales_data\nGROUP BY customer_id\nORDER BY total_sales DESC\nLIMIT 5"
            })
        ),
        (
            String = "text/event-stream",
            example = "data: {\"row_count\": 2, \"schema\": {...}, \"data\": [...], \"sql\": \"SELECT ...\"}\n\n"
        ))),
        (status = 400, description = "Invalid request parameters", content((
            String = "application/json", example = "Model nsql not found"
        ))),
        (status = 500, description = "Internal server error", content((
            String, example = "No query produced from NSQL model"
        )))
    )
))]
pub(crate) async fn post(
    Extension(rt): Extension<Arc<Runtime>>,
    Extension(llms): Extension<Arc<RwLock<LLMChatCompletionsModelStore>>>,
    accept: Option<TypedHeader<Accept>>,
    Json(payload): Json<Request>,
) -> Response {
    // track ai_inferences_with_spice_count metric
    let context = RequestContext::current(AsyncMarker::new().await);

    if payload.stream {
        let stream = futures::stream::once(handle_nsql_query(rt, context, llms, accept, payload))
            .map(|(status, _, body)| {
                if status.is_success() {
                    Ok(Event::default().data(body))
                } else {
                    Err(status.to_string())
                }
            });
        Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(NSQL_STREAM_KEEP_ALIVE))
                    .text("nsql still in progress"),
            )
            .into_response()
    } else {
        handle_nsql_query(rt, context, llms, accept, payload)
            .await
            .into_response()
    }
}

pub(crate) async fn handle_nsql_query(
    rt: Arc<Runtime>,
    context: Arc<RequestContext>,
    llms: Arc<RwLock<LLMChatCompletionsModelStore>>,
    accept: Option<TypedHeader<Accept>>,
    payload: Request,
) -> (StatusCode, HeaderMap, String) {
    let df = get_current_datafusion(&context);
    let headers = HeaderMap::new();

    // NSQL-scoped cancellation token (child of the request token). Used for
    // both the LLM race and as the per-query cancellation token passed to
    // `QueryBuilder`. This way `POST /v1/sql/{id}/cancel` against the
    // NSQL-issued query reliably cancels NSQL end-to-end (the inner query
    // registers this same token in the cancel registry).
    let nsql_token = context.child_cancellation_token();

    let Request {
        query,
        model: requested_model,
        sample_data_enabled,
        datasets,
        prompt_cache_key,
        ..
    } = payload;

    let model = match resolve_nsql_model_name(requested_model, &rt).await {
        Ok(model) => model,
        Err((status, message)) => return (status, headers, message),
    };

    let table_allowlist_opt = match table_allowlist(&model, &rt).await {
        Ok(ta) => ta,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, headers, e);
        }
    };

    // Validate that requested datasets are within the model's allowlist
    if let (Some(requested_datasets), Some(allowlist)) = (&datasets, &table_allowlist_opt) {
        for ds in requested_datasets {
            let table_ref = TableReference::parse_str(ds);
            if !allowlist.table_is_allowed(&table_ref) {
                return (
                    StatusCode::BAD_REQUEST,
                    headers,
                    format!("Dataset '{ds}' not found"),
                );
            }
        }
    }

    crate::model::add_tools_used(&context, 1);

    // Admission control: bound concurrent NSQL executions so a burst (or a pile of
    // client-abandoned-but-still-running requests) can't saturate the runtime worker
    // threads and stall all query execution. Held (RAII) for the whole request,
    // covering the heavy schema/sampling/model/execution work below.
    let _nsql_permit = NSQL_CONCURRENCY.acquire().await.ok();

    // Per-request wall-clock budget. A background timer trips the NSQL cancellation
    // token when the budget elapses; via the existing cooperative-cancellation
    // plumbing this stops the in-flight model call AND query execution, so a slow or
    // retrying request fails fast instead of pinning workers until a client timeout.
    let budget = nsql_budget();
    let deadline = tokio::time::Instant::now() + budget;
    let _nsql_budget_timer = {
        let token = nsql_token.clone();
        AbortOnDrop(tokio::spawn(async move {
            tokio::time::sleep_until(deadline).await;
            token.cancel();
        }))
    };

    let span = tracing::span!(target: "task_history", tracing::Level::INFO, "nsql", input = %query, model = %model, "labels");

    if let Some(traceparent) = context.trace_parent() {
        crate::http::traceparent::override_task_history_with_trace_parent(&span, traceparent);
    }

    // Default to all available tables if specific table(s) are not provided.
    let tables = datasets
        .map(|ds| ds.iter().map(TableReference::from).collect_vec())
        .unwrap_or(
            df.get_user_table_names()
                .into_iter()
                .filter(|t| {
                    table_allowlist_opt
                        .as_ref()
                        .is_none_or(|a| a.table_is_allowed(t))
                })
                .collect(),
        );

    // Assistant/tool messages for the `table_schema` tool over all/provided tables.
    // Cached by (model, table set) with a TTL so the schema fan-out isn't re-run on
    // every request; see NSQL_SAMPLE_CACHE.
    let schema_messages = match cached_schema_messages(
        &rt,
        &model,
        &tables,
        table_allowlist_opt.as_ref(),
        &span,
    )
    .await
    {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("Error getting schema messages: {e}");
            return (StatusCode::INTERNAL_SERVER_ERROR, headers, e);
        }
    };

    // Same for the (much heavier) data-sampling messages, only when requested. These
    // are the ~1-query-per-column DISTINCT + random-sample fan-out across every table.
    let sample_data_messages = if sample_data_enabled {
        match cached_sample_messages(&rt, &model, &tables, table_allowlist_opt.as_ref(), &span)
            .await
        {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("Error sampling datasets for NSQL messages: {e}");
                return (StatusCode::INTERNAL_SERVER_ERROR, headers, e);
            }
        }
    } else {
        Arc::new(Vec::new())
    };

    let nql_model = {
        let models = llms.read().await;
        let Some(nql_model) = models.get(&model) else {
            return (
                StatusCode::BAD_REQUEST,
                headers,
                format!("Model {model} not found"),
            );
        };
        Arc::clone(nql_model)
    };

    let default_sql_generation = DefaultSqlGeneration {};
    let sql_gen = nql_model.as_sql().unwrap_or(&default_sql_generation);
    // Tracks previously generated queries and associated errors to enable an efficient retry mechanism
    let mut sql_gen_ctx = QueryGenerationContext::default();
    // Tell the model which tables support vector_search (semantic search), so it
    // can answer semantic-similarity questions with vector_search(...) instead of
    // LIKE / invented functions.
    sql_gen_ctx.semantic_search_tables = crate::search::util::user_tables_that_can_search(&df)
        .await
        .map(|tbls| tbls.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    // If semantic tables exist AND a reranker is registered, tell the model it
    // may wrap vector_search in rerank(...) for higher-precision passage ranking.
    if !sql_gen_ctx.semantic_search_tables.is_empty() {
        let rerankers = rt.rerankers();
        let guard = rerankers.read().await;
        sql_gen_ctx.reranker = guard.keys().next().cloned();
    }
    // Tell the model which datasets are Neo4j graphs (queryable via
    // graph_query(...)), so it can traverse the graph in the same federated SQL
    // as vector_search. Auto-detected from datasets whose source is `neo4j:`.
    // The graph's ontology is deployment-specific, so an optional operator hint
    // (SPICE_NSQL_GRAPH_HINT) supplies the labels/anchors/scope conventions.
    if let Some(app) = rt.read_app().await {
        sql_gen_ctx.graph_datasets = app
            .datasets
            .iter()
            .filter(|d| d.from.starts_with("neo4j:"))
            .map(|d| d.name.clone())
            .collect();
    }
    if !sql_gen_ctx.graph_datasets.is_empty() {
        sql_gen_ctx.graph_hint = std::env::var("SPICE_NSQL_GRAPH_HINT")
            .ok()
            .filter(|s| !s.trim().is_empty());
    }
    let max_retries = nsql_max_retries();
    let mut num_retries = 0;

    loop {
        // Cooperative cancellation: bail out between LLM/query iterations if
        // the NSQL token was cancelled (request token cancel propagates to
        // this child, admin cancel via the inner query id cancels this token
        // directly, and the per-request budget timer cancels it on timeout).
        if nsql_token.is_cancelled() {
            let (code, msg) = nsql_cancel_outcome(deadline, budget);
            return (code, headers, msg);
        }

        let Ok(mut req) = sql_gen.create_request_for_query(&model, &query, &sql_gen_ctx) else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                headers,
                "Error preparing data for NQL model".to_string(),
            );
        };

        req.messages.extend(schema_messages.iter().cloned());
        req.messages.extend(sample_data_messages.iter().cloned());
        if let Some(prompt_cache_key) = &prompt_cache_key {
            req.prompt_cache_key = Some(prompt_cache_key.clone());
        }

        // Race the LLM call against the NSQL cancellation token so that a
        // long-running model inference does not pin the request after a
        // cancel/disconnect. Dropping the chat_request future tears down the
        // underlying client/network resources.
        let chat_fut = nql_model.chat_request(req).instrument(span.clone());
        let resp = tokio::select! {
            biased;
            () = nsql_token.cancelled() => {
                let (code, msg) = nsql_cancel_outcome(deadline, budget);
                return (code, headers, msg);
            }
            res = chat_fut => match res {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Error running NQL model: {e}");
                    return (StatusCode::INTERNAL_SERVER_ERROR, headers, e.to_string());
                }
            }
        };

        // Run the SQL from the NSQL model through datafusion.
        match sql_gen.parse_response(resp) {
            Ok(Some(model_sql_query)) => {
                let cleaned_query = clean_model_based_sql(&model_sql_query);

                if return_sql_only(accept.as_ref()) {
                    tracing::trace!("Not running query, requested SQL only:\n{cleaned_query}");
                    return (StatusCode::OK, headers, cleaned_query);
                }

                tracing::debug!("Running query:\n{cleaned_query}");

                // Run the SQL with table allowlist enforcement. LLM-generated SQL is
                // always executed in read-only mode: the runtime rejects any plan that
                // contains DDL, DML, COPY, or a `LogicalPlan::Statement` node (including
                // PREPARE/EXECUTE/DEALLOCATE) regardless of per-catalog writability,
                // which mitigates model-mediated SQL injection on `/v1/nsql`.
                let query_result = {
                    let mut builder = QueryBuilder::new(&cleaned_query, Arc::clone(&df))
                        .read_only(true)
                        .cancellation_token(nsql_token.clone());
                    if let Some(ref allowlist) = table_allowlist_opt {
                        builder = builder.allow_tables(allowlist.clone());
                    }
                    builder.build().run().await
                };

                match query_result {
                    Ok(result) => match result.data.try_collect::<Vec<RecordBatch>>().await {
                        Ok(data) => {
                            return to_http_response(
                                data,
                                result.cache_status,
                                ResponseMimeType::from_accept_header(accept.as_ref()),
                                ResponseMetadata::empty().with_sql(&cleaned_query),
                            )
                            .instrument(span.clone())
                            .await;
                        }
                        Err(e) => {
                            if num_retries >= max_retries {
                                tracing::error!("Error collecting query results: {e}");
                                return (StatusCode::BAD_REQUEST, headers, e.to_string());
                            }

                            tracing::debug!("Error collecting query results: {e}. Retrying...");

                            num_retries += 1;
                            sql_gen_ctx
                                .failed_attempts
                                .push(FailedAttempt::new(cleaned_query.clone(), e.to_string()));
                        }
                    },
                    Err(e) => {
                        // If query failed, retry with the updated context

                        if num_retries >= max_retries {
                            tracing::error!("Error executing query: {e}");
                            return (StatusCode::BAD_REQUEST, headers, e.to_string());
                        }

                        tracing::debug!("Error executing query: {e}. Retrying...");

                        num_retries += 1;
                        sql_gen_ctx
                            .failed_attempts
                            .push(FailedAttempt::new(cleaned_query.clone(), e.to_string()));
                    }
                }
            }
            Ok(None) => {
                tracing::trace!("No query produced from NSQL model");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    headers,
                    "No query produced from NSQL model".to_string(),
                );
            }
            Err(e) => {
                tracing::error!("Error running NSQL model: {e}");
                return (StatusCode::INTERNAL_SERVER_ERROR, headers, e.to_string());
            }
        }
    }
}

async fn resolve_nsql_model_name(
    requested_model: Option<String>,
    rt: &Arc<Runtime>,
) -> Result<String, (StatusCode, String)> {
    if let Some(model) = requested_model {
        return Ok(model);
    }

    let Some(app) = rt.read_app().await else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Unexpected internal error. App not prepared in runtime.".to_string(),
        ));
    };

    resolve_nsql_model_name_from_app(app.as_ref())
        .map_err(|message| (StatusCode::BAD_REQUEST, message))
}

fn resolve_nsql_model_name_from_app(app: &app::App) -> Result<String, String> {
    let compatible_models = compatible_nsql_model_names(app);

    match compatible_models.as_slice() {
        [] => Err(
            "No model specified and no compatible LLM model is configured. Add exactly one LLM model to the Spicepod or include the 'model' field in the request."
                .to_string(),
        ),
        [model] => Ok(model.clone()),
        models => Err(format!(
            "No model specified and multiple compatible LLM models are configured ({}). Include the 'model' field in the request.",
            models.join(", ")
        )),
    }
}

fn compatible_nsql_model_names(app: &app::App) -> Vec<String> {
    app.models
        .iter()
        .filter(|model| model.model_type() == Some(ModelType::Llm))
        .map(|model| model.name.clone())
        .collect()
}

/// Construct a [`ResolvedTableAwareAllowlist`] based on the `App`'s `model.datasets`.
async fn table_allowlist(
    model_name: &str,
    rt: &Arc<Runtime>,
) -> Result<Option<ResolvedTableAwareAllowlist>, String> {
    let Some(app) = rt.read_app().await else {
        return Err("Unexpected internal error. App not prepared in runtime.".to_string());
    };

    // Create table allowlist from the model's datasets configuration
    let model_datasets = app
        .models
        .iter()
        .find(|m| m.name == model_name)
        .map(|m| m.datasets.clone())
        .unwrap_or_default();

    let table_allowlist = if model_datasets.is_empty() {
        None
    } else {
        match ResolvedTableAwareAllowlist::with_defaults(
            SPICE_DEFAULT_CATALOG,
            SPICE_DEFAULT_SCHEMA,
        )
        .with_table_patterns(model_datasets)
        {
            Ok(allowlist) => Some(allowlist),
            Err(_) => {
                return Err(format!(
                    "Unexpected internal error. Model '{model_name}' datasets are invalid."
                ));
            }
        }
    };
    Ok(table_allowlist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use app::AppBuilder;
    use serde_json::json;
    use spicepod::component::model::Model;

    fn app_with_models(models: Vec<Model>) -> app::App {
        let mut builder = AppBuilder::new("test");
        for model in models {
            builder = builder.with_model(model);
        }
        builder.build()
    }

    #[test]
    fn request_defaults_to_no_model_and_no_sample_data() {
        let request: Request = serde_json::from_value(json!({
            "query": "show total sales"
        }))
        .expect("request should deserialize with omitted optional fields");

        assert_eq!(request.model, None);
        assert!(!request.sample_data_enabled);
    }

    #[test]
    fn omitted_model_uses_single_compatible_model() {
        let app = app_with_models(vec![Model::new("openai:gpt-4o-mini", "llm_model")]);

        let model_name = resolve_nsql_model_name_from_app(&app)
            .expect("single compatible model should be selected");

        assert_eq!(model_name, "llm_model");
    }

    #[test]
    fn omitted_model_ignores_non_llm_models() {
        let app = app_with_models(vec![
            Model::new("spiceai:my-org/my-app/models/runnable", "ml_model"),
            Model::new("openai:gpt-4o-mini", "llm_model"),
        ]);

        let model_name = resolve_nsql_model_name_from_app(&app)
            .expect("single compatible model should be selected");

        assert_eq!(model_name, "llm_model");
    }

    #[test]
    fn omitted_model_errors_when_no_compatible_model_exists() {
        let app = app_with_models(vec![]);

        let error = resolve_nsql_model_name_from_app(&app)
            .expect_err("omitted model should fail without compatible models");

        assert_eq!(
            error,
            "No model specified and no compatible LLM model is configured. Add exactly one LLM model to the Spicepod or include the 'model' field in the request."
        );
    }

    #[test]
    fn omitted_model_errors_when_multiple_compatible_models_exist() {
        let app = app_with_models(vec![
            Model::new("openai:gpt-4o-mini", "first_model"),
            Model::new("openai:gpt-4o", "second_model"),
        ]);

        let error = resolve_nsql_model_name_from_app(&app)
            .expect_err("omitted model should fail with multiple compatible models");

        assert_eq!(
            error,
            "No model specified and multiple compatible LLM models are configured (first_model, second_model). Include the 'model' field in the request."
        );
    }

    #[test]
    fn cache_key_is_order_independent_and_scoped() {
        let t1 = vec![
            TableReference::from("spice.public.a"),
            TableReference::from("spice.public.b"),
        ];
        let t2 = vec![
            TableReference::from("spice.public.b"),
            TableReference::from("spice.public.a"),
        ];

        // Same table set in any order -> identical key (so equivalent requests share an entry).
        assert_eq!(
            nsql_cache_key("sample", "m1", &t1),
            nsql_cache_key("sample", "m1", &t2)
        );

        // kind, model, and the table set each scope the key.
        assert_ne!(
            nsql_cache_key("schema", "m1", &t1),
            nsql_cache_key("sample", "m1", &t1)
        );
        assert_ne!(
            nsql_cache_key("sample", "m1", &t1),
            nsql_cache_key("sample", "m2", &t1)
        );
        assert_ne!(
            nsql_cache_key("sample", "m1", &t1),
            nsql_cache_key("sample", "m1", &[TableReference::from("spice.public.a")])
        );
    }
}
