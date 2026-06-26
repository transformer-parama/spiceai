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

//! Milvus as a Spice vector store (storage layer for the `VectorIndex` in the
//! `search` crate). Reuses the low-level `milvus-client` crate so it shares the
//! connector's hardened client without a `runtime` dependency cycle.

mod compute_query;
mod query_provider;

pub use compute_query::{CachedQueryVector, ComputeQueryVector};
pub use query_provider::MilvusQueryTable;

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::Constraints;
use milvus_client::{MilvusCollection, MilvusConnection};

/// A Milvus collection exposed as a vector index: the shared connection, the
/// collection descriptor (vector field + metric + output fields), and the
/// Arrow schema the search returns (output columns + a `score` Float32 column).
#[derive(Clone, Debug)]
pub struct MilvusVectorsTable {
    pub conn: Arc<MilvusConnection>,
    pub coll: MilvusCollection,
    /// Schema the search returns: the collection's scalar/output fields plus a
    /// synthetic `score` (Float32) column.
    pub schema: SchemaRef,
    pub constraints: Constraints,
    pub dimension: i64,
}

impl MilvusVectorsTable {
    #[must_use]
    pub fn new(
        conn: Arc<MilvusConnection>,
        coll: MilvusCollection,
        schema: SchemaRef,
        dimension: i64,
    ) -> Self {
        Self {
            conn,
            coll,
            schema,
            constraints: Constraints::default(),
            dimension,
        }
    }
}
