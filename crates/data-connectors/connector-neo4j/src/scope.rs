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

//! Per-tenant scope enforcement for `graph_query()` on a SHARED Neo4j graph.
//!
//! On a graph shared by many tenants, the only isolation boundary is the
//! per-tenant node label (`:asst_<id>`) every node carries. `graph_query()`
//! runs ARBITRARY read Cypher, so without enforcement a tenant's engine could
//! read another tenant's nodes. When a scope label is configured
//! ([`scope_label_from_env`]), [`enforce_scope`] makes the traversal fail
//! CLOSED:
//!
//!   * every node pattern in a `MATCH` / `OPTIONAL MATCH` gets the scope label
//!     injected as a required second label — `(n:Chunk)` -> `(n:Chunk:`asst_x`)`,
//!     `(n)` -> `(n:`asst_x`)`, so the traversal can only touch this tenant's nodes;
//!   * any construct that could ESCAPE that scoping is REJECTED rather than
//!     run unscoped — procedure calls (`CALL`, incl. `db.*` schema/`apoc.*`/
//!     fulltext which return cross-tenant nodes), `UNION`, subquery expressions
//!     (`EXISTS {`, `COUNT {`, `COLLECT {`), pattern comprehensions, quantified/
//!     grouped path patterns, and `shortestPath` are all refused.
//!
//! The rule of the module is: **when in doubt, reject.** A rejected valid query
//! is a usability problem the author can rephrase around; a mis-rewrite or a
//! missed node pattern would be a cross-tenant data leak. So the accepted Cypher
//! grammar is deliberately a small, safe subset (`MATCH`/`OPTIONAL MATCH` +
//! `WHERE`/`WITH`/`RETURN`/`UNWIND`/`ORDER BY`/`SKIP`/`LIMIT` with plain node and
//! relationship patterns). Enforcement is OFF (no rewrite, no rejection) when no
//! scope label is configured, preserving single-tenant/dev behaviour.

/// Read the process-wide tenant scope label (e.g. `asst_42`) from
/// `SPICE_NEO4J_SCOPE_LABEL`. Empty / whitespace resolves to `None` (enforcement
/// off). One env drives both label-mode scans and `graph_query()`.
#[must_use]
pub fn scope_label_from_env() -> Option<String> {
    std::env::var("SPICE_NEO4J_SCOPE_LABEL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Backtick-escape a Cypher label so an arbitrary scope string is safe to inject.
fn cypher_label(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Enforce `scope` on read Cypher: return the rewritten Cypher (node patterns
/// carry the scope label), or `Err(reason)` if the query uses a construct that
/// can't be safely scoped. Caller only invokes this when a scope is configured.
pub fn enforce_scope(cypher: &str, scope: &str) -> Result<String, String> {
    reject_unscopable(cypher)?;
    rewrite_match_patterns(cypher, scope)
}

/// A blanked copy of `cypher`: the CONTENTS of quoted string literals become
/// spaces (structure preserved) so keyword/procedure scanning never trips on
/// user text, and backtick-quoted identifiers are also blanked (they can't carry
/// an executable clause). Mirrors `graph_query::blank_string_literals`.
fn blank_literals(cypher: &str) -> String {
    let mut out = String::with_capacity(cypher.len());
    let mut quote: Option<char> = None;
    let mut chars = cypher.chars();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                out.push(' ');
                if c == '\\' {
                    if chars.next().is_some() {
                        out.push(' ');
                    }
                } else if c == q {
                    quote = None;
                }
            }
            None => {
                if c == '\'' || c == '"' || c == '`' {
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

/// Reject Cypher containing any construct that could read outside the scoped node
/// set. Scans a literal-blanked copy so user text can't cause a false reject.
fn reject_unscopable(cypher: &str) -> Result<(), String> {
    let blanked = blank_literals(cypher);
    let lower = blanked.to_ascii_lowercase();

    // Procedure / function namespaces that return or execute cross-tenant data.
    // These are dotted (survive blanking) so match as substrings.
    for ns in ["apoc.", "db.", "dbms.", "gds.", "spatial.", "algo."] {
        if lower.contains(ns) {
            return Err(format!(
                "graph_query(): procedure/function namespace '{ns}' is not allowed under tenant scoping"
            ));
        }
    }

    // Whole-word keywords that introduce unscopable control flow / subqueries.
    // Tokenise on non-identifier chars so `created_at` / `union_type` don't match.
    for token in blanked.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if token.is_empty() {
            continue;
        }
        let upper = token.to_ascii_uppercase();
        if matches!(
            upper.as_str(),
            "CALL"
                | "UNION"
                | "EXISTS"
                | "FOREACH"
                | "SHORTESTPATH"
                | "ALLSHORTESTPATHS"
        ) {
            return Err(format!(
                "graph_query(): '{token}' is not allowed under tenant scoping (cannot guarantee it stays within the tenant)"
            ));
        }
    }

    // Subquery expressions `COUNT { ... }` / `COLLECT { ... }` embed patterns in
    // expression position (EXISTS is already caught as a keyword above).
    if contains_word_then_brace(&lower, "count") || contains_word_then_brace(&lower, "collect") {
        return Err(
            "graph_query(): COUNT { } / COLLECT { } subqueries are not allowed under tenant scoping"
                .to_string(),
        );
    }

    // Pattern comprehension `[ (a)-->(b) | ... ]` embeds node patterns we don't
    // rewrite. Detect `[` followed (after whitespace) by `(`.
    if bracket_then_paren(&blanked) {
        return Err(
            "graph_query(): pattern comprehensions `[( ... )]` are not allowed under tenant scoping"
                .to_string(),
        );
    }

    Ok(())
}

/// True if `word` appears as a whole token immediately followed (after optional
/// whitespace) by `{` — i.e. a `word { ... }` subquery expression.
fn contains_word_then_brace(lower: &str, word: &str) -> bool {
    let bytes = lower.as_bytes();
    let mut from = 0;
    while let Some(rel) = lower[from..].find(word) {
        let start = from + rel;
        let end = start + word.len();
        let before_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        let mut j = end;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if before_ok && j < bytes.len() && bytes[j] == b'{' {
            return true;
        }
        from = end;
    }
    false
}

/// True if a `[` is followed (after only whitespace) by `(` — a pattern
/// comprehension opener, distinct from a list literal or a relationship `-[`.
fn bracket_then_paren(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'(' {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Clause {
    None,
    /// Inside a `MATCH` / `OPTIONAL MATCH` pattern span — node patterns here are
    /// rewritten. Ends at the next top-level clause keyword.
    Match,
    /// Inside `WHERE`/`WITH`/`RETURN`/`UNWIND`/`ORDER`/`SKIP`/`LIMIT` — expression
    /// context, `(` is a function call or grouping, never a node pattern.
    Other,
}

/// Walk the (real) Cypher and inject the scope label into every node pattern that
/// appears in a `MATCH` / `OPTIONAL MATCH` pattern span. String and backtick
/// contents are never touched. Returns `Err` if an unexpected shape is hit.
fn rewrite_match_patterns(cypher: &str, scope: &str) -> Result<String, String> {
    let chars: Vec<char> = cypher.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(cypher.len() + 16);
    let mut i = 0;
    let mut clause = Clause::None;
    // Nesting depth OUTSIDE of node patterns we consume wholesale. Clause keywords
    // and node-pattern starts are only recognised at depth 0.
    let mut depth: i32 = 0;

    while i < n {
        let c = chars[i];

        // Pass string / backtick literals through untouched.
        if c == '\'' || c == '"' || c == '`' {
            out.push(c);
            i += 1;
            while i < n {
                let d = chars[i];
                out.push(d);
                i += 1;
                if d == '\\' {
                    if i < n {
                        out.push(chars[i]);
                        i += 1;
                    }
                } else if d == c {
                    break;
                }
            }
            continue;
        }

        // A node pattern starts here only inside a MATCH span, at top level.
        if c == '(' && clause == Clause::Match && depth == 0 {
            // Reject a function call in pattern position (`(` glued to an ident)
            // or a grouped/quantified sub-pattern (`(` followed by `(`).
            //
            // A function call is GLUED to its name with no space (`count(`); a node
            // pattern after a clause keyword has a separator (`MATCH (`). So only the
            // immediately-preceding char matters: an ident char glued to `(` is a
            // function call and is rejected — UNLESS that word is a clause keyword
            // (handles a space-less `MATCH(n)` / `OPTIONAL MATCH(n)`).
            if i > 0 && is_ident_char(chars[i - 1]) {
                let w = preceding_word(&chars, i).to_ascii_uppercase();
                if w != "MATCH" {
                    return Err(
                        "graph_query(): unexpected '(' after an identifier in a MATCH pattern is not supported under tenant scoping"
                            .to_string(),
                    );
                }
            }
            let (inner, next) = take_balanced_paren(&chars, i)?;
            if inner.trim_start().starts_with('(') {
                return Err(
                    "graph_query(): grouped/quantified path patterns are not supported under tenant scoping"
                        .to_string(),
                );
            }
            out.push('(');
            out.push_str(&rewrite_node_inner(&inner, scope)?);
            out.push(')');
            i = next;
            continue;
        }

        // Track nesting so keyword/node detection only fires at top level.
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }

        // Read an identifier word to detect clause transitions (top level only).
        if is_ident_char(c) && depth == 0 {
            let start = i;
            while i < n && is_ident_char(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            match word.to_ascii_uppercase().as_str() {
                "MATCH" => clause = Clause::Match,
                "WHERE" | "RETURN" | "WITH" | "UNWIND" | "ORDER" | "SKIP" | "LIMIT" => {
                    clause = Clause::Other;
                }
                // OPTIONAL keeps the (soon to follow) MATCH; DISTINCT/AS/etc. don't
                // change the pattern-vs-expression context.
                _ => {}
            }
            out.push_str(&word);
            continue;
        }

        out.push(c);
        i += 1;
    }

    if depth != 0 {
        return Err("graph_query(): unbalanced parentheses/brackets".to_string());
    }
    Ok(out)
}

/// The identifier word ending immediately before `end` (exclusive), or empty if
/// the preceding char is not an identifier char.
fn preceding_word(chars: &[char], end: usize) -> String {
    let mut start = end;
    while start > 0 && is_ident_char(chars[start - 1]) {
        start -= 1;
    }
    chars[start..end].iter().collect()
}

/// Given `chars[open]` == '(', return (inner_without_parens, index_after_close).
/// Tracks strings/backticks and nested `()[]{}` so the matching `)` is correct.
fn take_balanced_paren(chars: &[char], open: usize) -> Result<(String, usize), String> {
    let n = chars.len();
    let mut i = open + 1;
    let mut depth = 1i32;
    let mut inner = String::new();
    while i < n {
        let c = chars[i];
        if c == '\'' || c == '"' || c == '`' {
            inner.push(c);
            i += 1;
            while i < n {
                let d = chars[i];
                inner.push(d);
                i += 1;
                if d == '\\' {
                    if i < n {
                        inner.push(chars[i]);
                        i += 1;
                    }
                } else if d == c {
                    break;
                }
            }
            continue;
        }
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok((inner, i + 1));
                }
            }
            _ => {}
        }
        inner.push(c);
        i += 1;
    }
    Err("graph_query(): unbalanced '(' in a MATCH pattern".to_string())
}

/// Rewrite the INSIDE of a node pattern (between its parens) to carry the scope
/// label, e.g. `n:Chunk` -> `n:Chunk:`asst_x``, `n` -> `n:`asst_x``,
/// `:Entity {p:1}` -> `:Entity:`asst_x` {p:1}`, `` (empty) `` -> `:`asst_x``.
fn rewrite_node_inner(inner: &str, scope: &str) -> Result<String, String> {
    // Inline predicates (`(n WHERE n.x > 1)`) and label-expression syntax
    // (`(n:A&B)`) can't be safely extended with an AND-ed label here.
    let scan = blank_literals(inner).to_ascii_lowercase();
    for token in scan.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if token == "where" {
            return Err(
                "graph_query(): inline WHERE inside a node pattern is not supported under tenant scoping"
                    .to_string(),
            );
        }
    }
    if inner.contains('&') || inner.contains('|') || inner.contains('!') {
        return Err(
            "graph_query(): label-expression syntax (& | !) inside a node pattern is not supported under tenant scoping"
                .to_string(),
        );
    }

    // Split off a trailing property map `{ ... }` (at brace-depth 0), so the scope
    // label is appended to the variable/label part, before the map.
    let label_part: String;
    let map_part: String;
    if let Some(idx) = top_level_brace(inner) {
        label_part = inner[..idx].to_string();
        map_part = inner[idx..].to_string();
    } else {
        label_part = inner.to_string();
        map_part = String::new();
    }

    let lp = label_part.trim();
    let mut new_inner = String::with_capacity(inner.len() + scope.len() + 3);
    new_inner.push_str(lp);
    new_inner.push(':');
    new_inner.push_str(&cypher_label(scope));
    let mp = map_part.trim();
    if !mp.is_empty() {
        new_inner.push(' ');
        new_inner.push_str(mp);
    }
    Ok(new_inner)
}

/// Byte index of the first `{` at brace/paren/bracket depth 0 (ignoring literals).
fn top_level_brace(s: &str) -> Option<usize> {
    let chars: Vec<char> = s.chars().collect();
    let mut byte = 0usize;
    let mut depth = 0i32;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' || c == '"' || c == '`' {
            byte += c.len_utf8();
            i += 1;
            while i < chars.len() {
                let d = chars[i];
                byte += d.len_utf8();
                i += 1;
                if d == c {
                    break;
                }
            }
            continue;
        }
        match c {
            '{' if depth == 0 => return Some(byte),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        byte += c.len_utf8();
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: &str = "asst_x";

    fn ok(cypher: &str) -> String {
        enforce_scope(cypher, S).expect("should be scopable")
    }

    #[test]
    fn injects_into_labeled_node() {
        assert_eq!(
            ok("MATCH (n:Chunk) RETURN n.id AS id"),
            "MATCH (n:Chunk:`asst_x`) RETURN n.id AS id"
        );
    }

    #[test]
    fn injects_into_bare_and_anonymous_nodes() {
        assert_eq!(ok("MATCH (n) RETURN n.id AS id"), "MATCH (n:`asst_x`) RETURN n.id AS id");
        assert_eq!(ok("MATCH () RETURN 1 AS x"), "MATCH (:`asst_x`) RETURN 1 AS x");
        assert_eq!(ok("MATCH (:Entity) RETURN 1 AS x"), "MATCH (:Entity:`asst_x`) RETURN 1 AS x");
    }

    #[test]
    fn injects_before_property_map() {
        assert_eq!(
            ok("MATCH (n:Entity {name:'intel'}) RETURN n.name AS name"),
            "MATCH (n:Entity:`asst_x` {name:'intel'}) RETURN n.name AS name"
        );
    }

    #[test]
    fn scopes_both_ends_of_a_relationship_traversal() {
        assert_eq!(
            ok("MATCH (a:Chunk)-[:NEXT*1..3]->(b:Chunk) RETURN a.id AS f, b.id AS t"),
            "MATCH (a:Chunk:`asst_x`)-[:NEXT*1..3]->(b:Chunk:`asst_x`) RETURN a.id AS f, b.id AS t"
        );
    }

    #[test]
    fn scopes_optional_match_and_multiple_clauses() {
        assert_eq!(
            ok("MATCH (a:Doc) OPTIONAL MATCH (a)-[:HAS]->(b) RETURN a.id AS i, b.id AS j"),
            "MATCH (a:Doc:`asst_x`) OPTIONAL MATCH (a:`asst_x`)-[:HAS]->(b:`asst_x`) RETURN a.id AS i, b.id AS j"
        );
    }

    #[test]
    fn leaves_return_functions_and_groupings_untouched() {
        assert_eq!(
            ok("MATCH (n:Chunk) RETURN count(n) AS c, (1 + 2) AS s"),
            "MATCH (n:Chunk:`asst_x`) RETURN count(n) AS c, (1 + 2) AS s"
        );
    }

    #[test]
    fn does_not_touch_parens_inside_string_literals() {
        assert_eq!(
            ok("MATCH (n:Chunk) WHERE n.name = '(not a node)' RETURN n.id AS id"),
            "MATCH (n:Chunk:`asst_x`) WHERE n.name = '(not a node)' RETURN n.id AS id"
        );
    }

    #[test]
    fn where_clause_grouping_not_scoped() {
        assert_eq!(
            ok("MATCH (n:E) WHERE (n.a = 1 OR n.b = 2) RETURN n.id AS id"),
            "MATCH (n:E:`asst_x`) WHERE (n.a = 1 OR n.b = 2) RETURN n.id AS id"
        );
    }

    #[test]
    fn rejects_procedure_calls() {
        for q in [
            "CALL db.labels() YIELD label RETURN label",
            "MATCH (n:E) CALL apoc.path.expand(n,'','',0,3) YIELD path RETURN path",
            "CALL db.index.fulltext.queryNodes('e_ft','intel') YIELD node RETURN node.name AS n",
        ] {
            assert!(enforce_scope(q, S).is_err(), "must reject: {q}");
        }
    }

    #[test]
    fn rejects_inline_apoc_and_db_functions() {
        assert!(enforce_scope(
            "MATCH (n:E) RETURN apoc.text.join(collect(n.name),',') AS names", S
        )
        .is_err());
        assert!(enforce_scope("MATCH (n:E) WHERE n.id IN db.foo() RETURN n.id AS id", S).is_err());
    }

    #[test]
    fn rejects_union_subqueries_and_comprehensions() {
        for q in [
            "MATCH (a:E) RETURN a.id AS id UNION MATCH (b:F) RETURN b.id AS id",
            "MATCH (a:E) WHERE EXISTS { MATCH (a)-[:R]->(:F) } RETURN a.id AS id",
            "MATCH (a:E) RETURN COUNT { (a)-[:R]->() } AS c",
            "MATCH (a:E) RETURN [ (a)-[:R]->(b) | b.id ] AS ids",
            "MATCH p = shortestPath((a:E)-[*]-(b:E)) RETURN p",
        ] {
            assert!(enforce_scope(q, S).is_err(), "must reject: {q}");
        }
    }

    #[test]
    fn rejects_inline_where_in_node_pattern() {
        assert!(enforce_scope("MATCH (n:E WHERE n.x > 1) RETURN n.id AS id", S).is_err());
    }

    #[test]
    fn write_words_inside_literals_do_not_break_scoping() {
        // 'union' / 'call' inside a string must not trigger a reject.
        assert_eq!(
            ok("MATCH (n:E) WHERE n.note = 'call the union rep' RETURN n.id AS id"),
            "MATCH (n:E:`asst_x`) WHERE n.note = 'call the union rep' RETURN n.id AS id"
        );
    }

    #[test]
    fn property_named_like_keyword_not_rejected() {
        // `.db_id` and `union_flag` are identifiers, not the db./UNION constructs.
        assert_eq!(
            ok("MATCH (n:E) WHERE n.union_flag = 1 RETURN n.id AS id"),
            "MATCH (n:E:`asst_x`) WHERE n.union_flag = 1 RETURN n.id AS id"
        );
    }
}
