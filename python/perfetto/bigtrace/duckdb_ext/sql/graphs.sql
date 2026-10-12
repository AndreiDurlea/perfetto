-- Graph operators (spec §7). Native kernel: pf_graph_tree_agg (src/graphs.cc).
--
--   GRAPH DFS TREE       NODES n EDGES e FROM s  ->  pf_graph_dfs_tree('n', 'e', 's' [, per := [...]])
--   GRAPH BFS TREE       NODES n EDGES e FROM s  ->  pf_graph_bfs_tree('n', 'e', 's' [, per := [...]])
--   GRAPH DOMINATOR TREE NODES n EDGES e FROM s  ->  pf_graph_dominator_tree('n', 'e', 's' [, per := [...]])
--
-- n: table with column id (+ payload), e: table with columns src, dst,
-- s: table with column id. per: partition column names present in all three
-- (e.g. ['trace_idx'] or ['trace_idx', 'upid']); each partition is an
-- independent graph. Output: per cols, id, parent_id, depth, node payload.
-- Payload columns named parent_id / depth are replaced. Columns starting with
-- _pf_ are reserved.
CREATE OR REPLACE MACRO pf_graph__src(r) AS
  CASE
    WHEN regexp_matches(trim(r), '(?is)^(select|with|from|values)\b|^\(')
    THEN '(' || regexp_replace(trim(r), ';\s*$', '') || ')'
    ELSE regexp_replace(trim(r), ';\s*$', '')
  END
-- @@
CREATE OR REPLACE MACRO pf_graph__per(per) AS
  CASE
    WHEN per IS NULL THEN []::VARCHAR[]
    WHEN json_type(to_json(per)) = 'ARRAY' THEN from_json(to_json(per), '["VARCHAR"]')
    WHEN trim(per::VARCHAR) = '' THEN []::VARCHAR[]
    ELSE [trim(per::VARCHAR)]
  END
-- @@
CREATE OR REPLACE MACRO pf_graph__per_sql(per) AS
  '[' || array_to_string(list_transform(pf_graph__per(per), c -> '''' || c || ''''), ', ') || ']::VARCHAR[]'
-- @@
CREATE OR REPLACE MACRO pf_graph__on(per, a, b) AS
  array_to_string(list_transform(pf_graph__per(per), c -> a || '.' || c || ' = ' || b || '.' || c || ' AND '), '')
-- @@
CREATE OR REPLACE MACRO _pf_graph_tree(kind, nodes, edges, seeds, per) AS TABLE
FROM query(
  'WITH _pf_n AS ('
  || 'SELECT 0 AS _pf_g, COLUMNS(c -> c NOT IN (''parent_id'', ''depth'')) '
  || 'FROM ' || pf_graph__src(nodes) || ' __pf_n0'
  || '), _pf_u AS ('
  || 'SELECT COLUMNS(c -> c = ''_pf_g'' OR list_contains(' || pf_graph__per_sql(per) || ', c)), '
  || '0 AS _pf_role, id::BIGINT AS _pf_a, NULL::BIGINT AS _pf_b FROM _pf_n '
  || 'UNION ALL BY NAME '
  || 'SELECT COLUMNS(c -> c = ''_pf_g'' OR list_contains(' || pf_graph__per_sql(per) || ', c)), '
  || '1 AS _pf_role, src::BIGINT AS _pf_a, dst::BIGINT AS _pf_b '
  || 'FROM (SELECT 0 AS _pf_g, * FROM ' || pf_graph__src(edges) || ' __pf_e0) '
  || 'UNION ALL BY NAME '
  || 'SELECT COLUMNS(c -> c = ''_pf_g'' OR list_contains(' || pf_graph__per_sql(per) || ', c)), '
  || '2 AS _pf_role, id::BIGINT AS _pf_a, NULL::BIGINT AS _pf_b '
  || 'FROM (SELECT 0 AS _pf_g, * FROM ' || pf_graph__src(seeds) || ' __pf_s0)'
  || '), _pf_agg AS ('
  || 'SELECT COLUMNS(c -> c = ''_pf_g'' OR list_contains(' || pf_graph__per_sql(per) || ', c)), '
  || 'pf_graph_tree_agg(' || kind::INTEGER || '::INTEGER, _pf_role::INTEGER, _pf_a, _pf_b) AS _pf_t '
  || 'FROM _pf_u GROUP BY ALL'
  || '), _pf_tree AS ('
  || 'SELECT * EXCLUDE (_pf_t, _pf_r), _pf_r.id AS id, _pf_r.parent_id AS _pf_parent, _pf_r.depth AS _pf_depth '
  || 'FROM (SELECT *, unnest(_pf_t) AS _pf_r FROM _pf_agg)'
  || ') SELECT _pf_tree.* EXCLUDE (_pf_g) RENAME (_pf_parent AS parent_id, _pf_depth AS depth), '
  || '_pf_n.* EXCLUDE (_pf_g, id' || array_to_string(list_transform(pf_graph__per(per), c -> ', ' || c), '') || ') '
  || 'FROM _pf_tree JOIN _pf_n ON ' || pf_graph__on(per, '_pf_tree', '_pf_n') || '_pf_tree.id = _pf_n.id'
)
-- @@
CREATE OR REPLACE MACRO pf_graph_dfs_tree(nodes, edges, seeds, per := []::VARCHAR[]) AS TABLE
SELECT * FROM _pf_graph_tree(0, nodes, edges, seeds, per)
-- @@
CREATE OR REPLACE MACRO pf_graph_bfs_tree(nodes, edges, seeds, per := []::VARCHAR[]) AS TABLE
SELECT * FROM _pf_graph_tree(1, nodes, edges, seeds, per)
-- @@
CREATE OR REPLACE MACRO pf_graph_dominator_tree(nodes, edges, seeds, per := []::VARCHAR[]) AS TABLE
SELECT * FROM _pf_graph_tree(2, nodes, edges, seeds, per)
