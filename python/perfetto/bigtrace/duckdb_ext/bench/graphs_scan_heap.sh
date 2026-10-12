#!/bin/bash
# Scan traces for heap graphs: prints "<objects> <refs> <path>".
# usage: graphs_scan_heap.sh N_LARGEST PARALLEL     (largest N traces by size)
#        graphs_scan_heap.sh - PARALLEL < paths.txt (explicit list)
TP=~/bt_bench/tp_arrow/out/rel/trace_processor_shell
if [ "$1" = "-" ]; then
  cat
else
  ls -S ~/bigtrace_10k/traces/*.pftrace.zst | head -n "$1"
fi | xargs -P "$2" -I{} bash -c '
    r=$('"$TP"' query --quiet {} "select (select count(*) from heap_graph_object) || \" \" || (select count(*) from heap_graph_reference) as x" 2>/dev/null | tail -1 | tr -d "\"")
    echo "$r {}"'
