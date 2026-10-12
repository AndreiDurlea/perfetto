// Shared declarations for the perfetto DuckDB extension. Every module includes
// this header; only extension.cc defines the C API global.
#ifndef PERFETTO_DUCKDB_COMMON_H_
#define PERFETTO_DUCKDB_COMMON_H_

#define DUCKDB_EXTENSION_NAME perfetto
#include "duckdb_extension.h"  // no-include-violation-check

DUCKDB_EXTENSION_EXTERN

// Each operator family registers its native functions here. Return false on
// failure. SQL macros live in sql/*.sql and are run after registration.
bool RegisterIntervalFunctions(duckdb_connection conn);
bool RegisterTreeFunctions(duckdb_connection conn);
bool RegisterGraphFunctions(duckdb_connection conn);
bool RegisterProtoIngestFunctions(duckdb_connection conn);

#endif  // PERFETTO_DUCKDB_COMMON_H_
