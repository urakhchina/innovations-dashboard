#!/usr/bin/env bash
set -euo pipefail

# Data pull script: Executes SQL queries and generates CSV files
# Requires PSQL_DSN environment variable to be set
#
# Usage:
#   export PSQL_DSN='postgresql://username:password@host:5432/ebdb?sslmode=require'
#   ./scripts/pull_data.sh

: "${PSQL_DSN:?ERROR: PSQL_DSN environment variable not set}"

echo "========================================="
echo "  Innovations Dashboard - Data Refresh"
echo "========================================="
echo ""

# Ensure data directory exists
mkdir -p data

# Helper function: run one SQL file to one CSV
run_sql() {
  local sql_file="$1"
  local out_csv="$2"
  echo ">> Generating $out_csv from $sql_file"

  if [ ! -f "$sql_file" ]; then
    echo "   ERROR: SQL file not found: $sql_file"
    exit 1
  fi

  psql "$PSQL_DSN" -v ON_ERROR_STOP=1 -f "$sql_file" > "$out_csv"

  # Verify CSV was created and has content
  if [ ! -s "$out_csv" ]; then
    echo "   WARNING: $out_csv is empty or was not created"
  else
    local line_count=$(wc -l < "$out_csv")
    echo "   ✓ Generated $line_count lines"
  fi
}

echo "Starting data extraction from AWS RDS..."
echo ""

# Execute all SQL queries
run_sql sql/products_2025_by_upc.sql           data/products_2025_by_upc.csv
run_sql sql/innovations_transactions_2025.sql  data/innovations_transactions_2025.csv
run_sql sql/summary_monthly.sql                data/summary_monthly.csv
run_sql sql/summary_distributor.sql       data/summary_distributor.csv
run_sql sql/summary_salesrep.sql          data/summary_salesrep.csv
run_sql sql/summary_accounts_top50.sql    data/summary_accounts_top50.csv
run_sql sql/top_10_overall_skus.sql       data/top_10_overall_skus.csv
run_sql sql/launches_performance.sql      data/launches_performance.csv

echo ""
echo "========================================="
echo "  ✓ Data extraction complete!"
echo "========================================="
echo ""
echo "Generated files:"
ls -lh data/*.csv | awk '{print "  - "$9" ("$5")"}'
echo ""
