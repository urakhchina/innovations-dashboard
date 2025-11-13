#!/usr/bin/env bash
set -euo pipefail

# Pipeline execution script: Runs ML/forecast pipelines
#
# Usage:
#   ./scripts/run_pipelines.sh

echo "========================================="
echo "  Running Analytics Pipelines"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
  echo "ERROR: python3 not found. Please install Python 3.8+"
  exit 1
fi

# Check if required input file exists
if [ ! -f "data/products_2025_by_upc.csv" ]; then
  echo "ERROR: data/products_2025_by_upc.csv not found"
  echo "Please run scripts/pull_data.sh first"
  exit 1
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo ">> Running churn/reorder prediction pipeline..."
if [ -f "churn_reorder_pipeline.py" ]; then
  # Use innovations-only dataset for innovation-specific churn model
  echo "   Training on INNOVATION PRODUCTS ONLY (8 SKUs)"

  # Automatically detect latest date in the data
  LATEST_DATE=$(python3 -c "import pandas as pd; df = pd.read_csv('data/innovations_transactions_2025.csv', usecols=['posting_date']); print(df['posting_date'].max())" 2>/dev/null)
  if [ -z "$LATEST_DATE" ] || [ "$LATEST_DATE" = "nan" ]; then
    LATEST_DATE="2025-09-30"
    echo "   WARNING: Could not detect latest date, using default: $LATEST_DATE"
  else
    echo "   Detected latest data date: $LATEST_DATE"
  fi

  # Calculate date ranges: For innovations (launched in 2025), adjust training period
  # Use shorter training window since products are new (no 2024 data for these SKUs)
  DATES=$(python3 -c "
from datetime import datetime, timedelta
latest = datetime.strptime('$LATEST_DATE', '%Y-%m-%d')
test_end = latest
test_start = test_end - timedelta(days=60)
valid_end = test_start - timedelta(days=1)
valid_start = valid_end - timedelta(days=45)  # Shorter validation (innovations are new)
train_end = valid_start - timedelta(days=1)
train_start = datetime(2025, 1, 1)  # Start from launch date (Jan 1, 2025)
print(f'{train_start:%Y-%m-%d}|{train_end:%Y-%m-%d}|{valid_start:%Y-%m-%d}|{valid_end:%Y-%m-%d}|{test_start:%Y-%m-%d}|{test_end:%Y-%m-%d}')
")

  IFS='|' read -r TRAIN_START TRAIN_END VALID_START VALID_END TEST_START TEST_END <<< "$DATES"

  echo "   Date ranges:"
  echo "     Train:      $TRAIN_START to $TRAIN_END"
  echo "     Validation: $VALID_START to $VALID_END"
  echo "     Test:       $TEST_START to $TEST_END"

  python3 churn_reorder_pipeline.py \
    --input data/innovations_transactions_2025.csv \
    --outdir outputs \
    --horizon_days 60 \
    --train_start "$TRAIN_START" \
    --train_end "$TRAIN_END" \
    --valid_start "$VALID_START" \
    --valid_end "$VALID_END" \
    --test_start "$TEST_START" \
    --test_end "$TEST_END"

  # Copy prediction output to data/ for dashboard
  if [ -f "outputs/churn_predictions_60d.csv" ]; then
    cp outputs/churn_predictions_60d.csv data/churn_predictions_60d.csv
    echo "   ✓ Churn predictions generated"
  else
    echo "   WARNING: Churn predictions file not generated"
  fi
else
  echo "   SKIPPED: churn_reorder_pipeline.py not found"
fi

echo ""
echo ">> Running sales forecast pipeline..."
if [ -f "sales_forecast_pipeline.py" ]; then
  python3 sales_forecast_pipeline.py \
    --input data/products_2025_by_upc.csv \
    --outdir outputs \
    --group product_distributor \
    --horizon_months 12 \
    --date_col posting_date \
    --revenue_col revenue \
    --product_col product_name \
    --distributor_col distributor \
    --salesrep_col sales_rep

  # Copy forecast outputs if they exist
  if ls outputs/forecast*.csv 1> /dev/null 2>&1; then
    cp outputs/forecast*.csv data/ 2>/dev/null || true
    echo "   ✓ Sales forecasts generated"
  else
    echo "   WARNING: No forecast files generated"
  fi
else
  echo "   SKIPPED: sales_forecast_pipeline.py not found"
fi

echo ""
echo "========================================="
echo "  ✓ Pipelines complete!"
echo "========================================="
echo ""

# List generated files
if ls data/*.csv 1> /dev/null 2>&1; then
  echo "Updated data files:"
  ls -lh data/*.csv | awk '{print "  - "$9" ("$5")"}'
  echo ""
fi
