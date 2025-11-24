#!/usr/bin/env bash
#
# Manual Data Update Script for Innovations Dashboard
# Run this from your office network when you need to refresh data
#

set -euo pipefail

echo "========================================="
echo "  Innovations Dashboard - Manual Update"
echo "========================================="
echo ""

# Check for PSQL_DSN
if [ -z "${PSQL_DSN:-}" ]; then
  echo "ERROR: PSQL_DSN environment variable not set"
  echo ""
  echo "Set it with:"
  echo "  export PSQL_DSN='postgresql://marioanoadmin:cycleclocktheory600\$@awseb-e-hhgfq9zcb9-stack-awsebrdsdatabase-vrbrjr69ej4v.cxqmysocizjq.us-west-2.rds.amazonaws.com:5432/ebdb?sslmode=require'"
  echo ""
  exit 1
fi

# Step 1: Pull data from RDS
echo "Step 1: Pulling fresh data from AWS RDS..."
./scripts/pull_data.sh

# Update last_updated timestamp
echo "{\"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > data/last_updated.json
echo "   ✓ Updated data/last_updated.json"
echo ""

# Step 2: Run analytics pipelines (with conda environment)
echo "Step 2: Running analytics pipelines..."
if command -v conda &> /dev/null; then
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate eb-app-env
  ./scripts/run_pipelines.sh || echo "WARNING: Pipelines had some errors, but continuing..."
  conda deactivate
else
  ./scripts/run_pipelines.sh || echo "WARNING: Pipelines had some errors, but continuing..."
fi
echo ""

# Step 3: Show what changed
echo "Step 3: Checking what changed..."
if git diff --quiet data/*.csv outputs/*.csv 2>/dev/null; then
  echo "No data changes detected. Nothing to commit."
  exit 0
fi

echo ""
git status --short data/ outputs/
echo ""

# Step 4: Offer to commit and push
read -p "Commit and push these changes? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  git add data/*.csv data/last_updated.json outputs/*.csv 2>/dev/null || git add data/*.csv data/last_updated.json
  git commit -m "Manual data refresh: $(date '+%Y-%m-%d %H:%M')"
  git push
  echo ""
  echo "✓ Data updated and pushed to GitHub!"
  echo "✓ Vercel will auto-deploy in ~1 minute"
else
  echo "Skipped commit. Changes remain staged locally."
fi
