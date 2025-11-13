-- Monthly aggregated sales data
COPY (
  WITH line AS (
    SELECT
      date_trunc('month', posting_date)::date AS month,
      description AS product_name,
      COALESCE(revenue, amount) AS revenue,
      quantity AS units
    FROM transactions
    WHERE posting_date >= '2025-01-01'
      AND posting_date < '2026-01-01'
  )
  SELECT
    month,
    product_name,
    SUM(revenue) AS revenue,
    SUM(units) AS units,
    COUNT(*) AS txn_count
  FROM line
  GROUP BY 1, 2
  ORDER BY 1, 2
) TO STDOUT WITH CSV HEADER;
