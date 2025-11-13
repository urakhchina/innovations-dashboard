-- Sales rep performance summary
COPY (
  SELECT
    sales_rep,
    description AS product_name,
    SUM(COALESCE(revenue, amount)) AS revenue_ytd,
    SUM(quantity) AS units_ytd,
    COUNT(*) AS txn_count
  FROM transactions
  WHERE posting_date >= '2025-01-01'
    AND posting_date < '2026-01-01'
  GROUP BY 1, 2
  ORDER BY 1, 2
) TO STDOUT WITH CSV HEADER;
