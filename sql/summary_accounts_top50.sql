-- Top 50 accounts per product
COPY (
  WITH acct AS (
    SELECT
      canonical_code,
      description AS product_name,
      SUM(COALESCE(revenue, amount)) AS revenue_ytd,
      SUM(quantity) AS units_ytd,
      COUNT(*) AS txn_count
    FROM transactions
    WHERE posting_date >= '2025-01-01'
      AND posting_date < '2026-01-01'
    GROUP BY 1, 2
  ),
  ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY product_name ORDER BY revenue_ytd DESC) AS rn
    FROM acct
  )
  SELECT
    canonical_code,
    product_name,
    revenue_ytd,
    units_ytd,
    txn_count,
    rn AS rank
  FROM ranked
  WHERE rn <= 50
  ORDER BY product_name, rn
) TO STDOUT WITH CSV HEADER;
