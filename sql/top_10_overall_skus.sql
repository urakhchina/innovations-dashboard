-- Top 10 SKUs by YTD revenue with growth comparison
COPY (
WITH norm AS (
  SELECT
    CASE
      WHEN length(regexp_replace(coalesce(item_code,''),'[^0-9]','','g')) < 12
        THEN lpad(regexp_replace(coalesce(item_code,''),'[^0-9]','','g'),12,'0')
      WHEN length(regexp_replace(coalesce(item_code,''),'[^0-9]','','g')) = 13
           AND left(regexp_replace(coalesce(item_code,''),'[^0-9]','','g'),1)='0'
        THEN substring(regexp_replace(coalesce(item_code,''),'[^0-9]','','g') from 2)
      ELSE regexp_replace(coalesce(item_code,''),'[^0-9]','','g')
    END AS upc_norm12,
    posting_date::date AS dt,
    COALESCE(revenue, amount)::numeric AS revenue,
    COALESCE(quantity,0)::numeric AS units,
    description
  FROM transactions
  WHERE posting_date >= '2024-01-01'
),
cy AS (
  SELECT
    upc_norm12,
    SUM(revenue) AS cytd_revenue,
    SUM(units) AS cytd_units
  FROM norm
  WHERE dt >= '2025-01-01'
  GROUP BY 1
),
ly AS (
  SELECT
    upc_norm12,
    SUM(revenue) AS lytd_revenue
  FROM norm
  WHERE dt >= '2024-01-01' AND dt < '2025-01-01'
  GROUP BY 1
),
names_2025 AS (
  SELECT DISTINCT ON (upc_norm12)
    upc_norm12,
    description AS sku_name
  FROM (
    SELECT
      upc_norm12,
      description,
      COUNT(*) c
    FROM norm
    WHERE dt >= '2025-01-01'
    GROUP BY 1, 2
  ) s
  ORDER BY upc_norm12, c DESC, sku_name ASC
),
ranked AS (
  SELECT
    cy.upc_norm12,
    COALESCE(n.sku_name, cy.upc_norm12) AS sku_name,
    cy.cytd_revenue,
    cy.cytd_units,
    COALESCE(ly.lytd_revenue,0) AS lytd_revenue
  FROM cy
  LEFT JOIN ly ON ly.upc_norm12 = cy.upc_norm12
  LEFT JOIN names_2025 n ON n.upc_norm12 = cy.upc_norm12
)
SELECT
  sku_name,
  upc_norm12,
  cytd_revenue AS ytd_revenue,
  cytd_units AS ytd_units,
  CASE
    WHEN lytd_revenue > 0 THEN (cytd_revenue / lytd_revenue) - 1
    ELSE NULL
  END AS growth_rate,
  NULL::numeric AS profit_margin
FROM ranked
ORDER BY ytd_revenue DESC
LIMIT 10
) TO STDOUT WITH CSV HEADER;
