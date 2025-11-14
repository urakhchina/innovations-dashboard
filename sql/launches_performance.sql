-- Performance tracking for 2025 new product launches
COPY (
WITH launch_list(product_label, upc_dashed) AS (
  VALUES
    ('Collagen Beauty','7-10363-59212-7'),
    ('Ashwagandha Healthy Brain Mood & Stress','8-40081-41231-2'),
    ('Magnesium + Beets & CoQ10','8-40081-41308-1'),
    ('Magnesium + Milk Thistle & Turmeric','8-40081-41312-8'),
    ('Power to Sleep Magnesium PM + Relaxing Flower Complex','8-40081-41316-6'),
    ('Turbo-Energy Libido-Max RED','8-40081-41304-3'),
    ('Milk Thistle Triple-Detox','8-40081-41326-5'),
    ('Alpha-Choline Brain & Muscle Support Extra-Strength','8-40081-41322-7'),
    -- Q4 Gummies Launch
    ('Apple Cider Vinegar + Metabolism','8-40081-41344-9'),
    ('Stress-Defy Balanced Calm Focused','8-40081-41348-7'),
    ('Milk Thistle Liver Detox','8-40081-41356-2'),
    ('Maca Root + Ashwagandha','8-40081-41352-4'),
    ('Testosterone Up Peak Performance','8-40081-41369'),
    ('Magnesium + Whole-Body Balance','8-40081-41368-5')
),
launches AS (
  SELECT
    product_label AS sku_name,
    CASE
      WHEN length(regexp_replace(upc_dashed,'[^0-9]','','g')) < 12
        THEN lpad(regexp_replace(upc_dashed,'[^0-9]','','g'),12,'0')
      WHEN length(regexp_replace(upc_dashed,'[^0-9]','','g')) = 13
           AND left(regexp_replace(upc_dashed,'[^0-9]','','g'),1)='0'
        THEN substring(regexp_replace(upc_dashed,'[^0-9]','','g') from 2)
      ELSE regexp_replace(upc_dashed,'[^0-9]','','g')
    END AS upc_norm12
  FROM launch_list
),
norm AS (
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
    COALESCE(quantity,0)::numeric AS units
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
)
SELECT
  l.sku_name,
  l.upc_norm12,
  COALESCE(cy.cytd_revenue,0) AS ytd_revenue,
  COALESCE(cy.cytd_units,0) AS ytd_units,
  CASE
    WHEN COALESCE(ly.lytd_revenue,0) > 0
    THEN (COALESCE(cy.cytd_revenue,0) / ly.lytd_revenue) - 1
    ELSE NULL
  END AS growth_rate,
  NULL::numeric AS profit_margin
FROM launches l
LEFT JOIN cy ON cy.upc_norm12 = l.upc_norm12
LEFT JOIN ly ON ly.upc_norm12 = l.upc_norm12
ORDER BY ytd_revenue DESC, l.sku_name
) TO STDOUT WITH CSV HEADER;
