-- Master detail file: all transaction lines for 2025
-- This is the primary source file that feeds all pipelines
COPY (
  SELECT
    -- Product and date fields
    posting_date::date     AS posting_date,
    EXTRACT(year FROM posting_date)::int AS year,
    CASE
      WHEN EXTRACT(month FROM posting_date) BETWEEN 1 AND 3 THEN 'Q1'
      WHEN EXTRACT(month FROM posting_date) BETWEEN 4 AND 6 THEN 'Q2'
      WHEN EXTRACT(month FROM posting_date) BETWEEN 7 AND 9 THEN 'Q3'
      ELSE 'Q4'
    END AS quarter,

    -- Product identification
    description AS product_name,
    item_code AS upc_item_code_raw,

    -- Normalized UPC as 12-digit
    CASE
      WHEN length(regexp_replace(coalesce(item_code,''),'[^0-9]','','g')) < 12
        THEN lpad(regexp_replace(coalesce(item_code,''),'[^0-9]','','g'),12,'0')
      WHEN length(regexp_replace(coalesce(item_code,''),'[^0-9]','','g')) = 13
           AND left(regexp_replace(coalesce(item_code,''),'[^0-9]','','g'),1)='0'
        THEN substring(regexp_replace(coalesce(item_code,''),'[^0-9]','','g') from 2)
      ELSE regexp_replace(coalesce(item_code,''),'[^0-9]','','g')
    END AS upc_item_code_norm12,

    distributor_item_code,

    -- Sales channels
    distributor,
    sales_rep,

    -- Account information
    canonical_code,
    base_card_code,
    ship_to_code,

    -- Transaction details
    quantity,
    COALESCE(revenue, amount) AS revenue,
    amount AS amount_raw,
    transaction_hash

  FROM transactions
  WHERE posting_date >= '2025-01-01'
    AND posting_date < '2026-01-01'
  ORDER BY posting_date DESC, canonical_code
) TO STDOUT WITH CSV HEADER;
