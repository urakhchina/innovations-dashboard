import pandas as pd
import json
import numpy as np

print("=" * 80)
print("ANALYZING OOS IMPACT FOR iHERB PRODUCTS")
print("=" * 80)

# Load catalog with monthly sales data
print("\nLoading iHerb catalog with sales history...")
catalog_df = pd.read_csv('/Users/natasha/Downloads/iherb_catalog.csv')

print(f"✓ Loaded {len(catalog_df)} products")

# Identify month columns (2024-03 through 2025-04)
month_cols = [col for col in catalog_df.columns if col.startswith('202')]
print(f"✓ Found sales data for {len(month_cols)} months: {month_cols[0]} to {month_cols[-1]}")

# Convert month columns to numeric, handling any non-numeric values
for col in month_cols:
    catalog_df[col] = pd.to_numeric(catalog_df[col], errors='coerce').fillna(0)

# Create a mapping of month column to readable month name
month_mapping = {}
for col in month_cols:
    year, month = col.split('-')
    month_names = {
        '01': 'January', '02': 'February', '03': 'March', '04': 'April',
        '05': 'May', '06': 'June', '07': 'July', '08': 'August',
        '09': 'September', '10': 'October', '11': 'November', '12': 'December'
    }
    month_mapping[col] = f"{month_names[month]} {year}"

print("\nAnalyzing OOS patterns...")

oos_products = []

# Analyze each product
for idx, row in catalog_df.iterrows():
    irwin_item = str(row['Irwin item']).strip()
    product_name = str(row['Product Description']).strip()

    # Skip if invalid data
    if irwin_item == 'nan' or not irwin_item:
        continue

    # Get sales history
    sales_history = []
    for col in month_cols:
        sales = float(row[col]) if pd.notna(row[col]) else 0
        sales_history.append(sales)

    # Find OOS periods (consecutive months with 0 sales)
    oos_periods = []
    in_oos = False
    oos_start_idx = None

    for i, sales in enumerate(sales_history):
        if sales == 0 and not in_oos:
            # Start of OOS period
            in_oos = True
            oos_start_idx = i
        elif sales > 0 and in_oos:
            # End of OOS period
            oos_end_idx = i - 1
            oos_periods.append((oos_start_idx, oos_end_idx))
            in_oos = False
            oos_start_idx = None

    # If still in OOS at the end
    if in_oos:
        oos_periods.append((oos_start_idx, len(sales_history) - 1))

    # Process each OOS period
    for oos_start, oos_end in oos_periods:
        # Only consider OOS periods of at least 1 month
        oos_duration = oos_end - oos_start + 1

        if oos_duration >= 1:
            # Calculate average sales from 3 months before OOS (if available)
            lookback_start = max(0, oos_start - 3)
            lookback_sales = sales_history[lookback_start:oos_start]

            # Only calculate if we have pre-OOS sales data
            if lookback_sales and sum(lookback_sales) > 0:
                avg_monthly_volume = sum(lookback_sales) / len(lookback_sales)

                # Calculate lost units during OOS period
                lost_units = avg_monthly_volume * oos_duration

                # Determine recovery status
                still_oos = (oos_end == len(sales_history) - 1)

                # Get month names
                oos_start_month = month_mapping[month_cols[oos_start]]

                if still_oos:
                    back_in_stock_month = None
                    status = "Still OOS"
                    recovery_time_days = None
                else:
                    back_in_stock_month = month_mapping[month_cols[oos_end + 1]]
                    status = "Restocked"
                    recovery_time_days = oos_duration * 30  # Approximate days

                # Estimate lost revenue (need to get price from invoice data or use estimate)
                # For now, we'll calculate this later from invoice data
                # Using a placeholder avg_price
                lost_revenue = lost_units * 12  # Placeholder: $12 avg wholesale price

                oos_products.append({
                    'irwin_item': irwin_item,
                    'product_name': product_name,
                    'oos_start_month': oos_start_month,
                    'back_in_stock_month': back_in_stock_month,
                    'months_oos': oos_duration,
                    'weeks_oos': round(oos_duration * 4.33, 1),
                    'avg_monthly_volume': round(avg_monthly_volume, 2),
                    'lost_units': round(lost_units, 2),
                    'lost_revenue': round(lost_revenue, 2),
                    'status': status,
                    'recovery_time_days': recovery_time_days
                })

print(f"✓ Found {len(oos_products)} OOS periods")

# Sort by lost revenue descending
oos_products.sort(key=lambda x: x['lost_revenue'], reverse=True)

# Calculate totals
total_lost_units = sum(p['lost_units'] for p in oos_products)
total_lost_revenue = sum(p['lost_revenue'] for p in oos_products)
still_oos_count = sum(1 for p in oos_products if p['status'] == 'Still OOS')

print(f"\nOOS Impact Summary:")
print(f"  Total OOS periods: {len(oos_products)}")
print(f"  Still OOS: {still_oos_count}")
print(f"  Restocked: {len(oos_products) - still_oos_count}")
print(f"  Total lost units: {total_lost_units:,.0f}")
print(f"  Estimated lost revenue: ${total_lost_revenue:,.2f}")

# Create output structure
output_data = {
    'oos_impact': oos_products,
    'summary': {
        'total_oos_periods': len(oos_products),
        'still_oos_count': still_oos_count,
        'restocked_count': len(oos_products) - still_oos_count,
        'total_lost_units': round(total_lost_units, 2),
        'total_lost_revenue': round(total_lost_revenue, 2)
    },
    'metadata': {
        'generated_at': pd.Timestamp.now().isoformat(),
        'analysis_period': f"{month_mapping[month_cols[0]]} to {month_mapping[month_cols[-1]]}",
        'note': 'Lost revenue estimated using $12 avg wholesale price placeholder'
    }
}

# Save to dashboard public folder
output_path = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/oos_impact.json'
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Saved OOS impact data to: {output_path}")

# Print top 10 most impactful OOS periods
print("\nTop 10 Most Impactful OOS Periods:")
print("-" * 80)
for i, product in enumerate(oos_products[:10], 1):
    status_badge = "⚠️ Still OOS" if product['status'] == "Still OOS" else f"✓ Restocked {product['back_in_stock_month']}"
    print(f"\n{i}. {product['product_name'][:50]}")
    print(f"   SKU: {product['irwin_item']}")
    print(f"   OOS Period: {product['oos_start_month']} ({product['months_oos']} months)")
    print(f"   Lost Units: {product['lost_units']:,.0f} | Lost Revenue: ${product['lost_revenue']:,.2f}")
    print(f"   Status: {status_badge}")

print("\n✓ OOS impact analysis complete!")
