import pandas as pd
import json

print("=" * 80)
print("ANALYZING OOS IMPACT FROM INVENTORY SNAPSHOTS")
print("=" * 80)

# Load inventory management data
print("\nLoading inventory management data...")
inventory_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/inventory_management.json'
with open(inventory_file, 'r') as f:
    inventory_data = json.load(f)

months = list(inventory_data['months'].keys())
print(f"✓ Loaded data for {len(months)} months: {', '.join(months)}")

# Build a product-level view across all months
products_by_sku = {}

for month in months:
    month_data = inventory_data['months'][month]
    for product in month_data['products']:
        sku = product['sku']

        if sku not in products_by_sku:
            products_by_sku[sku] = {
                'sku': sku,
                'product_name': product['productName'],
                'monthly_data': {}
            }

        products_by_sku[sku]['monthly_data'][month] = {
            'qty_available': product['qtyAvailable'],
            'velocity_monthly': product.get('velocityMonthly', 0)
        }

print(f"✓ Tracking {len(products_by_sku)} unique products across months")

# Identify OOS periods
oos_periods = []

for sku, product_info in products_by_sku.items():
    monthly_data = product_info['monthly_data']

    # Check each month for OOS (qty_available = 0)
    oos_start = None
    pre_oos_volumes = []

    for i, month in enumerate(months):
        if month not in monthly_data:
            continue

        qty = monthly_data[month]['qty_available']
        velocity = monthly_data[month]['velocity_monthly']

        if qty == 0:
            # Product is OOS this month
            if oos_start is None:
                # Start of OOS period
                oos_start = month
                oos_start_idx = i

                # Get pre-OOS volumes (up to 3 months before)
                for j in range(max(0, i-3), i):
                    prev_month = months[j]
                    if prev_month in monthly_data and monthly_data[prev_month]['velocity_monthly'] > 0:
                        pre_oos_volumes.append(monthly_data[prev_month]['velocity_monthly'])

        elif qty > 0 and oos_start is not None:
            # End of OOS period - product is back in stock
            oos_end_idx = i - 1
            oos_duration = oos_end_idx - oos_start_idx + 1

            # Calculate impact if we have pre-OOS velocity
            if pre_oos_volumes:
                avg_monthly_volume = sum(pre_oos_volumes) / len(pre_oos_volumes)
                lost_units = avg_monthly_volume * oos_duration

                oos_periods.append({
                    'irwin_item': sku,
                    'product_name': product_info['product_name'],
                    'oos_start_month': oos_start,
                    'back_in_stock_month': month,
                    'months_oos': oos_duration,
                    'weeks_oos': round(oos_duration * 4.33, 1),
                    'avg_monthly_volume': round(avg_monthly_volume, 2),
                    'lost_units': round(lost_units, 2),
                    'status': 'Restocked',
                    'recovery_time_days': oos_duration * 30
                })

            # Reset for next potential OOS period
            oos_start = None
            pre_oos_volumes = []

    # Check if still OOS at the end
    if oos_start is not None:
        last_month = months[-1]
        oos_end_idx = len(months) - 1
        oos_duration = oos_end_idx - oos_start_idx + 1

        if pre_oos_volumes:
            avg_monthly_volume = sum(pre_oos_volumes) / len(pre_oos_volumes)
            lost_units = avg_monthly_volume * oos_duration

            oos_periods.append({
                'irwin_item': sku,
                'product_name': product_info['product_name'],
                'oos_start_month': oos_start,
                'back_in_stock_month': None,
                'months_oos': oos_duration,
                'weeks_oos': round(oos_duration * 4.33, 1),
                'avg_monthly_volume': round(avg_monthly_volume, 2),
                'lost_units': round(lost_units, 2),
                'status': 'Still OOS',
                'recovery_time_days': None
            })

print(f"\n✓ Found {len(oos_periods)} OOS periods")

# Sort by lost units descending
oos_periods.sort(key=lambda x: x['lost_units'], reverse=True)

# Calculate totals
total_lost_units = sum(p['lost_units'] for p in oos_periods)
still_oos_count = sum(1 for p in oos_periods if p['status'] == 'Still OOS')

print(f"\nOOS Impact Summary:")
print(f"  Total OOS periods: {len(oos_periods)}")
print(f"  Still OOS: {still_oos_count}")
print(f"  Restocked: {len(oos_periods) - still_oos_count}")
print(f"  Total lost units: {total_lost_units:,.0f}")

# Create output structure
output_data = {
    'oos_impact': oos_periods,
    'summary': {
        'total_oos_periods': len(oos_periods),
        'still_oos_count': still_oos_count,
        'restocked_count': len(oos_periods) - still_oos_count,
        'total_lost_units': round(total_lost_units, 2)
    },
    'metadata': {
        'generated_at': pd.Timestamp.now().isoformat(),
        'analysis_period': f"{months[0]} to {months[-1]} 2025",
        'note': 'Lost units calculated based on average monthly sales velocity before OOS period',
        'data_source': 'inventory_management.json monthly snapshots'
    }
}

# Save to dashboard public folder
output_path = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/oos_impact.json'
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Saved OOS impact data to: {output_path}")

# Print top 10 most impactful OOS periods
print("\nTop 10 Most Impactful OOS Periods (by Lost Units):")
print("-" * 80)
for i, product in enumerate(oos_periods[:10], 1):
    status_badge = "⚠️ Still OOS" if product['status'] == "Still OOS" else f"✓ Restocked {product['back_in_stock_month']}"
    print(f"\n{i}. {product['product_name'][:60]}")
    print(f"   SKU: {product['irwin_item']}")
    print(f"   OOS Period: {product['oos_start_month']} ({product['months_oos']} months)")
    print(f"   Avg Monthly Volume: {product['avg_monthly_volume']:,.1f} units")
    print(f"   Lost Units: {product['lost_units']:,.0f}")
    print(f"   Status: {status_badge}")

print("\n✓ OOS impact analysis complete!")
