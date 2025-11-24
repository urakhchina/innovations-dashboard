import pandas as pd
import json
import math

print("Generating Q1 2025 dashboard data from forecast vs actuals...")

# Read the Q1 forecast vs actuals CSV
df = pd.read_csv('/Users/natasha/Downloads/q1_2025_forecast_vs_actuals_iherb.csv')

print(f"Total records: {len(df)}")
print(f"Months: {df['month'].unique()}")

# Group by month
months_data = {}

for month in df['month'].unique():
    # Remove " 2025" suffix to match preloaded-data.json format
    month_key = month.replace(' 2025', '')
    month_df = df[df['month'] == month].copy()

    # Sort by forecast descending
    month_df = month_df.sort_values('forecast', ascending=False)

    # Build products list
    products = []
    for _, row in month_df.iterrows():
        forecast_val = float(row['forecast']) if pd.notna(row['forecast']) else 0
        actual_val = float(row['actual']) if pd.notna(row['actual']) else 0

        # Calculate accuracy: min(actual/forecast * 100, 100) if forecast > 0
        if forecast_val > 0:
            accuracy = min((actual_val / forecast_val) * 100, 100)
        else:
            accuracy = 0

        product = {
            'product_name': row['product_name'] if pd.notna(row['product_name']) else 'Unknown',
            'sku': row['irwin_item'] if pd.notna(row['irwin_item']) else 'N/A',
            'upc': 'N/A',  # We don't have UPC in this dataset yet
            'distributor': 'iHerb',
            'forecast': forecast_val,
            'actual': actual_val,
            'actual_revenue': float(row['actual_revenue']) if pd.notna(row['actual_revenue']) else 0,
            'variance': float(row['variance']) if pd.notna(row['variance']) else 0,
            'variance_pct': float(row['variance_pct']) if pd.notna(row['variance_pct']) else 0,
            'accuracy': round(float(accuracy), 2)
        }
        products.append(product)

    # Calculate totals
    total_forecast = float(month_df['forecast'].sum())
    total_actual = float(month_df['actual'].sum())
    total_revenue = float(month_df['actual_revenue'].sum())
    total_variance = total_actual - total_forecast
    total_variance_pct = (total_variance / total_forecast * 100) if total_forecast > 0 else 0

    # Calculate month accuracy: min(actual/forecast * 100, 100)
    if total_forecast > 0:
        month_accuracy = min((total_actual / total_forecast) * 100, 100)
    else:
        month_accuracy = 0

    months_data[month_key] = {
        'month': month_key,
        'products': products,
        'total_units_planned': round(float(total_forecast), 2),
        'month_total_actual': round(float(total_actual), 2),
        'month_accuracy': round(float(month_accuracy), 2),
        'total_revenue': round(float(total_revenue), 2),
        'total_variance': round(float(total_variance), 2),
        'total_variance_pct': round(float(total_variance_pct), 2),
        'qty_products': int(len(products)),
        'products_compared': int(len(month_df[(month_df['forecast'] > 0) & (month_df['actual'] > 0)])),
        'qty_products_with_sales': int(len(month_df[month_df['actual'] > 0]))
    }

# Create final structure
output_data = {
    'by_month': months_data
}

# Save to dashboard public folder
output_path = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/forecast_data.json'
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved to: {output_path}")
print("\nSummary:")
for month in sorted(months_data.keys()):
    data = months_data[month]
    print(f"\n{month}:")
    print(f"  Products: {data['qty_products']}")
    print(f"  Products with sales: {data['qty_products_with_sales']}")
    print(f"  Products matched: {data['products_compared']}")
    print(f"  Forecast: {data['total_units_planned']:,.0f} units")
    print(f"  Actual: {data['month_total_actual']:,.0f} units")
    print(f"  Accuracy: {data['month_accuracy']:.1f}%")
    print(f"  Revenue: ${data['total_revenue']:,.2f}")
    print(f"  Variance: {data['total_variance']:,.0f} units ({data['total_variance_pct']:+.1f}%)")
