import pandas as pd
import json

print("=" * 80)
print("ADDING JANUARY 2025 TO DASHBOARD DATA")
print("=" * 80)

# Load January CSV
print("\nLoading January 2025 data...")
jan_df = pd.read_csv('/Users/natasha/Documents/Projects/IN_Reports/Innovations/january_2025_forecast_vs_actuals.csv')

# Build products array
products = []
total_forecast = 0
total_actual = 0

for _, row in jan_df.iterrows():
    forecast_val = float(row['ForecastQty']) if pd.notna(row['ForecastQty']) else 0
    actual_val = float(row['ActualQty']) if pd.notna(row['ActualQty']) else 0

    # Calculate accuracy (capped at 100%)
    if forecast_val > 0:
        accuracy = min((actual_val / forecast_val) * 100, 100)
    else:
        accuracy = 0

    products.append({
        'irwin_item': str(row['IrwinItem']),
        'product_name': str(row['Description']),
        'units_planned': round(forecast_val, 2),
        'units_actual': round(actual_val, 2),
        'variance': round(float(row['Variance']), 2) if pd.notna(row['Variance']) else 0,
        'variance_pct': round(float(row['VariancePct']), 2) if pd.notna(row['VariancePct']) else 0,
        'accuracy': round(accuracy, 2)
    })

    total_forecast += forecast_val
    total_actual += actual_val

# Calculate month-level metrics
if total_forecast > 0:
    month_accuracy = min((total_actual / total_forecast) * 100, 100)
else:
    month_accuracy = 0

variance = total_actual - total_forecast
variance_pct = (variance / total_forecast * 100) if total_forecast > 0 else 0

# Count products compared (those with actuals > 0)
products_compared = sum(1 for p in products if p['units_actual'] > 0)

january_data = {
    'month': 'January',
    'products': products,
    'total_units_planned': round(total_forecast, 2),
    'month_total_actual': round(total_actual, 2),
    'month_accuracy': round(month_accuracy, 2),
    'total_variance': round(variance, 2),
    'total_variance_pct': round(variance_pct, 2),
    'qty_products': int(len(products)),
    'products_compared': int(products_compared)
}

print(f"✓ January: {len(products)} products, {products_compared} with actuals")
print(f"  Forecast: {total_forecast:,.0f}, Actual: {total_actual:,.0f}, Variance: {variance_pct:+.1f}%")

# Load existing forecast data
print("\nLoading existing forecast data...")
forecast_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/forecast_data.json'
with open(forecast_file, 'r') as f:
    forecast_data = json.load(f)

# Add January to the data
forecast_data['by_month']['January'] = january_data

# Update metadata
forecast_data['metadata']['generated_at'] = pd.Timestamp.now().isoformat()
forecast_data['metadata']['months_included'] = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']

# Save updated data
with open(forecast_file, 'w') as f:
    json.dump(forecast_data, f, indent=2)

print(f"\n✓ Updated forecast data saved to: {forecast_file}")

# Print complete summary
print("\n" + "=" * 80)
print("COMPLETE SUMMARY (JANUARY - OCTOBER 2025)")
print("=" * 80)

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
for month_name in month_order:
    if month_name in forecast_data['by_month']:
        month_data = forecast_data['by_month'][month_name]
        print(f"\n{month_name:10} | Forecast: {month_data['total_units_planned']:>7,.0f} | Actual: {month_data['month_total_actual']:>7,.0f} | Variance: {month_data['total_variance_pct']:>+6.1f}%")

print("\n✓ Dashboard data update complete!")
