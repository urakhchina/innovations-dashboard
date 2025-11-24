import pandas as pd
import json

print("=" * 80)
print("ADDING OCTOBER 2025 ACTUALS")
print("=" * 80)

# Load October invoice data
print("\nLoading October 2025 invoice data...")
oct_invoices = pd.read_excel('/Users/natasha/Downloads/iHerbInvoices_October.xlsx')

# Check date range
oct_invoices['DocDate'] = pd.to_datetime(oct_invoices['DocDate'])
print(f"✓ Loaded {len(oct_invoices)} invoice lines")
print(f"  Date range: {oct_invoices['DocDate'].min()} to {oct_invoices['DocDate'].max()}")

# Aggregate by ItemCode
oct_actuals = oct_invoices.groupby('ItemCode')['Quantity'].sum().reset_index()
oct_actuals.columns = ['IrwinItem', 'ActualQty']

print(f"\n✓ Aggregated to {len(oct_actuals)} unique products")
print(f"  Total units: {oct_actuals['ActualQty'].sum():,.0f}")

# Load existing October forecast data
print("\nLoading October forecast data...")
oct_forecast = pd.read_csv('/Users/natasha/Documents/Projects/IN_Reports/Innovations/q2_q3_forecast_vs_actuals_october_2025.csv')

print(f"✓ Loaded {len(oct_forecast)} forecasted products")
print(f"  Total forecast: {oct_forecast['ForecastQty'].sum():,.0f} units")

# Update actuals in the forecast data
# First, create a dictionary of actuals
actuals_dict = dict(zip(oct_actuals['IrwinItem'], oct_actuals['ActualQty']))

# Update the forecast data with new actuals
oct_forecast['ActualQty'] = oct_forecast['IrwinItem'].map(actuals_dict).fillna(0)

# Recalculate variance and accuracy
oct_forecast['Variance'] = oct_forecast['ActualQty'] - oct_forecast['ForecastQty']
oct_forecast['VariancePct'] = ((oct_forecast['ActualQty'] - oct_forecast['ForecastQty']) / oct_forecast['ForecastQty'] * 100).round(2)
oct_forecast['Accuracy'] = oct_forecast.apply(
    lambda row: min((row['ActualQty'] / row['ForecastQty']) * 100, 100) if row['ForecastQty'] > 0 else 0,
    axis=1
).round(2)

# Save updated CSV
oct_forecast.to_csv('/Users/natasha/Documents/Projects/IN_Reports/Innovations/q2_q3_forecast_vs_actuals_october_2025.csv', index=False)

matched_count = (oct_forecast['ActualQty'] > 0).sum()
total_forecast = oct_forecast['ForecastQty'].sum()
total_actual = oct_forecast['ActualQty'].sum()
variance = total_actual - total_forecast
variance_pct = (variance / total_forecast * 100) if total_forecast > 0 else 0

print(f"\n✓ Updated October forecast vs actuals")
print(f"  Matched: {matched_count}/{len(oct_forecast)} products ({matched_count/len(oct_forecast)*100:.1f}%)")
print(f"\nOctober 2025 Summary:")
print(f"  Forecast: {total_forecast:,.0f} units")
print(f"  Actual:   {total_actual:,.0f} units")
print(f"  Variance: {variance:+,.0f} units ({variance_pct:+.1f}%)")

# Now update the dashboard JSON
print("\n" + "=" * 80)
print("UPDATING DASHBOARD DATA")
print("=" * 80)

# Build products array for dashboard
products = []
for _, row in oct_forecast.iterrows():
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

# Calculate month-level accuracy
if total_forecast > 0:
    month_accuracy = min((total_actual / total_forecast) * 100, 100)
else:
    month_accuracy = 0

october_data = {
    'month': 'October',
    'products': products,
    'total_units_planned': round(total_forecast, 2),
    'month_total_actual': round(total_actual, 2),
    'month_accuracy': round(month_accuracy, 2),
    'total_variance': round(variance, 2),
    'total_variance_pct': round(variance_pct, 2),
    'qty_products': int(len(products)),
    'products_compared': int(matched_count)
}

# Load and update dashboard data
forecast_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/forecast_data.json'
with open(forecast_file, 'r') as f:
    forecast_data = json.load(f)

# Update October data
forecast_data['by_month']['October'] = october_data

# Update metadata
forecast_data['metadata']['generated_at'] = pd.Timestamp.now().isoformat()

# Save updated data
with open(forecast_file, 'w') as f:
    json.dump(forecast_data, f, indent=2)

print(f"\n✓ Updated dashboard data saved to: {forecast_file}")

# Print complete summary
print("\n" + "=" * 80)
print("COMPLETE SUMMARY (JANUARY - OCTOBER 2025)")
print("=" * 80)

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
for month_name in month_order:
    if month_name in forecast_data['by_month']:
        month_data = forecast_data['by_month'][month_name]
        print(f"\n{month_name:10} | Forecast: {month_data['total_units_planned']:>7,.0f} | Actual: {month_data['month_total_actual']:>7,.0f} | Variance: {month_data['total_variance_pct']:>+6.1f}%")

print("\n✓ October actuals update complete!")
