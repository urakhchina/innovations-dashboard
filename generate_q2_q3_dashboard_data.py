import pandas as pd
import json

print("=" * 80)
print("GENERATING Q2-Q3 DASHBOARD DATA (MAY-OCTOBER)")
print("=" * 80)

months = ['May', 'June', 'July', 'August', 'September', 'October']
months_data = {}

for month in months:
    print(f"\nProcessing {month} 2025...")

    # Load the CSV
    csv_file = f'/Users/natasha/Documents/Projects/IN_Reports/Innovations/q2_q3_forecast_vs_actuals_{month.lower()}_2025.csv'
    df = pd.read_csv(csv_file)

    # Build products array
    products = []
    total_forecast = 0
    total_actual = 0

    for _, row in df.iterrows():
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

    # Calculate month-level accuracy
    if total_forecast > 0:
        month_accuracy = min((total_actual / total_forecast) * 100, 100)
    else:
        month_accuracy = 0

    # Calculate variance
    variance = total_actual - total_forecast
    variance_pct = (variance / total_forecast * 100) if total_forecast > 0 else 0

    # Count products compared (those with actuals > 0)
    products_compared = sum(1 for p in products if p['units_actual'] > 0)

    months_data[month] = {
        'month': month,
        'products': products,
        'total_units_planned': round(total_forecast, 2),
        'month_total_actual': round(total_actual, 2),
        'month_accuracy': round(month_accuracy, 2),
        'total_variance': round(variance, 2),
        'total_variance_pct': round(variance_pct, 2),
        'qty_products': int(len(products)),
        'products_compared': int(products_compared)
    }

    print(f"✓ {month}: {len(products)} products, {products_compared} with actuals")
    print(f"  Forecast: {total_forecast:,.0f}, Actual: {total_actual:,.0f}, Variance: {variance_pct:+.1f}%")

# Build final structure
dashboard_data = {
    'by_month': months_data,
    'metadata': {
        'generated_at': pd.Timestamp.now().isoformat(),
        'source': 'Q2-Q3 2025 Forecast Files + alliHerb.xlsx',
        'months_included': months
    }
}

# Save to JSON
output_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/forecast_data_q2_q3.json'
with open(output_file, 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print(f"\n✓ Saved to: {output_file}")

# Also merge with existing Q1 data to create complete forecast_data.json
print("\n" + "=" * 80)
print("MERGING WITH Q1 DATA")
print("=" * 80)

# Load existing Q1 data
q1_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/forecast_data.json'
with open(q1_file, 'r') as f:
    q1_data = json.load(f)

# Merge Q1 and Q2-Q3 data
all_months = {}
all_months.update(q1_data['by_month'])  # February, March, April
all_months.update(months_data)  # May, June, July, August, September, October

complete_data = {
    'by_month': all_months,
    'metadata': {
        'generated_at': pd.Timestamp.now().isoformat(),
        'source': 'Q1-Q3 2025 Forecast Files + alliHerb.xlsx',
        'months_included': ['February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
    }
}

# Save complete data
complete_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/forecast_data.json'
with open(complete_file, 'w') as f:
    json.dump(complete_data, f, indent=2)

print(f"✓ Saved complete forecast data (Feb-Oct) to: {complete_file}")

print("\n" + "=" * 80)
print("COMPLETE SUMMARY (FEBRUARY - OCTOBER 2025)")
print("=" * 80)

for month_name in ['February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']:
    month_data = all_months[month_name]
    print(f"\n{month_name:10} | Forecast: {month_data['total_units_planned']:>7,.0f} | Actual: {month_data['month_total_actual']:>7,.0f} | Variance: {month_data['total_variance_pct']:>+6.1f}%")

print("\n✓ Dashboard data generation complete!")
