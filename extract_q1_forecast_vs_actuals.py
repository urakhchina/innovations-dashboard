import pandas as pd
from datetime import datetime

print("="*80)
print("Q1 2025 iHerb Forecast vs Actuals Analysis")
print("="*80)

# 1. Load Q1 2025 Forecast Data
print("\n1. Loading Q1 2025 forecast data...")
q1_file = '/Users/natasha/Documents/Projects/IN_POS Claude/iHerb/raw_data/forecasts/2025/Irwin Naturals_IRW_iHerb Supplier Forecast Q1 2025_REV.xlsx'
forecast_df = pd.read_excel(q1_file, sheet_name=0, header=6)

print(f"   Columns in Q1 file: {list(forecast_df.columns)}")

# 2. Extract Feb, March, April forecast data
print("\n2. Extracting Feb, March, April forecast data...")

# Get relevant columns
forecast_data = forecast_df[['Irwin Item', 'Description', '25-Feb', '25-Mar', '25-Apr']].copy()

# Filter out rows with no Irwin Item
forecast_data = forecast_data[forecast_data['Irwin Item'].notna()]

print(f"   Total products with Irwin Item: {len(forecast_data)}")

# 3. Load alliHerb actuals
print("\n3. Loading alliHerb actuals...")
actuals_df = pd.read_excel('/Users/natasha/Downloads/alliHerb.xlsx')

print(f"   Total transactions in alliHerb: {len(actuals_df)}")
print(f"   Columns: {list(actuals_df.columns)}")

# Convert DocDate to datetime
actuals_df['DocDate'] = pd.to_datetime(actuals_df['DocDate'])

# Filter for Feb, March, April 2025
feb_actuals = actuals_df[(actuals_df['DocDate'].dt.month == 2) & (actuals_df['DocDate'].dt.year == 2025)]
mar_actuals = actuals_df[(actuals_df['DocDate'].dt.month == 3) & (actuals_df['DocDate'].dt.year == 2025)]
apr_actuals = actuals_df[(actuals_df['DocDate'].dt.month == 4) & (actuals_df['DocDate'].dt.year == 2025)]

print(f"   Feb 2025 transactions: {len(feb_actuals)}")
print(f"   Mar 2025 transactions: {len(mar_actuals)}")
print(f"   Apr 2025 transactions: {len(apr_actuals)}")

# Aggregate actuals by ItemCode for each month
feb_agg = feb_actuals.groupby('ItemCode').agg({
    'Quantity': 'sum',
    'LineTotal': 'sum'
}).reset_index()

mar_agg = mar_actuals.groupby('ItemCode').agg({
    'Quantity': 'sum',
    'LineTotal': 'sum'
}).reset_index()

apr_agg = apr_actuals.groupby('ItemCode').agg({
    'Quantity': 'sum',
    'LineTotal': 'sum'
}).reset_index()

print(f"\n   Unique products sold:")
print(f"   Feb 2025: {len(feb_agg)} products")
print(f"   Mar 2025: {len(mar_agg)} products")
print(f"   Apr 2025: {len(apr_agg)} products")

# 4. Match forecasts to actuals for each month
print("\n4. Matching forecasts to actuals...")

results = []

for _, row in forecast_data.iterrows():
    irwin_item = str(row['Irwin Item']).strip() if pd.notna(row['Irwin Item']) else None
    product_name = row['Description']

    if not irwin_item:
        continue

    # February
    feb_forecast = row['25-Feb'] if pd.notna(row['25-Feb']) else 0
    feb_actual_match = feb_agg[feb_agg['ItemCode'] == irwin_item]
    feb_actual = feb_actual_match['Quantity'].sum() if len(feb_actual_match) > 0 else 0
    feb_revenue = feb_actual_match['LineTotal'].sum() if len(feb_actual_match) > 0 else 0

    if feb_forecast > 0 or feb_actual > 0:
        feb_variance = feb_actual - feb_forecast
        feb_variance_pct = (feb_variance / feb_forecast * 100) if feb_forecast > 0 else 0

        results.append({
            'month': 'February 2025',
            'irwin_item': irwin_item,
            'product_name': product_name,
            'forecast': round(feb_forecast, 2),
            'actual': round(feb_actual, 2),
            'actual_revenue': round(feb_revenue, 2),
            'variance': round(feb_variance, 2),
            'variance_pct': round(feb_variance_pct, 2)
        })

    # March
    mar_forecast = row['25-Mar'] if pd.notna(row['25-Mar']) else 0
    mar_actual_match = mar_agg[mar_agg['ItemCode'] == irwin_item]
    mar_actual = mar_actual_match['Quantity'].sum() if len(mar_actual_match) > 0 else 0
    mar_revenue = mar_actual_match['LineTotal'].sum() if len(mar_actual_match) > 0 else 0

    if mar_forecast > 0 or mar_actual > 0:
        mar_variance = mar_actual - mar_forecast
        mar_variance_pct = (mar_variance / mar_forecast * 100) if mar_forecast > 0 else 0

        results.append({
            'month': 'March 2025',
            'irwin_item': irwin_item,
            'product_name': product_name,
            'forecast': round(mar_forecast, 2),
            'actual': round(mar_actual, 2),
            'actual_revenue': round(mar_revenue, 2),
            'variance': round(mar_variance, 2),
            'variance_pct': round(mar_variance_pct, 2)
        })

    # April
    apr_forecast = row['25-Apr'] if pd.notna(row['25-Apr']) else 0
    apr_actual_match = apr_agg[apr_agg['ItemCode'] == irwin_item]
    apr_actual = apr_actual_match['Quantity'].sum() if len(apr_actual_match) > 0 else 0
    apr_revenue = apr_actual_match['LineTotal'].sum() if len(apr_actual_match) > 0 else 0

    if apr_forecast > 0 or apr_actual > 0:
        apr_variance = apr_actual - apr_forecast
        apr_variance_pct = (apr_variance / apr_forecast * 100) if apr_forecast > 0 else 0

        results.append({
            'month': 'April 2025',
            'irwin_item': irwin_item,
            'product_name': product_name,
            'forecast': round(apr_forecast, 2),
            'actual': round(apr_actual, 2),
            'actual_revenue': round(apr_revenue, 2),
            'variance': round(apr_variance, 2),
            'variance_pct': round(apr_variance_pct, 2)
        })

# 5. Create results DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = '/Users/natasha/Downloads/q1_2025_forecast_vs_actuals_iherb.csv'
results_df.to_csv(output_path, index=False)

print(f"\n5. Results saved to: {output_path}")
print(f"   Total records: {len(results_df)}")

# 6. Summary Statistics
print("\n" + "="*80)
print("SUMMARY BY MONTH")
print("="*80)

for month in ['February 2025', 'March 2025', 'April 2025']:
    month_data = results_df[results_df['month'] == month]

    total_forecast = month_data['forecast'].sum()
    total_actual = month_data['actual'].sum()
    total_revenue = month_data['actual_revenue'].sum()
    variance = total_actual - total_forecast
    variance_pct = (variance / total_forecast * 100) if total_forecast > 0 else 0

    products_forecasted = len(month_data[month_data['forecast'] > 0])
    products_with_sales = len(month_data[month_data['actual'] > 0])
    products_matched = len(month_data[(month_data['forecast'] > 0) & (month_data['actual'] > 0)])

    print(f"\n{month}:")
    print(f"  Products forecasted: {products_forecasted}")
    print(f"  Products with actual sales: {products_with_sales}")
    print(f"  Products matched (forecast & sales): {products_matched}")
    print(f"  Total forecast: {total_forecast:,.0f} units")
    print(f"  Total actual: {total_actual:,.0f} units")
    print(f"  Total revenue: ${total_revenue:,.2f}")
    print(f"  Variance: {variance:,.0f} units ({variance_pct:+.1f}%)")

# 7. Top matched products
print("\n" + "="*80)
print("TOP 10 PRODUCTS WITH BOTH FORECAST & ACTUALS (February)")
print("="*80)

feb_matched = results_df[
    (results_df['month'] == 'February 2025') &
    (results_df['forecast'] > 0) &
    (results_df['actual'] > 0)
].sort_values('actual', ascending=False).head(10)

print("\n{:<15} {:<50} {:>10} {:>10} {:>10}".format(
    'Irwin Item', 'Product Name', 'Forecast', 'Actual', 'Variance%'
))
print("-"*100)

for _, row in feb_matched.iterrows():
    product_name = row['product_name'][:47] + '...' if len(str(row['product_name'])) > 50 else row['product_name']
    print("{:<15} {:<50} {:>10.0f} {:>10.0f} {:>9.1f}%".format(
        row['irwin_item'],
        product_name,
        row['forecast'],
        row['actual'],
        row['variance_pct']
    ))

print("\n" + "="*80)
