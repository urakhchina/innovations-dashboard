import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("EXTRACTING Q2-Q3 2025 FORECAST VS ACTUALS (MAY-OCTOBER)")
print("=" * 80)

# ========== STEP 1: Load iHerb Catalog Mapping ==========
print("\n[1/6] Loading iHerb catalog mapping...")
catalog_df = pd.read_csv('/Users/natasha/Downloads/iherb_catalog.csv')

# Create Part Number -> Irwin item mapping
# Note: Part Number appears twice in the CSV (columns 0 and 3), use column 0
catalog_df = catalog_df.rename(columns={catalog_df.columns[0]: 'PartNumber', catalog_df.columns[2]: 'IrwinItem'})
catalog_mapping = dict(zip(catalog_df['PartNumber'].astype(str).str.strip(),
                           catalog_df['IrwinItem'].astype(str).str.strip()))

print(f"✓ Loaded {len(catalog_mapping)} Part Number -> Irwin item mappings")
print(f"  Sample: {list(catalog_mapping.items())[:3]}")

# ========== STEP 2: Load Q2 2025 Forecast (May, June, July) ==========
print("\n[2/6] Loading Q2 2025 forecast data...")
q2_file = '/Users/natasha/Documents/Projects/IN_POS Claude/iHerb/raw_data/forecasts/2025/Irwin Naturals_IRW_iHerb Supplier Forecast Q2 2025.xlsx'
q2_df = pd.read_excel(q2_file, sheet_name=0, header=6)

# Extract relevant columns - Q2 has 'Irwin Item ' column (with trailing space)
q2_forecast = q2_df[['I Herb Part #', 'Irwin Item ', 'Description', '25-May', '25-Jun', '25-Jul']].copy()
q2_forecast.columns = ['PartNumber', 'IrwinItem', 'Description', 'May', 'June', 'July']

# Clean up
q2_forecast['PartNumber'] = q2_forecast['PartNumber'].astype(str).str.strip()
q2_forecast['IrwinItem'] = q2_forecast['IrwinItem'].astype(str).str.strip()

# If IrwinItem is missing/NaN, use catalog mapping
for idx, row in q2_forecast.iterrows():
    if pd.isna(row['IrwinItem']) or row['IrwinItem'] == 'nan' or row['IrwinItem'] == '':
        part_num = row['PartNumber']
        if part_num in catalog_mapping:
            q2_forecast.at[idx, 'IrwinItem'] = catalog_mapping[part_num]

# Filter out invalid rows
q2_forecast = q2_forecast[q2_forecast['IrwinItem'].notna() &
                          (q2_forecast['IrwinItem'] != 'nan') &
                          (q2_forecast['IrwinItem'] != '')].copy()

print(f"✓ Loaded Q2 forecast: {len(q2_forecast)} products")
print(f"  May forecasts: {q2_forecast['May'].notna().sum()}")
print(f"  June forecasts: {q2_forecast['June'].notna().sum()}")
print(f"  July forecasts: {q2_forecast['July'].notna().sum()}")

# ========== STEP 3: Load Q3 2025 Forecast (Aug, Sept, Oct) ==========
print("\n[3/6] Loading Q3 2025 forecast data...")
q3_file = '/Users/natasha/Documents/Projects/IN_POS Claude/iHerb/raw_data/forecasts/2025/Irwin Naturals_IRW_iHerb Supplier Forecast Q3 2025.xlsx'
q3_df = pd.read_excel(q3_file, sheet_name=0, header=6)

# Q3 only has 'I Herb Part #', we need to map to Irwin Item using catalog
q3_forecast = q3_df[['I Herb Part #', 'Description', '25-Aug', '25-Sep', '25-Oct']].copy()
q3_forecast.columns = ['PartNumber', 'Description', 'August', 'September', 'October']

# Clean up
q3_forecast['PartNumber'] = q3_forecast['PartNumber'].astype(str).str.strip()

# Map Part Number to Irwin Item using catalog
q3_forecast['IrwinItem'] = q3_forecast['PartNumber'].map(catalog_mapping)

# Filter out rows where mapping failed
q3_forecast = q3_forecast[q3_forecast['IrwinItem'].notna()].copy()

print(f"✓ Loaded Q3 forecast: {len(q3_forecast)} products")
print(f"  August forecasts: {q3_forecast['August'].notna().sum()}")
print(f"  September forecasts: {q3_forecast['September'].notna().sum()}")
print(f"  October forecasts: {q3_forecast['October'].notna().sum()}")

# ========== STEP 4: Load Actuals from alliHerb.xlsx ==========
print("\n[4/6] Loading actuals from alliHerb.xlsx...")
actuals_df = pd.read_excel('/Users/natasha/Downloads/alliHerb.xlsx')

# Ensure DocDate is datetime
actuals_df['DocDate'] = pd.to_datetime(actuals_df['DocDate'])

# Filter for 2025 data only
actuals_df = actuals_df[actuals_df['DocDate'].dt.year == 2025].copy()

print(f"✓ Loaded {len(actuals_df)} invoice records from 2025")

# ========== STEP 5: Match Forecast to Actuals by Month ==========
print("\n[5/6] Matching forecast to actuals...")

months_data = []

# Process May-July (Q2)
for month_name, month_num in [('May', 5), ('June', 6), ('July', 7)]:
    print(f"\n--- Processing {month_name} 2025 ---")

    # Filter actuals for this month
    month_actuals = actuals_df[actuals_df['DocDate'].dt.month == month_num].copy()
    month_actuals_agg = month_actuals.groupby('ItemCode')['Quantity'].sum().reset_index()
    month_actuals_agg.columns = ['IrwinItem', 'ActualQty']

    print(f"  Actuals: {len(month_actuals_agg)} unique products, {month_actuals_agg['ActualQty'].sum():.0f} total units")

    # Get forecasts for this month
    forecast_data = q2_forecast[['IrwinItem', 'Description', month_name]].copy()
    forecast_data.columns = ['IrwinItem', 'Description', 'ForecastQty']

    # Remove zero/negative/NaN forecasts
    forecast_data = forecast_data[(forecast_data['ForecastQty'] > 0) &
                                  forecast_data['ForecastQty'].notna()].copy()

    print(f"  Forecast: {len(forecast_data)} products, {forecast_data['ForecastQty'].sum():.0f} total units")

    # Merge forecast with actuals
    merged = forecast_data.merge(month_actuals_agg, on='IrwinItem', how='left')
    merged['ActualQty'] = merged['ActualQty'].fillna(0)

    # Calculate variance
    merged['Variance'] = merged['ActualQty'] - merged['ForecastQty']
    merged['VariancePct'] = ((merged['ActualQty'] - merged['ForecastQty']) / merged['ForecastQty'] * 100).round(2)

    # Calculate accuracy (capped at 100%)
    merged['Accuracy'] = merged.apply(
        lambda row: min((row['ActualQty'] / row['ForecastQty']) * 100, 100) if row['ForecastQty'] > 0 else 0,
        axis=1
    ).round(2)

    matched_count = (merged['ActualQty'] > 0).sum()
    print(f"  Matched: {matched_count}/{len(forecast_data)} products ({matched_count/len(forecast_data)*100:.1f}%)")

    months_data.append({
        'month': month_name,
        'data': merged,
        'total_forecast': forecast_data['ForecastQty'].sum(),
        'total_actual': merged['ActualQty'].sum()
    })

# Process Aug-Oct (Q3)
for month_name, month_num in [('August', 8), ('September', 9), ('October', 10)]:
    print(f"\n--- Processing {month_name} 2025 ---")

    # Filter actuals for this month
    month_actuals = actuals_df[actuals_df['DocDate'].dt.month == month_num].copy()
    month_actuals_agg = month_actuals.groupby('ItemCode')['Quantity'].sum().reset_index()
    month_actuals_agg.columns = ['IrwinItem', 'ActualQty']

    print(f"  Actuals: {len(month_actuals_agg)} unique products, {month_actuals_agg['ActualQty'].sum():.0f} total units")

    # Get forecasts for this month
    forecast_data = q3_forecast[['IrwinItem', 'Description', month_name]].copy()
    forecast_data.columns = ['IrwinItem', 'Description', 'ForecastQty']

    # Remove zero/negative/NaN forecasts
    forecast_data = forecast_data[(forecast_data['ForecastQty'] > 0) &
                                  forecast_data['ForecastQty'].notna()].copy()

    print(f"  Forecast: {len(forecast_data)} products, {forecast_data['ForecastQty'].sum():.0f} total units")

    # Merge forecast with actuals
    merged = forecast_data.merge(month_actuals_agg, on='IrwinItem', how='left')
    merged['ActualQty'] = merged['ActualQty'].fillna(0)

    # Calculate variance
    merged['Variance'] = merged['ActualQty'] - merged['ForecastQty']
    merged['VariancePct'] = ((merged['ActualQty'] - merged['ForecastQty']) / merged['ForecastQty'] * 100).round(2)

    # Calculate accuracy (capped at 100%)
    merged['Accuracy'] = merged.apply(
        lambda row: min((row['ActualQty'] / row['ForecastQty']) * 100, 100) if row['ForecastQty'] > 0 else 0,
        axis=1
    ).round(2)

    matched_count = (merged['ActualQty'] > 0).sum()
    print(f"  Matched: {matched_count}/{len(forecast_data)} products ({matched_count/len(forecast_data)*100:.1f}%)")

    months_data.append({
        'month': month_name,
        'data': merged,
        'total_forecast': forecast_data['ForecastQty'].sum(),
        'total_actual': merged['ActualQty'].sum()
    })

# ========== STEP 6: Save Results ==========
print("\n[6/6] Saving results...")

# Save detailed CSV for each month
for month_info in months_data:
    month = month_info['month']
    data = month_info['data']

    output_file = f'/Users/natasha/Documents/Projects/IN_Reports/Innovations/q2_q3_forecast_vs_actuals_{month.lower()}_2025.csv'
    data.to_csv(output_file, index=False)
    print(f"✓ Saved {month}: {output_file}")

print("\n" + "=" * 80)
print("SUMMARY BY MONTH")
print("=" * 80)

for month_info in months_data:
    month = month_info['month']
    total_forecast = month_info['total_forecast']
    total_actual = month_info['total_actual']
    variance = total_actual - total_forecast
    variance_pct = (variance / total_forecast * 100) if total_forecast > 0 else 0

    print(f"\n{month} 2025:")
    print(f"  Forecast: {total_forecast:,.0f} units")
    print(f"  Actual:   {total_actual:,.0f} units")
    print(f"  Variance: {variance:+,.0f} units ({variance_pct:+.1f}%)")

print("\n✓ Extraction complete!")
