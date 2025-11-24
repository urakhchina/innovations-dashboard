import pandas as pd
import numpy as np

print("=" * 80)
print("EXTRACTING JANUARY 2025 FORECAST FROM Q4 2024 FILE")
print("=" * 80)

# Load Q4 2024 forecast
q4_2024_file = '/Users/natasha/Documents/Projects/IN_POS Claude/iHerb/raw_data/forecasts/2024/Irwin Naturals_IRW_iHerb Supplier Forecast Q4 2024.xlsx'
df = pd.read_excel(q4_2024_file, sheet_name=0, header=6)

print(f"\nLoaded Q4 2024 file with {len(df.columns)} columns")

# Print column names with indices
print("\nColumn structure:")
for i, col in enumerate(df.columns[:20]):  # First 20 columns
    print(f"  {i}: {col}")

# Look for January 2025 column
# Based on the pattern, datetime(2023, 1, 25) likely represents Jan 2025 (25-Jan)
jan_2025_col = None
for col in df.columns:
    if isinstance(col, pd.Timestamp):
        # Check if this is January (month 1)
        if col.month == 1:
            print(f"\nFound potential January column: {col}")
            jan_2025_col = col
            break

if jan_2025_col is None:
    # Try looking for string that contains 'Jan' or '25-Jan' or similar
    for col in df.columns:
        col_str = str(col)
        if 'jan' in col_str.lower() or '01-25' in col_str or '25-01' in col_str:
            print(f"\nFound potential January column: {col}")
            jan_2025_col = col
            break

if jan_2025_col:
    print(f"\nUsing column: {jan_2025_col}")

    # Extract January forecast data
    forecast_data = df[['I Herb Part #', 'Description', jan_2025_col]].copy()
    forecast_data.columns = ['PartNumber', 'Description', 'January']

    # Clean up
    forecast_data['PartNumber'] = forecast_data['PartNumber'].astype(str).str.strip()

    # Load catalog mapping
    print("\nLoading catalog mapping...")
    catalog_df = pd.read_csv('/Users/natasha/Downloads/iherb_catalog.csv')
    catalog_df = catalog_df.rename(columns={catalog_df.columns[0]: 'PartNumber', catalog_df.columns[2]: 'IrwinItem'})
    catalog_mapping = dict(zip(catalog_df['PartNumber'].astype(str).str.strip(),
                               catalog_df['IrwinItem'].astype(str).str.strip()))

    # Map Part Number to Irwin Item
    forecast_data['IrwinItem'] = forecast_data['PartNumber'].map(catalog_mapping)

    # Filter out rows where mapping failed or forecast is 0/negative/NaN
    forecast_data = forecast_data[
        forecast_data['IrwinItem'].notna() &
        (forecast_data['January'] > 0) &
        forecast_data['January'].notna()
    ].copy()

    print(f"✓ Extracted {len(forecast_data)} products with January 2025 forecasts")
    print(f"  Total forecast: {forecast_data['January'].sum():,.0f} units")

    # Load actuals from alliHerb.xlsx
    print("\nLoading January 2025 actuals from alliHerb.xlsx...")
    actuals_df = pd.read_excel('/Users/natasha/Downloads/alliHerb.xlsx')
    actuals_df['DocDate'] = pd.to_datetime(actuals_df['DocDate'])

    # Filter for January 2025
    jan_actuals = actuals_df[(actuals_df['DocDate'].dt.month == 1) &
                             (actuals_df['DocDate'].dt.year == 2025)].copy()
    jan_actuals_agg = jan_actuals.groupby('ItemCode')['Quantity'].sum().reset_index()
    jan_actuals_agg.columns = ['IrwinItem', 'ActualQty']

    print(f"✓ Loaded {len(jan_actuals_agg)} products with actuals")
    print(f"  Total actual: {jan_actuals_agg['ActualQty'].sum():,.0f} units")

    # Merge forecast with actuals
    merged = forecast_data.merge(jan_actuals_agg, on='IrwinItem', how='left')
    merged['ActualQty'] = merged['ActualQty'].fillna(0)

    # Calculate variance
    merged['Variance'] = merged['ActualQty'] - merged['January']
    merged['VariancePct'] = ((merged['ActualQty'] - merged['January']) / merged['January'] * 100).round(2)

    # Calculate accuracy (capped at 100%)
    merged['Accuracy'] = merged.apply(
        lambda row: min((row['ActualQty'] / row['January']) * 100, 100) if row['January'] > 0 else 0,
        axis=1
    ).round(2)

    # Rename columns for consistency
    merged = merged.rename(columns={'January': 'ForecastQty'})

    matched_count = (merged['ActualQty'] > 0).sum()
    print(f"\n✓ Matched: {matched_count}/{len(merged)} products ({matched_count/len(merged)*100:.1f}%)")

    total_forecast = merged['ForecastQty'].sum()
    total_actual = merged['ActualQty'].sum()
    variance = total_actual - total_forecast
    variance_pct = (variance / total_forecast * 100) if total_forecast > 0 else 0

    print(f"\nJanuary 2025 Summary:")
    print(f"  Forecast: {total_forecast:,.0f} units")
    print(f"  Actual:   {total_actual:,.0f} units")
    print(f"  Variance: {variance:+,.0f} units ({variance_pct:+.1f}%)")

    # Save results
    output_file = '/Users/natasha/Documents/Projects/IN_Reports/Innovations/january_2025_forecast_vs_actuals.csv'
    merged.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")

else:
    print("\n❌ Could not find January 2025 column in Q4 2024 file")
    print("\nShowing first 10 column names for reference:")
    for i, col in enumerate(df.columns[:10]):
        print(f"  {i}: {col}")
