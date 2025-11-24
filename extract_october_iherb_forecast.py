import pandas as pd

# Extract October 2025 forecast from Q3 2025 file (has cleanest structure)
print("Extracting October 2025 iHerb forecast...")
q3_file = '/Users/natasha/Documents/Projects/IN_POS Claude/iHerb/raw_data/forecasts/2025/Irwin Naturals_IRW_iHerb Supplier Forecast Q3 2025.xlsx'
df = pd.read_excel(q3_file, sheet_name=0, header=6)

# Get October forecast with UPC
oct_forecast = df[['UPC', 'Description', '25-Oct']].copy()
oct_forecast = oct_forecast[oct_forecast['UPC'].notna()]
oct_forecast = oct_forecast[oct_forecast['25-Oct'].notna()]
oct_forecast = oct_forecast[oct_forecast['25-Oct'] > 0]  # Filter out zeros

# Clean UPC
oct_forecast['UPC'] = oct_forecast['UPC'].astype(str).str.replace('.0', '').str.zfill(12)

# Rename columns
oct_forecast.columns = ['upc', 'product_name', 'forecast_qty']
oct_forecast['forecast_qty'] = oct_forecast['forecast_qty'].round(2)

# Sort by forecast quantity
oct_forecast = oct_forecast.sort_values('forecast_qty', ascending=False)

# Save to CSV
output_path = '/Users/natasha/Downloads/october_2025_iherb_forecast_CLEAN.csv'
oct_forecast.to_csv(output_path, index=False)

print(f"\nResults:")
print(f"  Total forecasted products: {len(oct_forecast)}")
print(f"  Total units forecasted: {oct_forecast['forecast_qty'].sum():,.2f}")
print(f"\nSaved to: {output_path}")

print(f"\nTop 20 forecasted products:")
print(oct_forecast.head(20).to_string(index=False))

print(f"\nBottom 10 forecasted products:")
print(oct_forecast.tail(10).to_string(index=False))
