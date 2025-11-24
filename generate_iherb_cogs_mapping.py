import pandas as pd
import json

print("Generating iHerb COGS mapping from alliHerb.xlsx...")

# Load alliHerb data
df = pd.read_excel('/Users/natasha/Downloads/alliHerb.xlsx')

print(f"Total transactions: {len(df)}")

# Calculate per-unit COGS for each transaction
df['cogs_per_unit'] = df.apply(
    lambda row: row['COGS'] / row['Quantity'] if row['Quantity'] > 0 else 0,
    axis=1
)

# Filter out zero/negative quantities and zero COGS
df_valid = df[(df['Quantity'] > 0) & (df['COGS'] > 0)].copy()

print(f"Valid transactions (qty > 0, COGS > 0): {len(df_valid)}")

# Group by ItemCode and calculate average COGS per unit
cogs_mapping = df_valid.groupby('ItemCode').agg({
    'cogs_per_unit': 'mean',
    'Dscription': 'first',
    'Quantity': 'sum',
    'COGS': 'sum'
}).reset_index()

# Calculate total average COGS as well (to verify)
cogs_mapping['total_avg_cogs'] = cogs_mapping['COGS'] / cogs_mapping['Quantity']

# Create the mapping dictionary
cogs_dict = {}
for _, row in cogs_mapping.iterrows():
    item_code = row['ItemCode']
    cogs_dict[item_code] = {
        'itemCode': item_code,
        'productName': row['Dscription'],
        'avgCogsPerUnit': round(float(row['cogs_per_unit']), 2),
        'totalQuantity': int(row['Quantity']),
        'totalCOGS': round(float(row['COGS']), 2)
    }

# Save to JSON
output_path = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/iherb_cogs_mapping.json'
with open(output_path, 'w') as f:
    json.dump(cogs_dict, f, indent=2)

print(f"\nSaved COGS mapping to: {output_path}")
print(f"Total products with COGS: {len(cogs_dict)}")

# Show sample
print("\nSample COGS data:")
sample_items = list(cogs_dict.items())[:10]
for item_code, data in sample_items:
    print(f"\n{item_code}: {data['productName']}")
    print(f"  Avg COGS/unit: ${data['avgCogsPerUnit']:.2f}")
    print(f"  Total qty: {data['totalQuantity']:,}")
