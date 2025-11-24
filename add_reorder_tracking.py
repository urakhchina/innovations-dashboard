import pandas as pd
import json
from datetime import datetime

print("=" * 80)
print("ADDING REORDER TRACKING TO INVENTORY HEALTH DATA")
print("=" * 80)

# Load catalog to map Irwin items to iHerb part numbers
print("\nLoading catalog for SKU mapping...")
catalog_df = pd.read_csv('/Users/natasha/Downloads/iherb_catalog.csv')
catalog_df = catalog_df.rename(columns={catalog_df.columns[0]: 'PartNumber', catalog_df.columns[2]: 'IrwinItem'})

# Create mapping: Irwin item → iHerb Part Number
sku_mapping = {}
for _, row in catalog_df.iterrows():
    irwin_item = str(row['IrwinItem']).strip()
    part_number = str(row['PartNumber']).strip()
    if irwin_item and irwin_item != 'nan' and part_number and part_number != 'nan':
        sku_mapping[irwin_item] = part_number

print(f"✓ Created SKU mapping for {len(sku_mapping)} products")

# Load invoice data
print("\nLoading invoice data...")
invoices_df = pd.read_excel('/Users/natasha/Downloads/alliHerb.xlsx')
invoices_df['DocDate'] = pd.to_datetime(invoices_df['DocDate'])

print(f"✓ Loaded {len(invoices_df)} invoice records")
print(f"  Date range: {invoices_df['DocDate'].min().date()} to {invoices_df['DocDate'].max().date()}")

# Map Irwin items to iHerb part numbers
invoices_df['iHerbSKU'] = invoices_df['ItemCode'].map(sku_mapping)

# For each iHerb SKU, find the most recent order
print("\nFinding most recent orders for each product...")
latest_orders = invoices_df[invoices_df['iHerbSKU'].notna()].sort_values('DocDate', ascending=False).groupby('iHerbSKU').first().reset_index()
latest_orders = latest_orders[['iHerbSKU', 'DocDate', 'Quantity', 'ItemCode']]
latest_orders.columns = ['sku', 'last_order_date', 'last_order_qty', 'irwin_item']

# Create dictionary for quick lookup
reorder_info = {}
for _, row in latest_orders.iterrows():
    reorder_info[row['sku']] = {
        'last_order_date': row['last_order_date'].strftime('%Y-%m-%d'),
        'last_order_qty': int(row['last_order_qty']),
        'days_since_order': (datetime.now() - row['last_order_date']).days,
        'irwin_item': row['irwin_item']
    }

print(f"✓ Found reorder info for {len(reorder_info)} products")

# Load inventory management data
print("\nLoading inventory management data...")
inventory_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/inventory_management.json'
with open(inventory_file, 'r') as f:
    inventory_data = json.load(f)

# Add reorder info to each product in each month
print("\nEnhancing inventory data with reorder tracking...")
months_updated = 0
products_with_reorders = 0

for month_name, month_data in inventory_data['months'].items():
    for product in month_data['products']:
        sku = product['sku']

        # Add reorder information if available
        if sku in reorder_info:
            product['last_order_date'] = reorder_info[sku]['last_order_date']
            product['last_order_qty'] = reorder_info[sku]['last_order_qty']
            product['days_since_order'] = reorder_info[sku]['days_since_order']
            products_with_reorders += 1
        else:
            product['last_order_date'] = None
            product['last_order_qty'] = None
            product['days_since_order'] = None

    months_updated += 1

print(f"✓ Updated {months_updated} months")
print(f"✓ Added reorder info to {products_with_reorders} product records")

# Save updated data
with open(inventory_file, 'w') as f:
    json.dump(inventory_data, f, indent=2)

print(f"\n✓ Saved updated inventory data to: {inventory_file}")

# Show summary of at-risk products with recent orders
print("\n" + "=" * 80)
print("AT-RISK PRODUCTS WITH RECENT ORDERS")
print("=" * 80)

# Analyze October (most recent month)
october_data = inventory_data['months']['October']
at_risk_products = [p for p in october_data['products'] if p['atRisk']]

print(f"\nOctober: {len(at_risk_products)} products needing attention")
print(f"\nProducts with recent orders (within last 60 days):")

recent_reorders = []
for product in at_risk_products:
    if product.get('days_since_order') and product['days_since_order'] <= 60:
        recent_reorders.append(product)

recent_reorders.sort(key=lambda x: x['days_since_order'])

for i, product in enumerate(recent_reorders[:10], 1):
    print(f"\n{i}. {product['productName'][:50]}")
    print(f"   SKU: {product['sku']}")
    print(f"   Risk: {product['riskLabel']}")
    print(f"   Current Stock: {product['qtyAvailable']} units")
    print(f"   Last Reorder: {product['last_order_date']} ({product['days_since_order']} days ago)")
    print(f"   Reorder Qty: {product['last_order_qty']} units")
    print(f"   Weeks of Supply: {product['weeksOfSupply']:.1f} weeks")

# Count products at risk with no recent orders
no_recent_orders = [p for p in at_risk_products if not p.get('days_since_order') or p['days_since_order'] > 60]

print(f"\n⚠️  {len(no_recent_orders)} at-risk products with NO orders in last 60 days")
if no_recent_orders:
    print("\nMost critical (no recent reorders):")
    no_recent_orders.sort(key=lambda x: x.get('weeksOfSupply') or 999)
    for i, product in enumerate(no_recent_orders[:5], 1):
        days_text = f"{product['days_since_order']} days ago" if product.get('days_since_order') else "No orders found"
        print(f"  {i}. {product['sku']} - {product['riskLabel']} - {product['weeksOfSupply']:.1f} wks - Last order: {days_text}")

print("\n✓ Reorder tracking analysis complete!")
