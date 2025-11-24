import pandas as pd
import json
import argparse
from datetime import datetime

print("=" * 80)
print("MONTH-OVER-MONTH INVENTORY COMPARISON")
print("=" * 80)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare inventory health between two months')
parser.add_argument('--prev', type=str, required=True, help='Previous month name (e.g., October)')
parser.add_argument('--current', type=str, required=True, help='Current month name (e.g., November)')
args = parser.parse_args()

prev_month = args.prev
current_month = args.current

print(f"\nComparing: {prev_month} → {current_month}")
print("=" * 80)

# Load inventory management data
print("\nLoading inventory data...")
inventory_file = '/Users/natasha/Documents/Projects/IN_POS claude/iherb-dashboard/public/inventory_management.json'
with open(inventory_file, 'r') as f:
    inventory_data = json.load(f)

# Check if both months exist
available_months = list(inventory_data['months'].keys())
if prev_month not in available_months:
    print(f"❌ Error: {prev_month} not found in inventory data")
    print(f"   Available months: {', '.join(available_months)}")
    exit(1)

if current_month not in available_months:
    print(f"❌ Error: {current_month} not found in inventory data")
    print(f"   Available months: {', '.join(available_months)}")
    exit(1)

prev_data = inventory_data['months'][prev_month]
current_data = inventory_data['months'][current_month]

print(f"✓ Loaded {len(prev_data['products'])} products from {prev_month}")
print(f"✓ Loaded {len(current_data['products'])} products from {current_month}")

# Load catalog for SKU mapping
print("\nLoading catalog for SKU mapping...")
catalog_df = pd.read_csv('/Users/natasha/Downloads/iherb_catalog.csv')
catalog_df = catalog_df.rename(columns={catalog_df.columns[0]: 'PartNumber', catalog_df.columns[2]: 'IrwinItem'})

sku_mapping = {}
for _, row in catalog_df.iterrows():
    irwin_item = str(row['IrwinItem']).strip()
    part_number = str(row['PartNumber']).strip()
    if irwin_item and irwin_item != 'nan' and part_number and part_number != 'nan':
        sku_mapping[irwin_item] = part_number

print(f"✓ Created SKU mapping for {len(sku_mapping)} products")

# Load invoice data to see what was ordered
print("\nLoading invoice data...")
invoices_df = pd.read_excel('/Users/natasha/Downloads/alliHerb.xlsx')
invoices_df['DocDate'] = pd.to_datetime(invoices_df['DocDate'])

# Get the month number for filtering
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

current_month_num = month_map.get(current_month)
if current_month_num:
    # Filter invoices for the current month
    current_month_invoices = invoices_df[
        (invoices_df['DocDate'].dt.month == current_month_num) &
        (invoices_df['DocDate'].dt.year == 2025)
    ].copy()

    # Map to iHerb SKUs
    current_month_invoices['iHerbSKU'] = current_month_invoices['ItemCode'].map(sku_mapping)

    # Group by SKU to get total ordered
    orders_summary = current_month_invoices.groupby('iHerbSKU').agg({
        'Quantity': 'sum',
        'DocDate': 'max'
    }).reset_index()
    orders_summary.columns = ['sku', 'total_ordered', 'latest_order_date']

    orders_dict = {}
    for _, row in orders_summary.iterrows():
        orders_dict[row['sku']] = {
            'quantity': int(row['total_ordered']),
            'date': row['latest_order_date'].strftime('%Y-%m-%d')
        }

    print(f"✓ Found {len(orders_dict)} products ordered in {current_month}")
else:
    orders_dict = {}
    print(f"⚠️  Could not filter invoices for {current_month}")

# Create dictionaries for easy lookup
prev_products = {p['sku']: p for p in prev_data['products']}
current_products = {p['sku']: p for p in current_data['products']}

# Analysis categories
at_risk_reordered = []
oos_recovered = []
new_at_risk = []
new_oos = []
persistent_at_risk = []
persistent_oos = []
improved_products = []

# Compare products
all_skus = set(prev_products.keys()) | set(current_products.keys())

for sku in all_skus:
    prev = prev_products.get(sku)
    curr = current_products.get(sku)

    # Skip if product doesn't exist in both months
    if not prev or not curr:
        continue

    prev_at_risk = prev.get('atRisk', False)
    curr_at_risk = curr.get('atRisk', False)
    prev_qty = prev.get('qtyAvailable', 0)
    curr_qty = curr.get('qtyAvailable', 0)

    ordered_in_current = sku in orders_dict

    # 1. At-Risk Products That Were Reordered
    if prev_at_risk and ordered_in_current:
        at_risk_reordered.append({
            'sku': sku,
            'product_name': curr['productName'],
            'prev_qty': prev_qty,
            'curr_qty': curr_qty,
            'prev_weeks_supply': prev.get('weeksOfSupply', 0),
            'curr_weeks_supply': curr.get('weeksOfSupply', 0),
            'order_qty': orders_dict[sku]['quantity'],
            'order_date': orders_dict[sku]['date'],
            'still_at_risk': curr_at_risk,
            'status_change': 'Resolved' if not curr_at_risk else 'Still At-Risk'
        })

    # 2. OOS Products That Recovered
    if prev_qty == 0 and curr_qty > 0:
        oos_recovered.append({
            'sku': sku,
            'product_name': curr['productName'],
            'curr_qty': curr_qty,
            'curr_weeks_supply': curr.get('weeksOfSupply', 0),
            'was_ordered': ordered_in_current,
            'order_qty': orders_dict[sku]['quantity'] if ordered_in_current else None,
            'order_date': orders_dict[sku]['date'] if ordered_in_current else None
        })

    # 3. New Problems
    if not prev_at_risk and curr_at_risk:
        new_at_risk.append({
            'sku': sku,
            'product_name': curr['productName'],
            'prev_qty': prev_qty,
            'curr_qty': curr_qty,
            'prev_weeks_supply': prev.get('weeksOfSupply', 0),
            'curr_weeks_supply': curr.get('weeksOfSupply', 0),
            'risk_label': curr.get('riskLabel', 'Unknown')
        })

    if prev_qty > 0 and curr_qty == 0:
        new_oos.append({
            'sku': sku,
            'product_name': curr['productName'],
            'prev_qty': prev_qty,
            'prev_weeks_supply': prev.get('weeksOfSupply', 0),
            'velocity_monthly': curr.get('velocityMonthly', 0)
        })

    # 4. Persistent Issues
    if prev_at_risk and curr_at_risk and not ordered_in_current:
        persistent_at_risk.append({
            'sku': sku,
            'product_name': curr['productName'],
            'prev_qty': prev_qty,
            'curr_qty': curr_qty,
            'prev_weeks_supply': prev.get('weeksOfSupply', 0),
            'curr_weeks_supply': curr.get('weeksOfSupply', 0),
            'risk_label': curr.get('riskLabel', 'Unknown'),
            'change': curr_qty - prev_qty
        })

    if prev_qty == 0 and curr_qty == 0:
        persistent_oos.append({
            'sku': sku,
            'product_name': curr['productName'],
            'velocity_monthly': curr.get('velocityMonthly', 0),
            'was_ordered': ordered_in_current,
            'order_qty': orders_dict[sku]['quantity'] if ordered_in_current else None
        })

    # 5. Improved Products
    if prev_at_risk and not curr_at_risk:
        improved_products.append({
            'sku': sku,
            'product_name': curr['productName'],
            'prev_qty': prev_qty,
            'curr_qty': curr_qty,
            'prev_weeks_supply': prev.get('weeksOfSupply', 0),
            'curr_weeks_supply': curr.get('weeksOfSupply', 0),
            'was_ordered': ordered_in_current,
            'order_qty': orders_dict[sku]['quantity'] if ordered_in_current else None
        })

# Print Results
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

print(f"\n📊 SUMMARY")
print(f"   At-Risk Products Reordered: {len(at_risk_reordered)}")
print(f"   OOS Products Recovered: {len(oos_recovered)}")
print(f"   New At-Risk Products: {len(new_at_risk)}")
print(f"   New OOS Products: {len(new_oos)}")
print(f"   Persistent At-Risk (No Orders): {len(persistent_at_risk)}")
print(f"   Persistent OOS: {len(persistent_oos)}")
print(f"   Improved Products: {len(improved_products)}")

# 1. At-Risk Products That Were Reordered
if at_risk_reordered:
    print("\n" + "=" * 80)
    print(f"✅ AT-RISK PRODUCTS THAT WERE REORDERED ({len(at_risk_reordered)} products)")
    print("=" * 80)

    # Sort by status - still at-risk first
    at_risk_reordered.sort(key=lambda x: (not x['still_at_risk'], x['curr_weeks_supply'] if x['curr_weeks_supply'] is not None else 0))

    for i, product in enumerate(at_risk_reordered[:20], 1):
        status_emoji = "⚠️" if product['still_at_risk'] else "✓"
        prev_ws = product['prev_weeks_supply'] if product['prev_weeks_supply'] is not None else 0
        curr_ws = product['curr_weeks_supply'] if product['curr_weeks_supply'] is not None else 0
        print(f"\n{i}. {status_emoji} {product['product_name'][:60]}")
        print(f"   SKU: {product['sku']}")
        print(f"   Inventory Change: {product['prev_qty']} → {product['curr_qty']} units ({product['curr_qty'] - product['prev_qty']:+d})")
        print(f"   Weeks Supply: {prev_ws:.1f} → {curr_ws:.1f} weeks")
        print(f"   Order: {product['order_qty']} units on {product['order_date']}")
        print(f"   Status: {product['status_change']}")

# 2. OOS Products That Recovered
if oos_recovered:
    print("\n" + "=" * 80)
    print(f"🎉 OOS PRODUCTS THAT RECOVERED ({len(oos_recovered)} products)")
    print("=" * 80)

    for i, product in enumerate(oos_recovered, 1):
        order_info = f"Ordered {product['order_qty']} units on {product['order_date']}" if product['was_ordered'] else "No order found"
        weeks_supply = product['curr_weeks_supply'] if product['curr_weeks_supply'] is not None else 0
        print(f"\n{i}. {product['product_name'][:60]}")
        print(f"   SKU: {product['sku']}")
        print(f"   Current Stock: {product['curr_qty']} units ({weeks_supply:.1f} weeks)")
        print(f"   {order_info}")

# 3. New Problems
if new_at_risk or new_oos:
    print("\n" + "=" * 80)
    print(f"⚠️  NEW PROBLEMS")
    print("=" * 80)

    if new_at_risk:
        print(f"\nNew At-Risk Products ({len(new_at_risk)} products):")
        new_at_risk.sort(key=lambda x: x['curr_weeks_supply'] if x['curr_weeks_supply'] is not None else 0)
        for i, product in enumerate(new_at_risk[:10], 1):
            prev_ws = product['prev_weeks_supply'] if product['prev_weeks_supply'] is not None else 0
            curr_ws = product['curr_weeks_supply'] if product['curr_weeks_supply'] is not None else 0
            print(f"  {i}. {product['sku']} - {product['risk_label']}")
            print(f"     Weeks Supply: {prev_ws:.1f} → {curr_ws:.1f} weeks")

    if new_oos:
        print(f"\nNew OOS Products ({len(new_oos)} products):")
        for i, product in enumerate(new_oos, 1):
            prev_ws = product['prev_weeks_supply'] if product['prev_weeks_supply'] is not None else 0
            print(f"  {i}. {product['sku']} - {product['product_name'][:50]}")
            print(f"     Was: {product['prev_qty']} units ({prev_ws:.1f} weeks)")

# 4. Persistent Issues
if persistent_at_risk or persistent_oos:
    print("\n" + "=" * 80)
    print(f"🚨 PERSISTENT ISSUES (NEED IMMEDIATE ATTENTION)")
    print("=" * 80)

    if persistent_at_risk:
        print(f"\nStill At-Risk, NO Orders Placed ({len(persistent_at_risk)} products):")
        persistent_at_risk.sort(key=lambda x: x['curr_weeks_supply'] if x['curr_weeks_supply'] is not None else 0)
        for i, product in enumerate(persistent_at_risk[:10], 1):
            curr_ws = product['curr_weeks_supply'] if product['curr_weeks_supply'] is not None else 0
            print(f"\n  {i}. {product['product_name'][:60]}")
            print(f"     SKU: {product['sku']} - {product['risk_label']}")
            print(f"     Stock Change: {product['prev_qty']} → {product['curr_qty']} ({product['change']:+d})")
            print(f"     Weeks Supply: {curr_ws:.1f} weeks")

    if persistent_oos:
        print(f"\nStill OOS ({len(persistent_oos)} products):")
        for i, product in enumerate(persistent_oos[:10], 1):
            order_status = f"Ordered {product['order_qty']} (in transit?)" if product['was_ordered'] else "NO ORDER FOUND"
            print(f"  {i}. {product['sku']} - {product['product_name'][:50]}")
            print(f"     Status: {order_status}")

# 5. Improved Products
if improved_products:
    print("\n" + "=" * 80)
    print(f"✨ IMPROVED PRODUCTS ({len(improved_products)} products)")
    print("=" * 80)

    for i, product in enumerate(improved_products[:10], 1):
        order_info = f"Order: {product['order_qty']} units" if product['was_ordered'] else "Improved naturally"
        prev_ws = product['prev_weeks_supply'] if product['prev_weeks_supply'] is not None else 0
        curr_ws = product['curr_weeks_supply'] if product['curr_weeks_supply'] is not None else 0
        print(f"\n{i}. {product['product_name'][:60]}")
        print(f"   SKU: {product['sku']}")
        print(f"   Stock: {product['prev_qty']} → {product['curr_qty']} units")
        print(f"   Weeks Supply: {prev_ws:.1f} → {curr_ws:.1f} weeks")
        print(f"   {order_info}")

# Save detailed report as JSON
output_data = {
    'comparison': {
        'prev_month': prev_month,
        'current_month': current_month,
        'generated_at': datetime.now().isoformat()
    },
    'summary': {
        'at_risk_reordered': len(at_risk_reordered),
        'oos_recovered': len(oos_recovered),
        'new_at_risk': len(new_at_risk),
        'new_oos': len(new_oos),
        'persistent_at_risk': len(persistent_at_risk),
        'persistent_oos': len(persistent_oos),
        'improved': len(improved_products)
    },
    'details': {
        'at_risk_reordered': at_risk_reordered,
        'oos_recovered': oos_recovered,
        'new_at_risk': new_at_risk,
        'new_oos': new_oos,
        'persistent_at_risk': persistent_at_risk,
        'persistent_oos': persistent_oos,
        'improved': improved_products
    }
}

output_file = f'/Users/natasha/Documents/Projects/IN_Reports/Innovations/month_comparison_{prev_month}_{current_month}.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Saved detailed comparison to: {output_file}")
print("\n" + "=" * 80)
print("✓ COMPARISON COMPLETE")
print("=" * 80)
