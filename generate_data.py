import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

np.random.seed(42)

NUM_ORDERS = 10000


# ============================================================
# CUSTOMERS
# ============================================================

customer_ids = [
    f"CUST{str(i).zfill(4)}"
    for i in range(1, 1501)
]


# ============================================================
# PRODUCTS
# ============================================================

products = {

    "Electronics": [
        ("Wireless Mouse", 699),
        ("Mechanical Keyboard", 2499),
        ("Bluetooth Headphones", 1799),
        ("USB-C Hub", 1299),
        ("Smart Watch", 3499)
    ],

    "Clothing": [
        ("T-Shirt", 599),
        ("Jeans", 1499),
        ("Hoodie", 1799),
        ("Sneakers", 2499),
        ("Jacket", 2999)
    ],

    "Home & Kitchen": [
        ("Coffee Maker", 2499),
        ("Air Fryer", 4999),
        ("Water Bottle", 799),
        ("Cookware Set", 3499),
        ("Table Lamp", 1299)
    ],

    "Beauty": [
        ("Face Wash", 399),
        ("Moisturizer", 599),
        ("Perfume", 1499),
        ("Sunscreen", 699),
        ("Hair Serum", 549)
    ],

    "Books": [
        ("Python Programming", 799),
        ("Data Science Handbook", 999),
        ("Atomic Habits", 599),
        ("Machine Learning Guide", 899),
        ("Business Analytics", 749)
    ]
}


# ============================================================
# LOCATIONS
# ============================================================

locations = {

    "East": [
        "Kolkata",
        "Bhubaneswar",
        "Guwahati",
        "Patna"
    ],

    "West": [
        "Mumbai",
        "Pune",
        "Ahmedabad",
        "Surat"
    ],

    "North": [
        "Delhi",
        "Jaipur",
        "Lucknow",
        "Chandigarh"
    ],

    "South": [
        "Bangalore",
        "Chennai",
        "Hyderabad",
        "Kochi"
    ]
}


# ============================================================
# PAYMENT METHODS
# ============================================================

payment_methods = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Cash on Delivery"
]


# ============================================================
# ORDER STATUS
# ============================================================

order_statuses = [
    "Completed",
    "Completed",
    "Completed",
    "Completed",
    "Cancelled"
]


# ============================================================
# GENERATE ORDERS
# ============================================================

data = []


for i in range(NUM_ORDERS):

    # Order ID
    order_id = f"ORD{10001 + i}"


    # Random customer
    customer_id = np.random.choice(customer_ids)


    # Random region
    region = np.random.choice(
        list(locations.keys()),
        p=[0.25, 0.25, 0.30, 0.20]
    )


    # Random city from selected region
    city = np.random.choice(
        locations[region]
    )


    # Random category
    category = np.random.choice(
        list(products.keys()),
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )


    # Random product
    product_index = np.random.randint(
        len(products[category])
    )

    product, unit_price = products[category][product_index]


    # Quantity
    quantity = np.random.randint(1, 6)


    # Discount
    discount = np.random.choice(
        [0, 0.05, 0.10, 0.15, 0.20],
        p=[0.20, 0.25, 0.30, 0.15, 0.10]
    )


    # Revenue
    revenue = (
        quantity
        * unit_price
        * (1 - discount)
    )


    # Profit
    profit_margin = np.random.uniform(
        0.08,
        0.30
    )

    profit = revenue * profit_margin


    # Order date
    random_days = np.random.randint(0, 365)

    order_date = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(
            random_days,
            unit="D"
        )
    )


    # Payment method
    payment_method = np.random.choice(
        payment_methods
    )


    # Order status
    order_status = np.random.choice(
        order_statuses
    )


    # Add row
    data.append([
        order_id,
        order_date,
        customer_id,
        product,
        category,
        quantity,
        unit_price,
        discount,
        round(revenue, 2),
        round(profit, 2),
        region,
        city,
        payment_method,
        order_status
    ])


# ============================================================
# CREATE DATAFRAME
# ============================================================

columns = [
    "Order_ID",
    "Order_Date",
    "Customer_ID",
    "Product",
    "Category",
    "Quantity",
    "Unit_Price",
    "Discount",
    "Revenue",
    "Profit",
    "Region",
    "City",
    "Payment_Method",
    "Order_Status"
]


df = pd.DataFrame(
    data,
    columns=columns
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    "ecommerce_sales.csv",
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("E-COMMERCE DATASET GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"Customers  : {df['Customer_ID'].nunique():,}")
print(f"Products   : {df['Product'].nunique():,}")
print(f"Categories : {df['Category'].nunique()}")

print(
    f"Revenue    : ₹{df['Revenue'].sum():,.2f}"
)

print(
    f"Profit     : ₹{df['Profit'].sum():,.2f}"
)

print("\nSaved as:")
print("ecommerce_sales.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("=" * 60)