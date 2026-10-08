import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("E-COMMERCE SALES TREND ANALYSIS")
print("=" * 70)

df = pd.read_csv("ecommerce_sales.csv")

print("\nDataset loaded successfully.")

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 2. DATA CLEANING & VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

# Convert date column
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Check missing values
missing_values = df.isnull().sum().sum()

# Check duplicate rows
duplicate_rows = df.duplicated().sum()

print(f"\nMissing values : {missing_values}")
print(f"Duplicate rows : {duplicate_rows}")

if missing_values == 0:
    print("✓ No missing values found.")

if duplicate_rows == 0:
    print("✓ No duplicate rows found.")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Date features
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Quarter"] = df["Order_Date"].dt.quarter
df["Day"] = df["Order_Date"].dt.day
df["Day_Name"] = df["Order_Date"].dt.day_name()

# Profit margin
df["Profit_Margin"] = (
    df["Profit"] / df["Revenue"]
) * 100

print("\nCreated features:")
print("• Year")
print("• Month")
print("• Month_Name")
print("• Quarter")
print("• Day")
print("• Day_Name")
print("• Profit_Margin")


# ============================================================
# 4. BASIC BUSINESS KPIs
# ============================================================

print("\n" + "=" * 70)
print("KEY BUSINESS KPIs")
print("=" * 70)

# Only completed orders are considered for sales performance
completed_df = df[df["Order_Status"] == "Completed"].copy()

total_orders = len(completed_df)
total_revenue = completed_df["Revenue"].sum()
total_profit = completed_df["Profit"].sum()
total_customers = completed_df["Customer_ID"].nunique()
total_products = completed_df["Product"].nunique()
total_quantity = completed_df["Quantity"].sum()

average_order_value = total_revenue / total_orders
average_profit = total_profit / total_orders
overall_profit_margin = (
    total_profit / total_revenue
) * 100

cancellation_rate = (
    (df["Order_Status"] == "Cancelled").sum()
    / len(df)
) * 100

print(f"\nTotal Completed Orders : {total_orders:,}")
print(f"Total Revenue         : ₹{total_revenue:,.2f}")
print(f"Total Profit          : ₹{total_profit:,.2f}")
print(f"Total Customers       : {total_customers:,}")
print(f"Total Products        : {total_products:,}")
print(f"Total Quantity Sold   : {total_quantity:,}")
print(f"Average Order Value   : ₹{average_order_value:,.2f}")
print(f"Average Profit/Order  : ₹{average_profit:,.2f}")
print(f"Profit Margin         : {overall_profit_margin:.2f}%")
print(f"Cancellation Rate     : {cancellation_rate:.2f}%")


# ============================================================
# 5. MONTHLY SALES TREND
# ============================================================

print("\n" + "=" * 70)
print("MONTHLY SALES TREND")
print("=" * 70)

monthly_sales = (
    completed_df
    .groupby(["Year", "Month", "Month_Name"], sort=False)
    .agg(
        Orders=("Order_ID", "count"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    .reset_index()
)

# Sort chronologically
monthly_sales = monthly_sales.sort_values(
    ["Year", "Month"]
)

print("\nMonthly sales:")
print(
    monthly_sales[
        [
            "Month_Name",
            "Orders",
            "Revenue",
            "Profit"
        ]
    ].to_string(index=False)
)

# Best month
best_month = monthly_sales.loc[
    monthly_sales["Revenue"].idxmax()
]

# Lowest month
lowest_month = monthly_sales.loc[
    monthly_sales["Revenue"].idxmin()
]

print(
    f"\nBest month: {best_month['Month_Name']} "
    f"with revenue ₹{best_month['Revenue']:,.2f}"
)

print(
    f"Lowest month: {lowest_month['Month_Name']} "
    f"with revenue ₹{lowest_month['Revenue']:,.2f}"
)


# ============================================================
# 6. QUARTERLY SALES TREND
# ============================================================

print("\n" + "=" * 70)
print("QUARTERLY SALES TREND")
print("=" * 70)

quarterly_sales = (
    completed_df
    .groupby("Quarter")
    .agg(
        Orders=("Order_ID", "count"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    .reset_index()
)

print("\nQuarterly sales:")
print(quarterly_sales.to_string(index=False))

best_quarter = quarterly_sales.loc[
    quarterly_sales["Revenue"].idxmax()
]

print(
    f"\nBest quarter: Q{int(best_quarter['Quarter'])} "
    f"with revenue ₹{best_quarter['Revenue']:,.2f}"
)


# ============================================================
# 7. CATEGORY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY PERFORMANCE")
print("=" * 70)

category_sales = (
    completed_df
    .groupby("Category")
    .agg(
        Orders=("Order_ID", "count"),
        Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

category_sales["Profit_Margin"] = (
    category_sales["Profit"]
    / category_sales["Revenue"]
) * 100

category_sales = category_sales.sort_values(
    "Revenue",
    ascending=False
)

print("\nCategory performance:")
print(category_sales.to_string(index=False))

top_category = category_sales.iloc[0]

print(
    f"\nTop category: {top_category['Category']} "
    f"with revenue ₹{top_category['Revenue']:,.2f}"
)


# ============================================================
# 8. REGIONAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("REGIONAL PERFORMANCE")
print("=" * 70)

region_sales = (
    completed_df
    .groupby("Region")
    .agg(
        Orders=("Order_ID", "count"),
        Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

region_sales["Profit_Margin"] = (
    region_sales["Profit"]
    / region_sales["Revenue"]
) * 100

region_sales = region_sales.sort_values(
    "Revenue",
    ascending=False
)

print("\nRegional performance:")
print(region_sales.to_string(index=False))

top_region = region_sales.iloc[0]

print(
    f"\nTop region: {top_region['Region']} "
    f"with revenue ₹{top_region['Revenue']:,.2f}"
)


# ============================================================
# 9. CITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CITY PERFORMANCE")
print("=" * 70)

city_sales = (
    completed_df
    .groupby("City")
    .agg(
        Orders=("Order_ID", "count"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
    .sort_values("Revenue", ascending=False)
)

print("\nTop 10 cities by revenue:")
print(city_sales.head(10).to_string(index=False))


# ============================================================
# 10. TOP PRODUCTS
# ============================================================

print("\n" + "=" * 70)
print("TOP PRODUCTS")
print("=" * 70)

product_sales = (
    completed_df
    .groupby(["Product", "Category"])
    .agg(
        Orders=("Order_ID", "count"),
        Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
    .sort_values("Revenue", ascending=False)
)

print("\nTop 10 products by revenue:")
print(
    product_sales.head(10).to_string(index=False)
)

top_product = product_sales.iloc[0]

print(
    f"\nBest-selling product by revenue: "
    f"{top_product['Product']} "
    f"with revenue ₹{top_product['Revenue']:,.2f}"
)


# ============================================================
# 11. CUSTOMER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER ANALYSIS")
print("=" * 70)

customer_sales = (
    completed_df
    .groupby("Customer_ID")
    .agg(
        Orders=("Order_ID", "count"),
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum"),
        Last_Order_Date=("Order_Date", "max")
    )
    .reset_index()
)

customer_sales["Average_Order_Value"] = (
    customer_sales["Total_Revenue"]
    / customer_sales["Orders"]
)

customer_sales = customer_sales.sort_values(
    "Total_Revenue",
    ascending=False
)

print("\nTop 10 customers by revenue:")
print(
    customer_sales.head(10).to_string(index=False)
)


# ============================================================
# 12. CUSTOMER SEGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER SEGMENTATION")
print("=" * 70)

# Calculate frequency and monetary value
customer_sales["Frequency"] = customer_sales["Orders"]
customer_sales["Monetary"] = customer_sales["Total_Revenue"]

# Create simple segments based on customer behavior
frequency_median = customer_sales["Frequency"].median()
monetary_median = customer_sales["Monetary"].median()

def classify_customer(row):

    if (
        row["Frequency"] >= frequency_median
        and row["Monetary"] >= monetary_median
    ):
        return "High Value"

    elif (
        row["Frequency"] >= frequency_median
        and row["Monetary"] < monetary_median
    ):
        return "Loyal Budget"

    elif (
        row["Frequency"] < frequency_median
        and row["Monetary"] >= monetary_median
    ):
        return "High Spending Occasional"

    else:
        return "Low Engagement"


customer_sales["Customer_Segment"] = (
    customer_sales.apply(
        classify_customer,
        axis=1
    )
)

segment_summary = (
    customer_sales
    .groupby("Customer_Segment")
    .agg(
        Customers=("Customer_ID", "count"),
        Orders=("Orders", "sum"),
        Revenue=("Total_Revenue", "sum"),
        Profit=("Total_Profit", "sum")
    )
    .reset_index()
)

segment_summary["Revenue_Per_Customer"] = (
    segment_summary["Revenue"]
    / segment_summary["Customers"]
)

segment_summary = segment_summary.sort_values(
    "Revenue",
    ascending=False
)

print("\nCustomer segments:")
print(
    segment_summary.to_string(index=False)
)


# ============================================================
# 13. PAYMENT METHOD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PAYMENT METHOD ANALYSIS")
print("=" * 70)

payment_analysis = (
    completed_df
    .groupby("Payment_Method")
    .agg(
        Orders=("Order_ID", "count"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
    .sort_values("Revenue", ascending=False)
)

print("\nPayment method performance:")
print(
    payment_analysis.to_string(index=False)
)


# ============================================================
# 14. ORDER STATUS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ORDER STATUS ANALYSIS")
print("=" * 70)

status_analysis = (
    df
    .groupby("Order_Status")
    .agg(
        Orders=("Order_ID", "count"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reset_index()
)

print("\nOrder status:")
print(
    status_analysis.to_string(index=False)
)


# ============================================================
# 15. DAY OF WEEK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DAY OF WEEK ANALYSIS")
print("=" * 70)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_sales = (
    completed_df
    .groupby("Day_Name")
    .agg(
        Orders=("Order_ID", "count"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )
    .reindex(day_order)
    .reset_index()
)

print("\nSales by day:")
print(
    day_sales.to_string(index=False)
)


# ============================================================
# 16. PIVOT TABLES
# ============================================================

print("\n" + "=" * 70)
print("PIVOT TABLE ANALYSIS")
print("=" * 70)

# Category vs Region
category_region_pivot = pd.pivot_table(
    completed_df,
    values="Revenue",
    index="Category",
    columns="Region",
    aggfunc="sum",
    fill_value=0
)

print("\nRevenue by Category and Region:")
print(category_region_pivot)

# Monthly revenue by category
monthly_category_pivot = pd.pivot_table(
    completed_df,
    values="Revenue",
    index="Month_Name",
    columns="Category",
    aggfunc="sum",
    fill_value=0
)

print("\nMonthly Revenue by Category:")
print(monthly_category_pivot)


# ============================================================
# 17. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

# Highest profit category
highest_profit_category = category_sales.loc[
    category_sales["Profit"].idxmax()
]

# Highest profit region
highest_profit_region = region_sales.loc[
    region_sales["Profit"].idxmax()
]

# Highest AOV customer segment
highest_segment = segment_summary.loc[
    segment_summary["Revenue_Per_Customer"].idxmax()
]

print("\n1. Revenue Leader")
print(
    f"{top_category['Category']} is the highest revenue-generating "
    f"category with ₹{top_category['Revenue']:,.2f}."
)

print("\n2. Regional Leader")
print(
    f"{top_region['Region']} generates the highest revenue "
    f"with ₹{top_region['Revenue']:,.2f}."
)

print("\n3. Best Product")
print(
    f"{top_product['Product']} is the top product by revenue "
    f"with ₹{top_product['Revenue']:,.2f}."
)

print("\n4. Profit Leader")
print(
    f"{highest_profit_category['Category']} generates the "
    f"highest total profit of ₹{highest_profit_category['Profit']:,.2f}."
)

print("\n5. Best Profit Region")
print(
    f"{highest_profit_region['Region']} generates the highest "
    f"regional profit of ₹{highest_profit_region['Profit']:,.2f}."
)

print("\n6. Customer Segment")
print(
    f"{highest_segment['Customer_Segment']} customers have the "
    f"highest revenue per customer at "
    f"₹{highest_segment['Revenue_Per_Customer']:,.2f}."
)

print("\n7. Cancellation")
print(
    f"The overall cancellation rate is "
    f"{cancellation_rate:.2f}%."
)


# ============================================================
# 18. BUSINESS RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("ACTIONABLE BUSINESS RECOMMENDATIONS")
print("=" * 70)

print(
    f"""
1. Focus marketing campaigns on the {top_category['Category']}
   category because it generates the highest revenue.

2. Increase inventory and promotional activity for
   {top_product['Product']}, the highest-revenue product.

3. Strengthen operations and advertising in the
   {top_region['Region']} region, which is the strongest
   revenue-generating region.

4. Target High Value and High Spending Occasional customers
   with personalized offers, loyalty rewards and cross-selling.

5. Investigate the {cancellation_rate:.2f}% cancellation rate
   and identify possible causes such as payment failures,
   delivery problems or customer dissatisfaction.

6. Use monthly and quarterly sales trends to plan inventory,
   marketing campaigns and seasonal promotions.

7. Encourage higher-value purchases using product bundles,
   discounts on bulk purchases and cross-selling strategies.
"""
)


# ============================================================
# 19. SAVE ANALYSIS RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING ANALYSIS RESULTS")
print("=" * 70)

monthly_sales.to_csv(
    "monthly_sales_analysis.csv",
    index=False
)

quarterly_sales.to_csv(
    "quarterly_sales_analysis.csv",
    index=False
)

category_sales.to_csv(
    "category_analysis.csv",
    index=False
)

region_sales.to_csv(
    "region_analysis.csv",
    index=False
)

product_sales.to_csv(
    "product_analysis.csv",
    index=False
)

customer_sales.to_csv(
    "customer_analysis.csv",
    index=False
)

segment_summary.to_csv(
    "customer_segments.csv",
    index=False
)

payment_analysis.to_csv(
    "payment_analysis.csv",
    index=False
)

city_sales.to_csv(
    "city_analysis.csv",
    index=False
)

print("\nAnalysis files saved successfully.")


# ============================================================
# 20. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)


# ------------------------------------------------------------
# Chart 1: Monthly Revenue
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Month_Name"],
    monthly_sales["Revenue"],
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "monthly_revenue_trend.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Chart 2: Category Revenue
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    category_sales["Category"],
    category_sales["Revenue"]
)

plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "category_revenue.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Chart 3: Regional Revenue
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    region_sales["Region"],
    region_sales["Revenue"]
)

plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue (₹)")
plt.tight_layout()

plt.savefig(
    "regional_revenue.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Chart 4: Top 10 Products
# ------------------------------------------------------------

top_10_products = product_sales.head(10)

plt.figure(figsize=(12, 6))

plt.barh(
    top_10_products["Product"][::-1],
    top_10_products["Revenue"][::-1]
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue (₹)")
plt.ylabel("Product")
plt.tight_layout()

plt.savefig(
    "top_products.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Chart 5: Customer Segments
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    segment_summary["Customer_Segment"],
    segment_summary["Customers"]
)

plt.title("Customer Segments")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "customer_segments.png",
    dpi=300
)

plt.show()


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"""
Dataset:
    Orders             : {len(df):,}
    Completed Orders   : {total_orders:,}
    Customers          : {total_customers:,}
    Products           : {total_products:,}

Business Performance:
    Revenue            : ₹{total_revenue:,.2f}
    Profit             : ₹{total_profit:,.2f}
    Average Order Value: ₹{average_order_value:,.2f}
    Profit Margin      : {overall_profit_margin:.2f}%
    Cancellation Rate  : {cancellation_rate:.2f}%

Top Performers:
    Category           : {top_category['Category']}
    Region             : {top_region['Region']}
    Product            : {top_product['Product']}
    Best Month         : {best_month['Month_Name']}
    Best Quarter       : Q{int(best_quarter['Quarter'])}

The complete e-commerce sales analysis has been completed.
"""
)

print("=" * 70)