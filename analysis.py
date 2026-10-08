import pandas as pd
import numpy as np


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("ecommerce_sales.csv")


# ============================================================
# DATASET OVERVIEW
# ============================================================

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# DATA TYPES
# ============================================================

print("\nData types:")
print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# DUPLICATES
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())

# ============================================================
# CONVERT DATE COLUMN
# ============================================================

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nUpdated data types:")
print(df.dtypes)


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\nNumerical summary:")
print(df.describe())


# ============================================================
# CATEGORICAL DATA SUMMARY
# ============================================================

print("\nCategories:")
print(df["Category"].value_counts())

print("\nRegions:")
print(df["Region"].value_counts())

print("\nPayment methods:")
print(df["Payment_Method"].value_counts())

print("\nOrder status:")
print(df["Order_Status"].value_counts())

# FEATURE ENGINEERING
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Extract date-related features
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Quarter"] = df["Order_Date"].dt.quarter
df["Day"] = df["Order_Date"].dt.day
df["Day_Name"] = df["Order_Date"].dt.day_name()

# Average Order Value
df["AOV"] = df["Revenue"]

print("\nNew columns created:")
print([
    "Year",
    "Month",
    "Month_Name",
    "Quarter",
    "Day",
    "Day_Name",
    "AOV"
])

print("\nSample after feature engineering:")
print(
    df[
        [
            "Order_Date",
            "Year",
            "Month",
            "Month_Name",
            "Quarter",
            "Day",
            "Day_Name",
            "Revenue",
            "AOV"
        ]
    ].head()
)