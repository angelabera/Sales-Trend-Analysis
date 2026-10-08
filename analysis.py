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