
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 1. Load the existing sales dataset
df = pd.read_csv("ecommerce_sales.csv")

# 2. Keep completed orders for the model
df = df[df["Order_Status"] == "Completed"].copy()

# 3. Select features and target
features = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Category",
    "Region",
    "Payment_Method"
]

target = "Revenue"

X = df[features]
y = df[target]

# 4. Convert categorical columns into numeric columns
X = pd.get_dummies(
    X,
    columns=["Category", "Region", "Payment_Method"],
    dtype=int
)

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("=" * 60)
print("E-COMMERCE REVENUE PREDICTION")
print("=" * 60)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# 6. Create and train the Decision Tree Regressor
model = DecisionTreeRegressor(
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed.")

# 7. Predict revenue for the test dataset
y_pred = model.predict(X_test)

# 8. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-" * 60)

print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"Root Mean Squared Error  : ₹{rmse:.2f}")
print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
print(f"R-squared (R²)           : {r2:.4f}")

# 9. Compare actual and predicted revenue
results = pd.DataFrame({
    "Actual_Revenue": y_test.values,
    "Predicted_Revenue": y_pred
})

results["Prediction_Error"] = (
    results["Actual_Revenue"]
    - results["Predicted_Revenue"]
)

print("\nSample predictions:")
print(results.head(10).round(2).to_string(index=False))

# 10. Save the results
results.to_csv("revenue_predictions.csv", index=False)

print("\nSaved: revenue_predictions.csv")
print("Machine learning analysis completed.")
