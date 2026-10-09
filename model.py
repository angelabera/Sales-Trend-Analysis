import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 1. Load the dataset
df = pd.read_csv("ecommerce_sales.csv")

# 2. Keep completed orders
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

X = df[features]
y = df["Revenue"]

# 4. Preprocess categorical features
categorical_features = [
    "Category",
    "Region",
    "Payment_Method"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# 5. Build the ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        DecisionTreeRegressor(
            max_depth=8,
            random_state=42
        )
    )
])

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 7. Train
model.fit(X_train, y_train)

# 8. Evaluate
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MODEL EVALUATION")
print("-" * 40)
print(f"MSE  : {mse:.2f}")
print(f"RMSE : ₹{rmse:.2f}")
print(f"MAE  : ₹{mae:.2f}")
print(f"R²   : {r2:.4f}")

# 9. Save the trained pipeline
joblib.dump(model, "revenue_model.joblib")

# 10. Save test predictions
pd.DataFrame({
    "Actual_Revenue": y_test,
    "Predicted_Revenue": predictions
}).to_csv("revenue_predictions.csv", index=False)

print("\nModel saved as revenue_model.joblib")
print("Predictions saved as revenue_predictions.csv")
