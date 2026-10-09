# E-Commerce Sales Trend Analysis & Revenue Prediction

An end-to-end data analytics and machine learning project that analyzes e-commerce sales data to uncover revenue trends, identify valuable customer segments, evaluate business performance, and predict order revenue using Scikit-Learn.

## Project Overview

This project performs exploratory data analysis, feature engineering, business intelligence reporting, and supervised machine learning on an e-commerce sales dataset containing 10,000 synthetic order records.

The project aims to transform raw sales data into actionable business insights and demonstrate a complete machine learning workflow, from data preprocessing to model evaluation.

## Objectives

- Analyze overall sales, revenue, and profit performance.
- Identify monthly and quarterly revenue trends.
- Determine top-performing products and product categories.
- Compare sales performance across regions and cities.
- Analyze customer purchasing behavior and segment customers.
- Examine payment methods and order cancellation rates.
- Create pivot tables and visualizations for business reporting.
- Train a Decision Tree Regression model to predict order revenue.
- Evaluate model performance using standard regression metrics.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-Learn
- **Development Environment:** Visual Studio Code
- **Dataset Format:** CSV

## Project Structure

```text
Sales-Trend-Analysis/
│
├── generate_data.py
├── analysis.py
├── ml_model.py
├── ecommerce_sales.csv
├── README.md
│
├── monthly_sales_analysis.csv
├── quarterly_sales_analysis.csv
├── category_analysis.csv
├── region_analysis.csv
├── product_analysis.csv
├── customer_analysis.csv
├── customer_segments.csv
├── payment_analysis.csv
├── city_analysis.csv
│
├── monthly_revenue_trend.png
├── category_revenue.png
├── regional_revenue.png
├── top_products.png
├── customer_segments.png
│
└── revenue_predictions.csv
```

*Note: The analysis output files and visualizations are generated when the relevant Python scripts run.*

## Dataset Description

The dataset is synthetically generated using Python, NumPy, and Pandas. It contains 10,000 order records representing a simulated e-commerce business operating across multiple regions in India.

| Feature | Description |
|---|---|
| `Order_ID` | Unique order identifier |
| `Order_Date` | Date the order was placed |
| `Customer_ID` | Customer identifier |
| `Product` | Purchased product |
| `Category` | Product category |
| `Quantity` | Number of units ordered |
| `Unit_Price` | Price per unit |
| `Discount` | Discount applied to the order |
| `Revenue` | Revenue recorded for the order |
| `Profit` | Estimated profit from the order |
| `Region` | Geographic sales region |
| `City` | Customer city |
| `Payment_Method` | Payment method used |
| `Order_Status` | Completed or cancelled status |

The dataset includes five product categories: Electronics, Clothing, Home & Kitchen, Beauty, and Books.

**Data note:** The dataset is synthetic and intended for educational and analytical purposes. Its patterns do not represent verified real-world business performance.

## Key Features

### 1. Data Generation and Validation

- Generate a reproducible synthetic e-commerce dataset.
- Inspect dataset dimensions and data types.
- Convert order dates into appropriate datetime values.
- Check for missing values and duplicate records.

### 2. Feature Engineering

Extract useful time-based features from the order date:

- Year
- Month
- Month name
- Quarter
- Day of month
- Day of week

Calculate profit margin to support profitability analysis.

### 3. Business KPI Analysis

Calculate important business performance indicators:

- Total completed orders
- Total revenue
- Total profit
- Number of unique customers
- Number of products sold
- Average order value
- Average profit per order
- Overall profit margin
- Order cancellation rate

### 4. Sales Trend Analysis

Analyze sales performance across different time periods:

- Monthly revenue and profit
- Quarterly revenue and profit
- Highest- and lowest-revenue months
- Best-performing quarter
- Sales performance by day of the week

### 5. Product and Regional Analysis

- Compare revenue and profit across product categories.
- Identify top-performing products.
- Analyze sales across regions and cities.
- Compare category performance across geographic regions.
- Identify potential areas for business growth.

### 6. Customer Segmentation

Aggregate customer-level purchasing data to calculate order frequency, total revenue, total profit, and average order value.

Classify customers into four basic segments:

- **High Value:** Higher purchase frequency and spending.
- **Loyal Budget:** Higher purchase frequency but lower spending.
- **High Spending Occasional:** Lower purchase frequency but higher spending.
- **Low Engagement:** Lower purchase frequency and spending.

The segments are created using median-based rules rather than a trained clustering algorithm.

### 7. Pivot Table Analysis

Generate pivot tables to compare:

- Revenue by category and region.
- Monthly revenue by product category.

These summaries make it easier to compare business performance across multiple dimensions.

### 8. Data Visualization

Generate charts to communicate business trends:

- Monthly revenue trend
- Revenue by product category
- Revenue by region
- Top 10 products by revenue
- Customer segment distribution

### 9. Machine Learning: Revenue Prediction

The `ml_model.py` script uses a Decision Tree Regressor to predict order revenue.

**Machine learning workflow:**

1. Load the existing sales dataset.
2. Filter completed orders.
3. Select relevant input features.
4. Encode categorical variables using one-hot encoding.
5. Split the data into training and testing sets.
6. Train a Decision Tree Regression model.
7. Predict revenue for the test dataset.
8. Evaluate the model.
9. Export actual and predicted revenue for comparison.

**Input features:**

- Quantity
- Unit price
- Discount
- Product category
- Region
- Payment method

**Target variable:** `Revenue`

**Evaluation metrics:**

| Metric | Purpose |
|---|---|
| Mean Squared Error (MSE) | Measures the average squared prediction error |
| Root Mean Squared Error (RMSE) | Expresses prediction error in revenue units |
| Mean Absolute Error (MAE) | Measures average absolute prediction error |
| R-squared (R²) | Measures the proportion of target variance explained by the model |

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Visual Studio Code
- Python extension for VS Code

### Step 1: Open the project folder

Open the `Sales-Trend-Analysis` folder in VS Code.

### Step 2: Install dependencies

Run the following command in the VS Code terminal:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

### Step 3: Generate the dataset

To regenerate the synthetic dataset, run:

```bash
python generate_data.py
```

This creates `ecommerce_sales.csv` in the project directory.

### Step 4: Run the sales analysis

```bash
python analysis.py
```

This performs the business analysis and generates summary CSV files and visualizations.

### Step 5: Run the machine learning model

```bash
python ml_model.py
```

This trains the Decision Tree Regressor, prints evaluation metrics, and generates `revenue_predictions.csv`.

## Business Insights and Recommendations

The analysis can help a business:

1. Focus marketing efforts on high-revenue product categories.
2. Improve inventory planning based on monthly and quarterly trends.
3. Identify regions and cities with strong sales performance.
4. Promote high-performing products and complementary products.
5. Develop loyalty campaigns for valuable customer segments.
6. Investigate cancellation rates and potential operational issues.
7. Compare revenue with profit to avoid prioritizing sales that contribute less to profitability.

Actual recommendations should be based on the results generated by the scripts, rather than assumed in advance.

## Machine Learning Limitations

The revenue prediction model is a learning demonstration, not a production forecasting system.

- The dataset is synthetic rather than collected from real customers.
- Revenue is directly calculated from quantity, price, and discount in the data generator.
- The selected features therefore contain much of the information used to calculate the target.
- A random train/test split evaluates performance on held-out synthetic orders, but does not establish future-sales forecasting performance.
- Real-world deployment would require representative historical data, stronger validation, and features available at prediction time.

## Future Improvements

- Use a real-world e-commerce dataset to validate findings.
- Improve the synthetic dataset with more realistic customer purchasing patterns.
- Add time-series forecasting for future monthly revenue.
- Compare Decision Tree Regression with Linear Regression and Random Forest Regression.
- Use cross-validation and hyperparameter tuning.
- Add interactive dashboards using Streamlit or Power BI.
- Explore clustering algorithms for customer segmentation.
- Add automated reports and business KPI dashboards.

## Learning Outcomes

Through this project, the following skills are demonstrated:

- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis
- Business KPI calculation
- Pivot tables and aggregation
- Data visualization
- Customer segmentation
- Supervised machine learning
- Training and testing data separation
- Regression model evaluation
- Interpretation of business results

## Conclusion

This project demonstrates an end-to-end workflow that combines e-commerce data analysis with supervised machine learning. It transforms a synthetic sales dataset into business summaries, customer segments, visualizations, and a baseline revenue prediction model.

It provides practical experience in Python-based analytics, business intelligence, and core machine learning concepts.

## Author

**Angela Bera**

B.Tech Computer Science and Engineering

**Project:** E-Commerce Sales Trend Analysis & Revenue Prediction
