## 📈 Store Sales Forecasting (Large Scale)

This project focuses on predicting future sales based on historical data using a **Time-Series to Supervised Learning** approach. By transforming date-based trends into numerical features, the model learns to anticipate monthly sales fluctuations.

## 🏗️ Project Workflow

### 1. Data Cleaning & Transformation

* **Aggregation:** Sales data is grouped by month to reduce noise and focus on long-term trends.
* **Stationarity:** To make the data predictable for Linear Regression, we apply a **difference transformation** (subtracting the previous month's sales from the current month) to remove seasonality and trends.
* **Supervised Learning Setup:** The time-series data is converted into a supervised learning format using a **12-month sliding window** (lag features).

### 2. Feature Engineering

* **Scaling:** Data is normalized using `MinMaxScaler` to a range of `[-1, 1]` to help the model converge faster.
* **Train-Test Split:** The last 12 months of data are reserved for testing to evaluate the "real-world" forecasting ability of the model.

### 3. Model Architecture

The primary model used in this script is **Linear Regression**. While the imports include XGBoost and LSTMs, the current implementation focuses on the Linear Regression baseline:

* **Input:** 12 months of lagged sales differences.
* **Output:** Predicted sales difference for the next month.
* **Reversion:** The predicted "difference" is added back to the actual sales value to get the final forecasted number.

---

## 📊 Evaluation Metrics

The model performance is measured using:

* **Mean Absolute Error (MAE):** The average magnitude of errors.
* **Mean Squared Error (MSE):** Penalizes larger errors more heavily.
* **R² Score:** Indicates how well the independent variables explain the variance in sales.

---

## 🛠️ Usage

### Installation

```bash
pip install pandas numpy matplotlib xgboost scikit-learn tensorflow joblib

```

### Running the Forecast

1. **Prepare Data:** Ensure `train.csv` is in the root directory.
2. **Train:** Run the script to process the data and train the Linear Regression model.
3. **Predict:** The script will output a plot comparing "Actual Sales" vs. "Predicted Sales."
4. **Save/Load:** The model is saved as both a `.txt` (joblib) and a `.json` file for portability.

---

## 📂 Exported Files

* `linear_regression_model.txt`: Serialized model for Python use.
* `linear_regression_model.json`: Model coefficients and intercept for use in other environments (like JavaScript or C++).

---

## 🚀 Next Steps

* **Implement LSTM:** The current script has placeholders for Long Short-Term Memory (LSTM) networks, which could better capture complex non-linear patterns.
* **Hyperparameter Tuning:** Use XGBoost with GridSearchCV to improve the R² score.
