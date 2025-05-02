# Inventory Demand Forecasting and Restocking Decision System

This project predicts inventory demand using an LSTM neural network and provides restocking urgency recommendations using fuzzy logic. It is designed for e-commerce businesses to make intelligent, data-driven inventory decisions based on historical sales data.

---

## Features

- **LSTM-based Demand Forecasting**: Uses a sequence of past sales data to predict future demand.
- **Fuzzy Logic-Based Restocking Urgency**: Determines how urgent restocking is based on demand and current stock level.
- **Data Preprocessing**: Filters successful transactions and normalizes sales data.
- **Visualization**: Displays past sales, forecasted demand, and urgency score.

---

## Dataset

- The script expects a CSV file named `Amazon Sale Report.csv` with at least the following columns:
  - `Date`: Date of transaction (format: `MM-DD-YY`)
  - `Qty`: Quantity sold
  - `Status`: Shipping status (`Shipped`, `Delivered`, etc.)

Update the file path in the script if your dataset location is different:

file_path = "/Users/meghanachada/Downloads/inventorydata/Amazon Sale Report.csv"

### Installation

Install all dependencies using pip:
pip install numpy pandas matplotlib scikit-learn tensorflow scikit-fuzzy

### How to Run
Place the sales dataset in the correct path and update in the code, then run python inventoryforecast.py

### Output
Forecasted Demand for the next 10 days (printed to console).

Restocking Urgency Score (0-100).

Visualization of both forecasts and urgency.

