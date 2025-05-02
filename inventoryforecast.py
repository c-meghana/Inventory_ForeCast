# inventory_forecast_fuzzy.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load Real Sales Data from Amazon Sale Report
file_path = r"/Users/meghanachada/Downloads/inventorydata/Amazon Sale Report.csv"
amazon_df = pd.read_csv(file_path)
#amazon_df = pd.read_csv("/mnt/data/inventory_dataset/Amazon Sale Report.csv")
amazon_df['Date'] = pd.to_datetime(amazon_df['Date'], format='%m-%d-%y')

# Filter only successful sales (e.g., Shipped or Delivered)
status_filter = amazon_df['Status'].str.contains("Shipped|Delivered", na=False)
filtered_df = amazon_df[status_filter]

# Group by date and sum quantity
daily_sales = filtered_df.groupby('Date')['Qty'].sum().reset_index()
daily_sales = daily_sales.sort_values('Date')
daily_sales.rename(columns={'Qty': 'sales'}, inplace=True)

#Normalize Sales Data
scaler = MinMaxScaler()
daily_sales['sales_scaled'] = scaler.fit_transform(daily_sales[['sales']])

#Create LSTM Sequences
SEQ_LEN = 10
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(daily_sales['sales_scaled'].values, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

#Build LSTM Model
model = Sequential()
model.add(LSTM(64, input_shape=(SEQ_LEN, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=16, verbose=0)

#Predict Next 10 Days
last_seq = daily_sales['sales_scaled'].values[-SEQ_LEN:]
preds = []
input_seq = last_seq.copy()
for _ in range(10):
    input_reshaped = input_seq.reshape((1, SEQ_LEN, 1))
    pred = model.predict(input_reshaped, verbose=0)
    preds.append(pred[0][0])
    input_seq = np.append(input_seq[1:], pred[0][0])

preds_unscaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
print("\nPredicted next 10 days demand:", preds_unscaled)

#Fuzzy Logic for Restocking
# Define fuzzy variables
demand = ctrl.Antecedent(np.arange(0, 101, 1), 'demand')
stock = ctrl.Antecedent(np.arange(0, 101, 1), 'stock')
urgency = ctrl.Consequent(np.arange(0, 101, 1), 'urgency')

# Membership functions
demand['low'] = fuzz.trimf(demand.universe, [0, 0, 50])
demand['medium'] = fuzz.trimf(demand.universe, [20, 50, 80])
demand['high'] = fuzz.trimf(demand.universe, [50, 100, 100])

stock['low'] = fuzz.trimf(stock.universe, [0, 0, 50])
stock['medium'] = fuzz.trimf(stock.universe, [20, 50, 80])
stock['high'] = fuzz.trimf(stock.universe, [50, 100, 100])

urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 50])
urgency['medium'] = fuzz.trimf(urgency.universe, [20, 50, 80])
urgency['high'] = fuzz.trimf(urgency.universe, [50, 100, 100])

# Fuzzy Rules
rule1 = ctrl.Rule(demand['high'] & stock['low'], urgency['high'])
rule2 = ctrl.Rule(demand['medium'] & stock['medium'], urgency['medium'])
rule3 = ctrl.Rule(demand['low'] & stock['high'], urgency['low'])

urgency_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
urgency_sim = ctrl.ControlSystemSimulation(urgency_ctrl)

# Simulate Restocking Decision
example_demand = np.mean(preds_unscaled)
example_stock = 30  # assume current stock is low

urgency_sim.input['demand'] = example_demand
urgency_sim.input['stock'] = example_stock
urgency_sim.compute()

print("\nAverage Predicted Demand:", example_demand)
print("Current Stock Level:", example_stock)
print("Restocking Urgency (0-100):", urgency_sim.output['urgency'])

#Visualization
plt.figure(figsize=(12, 6))

# Plot historical sales
plt.plot(daily_sales['Date'], daily_sales['sales'], label='Historical Sales')

# Plot forecasted sales
future_dates = pd.date_range(start=daily_sales['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)
plt.plot(future_dates, preds_unscaled, label='Forecasted Sales', linestyle='--', marker='o')

plt.title('Sales Forecast with LSTM')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot restocking urgency as a bar
plt.figure(figsize=(6, 4))
plt.bar(['Restocking Urgency'], [urgency_sim.output['urgency']], color='orange')
plt.ylim(0, 100)
plt.title('Restocking Urgency Score')
plt.ylabel('Urgency (0-100)')
plt.tight_layout()
plt.show()
