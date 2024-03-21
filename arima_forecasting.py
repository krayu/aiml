import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate example time series data (simple increasing trend with noise)
np.random.seed(42)
data = np.linspace(10, 50, 100) + np.random.normal(0, 5, 100)

# Create a pandas Series
time_index = pd.date_range(start='2022-01-01', periods=100, freq='D')
series = pd.Series(data, index=time_index)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(series)
plt.title('Original Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Split data into training and test sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit ARIMA model (p=5, d=1, q=0 as an example, can be adjusted based on data)
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecasting the next 20 periods (test set)
forecast = model_fit.forecast(steps=len(test))

# Plot the forecasted values against the actual test set
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data')
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(test.index, forecast, label='Forecasted Data', linestyle='--')
plt.title('ARIMA Forecast vs Actual Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Print forecasted values
print(f"Forecasted values: \n{forecast}")