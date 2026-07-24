from feature_engineering import train_series, test_series, train_prices, test_prices
from statsmodels.tsa.arima.model import ARIMA
from backtest import backtester
from AIC import best_order
import matplotlib.pyplot as plt

# MODEL FITTING
model = ARIMA(train_series, order = best_order)
results = model.fit()

print(results.summary())

# FORECASTING

forecast = results.forecast(steps = len(test_series))

# TRADING

signals = []
threshold = 0.001

for pred in forecast:
    if pred > threshold:
        signals.append(1)
    elif pred < -threshold:
        signals.append(-1)
    else:
        signals.append(0)

# SIMULATION

arima_portfolio = backtester(
    signals,
    test_prices
)

print(arima_portfolio.head())

plt.figure(figsize=(12,6))
plt.plot(arima_portfolio["PortfolioValue"])
plt.title("ARIMA Portfolio Value")
plt.xlabel("Trading Day")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()