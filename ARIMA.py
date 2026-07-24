from feature_engineering import train_series, test_series, train_prices, test_prices
from statsmodels.tsa.arima.model import ARIMA
from backtest import backtester
from AIC import best_order
import matplotlib.pyplot as plt
import numpy as np

# MODEL FITTING
model = ARIMA(train_series, order = best_order)
results = model.fit()

print(results.summary())

# FORECASTING

test_results = results.apply(test_series)
forecast = test_results.fittedvalues

# TRADING:
# One-way cost = 0.01% fee + 0.02% slippage = 0.03% (0.0003)
# Round-trip cost = 0.06% (0.0006). Demand at least 0.08% return to enter a trade.
min_expected_return = 0.0008

signals = []
for val in forecast:
    if val > min_expected_return:
        signals.append(1)    # Strong Buy: expected return beats fees
    elif val < -min_expected_return:
        signals.append(-1)   # Strong Sell: exit position
    else:
        signals.append(0)    # Hold: forecast is too weak to justify trading costs

# SIMULATION

arima_portfolio = backtester(
    signals,
    test_prices
)

print(arima_portfolio.head())
print(forecast.head())

values, counts = np.unique(signals, return_counts=True)

for v, c in zip(values, counts):
    print(v, c)

plt.figure(figsize=(12,6))
plt.plot(arima_portfolio["PortfolioValue"])
plt.title("ARIMA Portfolio Value")
plt.xlabel("Trading Day")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()