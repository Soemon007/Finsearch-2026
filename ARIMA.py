from feature_engineering import series
from statsmodels.tsa.arima.model import ARIMA

# MODEL FITTING
model = ARIMA(series, order=(1,0,0))
results = model.fit()

print(results.summary())

# FORECASTING

forecast = results.forecast(steps = 10)

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

cash = 100000
shares = 0

portfolio_values = []