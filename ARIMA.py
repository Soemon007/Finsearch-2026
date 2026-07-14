from feature_engineering import series
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(1,0,0))
results = model.fit()

print(results.summary())