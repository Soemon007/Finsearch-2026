from preprocessing import OHLCV_data
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Transforming Data
OHLCV_data['Returns'] = OHLCV_data["Open"].pct_change()
OHLCV_data['LogReturns'] = np.log(1 + OHLCV_data['Returns'])

series = OHLCV_data['LogReturns'].dropna()
prices = OHLCV_data['Open'].loc[series.index]

# Test - Train split
split =  int(0.8 * len(series))

train_series = series[:split]
test_series = series[split:]

train_prices = prices[:split]
test_prices= prices[split:]

result = adfuller(series)



