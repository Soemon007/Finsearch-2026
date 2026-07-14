from preprocessing import OHLCV_data
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


OHLCV_data['Returns'] = OHLCV_data["Open"].pct_change()
OHLCV_data['LogReturns'] = np.log(1 + OHLCV_data['Returns'])

series = OHLCV_data['LogReturns'].dropna()

result = adfuller(series)



