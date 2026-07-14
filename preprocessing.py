import pandas as pd
import numpy as np

OHLCV_data = pd.read_csv("Historical_Data.csv")


column_list = ['Price', 'Open', 'High', 'Low']

OHLCV_data['Date'] = pd.to_datetime(OHLCV_data['Date'], errors='coerce')
for cols in column_list:
    OHLCV_data[cols] = OHLCV_data[cols].str.replace(",","")
    OHLCV_data[cols] = pd.to_numeric(OHLCV_data[cols], errors= 'coerce')
