import pandas as pd
import numpy as np

def preprocessing(file_path):
    data = pd.read_csv(file_path)
    column_list = ['Price', 'Open', 'High', 'Low']

    data['Date'] = pd.to_datetime(data['Date'], errors = 'coerce')
    for cols in column_list:
        data[cols] = data[cols].str.replace(",", "")
        data[cols] = pd.to_numeric(data[cols], errors= 'coerce')

    cleaned_data = data
    return cleaned_data