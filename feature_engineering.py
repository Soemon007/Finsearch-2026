import numpy as np


def feature_engineering(data):
    # Transforming Data
    data['Returns'] = data["Open"].pct_change()
    data['LogReturns'] = np.log(1 + data['Returns'])

    series = data['LogReturns'].dropna()
    prices = data['Open'].loc[series.index]

    # Test - Train split
    split =  int(0.8 * len(series))

    train_series = series[:split]
    test_series = series[split:]

    train_prices = prices[:split]
    test_prices= prices[split:]

    return train_series, test_series, train_prices, test_prices