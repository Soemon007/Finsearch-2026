from statsmodels.tsa.arima.model import ARIMA
from backtest import backtester
from metric import evaluate_strategy, compare_strategies

import matplotlib.pyplot as plt
import numpy as np

def ARIMA_modelling(best_order, train_series):

    # MODEL FITTING
    model = ARIMA(train_series, order = best_order)
    results = model.fit()

    print(results.summary())
    return results

def Forecast(results, test_series, test_prices):

    # FORECASTING
    test_results = results.apply(test_series)
    forecast = test_results.fittedvalues

    # TRADING
    forecast = forecast.rolling(3, min_periods=1).mean()
    mu = forecast.mean()
    sigma = forecast.std()

    # Avoid divide-by-zero
    sigma = max(sigma, 1e-8)

    # Hyperparameter
    z_threshold = 1.0

    signals = []

    for pred in forecast:

        z = (pred - mu) / sigma

        if z > z_threshold:
            signals.append(1)

        elif z < -z_threshold:
            signals.append(-1)

        else:
            signals.append(0)

    # Remove repeated entries
    filtered = []
    position = 0

    for s in signals:

        if s == position:
            filtered.append(0)

        else:
            filtered.append(s)
            position = s

    signals = filtered
    # SIMULATION

    arima_portfolio = backtester(
        signals,
        test_prices
    )

    return arima_portfolio, forecast, signals

def ARIMA_summary(arima_portfolio, forecast, signals):

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

    evaluate_strategy(arima_portfolio, "ARIMA")