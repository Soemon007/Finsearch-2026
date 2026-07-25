from statsmodels.tsa.arima.model import ARIMA
from backtest import backtester
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

    # TRADING:
    min_expected_return = 0.0008

    signals = []
    for val in forecast:
        if val > min_expected_return:
            signals.append(1)   
        elif val < -min_expected_return:
            signals.append(-1)  
        else:
            signals.append(0) 
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