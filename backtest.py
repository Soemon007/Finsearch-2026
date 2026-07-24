from ARIMA import signals
from feature_engineering import signals, prices
import pandas as pd

def backtest(signals, prices, initial_cash=100000):
    """
    Backtest a trading strategy.

    Parameters
    ----------
    signals : list or array-like
        1 = Buy
        0 = Hold
        -1 = Sell

    prices : list, numpy array or pandas Series
        Actual market prices.

    initial_cash : float
        Starting capital.

    Returns
    -------
    portfolio : pandas.DataFrame
        Daily portfolio history.
    """

    cash = initial_cash
    shares = 0

    history = []

    for signal, price in zip(signals, prices):

        # BUY
        if signal == 1 and cash >= price:
            shares += 1
            cash -= price

        # SELL (sell all shares)
        elif signal == -1 and shares > 0:
            cash += shares * price
            shares = 0

        portfolio_value = cash + shares * price

        history.append({
            "Price": price,
            "Signal": signal,
            "Cash": cash,
            "Shares": shares,
            "PortfolioValue": portfolio_value
        })

    portfolio = pd.DataFrame(history)

    return portfolio

