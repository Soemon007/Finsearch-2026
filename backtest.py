import pandas as pd
import numpy as np


def backtester(
    signals,
    prices,
    initial_cash=100000,
    position_size=1.0,
    transaction_cost=0.001,
    slippage=0.0005,
    stop_loss=0.05,
    take_profit=0.10
):
    """
    Advanced trading strategy backtester.

    Parameters
    ----------
    signals : array-like
        1  = Buy
        0  = Hold
        -1 = Sell

    prices : pandas Series
        Market prices

    initial_cash : float
        Starting capital

    position_size : float
        Fraction of capital used per trade
        1.0 = all capital

    transaction_cost : float
        Trading fee percentage

    slippage : float
        Price execution disadvantage

    stop_loss : float
        Exit loss percentage

    take_profit : float
        Exit profit percentage


    Returns
    -------
    portfolio : DataFrame
        Portfolio history

    trades : DataFrame
        Trade history
    """

    cash = initial_cash
    shares = 0

    entry_price = None

    portfolio_history = []
    trade_history = []


    for i, (signal, price) in enumerate(zip(signals, prices)):

        current_value = cash + shares * price


        # ---------------------------
        # CHECK STOP LOSS / TAKE PROFIT
        # ---------------------------

        if shares > 0:

            return_pct = (price - entry_price) / entry_price


            if return_pct <= -stop_loss:

                execution_price = price * (1 - slippage)

                cash += shares * execution_price * (
                    1 - transaction_cost
                )

                trade_history.append({
                    "Day": i,
                    "Type": "STOP LOSS",
                    "Price": execution_price,
                    "Shares": shares
                })

                shares = 0
                entry_price = None


            elif return_pct >= take_profit:

                execution_price = price * (1 - slippage)

                cash += shares * execution_price * (
                    1 - transaction_cost
                )

                trade_history.append({
                    "Day": i,
                    "Type": "TAKE PROFIT",
                    "Price": execution_price,
                    "Shares": shares
                })

                shares = 0
                entry_price = None



        # ---------------------------
        # BUY
        # ---------------------------

        if signal == 1 and shares == 0:

            execution_price = price * (1 + slippage)


            capital_used = cash * position_size


            shares = capital_used / execution_price*(1 + transaction_cost)


            cost = shares * execution_price


            fee = cost * transaction_cost


            cash -= cost + fee


            entry_price = execution_price


            trade_history.append({

                "Day": i,
                "Type": "BUY",
                "Price": execution_price,
                "Shares": shares

            })


        # ---------------------------
        # SELL
        # ---------------------------

        elif signal == -1 and shares > 0:

            execution_price = price * (1 - slippage)


            revenue = shares * execution_price


            fee = revenue * transaction_cost


            cash += revenue - fee


            trade_history.append({

                "Day": i,
                "Type": "SELL",
                "Price": execution_price,
                "Shares": shares

            })


            shares = 0
            entry_price = None



        # ---------------------------
        # PORTFOLIO VALUE
        # ---------------------------

        portfolio_value = cash + shares * price


        portfolio_history.append({

            "Day": i,
            "Price": price,
            "Signal": signal,
            "Cash": cash,
            "Shares": shares,
            "PortfolioValue": portfolio_value

        })


    portfolio = pd.DataFrame(portfolio_history)

    trades = pd.DataFrame(trade_history)


    return portfolio