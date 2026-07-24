import numpy as np
import pandas as pd
from ARIMA import arima_portfolio
import import_ipynb

# Function Definitions

def total_return(portfolio):
    """
    Returns total percentage return.
    """
    initial = portfolio["PortfolioValue"].iloc[0]
    final = portfolio["PortfolioValue"].iloc[-1]

    return (final - initial) / initial


def daily_returns(portfolio):
    """
    Returns daily percentage returns.
    """
    return portfolio["PortfolioValue"].pct_change().dropna()


def volatility(portfolio):
    """
    Standard deviation of daily returns.
    """
    returns = daily_returns(portfolio)

    return returns.std()


def annualized_volatility(portfolio, periods_per_year=252):
    """
    Annualized volatility.
    """
    returns = daily_returns(portfolio)

    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(portfolio,
                 risk_free_rate=0,
                 periods_per_year=252):
    """
    Annualized Sharpe Ratio.
    """

    returns = daily_returns(portfolio)

    excess_returns = returns - (risk_free_rate / periods_per_year)

    if excess_returns.std() == 0:
        return 0

    return (
        np.sqrt(periods_per_year)
        * excess_returns.mean()
        / excess_returns.std()
    )


def max_drawdown(portfolio):
    """
    Maximum Drawdown.
    """

    values = portfolio["PortfolioValue"]

    running_max = values.cummax()

    drawdown = (values - running_max) / running_max

    return drawdown.min()


def final_portfolio_value(portfolio):
    """
    Final portfolio value.
    """
    return portfolio["PortfolioValue"].iloc[-1]


def evaluate_strategy(portfolio,
                      strategy_name="Strategy"):
    """
    Prints all important metrics.
    """

    print("=" * 40)
    print(strategy_name)
    print("=" * 40)

    print(f"Final Portfolio Value : ₹{final_portfolio_value(portfolio):.2f}")

    print(f"Total Return          : {total_return(portfolio)*100:.2f}%")

    print(f"Volatility            : {volatility(portfolio):.4f}")

    print(f"Annualized Volatility : {annualized_volatility(portfolio):.4f}")

    print(f"Sharpe Ratio          : {sharpe_ratio(portfolio):.4f}")

    print(f"Maximum Drawdown      : {max_drawdown(portfolio)*100:.2f}%")

    print()


def compare_strategies(arima_portfolio,
                       dqn_portfolio):
    """
    Returns a comparison DataFrame.
    """

    comparison = pd.DataFrame({

        "Metric": [
            "Final Portfolio Value",
            "Total Return (%)",
            "Volatility",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown (%)"
        ],

        "ARIMA": [

            final_portfolio_value(arima_portfolio),

            total_return(arima_portfolio) * 100,

            volatility(arima_portfolio),

            annualized_volatility(arima_portfolio),

            sharpe_ratio(arima_portfolio),

            max_drawdown(arima_portfolio) * 100
        ],

        "DQN": [

            final_portfolio_value(dqn_portfolio),

            total_return(dqn_portfolio) * 100,

            volatility(dqn_portfolio),

            annualized_volatility(dqn_portfolio),

            sharpe_ratio(dqn_portfolio),

            max_drawdown(dqn_portfolio) * 100
        ]

    })

    return comparison

# Evaluations

dqn_portfolio = pd.read_csv("dqn_portfolio.csv")

evaluate_strategy(arima_portfolio, "ARIMA")
evaluate_strategy(dqn_portfolio, "DQN")

comparison = compare_strategies(
    arima_portfolio,
    dqn_portfolio
)

print(comparison)