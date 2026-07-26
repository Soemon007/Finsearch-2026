# FinSearch | End-Term Project 2026

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4B8BBE?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

An algorithmic trading project comparing a classical statistical forecasting model (**ARIMA**) with a **Double Dueling Deep Q-Network (D3QN)** for trading on historical **NIFTY 50** market data.

Both strategies are evaluated using a common backtesting engine, allowing a direct comparison between statistical forecasting and reinforcement learning.

---

## Contributors

* Rehan (@Soemon007)
* Arnav (@VadapavR9)
* Nishant (@nish2007-ui)

---

## Features

* ARIMA-based market forecasting
* Double Dueling Deep Q-Network (D3QN)
* Technical indicator feature engineering
* Custom reinforcement learning trading environment
* Common backtesting framework
* Portfolio performance evaluation
* Risk-adjusted performance comparison

---

## Repository Structure

```text
.
├── preprocessing.py          # Data preprocessing
├── feature_engineering.py    # Technical indicator generation
├── ARIMA.py                  # ARIMA implementation
├── DQN.py                    # Double Dueling DQN
├── backtest.py               # Trading simulator
├── metric.py                 # Performance metrics
├── Historical_Data.csv
└── README.md
```

---

## Dataset

The project uses approximately six weeks of historical **NIFTY 50** market data containing:

* Open
* High
* Low
* Volume

Since closing prices were unavailable, **opening prices** were consistently used for:

* Feature engineering
* Forecasting
* Reinforcement learning
* Trading simulation
* Portfolio evaluation

---

## Methodology

### ARIMA

The statistical benchmark consists of:

* Augmented Dickey-Fuller (ADF) stationarity testing
* Automatic order selection using AIC
* Log-return forecasting
* Buy/Hold/Sell signal generation

### Double Dueling DQN

The reinforcement learning agent learns a trading policy by interacting with a simulated trading environment.

#### State

* Moving-average ratios
* Momentum indicators
* Rolling volatility
* RSI
* MACD
* Bollinger %B
* Current portfolio position

#### Actions

* Buy
* Hold
* Sell

#### Training Improvements

* Double Q-Learning
* Dueling Network Architecture
* Experience Replay
* Soft Target Updates
* Multi-seed model selection

---

## Backtesting

Both models are evaluated using the same execution engine with:

* Initial capital: **₹100,000**
* Transaction costs
* Slippage
* Stop-loss
* Take-profit
* Fractional share trading

---

## Results

| Metric                |       ARIMA |            D3QN |
| :-------------------- | ----------: | --------------: |
| Final Portfolio Value | ₹117,627.33 | **₹117,932.31** |
| Total Return          |      17.63% |      **17.93%** |
| Daily Volatility      |  **0.0078** |          0.0083 |
| Annualized Volatility |  **0.1241** |          0.1314 |
| Sharpe Ratio          |  **0.7690** |          0.7457 |
| Maximum Drawdown      | **−13.75%** |         −15.77% |

The D3QN achieved a slightly higher absolute return, while ARIMA produced stronger risk-adjusted performance through lower volatility, a smaller maximum drawdown, and a higher Sharpe ratio.

---

## Tech Stack

* Python
* NumPy
* Pandas
* PyTorch
* Statsmodels
* Scikit-learn
* Matplotlib

---

## License

This repository was developed as part of the **FinSearch 2026 End-Term Project** at IIT Bombay.
