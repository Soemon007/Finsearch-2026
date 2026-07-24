# FinSearch | 2026 | End-Term Project

## Contributors

- Rehan (@Soemon007)
- Arnav
- Nishant

---

# Overview

This project investigates the application of statistical forecasting and deep reinforcement learning to algorithmic stock trading using historical Nifty100 market data.

Two fundamentally different approaches are implemented and compared:

- **AutoRegressive Integrated Moving Average (ARIMA)** — a classical statistical time-series forecasting model.
- **Deep Q-Network (DQN)** — a reinforcement learning agent that learns an optimal trading policy through interaction with a simulated trading environment.

Both models are evaluated using a common backtesting framework to ensure a fair comparison of trading performance.

---

# Project Objectives

- Forecast market behaviour using ARIMA.
- Learn an optimal trading strategy using Deep Reinforcement Learning.
- Generate Buy/Hold/Sell trading signals.
- Backtest both strategies on historical Nifty100 data.
- Compare performance using standard financial metrics.

---

# Dataset

The project uses approximately six weeks of historical **Nifty100** market data.

The dataset contains:

- Open Price
- High Price
- Low Price
- Volume

Since closing prices were unavailable in the provided dataset, **opening prices were used consistently throughout the project** for:

- Log return computation
- Forecasting
- Trading simulation
- Portfolio evaluation

---

# Project Structure

```
Finsearch-2026/
│
├── preprocessing.py          # Data cleaning and preprocessing
├── feature_engineering.py    # Technical indicator generation
├── ARIMA.py                  # ARIMA forecasting model
├── FinSearch_End.ipynb       # DQN implementation
├── backtest.py               # Common backtesting framework
├── metric.py                 # Performance evaluation metrics
├── Historical_Data.csv
└── README.md
```

---

# Methodology

## 1. Data Preprocessing

- Clean missing values
- Convert numerical columns
- Compute logarithmic returns
- Generate technical indicators

---

## 2. ARIMA Benchmark

- Stationarity testing using the Augmented Dickey-Fuller (ADF) test
- Model selection using Akaike Information Criterion (AIC)
- Forecast future log returns
- Convert forecasts into Buy/Hold/Sell signals

---

## 3. Deep Reinforcement Learning

A custom stock trading environment is developed where a Deep Q-Network learns an optimal trading policy.

### State Space

The agent observes engineered market features and portfolio information.

### Action Space

- Buy
- Hold
- Sell

### Reward

Portfolio value improvement after executing an action.

---

## 4. Backtesting

Both strategies are evaluated using the same execution engine.

Initial capital:

```
₹100,000
```

Portfolio performance is computed using historical opening prices.

---

# Performance Metrics

The following evaluation metrics are computed for both strategies:

- Final Portfolio Value
- Total Return
- Daily Volatility
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown

---

# Results

The project compares both models using:

- Portfolio value over time
- Buy/Sell signal visualization
- Total return
- Risk-adjusted performance
- Drawdown analysis

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- TensorFlow / Keras
- Scikit-learn

---

# References

1. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. *Time Series Analysis: Forecasting and Control.*

2. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction.*

3. Mnih, V. et al. *Human-level Control through Deep Reinforcement Learning.* Nature, 2015.

---

## Authors

Rehan  
Arnav  
Nishant

Department of Chemical Engineering  
Indian Institute of Technology Bombay  
2026
