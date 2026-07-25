from preprocessing import preprocessing
from feature_engineering import feature_engineering
from AIC import AIC
from ARIMA import ARIMA_modelling, Forecast, ARIMA_summary
from metric import evaluate_strategy, compare_strategies

import pandas as pd

def main():
    # Loading the Data
    OHLCV_data = "Historical_Data.csv"

    # Preprocessing Data
    df = preprocessing(OHLCV_data)
    
    # Feature Engineering
    train_series, test_series, train_prices, test_prices = feature_engineering(df)

    # AIC testing
    best_order = AIC(train_series)

    #ARIMA
    results = ARIMA_modelling(best_order, train_series)
    arima_portfolio, forecast, signals = Forecast(results, test_series, test_prices)

    ARIMA_summary(arima_portfolio, forecast, signals)

    # Evaluations

    dqn_portfolio = pd.read_csv("dqn_portfolio.csv")

    evaluate_strategy(arima_portfolio, "ARIMA")
    evaluate_strategy(dqn_portfolio, "DQN")

    comparison = compare_strategies(
        arima_portfolio,
        dqn_portfolio
    )

    print(comparison)

if __name__ == "__main__":
    main()