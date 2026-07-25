from preprocessing import preprocessing
from feature_engineering import feature_engineering
from AIC import AIC
from ARIMA import ARIMA_modelling, Forecast, ARIMA_summary
from DQN import DQN_modelling, DQN_summary

import pandas as pd

def main():
    # Loading the Data
    OHLCV_data = "Historical_Data.csv"

    # Preprocessing Data
    df = preprocessing(OHLCV_data)
    
    # Feature Engineering
    train_series, test_series, train_prices, test_prices = feature_engineering(df)

    # AIC testing
    best_order, best_aic = AIC(train_series)

    #ARIMA
    results = ARIMA_modelling(best_order, train_series)
    arima_portfolio, forecast, signals = Forecast(results, test_series, test_prices)

    ARIMA_summary(arima_portfolio, forecast, signals)

    #DQN
    dqn_portfolio, best_overall, seed_results = DQN_modelling(train_prices, test_prices)
 
    DQN_summary(dqn_portfolio, best_overall, seed_results)

if __name__ == "__main__":
    main()