from preprocessing import preprocessing
from feature_engineering import feature_engineering

def main():
    # Loading the Data
    OHLCV_data = "Historical_Data.csv"

    # Preprocessing Data
    df = preprocessing(OHLCV_data)
    
    # Feature Engineering
    train_series, test_series, train_prices, test_prices = feature_engineering(df)


if __name__ == "__main__":
    main()