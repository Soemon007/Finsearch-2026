from preprocessing import preprocessing

def main():
    # Loading the Data
    OHLCV_data = "Historical_Data.csv"

    #Preprocessing Data
    preprocessing(OHLCV_data)
    


if __name__ == "__main__":
    main()