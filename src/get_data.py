import yfinance as yf


def get_etf_data(ticker_symbol):
    try:
        # Create a ticker object
        etf = yf.Ticker(ticker_symbol)

        # Get historical market data (daily)
        # Options for period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        history = etf.history(period="3y", interval="1d")

        if history.empty:
            print(f"No data found for ticker: {ticker_symbol}")
            return

        return history
    except Exception as e:
        print(f"An error occurred: {e}")