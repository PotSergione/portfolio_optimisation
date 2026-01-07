from get_data import  get_etf_data
import datetime
import pandas as pd
from pathlib import Path
import os
import numpy as np


RUN_DATE = datetime.date.today().isoformat()

BASE_OUTPUT = Path("outputs")
LATEST_DIR = BASE_OUTPUT / "latest"
HISTORY_DIR = BASE_OUTPUT / "history" / RUN_DATE


def test_against_benchmarks(benchmarks :list = [] ):

    # gets the portoflios from the history folder and computes the returns
    # of the portfolio. We then save the returns from the benchmarks and the
    # return from the portfolio for plotting in the dashboard.

    historical_portfolios = sorted(os.listdir(BASE_OUTPUT / "history"))

    returns = {}
    for benchmark in benchmarks:
        data = get_etf_data(benchmark)['Close']
        filter_date = pd.to_datetime(historical_portfolios[0]).tz_localize('Europe/Berlin')
        returns[benchmark] = data.loc[filter_date:].pct_change().fillna(0.0) + 1.0

    # now we filter the data from the portfolio where the dates match
    # this is necessary because the portfolio is run daily
    # but the markets run only 252 days a year. 

    portfolio_returns = {}
    for date in data.index:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in historical_portfolios:
            portfolio_data = pd.read_csv(BASE_OUTPUT / "history" / date_str / "weights.csv")
            # we now get the value of each asset in that date and multiply it by its weight. 
            # in this way, we get the value of the portfolio in that date. 
            # to get the final returns, normalising with respect to the initial value.
            for _, row in portfolio_data.iterrows():
                ticker = row['ticker']
                weight = row['weight']
                asset_price = get_etf_data(ticker).loc[date]['Close']
                if date_str not in portfolio_returns: 
                    portfolio_returns[date_str] = 0.0
                
                portfolio_returns[date_str] += weight * asset_price

    df = pd.DataFrame.from_dict(portfolio_returns, orient='index', columns=['portfolio_value'])
    df_rescaled = df.pct_change().fillna(0.0) + 1.0
    df_rescaled.index = pd.to_datetime(df_rescaled.index).tz_localize('Europe/Berlin')

    df_benchmarks = pd.DataFrame.from_dict(returns)
    df_benchmarks['CVAR_STRAT'] = df_rescaled['portfolio_value']
    # now saving to latest for plotting in dashboard

    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    df_benchmarks.to_csv(LATEST_DIR / "benchmark_comparison.csv")




if __name__ == "__main__":  
    test_against_benchmarks(["AUM5.DE", "F500.DE", "EMXC.DE"])