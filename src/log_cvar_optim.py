import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import t, norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import datetime
import os
from pathlib import Path
import json

RUN_DATE = datetime.date.today().isoformat()

BASE_OUTPUT = Path("outputs")
LATEST_DIR = BASE_OUTPUT / "latest"
HISTORY_DIR = BASE_OUTPUT / "history" / RUN_DATE


def solve_optimisation(confidence_level=0.95, max_weight=1.0, min_weight=0.0, max_daily_cvar=0.01):

  # --- Constraints --- #
  constraints = [
      {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},           # sum of weights = 1
      {'type': 'ineq', 'fun': lambda w: -portfolio_cvar(w, sim_final, confidence_level) - (-max_daily_cvar)}  # CVaR <= max_daily_cvar% dayly loss
  ]

  # --- Bounds: min/max allocation per asset --- #
  bounds = tuple((min_weight, max_weight) for _ in range(sim_final.shape[1]))

  # --- Initial guess --- #
  init_guess = np.ones(sim_final.shape[1]) / sim_final.shape[1]

  # --- Optimization --- #
  res = minimize(
      negative_expected_utility,
      init_guess,
      args=(sim_final, identity),
      method='SLSQP',
      bounds=bounds,
      constraints=constraints,
      options={'disp': True, 'maxiter': 10000, 'ftol':1e-9}
  )

  return res


def identity(portfolio_returns):
  return portfolio_returns


# --- CVaR Calculation --- #
def portfolio_cvar(weights, returns, alpha=0.95):
    port_returns = returns @ weights
    var_alpha = np.percentile(port_returns, (1 - alpha) * 100)
    cvar = port_returns[port_returns <= var_alpha].mean()
    return -cvar  # negative because used as constraint


# --- Objective: Negative Expected Utility --- #
def negative_expected_utility(weights, returns, utility_fn=identity):
    port_returns = returns @ weights
    return -np.mean(utility_fn(port_returns))


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


if __name__ == "__main__":


    for d in [LATEST_DIR, HISTORY_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(d)


    tickers=["MWRE.DE", "LYMS.DE", "LYP6.DE", "LYYA.DE", "AHYQ.DE", "10AL.DE",
            "F500.DE", "AE5A.DE", "A4H8.DE", "AMEW.DE", "XAMB.DE", "CB3G.DE",
            "LCUJ.DE", "MWSH.DE", "LYPS.DE", "LYSX.DE", "LYMS.DE", "L8I3.DE",
            "ECR3.DE", "6TVM.DE", "AUM5.DE", "MIVB.DE", "ZPAB.DE", "LYBK.DE",
            "AMEM.DE", "PABW.DE", "PR1R.DE", "V50A.DE", "EGV3.DE", "CEUG.DE",
            "LYOR.DE", "PRAR.DE", "A4HC.DE", "LYQ3.DE", "EGV5.DE", "LYM8.DE",
            "LYXD.DE", "MTDD.DE", "LYTR.DE", "CG1G.DE", "LWCR.DE", "INDA.DE",
            "EMXC.DE", "6AQQ.DE", "AMEI.DE", "LYMD.DE", "LYQ6.DE", "NADB.DE",
            "C001.DE", "LYY0.DE", "E15H.DE", "UCT2.DE", "SADM.DE", "WEBD.DE",
            "LYY7.DE", "PR10.DE", "PR1J.DE", "FRNE.DE", "LYEB.DE", "KX1G.DE",
            "X1GD.DE", "AME6.DE", "LYP2.DE", "PR1C.DE", "LYM7.DE", "NADQ.DE",
            "AMEA.DE", "H1D5.DE", "GC40.DE", "SADU.DE", "ACU2.DE", "AE50.DE",
            "UCRP.DE", "V50D.DE", "LYY4.DE", "HNDX.DE", "LCUK.DE", "LHKG.DE",
            "LCHI.DE", "LYM9.DE", "AMEQ.DE", "JARI.DE", "TIUP.DE", "WELH.DE",
            "WELW.DE", "DJAD.DE", "WELJ.DE", "LEER.DE", "TTPX.DE", "J1GR.DE"]

    data = {}
    for ticker in tickers:
        # Use log returns instead of pct_change
        prices = get_etf_data(ticker)['Close']
        data[ticker] = np.log(prices / prices.shift(1)).dropna()

    print(data.keys(), len(data.keys()))

    df = pd.DataFrame()
    df = df.from_dict(data)
    # drop columns with no data (all NaNs) so tickers list and data align
    df = df.dropna(axis=1, how='all')

    # preserve original tickers order and identify removed tickers
    original_tickers = tickers.copy()
    assets_sorted = [t for t in original_tickers if t in df.columns]
    removed_tickers = [t for t in original_tickers if t not in df.columns]

    if removed_tickers:
        print(f"Removed {len(removed_tickers)} tickers due to missing data: {removed_tickers}")
        # record removed tickers for traceability
        (LATEST_DIR / "removed_tickers.json").write_text(json.dumps({"removed": removed_tickers}, indent=2))
        (HISTORY_DIR / "removed_tickers.json").write_text(json.dumps({"removed": removed_tickers}, indent=2))

    # 1. COLLAPSE DATA
    df_daily = df.groupby(df.index.date).first().dropna()
    df_daily.index = pd.to_datetime(df_daily.index)

    # reorder columns to match original tickers order (filtered)
    if assets_sorted:
        df_daily = df_daily[assets_sorted]

    # 2. FIT STUDENT'S T AND GAUSSIAN COPULA
    n_vars = df_daily.shape[1]
    fitted_params = [] # Store (df, loc, scale) for each column

    # Step: Transform historical data to Uniform [0, 1] using fitted t-distributions
    u_data = np.zeros_like(df_daily.values)

    for i in range(n_vars):
        col_data = df_daily.iloc[:, i].values

        # Fit the Student's t-distribution
        # returns: df (degrees of freedom), loc (mean), scale (stdev)
        params = t.fit(col_data)
        fitted_params.append(params)

        # Transform data to [0, 1] space using the fitted CDF
        u_data[:, i] = t.cdf(col_data, *params)

    # Step: Transform to Gaussian space to find the Correlation Matrix
    z_data = norm.ppf(np.clip(u_data, 1e-7, 1-1e-7))
    corr_matrix = np.corrcoef(z_data, rowvar=False)

    # GENERATE NEW SAMPLES (Simulation)
    n_sim = int(1e5)
    # Simulate correlated normal variables
    sim_z = np.random.multivariate_normal(np.zeros(n_vars), corr_matrix, n_sim)
    # Convert back to Uniform [0, 1]
    sim_u = norm.cdf(sim_z)

    # Convert back to original scale using the Inverse CDF (ppf) of Student's t
    sim_final = np.zeros_like(sim_u)
    for i in range(n_vars):
        sim_final[:, i] = t.ppf(sim_u[:, i], *fitted_params[i])

    # use the assets that survived cleaning / aggregation (preserving original order)
    assets = assets_sorted

    df_sim = pd.DataFrame(sim_final, columns=assets)
    df_simulated = pd.DataFrame(sim_final, columns=assets)

    confidence_level=0.95
    max_weight=1.0
    min_weight=0.0
    max_daily_cvar=0.015

    res = solve_optimisation(confidence_level,
                            max_weight,
                            min_weight,
                            max_daily_cvar)

    # --- Extract Results --- #
    # --- Inside the if res.success: block --- #
    if res.success:
        opt_weights = res.x
        port_log_returns = sim_final @ opt_weights # This is a distribution of log returns
        expected_log_return = np.mean(port_log_returns)
        daily_vol = np.std(port_log_returns)

        # CVaR (Calculated in log space)
        var_alpha = np.percentile(port_log_returns, (1-confidence_level)*100)
        cvar_alpha_log = port_log_returns[port_log_returns <= var_alpha].mean()

        # Annualize Log Returns (Additive)
        annual_log_return = expected_log_return * 252
        annual_vol = daily_vol * np.sqrt(252)
        
        # Convert Log Annual Return back to Arithmetic Annual Return for reporting
        # This represents the actual percentage growth you expect to see
        annual_return_arithmetic = np.exp(annual_log_return + 0.5 * annual_vol**2) - 1
        
        # Daily CVaR expressed as a simple percentage loss (more intuitive)
        cvar_alpha_pct = np.exp(cvar_alpha_log) - 1

        print("\n--- Optimal Hybrid Portfolio ---\n")
        nonzero_weights = {}

        if len(opt_weights) != len(assets):
            print(f"Warning: number of optimized weights ({len(opt_weights)}) != number of assets ({len(assets)}).\n"
                  "Iterating only over the minimum to avoid indexing errors.")

        for i, w in enumerate(opt_weights):
            if i >= len(assets):
                break
            if np.round(w, 2) > 0.00:
                print(f"{assets[i]} Weight: {w:.2%}")
                nonzero_weights[assets[i]] = w
    else:
        print("Optimization failed:", res.message)


    # Compare Rank Correlations
    print('\n--- Testing how well the copula model fits the data ---\n')
    print("Original Spearman Correlation:")
    print(df_daily.corr(method='spearman').iloc[0,1])

    print("Simulated Spearman Correlation:")
    print(df_sim.corr(method='spearman').iloc[0,1])

    # ensure weights/tickers align when saving
    num_to_use = min(len(opt_weights), len(assets))
    weights_df = pd.DataFrame({
    "ticker": assets[:num_to_use],
    "weight": opt_weights[:num_to_use]
    }).query("weight > 0.0001")

    weights_df.to_csv(LATEST_DIR / "weights.csv", index=False)
    weights_df.to_csv(HISTORY_DIR / "weights.csv", index=False)

    metrics_df = pd.DataFrame([{
    "expected_daily_return": expected_return,
    "daily_volatility": daily_vol,
    "daily_cvar": cvar_alpha,
    "annual_return": annual_return,
    "annual_volatility": annual_vol,
    "annual_cvar": annual_cvar,
    "confidence_level": confidence_level,
    "max_daily_cvar_constraint": max_daily_cvar,
    }])

    metrics_df.to_csv(LATEST_DIR / "metrics.csv", index=False)
    metrics_df.to_csv(HISTORY_DIR / "metrics.csv", index=False)

    corr_df = pd.DataFrame({
    "original_spearman": [df_daily.corr(method="spearman").iloc[0,1]],
    "simulated_spearman": [df_sim.corr(method="spearman").iloc[0,1]]
    })

    corr_df.to_csv(LATEST_DIR / "correlation.csv", index=False)
    corr_df.to_csv(HISTORY_DIR / "correlation.csv", index=False)

    

    summary = {
        "run_date": RUN_DATE,
        "expected_daily_return": expected_return,
        "daily_volatility": daily_vol,
        "daily_cvar": cvar_alpha,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "annual_cvar": annual_cvar,
        "num_assets": len(weights_df),
    }

    with open(LATEST_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(HISTORY_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


    # on so now we could also save the implied return distribution. 
    # to do so, we have to isolate, fro the simulated outputs, the realisations
    # of the assets present in the portfolio and then combine them linearly 
    # with the optimal weights found, to get the implied portfolio return distribution.
    print(np.sum(df_simulated[nonzero_weights.keys()].values, axis=1).shape)

    plt.hist(np.sum(df_simulated[nonzero_weights.keys()].values, axis=1))
    plt.savefig(LATEST_DIR / "implied_portfolio_return_distribution.png")