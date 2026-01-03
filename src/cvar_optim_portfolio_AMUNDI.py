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


def plot_return_distributions(simulated_returns, nonzero_weights):

    # 1. Calculate the returns
    simulated_returns = np.sum(df_simulated[list(nonzero_weights.keys())], axis=1).values

    np_weights = np.array(list(nonzero_weights.values()))
    simulated_returns = df_simulated[list(nonzero_weights.keys())].values @ np_weights

    # 2. Fit the Student's t-distribution
    # params = (df, loc, scale)
    params = t.fit(simulated_returns)
    fitted_params.append(params)

    # # 3. Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the histogram of actual data for context
    plt.hist(simulated_returns, bins=50, density=True, alpha=0.5, color='gray', label='Simulated Returns')

    # Create a smooth x-axis range for the PDF curve
    x = np.linspace(min(simulated_returns), max(simulated_returns), 100)
    pdf_fitted = t.pdf(x, *params)

    # Plot the fitted T-distribution
    plt.plot(x, pdf_fitted, '-', lw=2, label=f'Fitted T-Dist (df={params[0]:.2f})')

    plt.title("Implied Portfolio Return Distribution")
    plt.xlabel("Returns")
    plt.ylabel("Density")
    plt.grid()
    plt.legend()

    plt.savefig(LATEST_DIR / "implied_portfolio_return_distribution.png")

    # rewards is a pandas Series
    nu, mu, sigma = params

    pd.DataFrame({
        "nu": [nu],
        "mu": [mu],
        "sigma": [sigma]
    }).to_json(LATEST_DIR / "reward_t_fit.json", orient="records")

    # Generate time series data from the distribution
    n_samples = 252  # e.g., one trading year
    n_trajectories = 100
    data = []
    for _ in range(n_trajectories):
        temp = t.rvs(df=nu, loc=mu, scale=sigma, size=n_samples) + 1
        data.append(np.cumprod(temp, axis=0))
    
    time = np.arange(n_samples)


    # Plot
    plt.figure(figsize=(12, 6))
    for trajectory in data:
        plt.plot(time, trajectory, label='Simulated Returns', linewidth=1.5, color='steelblue')

    plt.xlabel('Time (days)')
    plt.ylabel('Returns')
    plt.title(f'Simulated portfolio trajectories')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LATEST_DIR / 'forecasted_returns.png', dpi=150)
    plt.show()

    return simulated_returns
    

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
        data[ticker] = get_etf_data(ticker)['Close'].pct_change().dropna()

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
    if res.success:
        opt_weights = res.x
        port_returns = sim_final @ opt_weights
        expected_return = np.mean(port_returns)
        daily_vol = np.std(port_returns)

        # CVaR
        var_alpha = np.percentile(port_returns, (1-confidence_level)*100)
        cvar_alpha = port_returns[port_returns <= var_alpha].mean()

        # Annualize
        annual_return = (1 + expected_return)**252 - 1
        annual_vol = daily_vol * np.sqrt(252)
        annual_cvar = cvar_alpha * np.sqrt(252)


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

        print(f"\nExpected Daily Return: {expected_return:.2%}")
        print(f"Daily Volatility: {daily_vol:.2%}")
        print(f"Daily CVaR ({int(confidence_level*100)}%): {cvar_alpha:.2%}")

        print(f"\nAnnualized Return: {annual_return:.2%}")
        print(f"Annualized Volatility: {annual_vol:.2%}")
        print(f"Annualized CVaR ({int(confidence_level*100)}%): {annual_cvar:.2%}")
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


    simulated_returns = plot_return_distributions(df_simulated, nonzero_weights)

    df_return_distribution = pd.DataFrame({'realised_returns': simulated_returns})
    df_return_distribution.to_csv(LATEST_DIR / "implied_portfolio_return_distribution.csv", index=False)
