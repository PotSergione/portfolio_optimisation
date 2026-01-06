import pandas as pd
import numpy as np
from scipy.stats import t, norm, multivariate_t
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
import datetime
from pathlib import Path
import json
from send_email import send_email
from get_data import get_etf_data


RUN_DATE = datetime.date.today().isoformat()

BASE_OUTPUT = Path("outputs")
LATEST_DIR = BASE_OUTPUT / "latest"
HISTORY_DIR = BASE_OUTPUT / "history" / RUN_DATE


def check_lower_tail_dependence_robust(data, threshold=0.05):
    # 1. Transform to ranks [0, 1]
    u = data.rank(pct=True).dropna()
    n_vars = u.shape[1]
    
    tail_probs = []
    # Check first 10 pairs (or fewer if n_vars < 10) to get an average
    for i in range(min(10, n_vars - 1)):
        # 2. Extract raw numpy arrays to bypass the "duplicate labels" error
        a_vals = u.iloc[:, i].values
        b_vals = u.iloc[:, i+1].values
        
        # 3. Perform boolean logic on numpy arrays
        a_in_tail = a_vals <= threshold
        
        if a_in_tail.any():
            # How many times was B also in the tail when A was?
            joint_tail = (a_vals <= threshold) & (b_vals <= threshold)
            prob = joint_tail.sum() / a_in_tail.sum()
            tail_probs.append(prob)
            
    return np.mean(tail_probs) if tail_probs else 0.0


def t_copula_loglik(nu):
    # Transform Uniform data to t-scales with candidate degrees of freedom = nu
    t_quantiles = t.ppf(np.clip(u_data, 1e-7, 1-1e-7), df=nu)
    
    # Calculate Log Likelihood
    mv_logpdf = multivariate_t.logpdf(t_quantiles, df=nu, shape=R)
    univariate_logpdf = t.logpdf(t_quantiles, df=nu).sum(axis=1)
    
    return -np.sum(mv_logpdf - univariate_logpdf)


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
    

def solve_optimisation(confidence_level=0.95, max_weight=1.0, min_weight=0.0, max_daily_cvar=0.01, sim_final=None):

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
    assets = [t for t in original_tickers if t in df.columns]
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
    if assets:
        df_daily = df_daily[assets]

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

    # --- ROBUST CORRELATION ESTIMATION ---

    # 2. Estimate R using Kendall's Tau
    # This removes the "Gaussian proxy" bias.
    print("Calculating Kendall's Tau matrix (this may take a moment)...")
    tau_matrix = df_daily.corr(method='kendall').values

    # Convert Kendall's Tau to Pearson Correlation for Elliptical Copulas
    # Formula: Rho = sin(pi/2 * Tau)
    R = np.sin((np.pi / 2) * tau_matrix)

    # 3. Fix Non-Positive Definite Matrix (Common with Kendall's transform)
    # High-dimensional R constructed element-wise is often not positive definite.
    eigvals, eigvecs = np.linalg.eigh(R)
    if np.min(eigvals) < 0:
        print("Adjusting matrix to Positive Definite...")
        # Floor small negative eigenvalues to a tiny positive number
        eigvals = np.maximum(eigvals, 1e-7) 
        # Reconstruct R
        R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Normalize so diagonal is exactly 1.0 again
        d = np.sqrt(np.diag(R))
        R /= np.outer(d, d)

    # Increase upper bound to 300 to see if it truly wants to be Gaussian
    # If it stops at ~50-300, it's effectively Gaussian.
    res = minimize_scalar(t_copula_loglik, bounds=(2.5, 300), method='bounded')
    copula_nu = res.x

    print(f"Optimal Copula Degrees of Freedom: {copula_nu:.2f}")

    # --- STEP 3: HIGH-SPEED SIMULATION ---
    n_sim = int(1e6) 

    print(f"Simulating {n_sim} scenarios with nu={copula_nu:.2f}...")

    # 1. Simulate Multivariate Normal samples Z ~ N(0, R)
    # We use the Cholesky decomposition of R for speed (R = L @ L.T)
    # This is faster than np.random.multivariate_normal for large N
    L = np.linalg.cholesky(R) 
    uncorrelated_z = np.random.standard_normal((n_sim, n_vars))
    sim_z_gaussian = uncorrelated_z @ L.T

    # 2. Simulate Chi-Square variable W ~ Chi2(nu)
    # We need one W per simulation scenario (shared across all N assets in that scenario)
    # This is the "shock" factor that scales all assets simultaneously
    w = np.random.chisquare(df=copula_nu, size=(n_sim, 1))

    # 3. Construct Multivariate t samples
    # Formula: T = Z / sqrt(W / nu)
    sim_t_raw = sim_z_gaussian / np.sqrt(w / copula_nu)

    # 4. Convert to Uniform [0, 1] (The Copula Step)
    # We use the t-CDF with the COPULA'S degrees of freedom
    sim_u = t.cdf(sim_t_raw, df=copula_nu)

    # 5. Convert to Prices (The Marginal Step)
    # We use the Inverse CDF (ppf) of the MARGINALS
    sim_final = np.zeros_like(sim_u)

    for i in range(n_vars):
        # Retrieve the marginal params (df, loc, scale) we fitted earlier
        m_df, m_loc, m_scale = fitted_params[i]
        
        # Map u -> returns/prices using the specific marginal distribution
        sim_final[:, i] = t.ppf(sim_u[:, i], df=m_df, loc=m_loc, scale=m_scale)

    # Create DataFrame
    df_sim = pd.DataFrame(sim_final, columns=df_daily.columns)

    print("Simulation Complete.")

    for i in range(n_vars):
        sim_final[:, i] = t.ppf(sim_u[:, i], *fitted_params[i])

    df_sim = pd.DataFrame(sim_final, columns=assets)
    df_simulated = pd.DataFrame(sim_final, columns=assets)

    confidence_level=0.95
    max_weight=1.0
    min_weight=0.0
    max_daily_cvar=0.015

    res = solve_optimisation(confidence_level,
                            max_weight,
                            min_weight,
                            max_daily_cvar, 
                            sim_final)

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

    # Remove duplicate timestamps by keeping the first occurrence
    df_daily = df_daily[~df_daily.index.duplicated(keep='first')]

    # Now call the function
    print(f"\nHistorical Avg Tail Prob: {check_lower_tail_dependence_robust(df_daily):.4f}")
    print(f"Simulated Tail Prob:  {check_lower_tail_dependence_robust(df_sim):.4f}")


    # BUILDING EMAIL MESSAGE TO STREAM RESULTS
    MSG = f"{datetime.datetime.now()}\n" +\
    '\n--- Optimal Hybrid Portfolio Found --- \n' +\
    f"\nExpected Daily Return: {expected_return:.2%}\n" +\
    f"Daily Volatility: {daily_vol:.2%}\n"+\
    f"Daily CVaR ({int(confidence_level*100)}%): {cvar_alpha:.2%}\n"+\
    f"\nAnnualized Return: {annual_return:.2%}\n"+\
    f"Annualized Volatility: {annual_vol:.2%}\n"+\
    f"Annualized CVaR ({int(confidence_level*100)}%): {annual_cvar:.2%}\n" +\
    '\n---  Copula model fit of the data ---\n' +\
    "\nOriginal Spearman Correlation:\n" +\
    f"{df_daily.corr(method='spearman').iloc[0,1]}\n"+\
    "Simulated Spearman Correlation:\n"+\
    f"{df_sim.corr(method='spearman').iloc[0,1]}\n"+\
    "\n--- Optimal picks ---\n\n"

    for i, w in enumerate(opt_weights):
        if np.round(w, 2) > 0.00:
            MSG += f"{tickers[i]} Weight: {w:.2%}\n"

    df_corr_picks = df_simulated[weights_df['ticker']].corr()
    df_corr_picks.to_csv(LATEST_DIR / "correlation_selected_assets.csv", index=True)

    MSG += f"\n--- Correlation matrix of selected assets ---\n\n {df_simulated[weights_df['ticker']].corr()}\n"
    MSG += f"\nHistorical Avg Tail Prob for 10 pairs: {check_lower_tail_dependence_robust(df_daily):.4f}\n"
    MSG += f"Simulated Tail Prob for 10 pairs:  {check_lower_tail_dependence_robust(df_sim):.4f}"
    MSG += "\n\nRegards,\nHybrid CVaR Optimizer Bot"

    send_email(MSG)