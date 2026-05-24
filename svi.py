import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

# Loading Data
def load_iv_data(ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", "iv", f"{ticker}_iv.parquet")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} contracts")
    return df

# SVI Formula
def svi(k, a, b, rho, m, sigma):
    """
    k     : log-moneyness array
    Returns total implied variance w(k)
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# Fitting SVI for One Expiration
def fit_svi_single(df_exp):
    """
    df_exp: DataFrame for one expiration
    Returns fitted parameters (a, b, rho, m, sigma) or None if fit fails
    """
    k = df_exp["log_moneyness"].values
    T = df_exp["dte"].iloc[0] / 365.0
    iv = df_exp["iv_computed"].values

    # Convert IV to total variance
    w = iv**2 * T

    # Initial parameter guess
    p0 = [0.04, 0.1, -0.7, 0.0, 0.1]

    # Parameter bounds
    bounds = (
        [-1,    0,   -0.999,  -1,   1e-4],   # lower
        [ 1,    2,    0.999,   1,   1.0 ]    # upper
    )

    try:
        params, _ = curve_fit(svi, k, w, p0=p0, bounds=bounds,
                              maxfev=10000, method="trf")
        return params
    except Exception:
        return None
    
# Fit Across All Expiration
def fit_all_expirations(df):
    results = []
    expirations = df["expiration"].unique()
    print(f"Fitting SVI for {len(expirations)} expirations...")

    for exp in sorted(expirations):
        df_exp = df[df["expiration"] == exp].copy()
        dte = df_exp["dte"].iloc[0]

        if len(df_exp) < 5:
            print(f"  {exp} ({dte} DTE): skipped — too few points ({len(df_exp)})")
            continue

        params = fit_svi_single(df_exp)

        if params is None:
            print(f"  {exp} ({dte} DTE): fit failed")
            continue

        a, b, rho, m, sigma = params
        results.append({
            "expiration": exp,
            "dte": dte,
            "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma
        })
        print(f"  {exp} ({dte} DTE): fitted — a={a:.4f} b={b:.4f} rho={rho:.4f}")

    return pd.DataFrame(results)

# Generate Smooth Surface
def generate_surface(params_df, k_grid=None):
    if k_grid is None:
        k_grid = np.linspace(-0.3, 0.3, 100)

    surface_rows = []
    for _, row in params_df.iterrows():
        T = row["dte"] / 365.0
        w_grid = svi(k_grid, row["a"], row["b"], row["rho"], row["m"], row["sigma"])
        w_grid = np.maximum(w_grid, 0)  # Ensure non-negative variance
        iv_grid = np.sqrt(w_grid / T)

        for k, iv in zip(k_grid, iv_grid):
            surface_rows.append({
                "expiration": row["expiration"],
                "dte": row["dte"],
                "log_moneyness": k,
                "iv_surface": iv
            })

    return pd.DataFrame(surface_rows)

# Saving
def save_surface(params_df, surface_df, ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "data", "surface")
    os.makedirs(out_dir, exist_ok=True)

    params_df.to_parquet(os.path.join(out_dir, f"{ticker}_svi_params.parquet"))
    surface_df.to_parquet(os.path.join(out_dir, f"{ticker}_svi_surface.parquet"))
    print(f"Saved SVI params and surface to {out_dir}")

# Main Runner
if __name__ == "__main__":
    df = load_iv_data("SPY")

    params_df = fit_all_expirations(df)
    print(f"\nSuccessfully fitted {len(params_df)} expirations")
    print(params_df[["expiration", "dte", "a", "b", "rho"]].to_string(index=False))

    surface_df = generate_surface(params_df)
    print(f"\nGenerated surface with {len(surface_df)} points")

    save_surface(params_df, surface_df, "SPY")