from scipy.stats import norm
import numpy as np
from scipy.optimize import brentq

# Black-Scholes Formula
def black_scholes(S, K, T, r, sigma, option_type="call", q=0.013):
    """
    S     : spot price
    K     : strike price
    T     : time to expiry in years
    r     : risk-free rate (annualized)
    sigma : volatility (annualized)

    q: continuous dividend yield — SPY pays ~1.3% annually
    """
    if sigma <= 0 or T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


# Vega
def vega(S, K, T, r, sigma, q=0.013):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

# Newton-Raphson IV Solver
def newton_raphson_iv(market_price, S, K, T, r, option_type="call",
                      q=0.013, max_iter=100, tol=1e-6):
    sigma = 0.3
    for _ in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type, q)
        v = vega(S, K, T, r, sigma, q)
        if v < 1e-10:
            return None
        sigma_new = sigma - (price - market_price) / v
        if abs(sigma_new - sigma) < tol:
            return sigma_new if 0 < sigma_new < 10 else None
        sigma = sigma_new
        if sigma <= 0:
            return None
    return None

# Brent's Method Fallback
def brent_iv(market_price, S, K, T, r, option_type="call", q=0.013):
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type, q) - market_price
    try:
        return brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=500)
    except (ValueError, RuntimeError):
        return np.nan
    
# Combined Solver
def compute_iv(market_price, S, K, T, r, option_type="call", q=0.013):
    iv = newton_raphson_iv(market_price, S, K, T, r, option_type, q)
    if iv is None:
        iv = brent_iv(market_price, S, K, T, r, option_type, q)
    return iv

# Apply Across DataFrame
def compute_iv_surface(df, r=0.043, q=0.013):
    ivs = []
    for _, row in df.iterrows():
        T = row["dte"] / 365.0
        if T <= 0:
            ivs.append(np.nan)
            continue
        iv = compute_iv(
            market_price=row["mid"],
            S=row["spot"],
            K=row["strike"],
            T=T,
            r=r,
            option_type=row["type"],
            q=q
        )
        ivs.append(iv)
    df = df.copy()
    df["iv_computed"] = ivs
    df = df.dropna(subset=["iv_computed"])
    df = df[df["iv_computed"] > 0]
    print(f"Computed IV for {len(df)} contracts")
    return df

# Save and Main Runner
def save_iv(df, ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "data", "iv")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}_iv.parquet")
    df.to_parquet(path)
    print(f"Saved IV data to {path}")

if __name__ == "__main__":
    import pandas as pd
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", "clean", "SPY_clean.parquet")
    df = pd.read_parquet(path)

    print(f"Computing IV for {len(df)} contracts...")
    df = compute_iv_surface(df, r=0.043, q=0.013)

    print("\nSample output:")
    print(df[["strike", "dte", "type", "mid", "impliedVolatility", "iv_computed"]].head(10))

    # Diagnostic
    df["iv_ratio"] = df["iv_computed"] / df["impliedVolatility"]
    print(f"\nIV ratio stats (our IV / yfinance IV):")
    print(f"  Mean:   {df['iv_ratio'].mean():.3f}")
    print(f"  Median: {df['iv_ratio'].median():.3f}")
    print(f"  Std:    {df['iv_ratio'].std():.3f}")
    print(f"\nIV ratio by type:")
    print(df.groupby("type")["iv_ratio"].mean())

    # Filter out bad IV ratios (keep contracts where our IV is within 50% of yfinance)
    df = df[(df["iv_ratio"] > 0.5) & (df["iv_ratio"] < 2.0)]
    print(f"\nContracts after IV ratio filter: {len(df)}")

    # Drop diagnostic columns before saving
    df = df.drop(columns=["iv_ratio"], errors="ignore")

    print(f"\nCorrelation with yfinance IV: {df['impliedVolatility'].corr(df['iv_computed']):.4f}")

    save_iv(df, "SPY")