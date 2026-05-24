import pandas as pd
import numpy as np
import os

# Loading Raw Data
def load_raw(ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", "raw", f"{ticker}_raw.parquet")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} contracts")
    return df

# The 6 Filters
def apply_filters(df):
    n0 = len(df)

    # Filter 1 — Remove zero bid (no market)
    df = df[df["bid"] > 0]
    print(f"  After zero-bid filter:       {len(df)} ({len(df)/n0:.1%})")

    # Filter 2 — Remove zero volume (stale contracts)
    df = df[df["volume"] > 0]
    print(f"  After zero-volume filter:    {len(df)} ({len(df)/n0:.1%})")

    # Filter 3 — Compute mid price, remove non-positive
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[df["mid"] > 0]
    print(f"  After zero-mid filter:       {len(df)} ({len(df)/n0:.1%})")

    # Filter 4 — Remove wide spreads (ask - bid > 50% of mid)
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[df["spread_pct"] < 0.5]
    print(f"  After wide-spread filter:    {len(df)} ({len(df)/n0:.1%})")

    # Filter 5 — Remove extreme moneyness (keep 0.7 to 1.3 spot/strike)
    df["moneyness"] = df["spot"] / df["strike"]
    df = df[(df["moneyness"] >= 0.7) & (df["moneyness"] <= 1.3)]
    print(f"  After moneyness filter:      {len(df)} ({len(df)/n0:.1%})")

    # Filter 6 — Remove contracts with impliedVolatility > 5 (500%) or = 0
    df = df[(df["impliedVolatility"] > 0) & (df["impliedVolatility"] < 5)]
    print(f"  After IV sanity filter:      {len(df)} ({len(df)/n0:.1%})")

    return df

# Compute Log-Moneyness
def compute_log_moneyness(df):
    df["log_moneyness"] = np.log(df["spot"] / df["strike"])
    return df

# Saving
def save_clean(df, ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "data", "clean")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}_clean.parquet")
    df.to_parquet(path)
    print(f"Saved {len(df)} clean contracts to {path}")

# Main Runner
if __name__ == "__main__":
    df = load_raw("SPY")
    print("\nApplying filters...")
    df = apply_filters(df)
    df = compute_log_moneyness(df)
    save_clean(df, "SPY")
    print(f"\nFinal columns: {df.columns.tolist()}")
    print(df[["strike", "dte", "bid", "ask", "mid", "log_moneyness", "impliedVolatility", "type"]].head(10))