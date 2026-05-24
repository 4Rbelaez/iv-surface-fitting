import yfinance as yf
import pandas as pd
from datetime import datetime
import os 

# Underlying Price
def get_underlying_price(ticker="SPY"):
    stock = yf.Ticker(ticker)
    price = stock.fast_info["last_price"]
    return price

# Get Expirations
def get_expirations(ticker="SPY", min_dte=7, max_dte=365):
    stock = yf.Ticker(ticker)
    expirations = stock.options  # Returns tuple of date strings "YYYY-MM-DD"

    today = datetime.today()
    filtered = []
    for exp in expirations:
        dte = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        if min_dte <= dte <= max_dte:
            filtered.append((exp, dte))

    return filtered  # Lift of (expiration_string, dte) tuples

# Fetch One Expiration
def fetch_chain(ticker, expiration):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)

    calls_df = chain.calls.copy()
    puts_df = chain.puts.copy()

    calls_df["type"] = "call"
    puts_df["type"] = "put"

    combined = pd.concat([calls_df, puts_df], ignore_index=True)
    combined["expiration"] = expiration
    return combined

# Fetch Full Surface
def fetch_full_surface(ticker="SPY", min_dte=7, max_dte=365):
    print(f"Fetching {ticker} underlying price...")
    spot = get_underlying_price(ticker)
    print(f"  Spot: {spot: .2f}")

    expirations = get_expirations(ticker, min_dte, max_dte)
    print(f"   Found {len(expirations)} expirations between {min_dte} and {max_dte} DTE")

    frames = []
    for exp, dte in expirations:
        print(f" Fetching {exp} ({dte} DTE)...")
        try:
            df = fetch_chain(ticker, exp)
            df["dte"] = dte
            df["spot"] = spot 
            frames.append(df)
        except Exception as e:
            print(f"      Failed: {e}")
            continue

    full = pd.concat(frames, ignore_index=True)
    print(f"   Total contracts fetched: {len(full)}")
    return full, spot 

# Saving
def save_raw(df, ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}_raw.parquet")
    df.to_parquet(path)
    print(f"Saved raw chain to {path}")

# Main Runner
if __name__ == "__main__":
    df, spot = fetch_full_surface("SPY")
    save_raw(df, "SPY")
    print(df.head())
    print(df.columns.tolist())



