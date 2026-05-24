import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Loading Data
def load_data(ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    iv_df = pd.read_parquet(os.path.join(data_dir, "iv", f"{ticker}_iv.parquet"))
    surface_df = pd.read_parquet(os.path.join(data_dir, "surface", f"{ticker}_svi_surface.parquet"))
    params_df = pd.read_parquet(os.path.join(data_dir, "surface", f"{ticker}_svi_params.parquet"))

    return iv_df, surface_df, params_df

# 3D Surface Plot
def plot_3d_surface(surface_df, iv_df, ticker="SPY"):
    fig = plt.figure(figsize=(14, 9), facecolor="#0f0f0f")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f0f0f")

    # Pivot surface into grid
    pivot = surface_df.pivot(index="dte", columns="log_moneyness", values="iv_surface")
    K = pivot.columns.values
    T = pivot.index.values
    K_grid, T_grid = np.meshgrid(K, T)
    IV_grid = pivot.values

    # Plot smooth surface
    surf = ax.plot_surface(K_grid, T_grid, IV_grid,
                           cmap="plasma", alpha=0.85,
                           linewidth=0, antialiased=True)

    # Scatter raw IV points on top
    ax.scatter(iv_df["log_moneyness"], iv_df["dte"], iv_df["iv_computed"],
               color="white", s=1, alpha=0.3, zorder=5)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1)
    cbar.set_label("Implied Volatility", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Labels
    ax.set_xlabel("Log-Moneyness", color="white", labelpad=10)
    ax.set_ylabel("DTE", color="white", labelpad=10)
    ax.set_zlabel("Implied Volatility", color="white", labelpad=10)
    ax.set_title(f"{ticker} Implied Volatility Surface (SVI Fit)",
                 color="white", fontsize=13, pad=20)

    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    return fig

# Vol Skew Slices
def plot_skew_slices(surface_df, iv_df, params_df, ticker="SPY"):
    # Pick 4 expirations spread across the term structure
    dtes = sorted(params_df["dte"].unique())
    selected = [dtes[i] for i in [0, 4, 10, -1]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0f0f0f")
    fig.suptitle(f"{ticker} Volatility Skew by Expiration",
                 color="white", fontsize=13)
    axes = axes.flatten()

    for i, dte in enumerate(selected):
        ax = axes[i]
        ax.set_facecolor("#1a1a1a")

        # SVI fitted curve
        slice_df = surface_df[surface_df["dte"] == dte]
        ax.plot(slice_df["log_moneyness"], slice_df["iv_surface"],
                color="#00ff88", linewidth=2, label="SVI fit")

        # Raw IV points
        raw = iv_df[iv_df["dte"] == dte]
        calls = raw[raw["type"] == "call"]
        puts = raw[raw["type"] == "put"]
        ax.scatter(calls["log_moneyness"], calls["iv_computed"],
                   color="#4fc3f7", s=15, alpha=0.7, label="Calls")
        ax.scatter(puts["log_moneyness"], puts["iv_computed"],
                   color="#ff8a65", s=15, alpha=0.7, label="Puts")

        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_title(f"{dte} DTE", color="white", fontsize=10)
        ax.set_xlabel("Log-Moneyness", color="#888888", fontsize=8)
        ax.set_ylabel("Implied Volatility", color="#888888", fontsize=8)
        ax.tick_params(colors="#888888")
        ax.legend(fontsize=7, facecolor="#222222", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    fig.tight_layout()
    return fig

# ATM Term Structure
def plot_term_structure(surface_df, ticker="SPY"):
    # ATM IV = IV at log_moneyness closest to 0 for each expiration
    atm = surface_df.groupby("dte").apply(
        lambda x: x.loc[x["log_moneyness"].abs().idxmin(), "iv_surface"]
    ).reset_index()
    atm.columns = ["dte", "atm_iv"]
    atm = atm.sort_values("dte")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a1a")
    ax.plot(atm["dte"], atm["atm_iv"], color="#00ff88",
            linewidth=2, marker="o", markersize=5)
    ax.fill_between(atm["dte"], atm["atm_iv"], alpha=0.15, color="#00ff88")
    ax.set_xlabel("Days to Expiration", color="#888888")
    ax.set_ylabel("ATM Implied Volatility", color="#888888")
    ax.set_title(f"{ticker} ATM Volatility Term Structure",
                 color="white", fontsize=12)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    fig.tight_layout()
    return fig

# Saving All Plots
def save_plots(figs, names, ticker="SPY"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for fig, name in zip(figs, names):
        path = os.path.join(out_dir, f"{ticker}_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved {path}")
        plt.close(fig)

# Main Runner
if __name__ == "__main__":
    print("Loading data...")
    iv_df, surface_df, params_df = load_data("SPY")

    print("Plotting 3D surface...")
    fig1 = plot_3d_surface(surface_df, iv_df)

    print("Plotting skew slices...")
    fig2 = plot_skew_slices(surface_df, iv_df, params_df)

    print("Plotting term structure...")
    fig3 = plot_term_structure(surface_df)

    print("Saving plots...")
    save_plots([fig1, fig2, fig3],
               ["3d_surface", "skew_slices", "term_structure"])

    print("Done. Check the plots/ folder.")