# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing a Profitable Trading Strategy
#
# Source: https://ai.gopubby.com/the-minimalists-guide-to-testing-a-profitable-trading-strategy-in-python-38b663223bf2

# %%
import pandas as pd
import numpy as np
import yfinance as yf

# %% [markdown]
# ## The Edge: Window Dressing

# %%
# Download TLT (Long-term Treasury Bond ETF) data
tlt = yf.download("TLT", start="2002-01-01", end="2024-06-30")
tlt

# %%
tlt["log_return"] = np.log(tlt["Close"] / tlt["Close"].shift(1))
tlt

# %%
tlt["day_of_month"] = tlt.index.day
tlt["year"] = tlt.index.year
grouped_by_day = tlt.groupby("day_of_month").log_return.mean()
grouped_by_day.plot.bar(title="Mean Log Returns by Calendar Day of Month")

# %%
tlt["first_week_returns"] = 0.0
tlt.loc[tlt.day_of_month <= 7, "first_week_returns"] = tlt[tlt.day_of_month <= 7].log_return
tlt["first_week_returns"].plot()

# %%
tlt["last_week_returns"] = 0.0
tlt.loc[tlt.day_of_month >= 23, "last_week_returns"] = tlt[tlt.day_of_month >= 23].log_return
tlt["last_week_returns"].plot()

# %%
tlt["last_week_less_first_week"] = tlt.last_week_returns - tlt.first_week_returns
tlt["last_week_less_first_week"].plot()

# %%
tlt.groupby("year")["last_week_less_first_week"].sum().cumsum().plot(title="Cumulative Sum of Returns By Year")

# %% [markdown]
# ## Finding the Best ETFs for Month-End Trading

# %%
etfs = [
    "TLT",  # Long-term Treasury bonds
    "IEF",  # Intermediate Treasury bonds
    "SHY",  # Short-term Treasury bonds
    "LQD",  # Corporate bonds
    "HYG",  # High-yield bonds
    "AGG",  # Aggregate bonds
    "GLD",  # Gold
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM"   # Russell 2000
]


# %%
def calculate_strategy_returns(ticker):
    try:
        # Download data
        df = yf.download(ticker, start="2002-01-01", end="2024-06-30")
        
        # Calculate log returns
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        
        # Add calendar features
        df["day_of_month"] = df.index.day
        
        # Calculate strategy returns
        df["first_week_returns"] = 0.0
        df.loc[df.day_of_month <= 7, "first_week_returns"] = df[df.day_of_month <= 7].log_return
        
        df["last_week_returns"] = 0.0
        df.loc[df.day_of_month >= 23, "last_week_returns"] = df[df.day_of_month >= 23].log_return
        
        # Calculate strategy metrics
        total_return = (df.last_week_returns - df.first_week_returns).sum()
        annual_return = total_return / (len(df) / 252)  # Annualized
        sharpe = np.sqrt(252) * (df.last_week_returns - df.first_week_returns).mean() / (df.last_week_returns - df.first_week_returns).std()
        
        return {
            "ticker": ticker,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe
        }
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


# %%
# Run analysis for all ETFs
results = []
for etf in etfs:
    result = calculate_strategy_returns(etf)
    if result:
        results.append(result)

results

# %%
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("sharpe", ascending=False)
results_df

# %%
# Format results for display
results_df["total_return"] = results_df["total_return"].map("{:.2%}".format)
results_df["annual_return"] = results_df["annual_return"].map("{:.2%}".format)
results_df["sharpe"] = results_df["sharpe"].map("{:.2f}".format)

# %%
print("\nStrategy Performance Across ETFs:")
print(results_df.to_string(index=False))

# %%
