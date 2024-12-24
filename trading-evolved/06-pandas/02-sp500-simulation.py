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

# %%
# %matplotlib inline
import pandas as pd
import numpy as np


# %%
data = pd.read_csv('./06-pandas/sp500.csv', index_col='Date', parse_dates=['Date'])
data

# %%
data['SMA50'] = data['SP500'].rolling(50).mean()
data['SMA100'] = data['SP500'].rolling(100).mean()
data


# %%
data['Position'] = np.where(data['SMA50'] > data['SMA100'], 1, 0)
data


# %%
# Buy a day delayed, shift the column 
data['Position'] = data['Position'].shift()
data

# %%
# Calculate the daily percent returns of strategy 
data['StrategyPct'] = data['SP500'].pct_change(1) * data['Position']
data


# %%
# Calculate the cumulative returns of strategy
data['Strategy'] = (data['StrategyPct'] + 1).cumprod()
data


# %%
# Calculate index cumulative returns data[' BuyHold'] = (data[' SP500']. pct_change( 1) + 1). cumprod()
data['BuyHold'] = (data['SP500'].pct_change(1) + 1).cumprod()
data

# %%
# Plot the strategy and buy-and-hold returns
data[['Strategy', 'BuyHold']].plot()

# %% [markdown]
# NOTE: this is not a real backtest, it's just a simulation to practice pandas and make sure the environment is set up correctly.
#
# We can't trade indexes directly, and we haven't taken into account fees, slippage, etc.

# %% [markdown]
#
