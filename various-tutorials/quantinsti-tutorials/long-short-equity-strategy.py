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
# # Long-Short Equity Strategy: A Comprehensive Guide
#
# Source: https://blog.quantinsti.com/long-short-equity-strategy/
#

# %% [markdown]
# The long-short equity strategy involves buying the stocks expected to rise (long positions) and selling the stocks expected to fall (short positions). It aims to gain from both market upswings and downturns while minimising overall market exposure.
#
# A tactic commonly utilised by hedge funds to enhance risk adjusted returns given its inherently lower risk profile.

# %%
# Top 5 most overvalued stock tickers, source: https://www.morningstar.com/stocks/top-5-overvalued-stocks

top_overvalued_tickers = ["WING", "CELH", "LUV", "VST", "DELL"]

# %%
# Top 5 most undervalued stock tickers, source: https://www.morningstar.com/stocks/33-undervalued-stocks-2024

top_undervalued_tickers = ["ALB", "GOOGL", "APA", "BBWI", "BAX"]


# %% [markdown]
# ## Steps to build a long short equity strategy
#
# ### 1. Define the universe
#
# We'll need a universe of stocks to trade, it can be based on dollar-volume, market cap, price and/or impact costs.
#
# In this example we'll use market cap.
#
# ### 2. Bucketing stock
#
# We will bucket stocks based on the sector. In this example we'll use the technology sector.
#
# ### 3. Define paratemer to long or short security
#
# We will rank stocks based on the previous day's returns, if they perform well we'll rank them higher, if they perform poorly we'll rank them lower.
#
# We will use mean reversion to take long postitions with the lower rank or short positions with the higher rank.
#
# ### 4. Capital allocation
#
# Allocating an equal amount of capital to each stock shortlisted from step 3 is a popular capital allocation strategy. 
#
# An equal weight approach helps to avoid a concentration on a particular stock in the portfolio.
#
# > Note: A combination of parameters such as quarterly earnings growth, PE ratio, P/BV, moving averages, and RSI could be used here with different weights on each parameter to create a profitable strategy.

# %% [markdown]
# ## Building a Long short equity strategy in Python

# %%
# 1. Import libraries and fetch historical data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import yfinance as yf

# Reading historical data for 38 large-cap tech stocks
tickers = ['AAPL', 'ACN', 'ADI', 'ADP', 'ADSK', 'ANSS', 'APH', 'BABA', 'BIDU', 'BR', 'CRM',
'FFIV', 'FIS', 'FISV','GOOG', 'GPN', 'IBM','INTC', 'INTU', 'IPGP', 'IT', 'JKHY', 'KEYS', 'KLAC', 'LRCX', 'MA', 'MCHP', 'MSFT',
'MSI', 'NVDA', 'NXPI', 'PYPL', 'SNPS', 'TEL', 'TTWO', 'TXN', 'V', 'VRSN']

data = yf.download(tickers,'2018-1-1', '2024-3-1')['Adj Close']

# %% [markdown]
#

# %%
data.head()

# %%
# 2. Calculate returns

daily_stock_returns = (data-data.shift(1))/data.shift(1)
daily_stock_returns.dropna(inplace=True)

# Assigning ranks in order of decreasing daily returns
df_rank = daily_stock_returns.rank(axis=1, ascending=False, method='min')
df_rank.head()

# %%
# 3. Generating signals

df_signal = df_rank.copy()
for ticker in tickers:
    df_signal[ticker] = np.where(df_signal[ticker] < 22, -1, 1)

# Calculating returns on the basis of our trade signals
returns = df_signal.mul(daily_stock_returns.shift(-1), axis=0)

# Assuming returns overall stocks to get final returns
strategy_returns = np.sum(returns, axis=1)/len(tickers)
df_signal.head(3)

# %%
# 4. Print cumulative returns, sharpe ratio and maximum drawdown

if not strategy_returns.empty:
   # Cumulative Returns
   cumulative_returns = (strategy_returns + 1).cumprod()


   # Sharpe Ratio
   # Assuming risk-free rate as 0 for simplicity
   daily_rf_rate = 0
   annual_rf_rate = daily_rf_rate * 252
   strategy_volatility = strategy_returns.std() * np.sqrt(252)
   sharpe_ratio = (strategy_returns.mean() - annual_rf_rate) / strategy_volatility


   # Drawdown
   cum_max = cumulative_returns.cummax()
   drawdown = (cumulative_returns - cum_max) / cum_max
   max_drawdown = drawdown.min()


   # Print the results
   print("Cumulative Returns:")
   print(cumulative_returns[-1] if not cumulative_returns.empty else "No trades executed.")
   print("\nSharpe Ratio:")
   print(sharpe_ratio)
   print("\nMax Drawdown:")
   print(max_drawdown)
else:
   print("No trades executed. Cannot compute performance metrics.")

# %%
# 5. Visualisation

import matplotlib.pyplot as plt

# Cumulative Returns
if not strategy_returns.empty:
   cumulative_returns = (strategy_returns + 1).cumprod()
   plt.figure(figsize=(10, 6))
   cumulative_returns.plot()
   plt.title('Cumulative Returns')
   plt.xlabel('Date')
   plt.ylabel('Cumulative Return')
   plt.grid(True)
   plt.show()


   # Define the rolling window for 6 months (126 trading days)
rolling_window = 126


# Calculate rolling Sharpe Ratio
rolling_sharpe_ratio = strategy_returns.rolling(window=rolling_window).mean() / strategy_returns.rolling(window=rolling_window).std() * np.sqrt(252)


# Plot Sharpe Ratio
plt.figure(figsize=(10, 6))
rolling_sharpe_ratio.plot()
plt.title('Rolling 6-Month Sharpe Ratio')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.show()


# Plot Maximum Drawdown
plt.figure(figsize=(10, 6))
drawdown.plot()
plt.title('Maximum Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.axhline(max_drawdown, color='red', linestyle='--', label='Max Drawdown')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
#
