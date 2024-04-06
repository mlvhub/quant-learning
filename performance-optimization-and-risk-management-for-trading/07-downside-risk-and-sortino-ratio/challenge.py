# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Coding Challenge

# 1. Calculate and compare the Sortino Ratio (assumption: TMR == 0) for 30 large US stocks that currently form the Dow Jones Industrial Average Index ("Dow Jones") for the time period between April 2019 and June 2021.
#
# 2. Determine the best-performing stock and the worst-performing stock according to the Sortino Ratio.
#
# 3. Compare Sortino Ratio and Sharpe Ratio. Does the ranking change?

import pandas as pd
import numpy as np
import yfinance as yf
pd.options.display.float_format = '{:.6f}'.format

dow_jones = pd.read_csv("dow-jones.csv")
dow_jones

symbols = dow_jones.Symbol.to_list()
symbols

start = "2019-03-01"
end = "2021-06-30"

df = yf.download(symbols, start, end)
df

returns = np.log(df.Close / df.Close.shift()) # daily log returns
returns


def sortino(series, TMR = 0):
    excess_returns = (series - TMR)
    downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0) ** 2))
    
    if downside_deviation == 0:
        return np.nan
    else:
        sortino = (series.mean() - TMR) / downside_deviation * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))
        return sortino


sortino_rank = returns.apply(sortino).sort_values(ascending = False)
sortino_rank

sortino_rank.head(1)

sortino_rank.tail(1)


def sharpe(series, rf = 0):
    if series.std() == 0:
        return np.nan
    else:
        return (series.mean() - rf) / series.std() * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))


sharpe_rank = returns.apply(sharpe).sort_values(ascending = False)
sharpe_rank

sharpe_rank.head(1)

sharpe_rank.tail(1)
