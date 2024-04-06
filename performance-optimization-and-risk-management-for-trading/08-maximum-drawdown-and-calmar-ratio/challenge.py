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

# 1. Calculate and compare:
# - Maximum Drawdown
# - Calmar Ratio
# - Maximum Drawdown Duration
#
# for 30 large US stocks that currently form the Dow Jones Industrial Average Index ("Dow Jones") for the time period between April 2019 and June 2021.

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


def max_drawdown(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns) / cummax
    max_dd = drawdown.max()
    return max_dd


returns.apply(max_drawdown).sort_values()


def calmar(series):
    max_dd = max_drawdown(series)
    if max_dd == 0:
        return np.nan
    else:
        cagr = np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
        calmar = cagr / max_dd
        return calmar


calmar_rank = returns.apply(calmar).sort_values(ascending=False)
calmar_rank


def max_dd_duration(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns) / cummax
    
    begin = drawdown[drawdown == 0].index
    end = begin[1:]
    end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
    periods = end - begin
    max_ddd = periods.max()
    return max_ddd.days


returns.apply(max_dd_duration).sort_values()

# 2. Determine the best-performing stock and the worst-performing stock according to the Calmar Ratio.

calmar_rank.head(1) # best

calmar_rank.tail(1) # worst


# 3. Compare Calmar Ratio and Sharpe Ratio. Does the ranking change?

def sharpe(series, rf = 0):
    if series.std() == 0:
        return np.nan
    else:
        return (series.mean() - rf) / series.std() * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))


sharpe_rank = returns.apply(sharpe).sort_values(ascending = False)
sharpe_rank

calmar_rank.plot()
sharpe_rank.plot()

merged = pd.concat([calmar_rank, sharpe_rank], axis = 1)
merged

merged.columns = ["Sortino", "Sharpe"]
