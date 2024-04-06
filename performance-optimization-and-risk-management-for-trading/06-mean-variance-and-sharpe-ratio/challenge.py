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

# 1. Calculate and compare the Sharpe Ratio (assumption: rf == 0) for 30 large US stocks that currently form the Dow Jones Industrial Average Index ("Dow Jones") for the time period between April 2019 and June 2021.
#
# 2. Determine the best-performing stock and the worst-performing stock according to the Sharpe Ratio.

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

df.info()

# +
# TODO: figure out how to do the blow
#df["Returns"] = df.Close.pct_change()
#df
# -

returns = np.log(df.Close / df.Close.shift()) # daily log returns
returns

td_year = returns.count() / ((returns.index[-1] - returns.index[0]).days / 365.25)
td_year

ann_mean = returns.mean() * td_year
ann_mean

ann_std = returns.std() * np.sqrt(td_year)
ann_std

summary = pd.DataFrame(data = {"ann_mean": ann_mean, "ann_std": ann_std})
summary

rf = 0 # simplification, don't use this assumption for portfolio management!

summary["Sharpe"] = (summary.ann_mean - rf) / summary.ann_std
summary

summary.sort_values(by = "Sharpe", ascending=False)
