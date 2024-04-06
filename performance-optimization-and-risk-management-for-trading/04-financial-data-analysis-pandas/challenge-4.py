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

# # Coding Challenge #4

# 1. Calculate daily log returns for Boeing.
#
# 2. Use Boeing´s log returns to calculate
#
#     Investment Multiple
#
#     CAGR (assuming 252 trading days)
#
#     Normalized Prices (Base = 1)

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

ba = close.BA.dropna().copy().to_frame().rename(columns = {"BA": "Price"})
ba

ba["log_return"] = np.log(ba.Price / ba.Price.shift()) # daily log returns (log of current price divided by the previous price)
ba

investment_multiple = np.exp(ba.log_return.sum()) # adding log returns ("cumulative returns")
investment_multiple 

cagr = (ba.Price[-1] / ba.Price[0]) ** (1 / ((ba.index[-1] - ba.index[0]).days / 365.25)) - 1
cagr

cagr = np.exp(ba.log_return.mean() * 252) - 1 # good approximation (for US stocks)
cagr

normalised_prices = np.exp(ba.log_return.cumsum()) # adding log returns ("cumulative returns")
normalised_prices

(60 / 20) ** (1 / (365 / 365.25)) - 1
