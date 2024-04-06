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

# # Coding Challenge #6

# 1. Calculate daily log returns for Boeing.
#
# 2. Calculate the annualized mean and annualized std (assume 252 trading days per year) for a short position in Boeing (ignore Trading and Borrowing Costs).

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

ba = close.BA.dropna().copy().to_frame().rename(columns = {"BA": "Price"})
ba

ba["log_return"] = np.log(ba.Price / ba.Price.shift()) # daily log returns (log of current price divided by the previous price)
ba

ba["short"] = ba.log_return * (-1)
ba

ann_mu = ba.short.mean() * 252 # annualised mean return
ann_mu

ann_std = ba.short.std() * np.sqrt(252) # annualised std of returns
ann_std


