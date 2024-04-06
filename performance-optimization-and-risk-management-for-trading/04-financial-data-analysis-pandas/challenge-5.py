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

# # Coding Challenge 5
#
# 1. Calculate daily log returns for Boeing.
#
# 2. Use Boeing´s daily log returns to calculate the annualized mean and annualized std (assume 252 trading days per year).
#
# 3. Resample to monthly prices and compare the annualized std (monthly) with the annualized std (daily). Any differences?
#
# 4. Keep working with monthly data and calculate/visualize the rolling 36 months mean return (annualized).

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

ba = close.BA.dropna().copy().to_frame().rename(columns = {"BA": "Price"})
ba

ba["log_return"] = np.log(ba.Price / ba.Price.shift()) # daily log returns (log of current price divided by the previous price)
ba

ba.Price.plot(figsize = (12, 8))

ann_mu = ba.log_return.mean() * 252 # annualised mean return
ann_mu

ann_std = ba.log_return.std() * np.sqrt(252) # annualised std of returns
ann_std

monthly = ba.Price.resample("M").last() # resample to monthly (month end)
monthly

monthly.plot(figsize = (12, 8))

monthly_ann_std = monthly.std() * np.sqrt(12) # annualised std of returns
monthly_ann_std

monthly.rolling(36).mean().plot()
