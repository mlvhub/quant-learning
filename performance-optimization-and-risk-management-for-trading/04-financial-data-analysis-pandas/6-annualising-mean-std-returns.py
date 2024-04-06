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

# # Annualising Mean Return and Std of Returns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft["log_return"] = np.log(msft.Price / msft.Price.shift()) # daily log returns (log of current price divided by the previous price)
msft

msft.log_return.agg(["mean", "std"]) # mean and std based on daily returns

ann_mu = msft.log_return.mean() * 252 # annualised mean return
ann_mu

cagr = np.exp(ann_mu) # don't mix up with cagr
cagr

ann_std = msft.log_return.std() * np.sqrt(252) # annualised std of returns
ann_std

ann_std = np.sqrt(msft.log_return.var() * 252) # annualised std of returns
ann_std

# ## Resampling/Smoothing

msft.head(25)

msft.Price.plot(figsize = (12, 8))
plt.legend()

monthly = msft.Price.resample("M").last() # resample to monthly (month end)
monthly

monthly.plot(figsize = (12, 8))
plt.legend()

# ### How will the Mean-Variance analysis change with smoothed data?

freqs = ["A", "Q", "M", "W-Fri", "D"]
periods = [1, 4, 12, 52, 252]
ann_mean = []
ann_std = []

for i in range(5):
    resample = msft.Price.resample(freqs[i]).last()
    ann_mean.append(np.log(resample / resample.shift()).mean() * periods[i]) # annualised mean return
    ann_std.append(np.log(resample / resample.shift()).std() * np.sqrt(periods[i])) # annualised std of returns

ann_mean

ann_std

summary = pd.DataFrame(data = {"ann_std": ann_std, "ann_mean": ann_mean}, index = freqs)
summary

summary.plot(kind = "scatter", x = "ann_std", y = "ann_mean", figsize = (15, 12), s = 50, fontsize = 15)
for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i, "ann_std"]+0.001, summary.loc[i, "ann_mean"]+0.001), size = 15)
plt.ylim(0, 0.3)
plt.xlabel("Ann Risk (std)", fontsize = 15)
plt.ylabel("Ann Return", fontsize = 15)
plt.title("Risk/Return", fontsize = 20)

# **Smoothing reduces (observed) risk.**

# Dubious practices:
# - managing (manipulating) performance in reportings.
# - adjusting frequency to investor's (average) holding period: volatility is still there.
# - comparing assets with different pricing frequency and pricing mechanisms: e.g. real estate with quarterly valuation vs. listed stocks (minutely/hourly/daily prices)

# **Take home**: when comparing instruments the frequency of underlying data must be the same! Don't compare apples and oranges.
