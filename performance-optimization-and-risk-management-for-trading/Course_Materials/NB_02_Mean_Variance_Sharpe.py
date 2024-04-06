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

# # Mean-Variance Analysis and the Sharpe Ratio

# ## Getting started

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns

# GBP_USD: Long Position in GBP (denominated in USD) <br>
# USD_GBP: Short Position in GBP (== Long Position in USD; denominated in GBP) <br>
# Levered: USD_GBP with Leverage ("Trading USD_GBP on Margin") <br>
# Neutral: Neutral Positions only (no Investments / Trades)  <br>
# Low_Vol: Active Strategy for USD_GBP with Long, Short and Neutral Positions <br>
# Random: Random "Strategy" for USD_GBP with random Long, Short and Neutral Positions

returns.info()

returns.cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.show()

returns.Low_Vol.cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.show()

returns.Low_Vol.value_counts()

returns[["Low_Vol", "Levered"]].cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.show()

# __Which one would you (intuitively) prefer?__

# __Low_Vol__, right? Let´s create a __risk-adjusted return metric__ that reflects/confirms this intuition!



# ## Mean Return (Reward)

returns

# __mean return__

returns.mean()

# __annualized mean return__

td_year = returns.count() / ((returns.index[-1] - returns.index[0]).days / 365.25)
td_year

ann_mean = returns.mean() * td_year
ann_mean

np.exp(ann_mean) - 1 # CAGR

summary = pd.DataFrame(data = {"ann_mean":ann_mean})
summary

summary.rank(ascending = False)



# ## Standard Deviation (Risk)

returns

# __Standard Deviation of Returns__

returns.std()

# __Annualized Standard Deviation__

td_year

ann_std = returns.std() * np.sqrt(td_year)
ann_std

summary["ann_std"] = returns.std() * np.sqrt(td_year)

summary.sort_values(by = "ann_std")



# ## Risk-adjusted Return ("Sharpe Ratio")

summary

# __Graphical Solution__

summary.plot(kind = "scatter", x = "ann_std", y = "ann_mean", figsize = (15,12), s = 50, fontsize = 15)
for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i, "ann_std"]+0.001, summary.loc[i, "ann_mean"]+0.001), size = 15)
plt.xlim(-0.01, 0.23)
plt.ylim(-0.02, 0.03)
plt.xlabel("Risk(std)", fontsize = 15)
plt.ylabel("Return", fontsize = 15)
plt.title("Risk/Return", fontsize = 20)
plt.show()

# __Risk-adjusted Return Metric__ ("Sharpe Ratio light")

rf = 0 # simplification, don´t use this assumption for Portfolio Management!

summary["Sharpe"] = (summary.ann_mean - rf) / summary.ann_std

summary.sort_values(by = "Sharpe", ascending = False)

td_year

returns.mean() / returns.std() * np.sqrt(td_year) # alternative: annualizing daily sharpe



# ## Putting everything together

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns


def sharpe(series, rf = 0):
    
    if series.std() == 0:
        return np.nan
    else:
        return (series.mean() - rf) / series.std() * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))


returns.apply(sharpe, rf = 0)

sharpe(series = returns.Levered, rf = 0)



# ------------------------------

# ## Coding Challenge

# __Calculate and compare__ the __Sharpe Ratio__ (assumption: rf == 0) for __30 large US stocks__ that currently form the Dow Jones Industrial Average Index ("Dow Jones") for the time period between April 2019 and June 2021. 

# __Hint:__ You can __import__ the price data from __"Dow_Jones.csv"__.
#  

# Determine the __best performing stock__ and the __worst performing stock__ according to the Sharpe Ratio.

# (Remark: Dividends are ignored here. Hence, for simplicity reasons, the Sharpe Ratio is based on Price Returns only. As a consequence, dividend-paying stocks are getting penalized.) 

# ## +++ Please stop here in case you don´t want to see the solution!!! +++++









# ## Coding Challenge Solution

import pandas as pd
import numpy as np

df = pd.read_csv("Dow_Jones.csv", index_col = "Date", parse_dates = ["Date"])
df

df.info()

returns = np.log(df / df.shift()) # daily log returns
returns


def sharpe(series, rf = 0):
    
    if series.std() == 0:
        return np.nan
    else:
        return (series.mean() - rf) / series.std() * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))


returns.apply(sharpe).sort_values(ascending = False)

# Best Performing Stock: __Apple__ (AAPL) <br>
# Worst Performing Stock: __Non-determinable__ (note: you can´t compare negative Sharpe Ratios)


