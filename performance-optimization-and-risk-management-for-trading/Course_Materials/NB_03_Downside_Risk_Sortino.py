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

# # Downside Risk and Sortino Ratio

# ## Getting Ready

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns.head(50)

returns.cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.show()

td_year = (returns.count() / ((returns.index[-1] - returns.index[0]).days / 365.25)).iloc[0]
td_year



# ## Downside Deviation (Semi-Deviation)

returns

symbol = "USD_GBP"

TMR = 0 # target minimum return

excess_returns = returns[symbol] - TMR # excess returns over TMR
excess_returns

excess_returns = np.where(excess_returns < 0, excess_returns, 0) # setting postive excess returns to zero. 
excess_returns

downside_deviation = np.sqrt(np.mean(excess_returns**2)) 
downside_deviation



# ## Sortino Ratio

downside_deviation

mean = returns[symbol].mean() 
mean

sortino = (mean - TMR) / downside_deviation * np.sqrt(td_year)
sortino



# ## Putting everything toghether

import pandas as pd
import numpy as np

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns


def sortino(series, TMR = 0):
    excess_returns = (series - TMR)
    downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0)**2))
    if downside_deviation == 0:
        return np.nan
    else:
        sortino = (series.mean() - TMR) / downside_deviation * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))
        return sortino


returns.apply(sortino).sort_values(ascending = False)

sortino(series = returns.USD_GBP, TMR = 0)



# -------------------------------------

# ## Coding Challenge

# __Calculate and compare__ the __Sortino Ratio__ (assumption: TMR == 0) for __30 large US stocks__ that currently form the Dow Jones Industrial Average Index ("Dow Jones") for the time period between April 2019 and June 2021. 

# __Hint:__ You can __import__ the price data from __"Dow_Jones.csv"__.
#  

# Determine the __best performing stock__ and the __worst performing stock__ according to the Sortino Ratio.

# __Compare__ Sortino Ratio and Sharpe Ratio. Does the __ranking change__?

# (Remark: Dividends are ignored here. Hence, for simplicity reasons, the Sortino Ratio is based on Price Returns only. As a consequence, dividend-paying stocks are getting penalized.) 

# ## +++ Please stop here in case you don´t want to see the solution!!! +++++









# ## Coding Challenge Solution

import pandas as pd
import numpy as np

df = pd.read_csv("Dow_Jones.csv", index_col = "Date", parse_dates = ["Date"])
df

returns = np.log(df / df.shift()) # daily log returns
returns


def sortino(series, TMR = 0):
    excess_returns = (series - TMR)
    downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0)**2))
    if downside_deviation == 0:
        return np.nan
    else:
        sortino = (series.mean() - TMR) / downside_deviation * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))
        return sortino


sort = returns.apply(sortino).sort_values(ascending = False)
sort


# Best Performing Stock: __Apple__ (AAPL) <br>
# Worst Performing Stock: __Non-determinable__ (note: you can´t compare negative Sortino Ratios)

def sharpe(series, rf = 0):
    
    if series.std() == 0:
        return np.nan
    else:
        return (series.mean() - rf) / series.std() * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))


sha = returns.apply(sharpe).sort_values(ascending = False)
sha

merged = pd.concat([sort, sha], axis = 1)
merged

merged.columns = ["Sortino", "Sharpe"]

merged.rank(ascending = False)

# -> Few Differences. __Disney gets better ranked__ with Sortino (-3) while __The Home Depot gets penalized__ by Sortino (+3).


