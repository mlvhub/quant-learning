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

# # Trading with Leverage and the Kelly Criterion

# ## Getting started

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])

returns.info()

returns.cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.show()

simple = np.exp(returns) - 1 # simple returns
simple



# ## Recap:  Leverage and Margin Trading

simple

symbol = "USD_GBP"

leverage = 2 # equivalent to a margin of 50%

instr = simple[symbol].to_frame().copy()
instr

instr["Lev_Returns"] = instr[symbol].mul(leverage) # multiply simple returns with leverage
instr

instr["Lev_Returns"] = np.where(instr["Lev_Returns"] < -1, -1, instr["Lev_Returns"]) # loss limited to 100%
instr

instr.add(1).cumprod().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()



# ## Finding the optimal degree of Leverage

simple

# We can either use __kelly criterion__ or we just __run the backtest for many different leverage settings__.

leverage = np.arange(1, 5, 0.01)
leverage

multiple = []
for lever in leverage:
    levered_returns = simple[symbol].mul(lever)
    levered_returns = pd.Series(np.where(levered_returns < -1, -1, levered_returns))
    multiple.append(levered_returns.add(1).prod())
results = pd.DataFrame(data = {"Leverage":list(leverage), "Multiple":multiple})

results.set_index("Leverage", inplace = True)

results

results.min()

max_multiple = results.max()
max_multiple

optimal_lev = results.idxmax()
optimal_lev

results.plot(figsize = (12, 8));
plt.scatter(x = optimal_lev, y = max_multiple, color = "r", s = 50)
plt.xlabel("Leverage", fontsize = 13)
plt.ylabel("Multiple", fontsize = 13)
plt.title("The optimal degree of Leverage", fontsize = 15)
plt.show()



# ## The Kelly Criterion

optimal_lev # true/correct value for the optimal leverage

# The Kelly Criterion closely approaches the true/correct value, if
# - simple returns are used (Yes)
# - dataset is sufficiently large (OK)

instr = simple[symbol].to_frame().copy()
instr

mu = instr.mean() # mean return (simple)
mu

var = instr.var() # variance of returns (simple)
var

kelly = mu / var
kelly

# -> Good approximation by __Kelly criterion__



# ## The impact of Leverage on Reward & Risk

simple # simple returns

# __Reward: 1) Mean of Simple Returns__

leverage = np.arange(1, 5, 0.01)

mu = []
sigma = []
sharpe = []
for lever in leverage:
    levered_returns = simple[symbol].mul(lever)
    levered_returns = pd.Series(np.where(levered_returns < -1, -1, levered_returns))
    mu.append(levered_returns.mean()) # mean of simple returns
    sigma.append(levered_returns.std())
    sharpe.append(levered_returns.mean() / levered_returns.std())
results = pd.DataFrame(data = {"Leverage":list(leverage), "Mean": mu, "Std": sigma, "Sharpe": sharpe})

results.set_index("Leverage", inplace = True)

results

results.plot(subplots = True, figsize = (12, 8), fontsize = 12);
plt.show()

# __Mean of simple Returns is steadily increasing with higher leverage -> misleading__

# __Sharpe Ratio remains constant -> misleading__



# __Reward: 2) Mean of Log Returns__

leverage = np.arange(1, 5, 0.01)

mu = []
sigma = []
sharpe = []
for lever in leverage:
    levered_returns = simple[symbol].mul(lever)
    levered_returns = pd.Series(np.where(levered_returns < -1, -1, levered_returns))
    levered_returns = np.log(levered_returns + 1) # convert to log returns
    mu.append(levered_returns.mean()) # mean of log returns
    sigma.append(levered_returns.std())
    sharpe.append(levered_returns.mean() / levered_returns.std())
results = pd.DataFrame(data = {"Leverage":list(leverage), "Mean": mu, "Std": sigma, "Sharpe": sharpe})

results.set_index("Leverage", inplace = True)

results

results.plot(subplots = True, figsize = (12, 8), fontsize = 12);
plt.show()

# - __Maximum Return @ Kelly__
# - __Sharpe Ratio steadily decreasing with higher leverage__.
# - __Leverage amplifies losses more than it amplifies gains__.
# - __Don´t use leverage if your goal is to maximize risk-adjusted return__
# - __If you want to increase return/income with leverage -> Trade-off__
# - __Rule of Thumb: Leverage shouldn´t be higher than "Half Kelly".__ 



# ## Putting everything together

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns


def kelly_criterion(series): # assuming series with log returns
    
    series = np.exp(series) - 1
    if series.var() == 0:
        return np.nan
    else:
        return series.mean() / series.var()


returns.apply(kelly_criterion).sort_values(ascending = False)

kelly_criterion(returns.Low_Vol)

# Side Note: For "Low_Vol", Kelly is not a good approximation because:
# - majority of daily returns is zero (neutral)
# - only very few "real" observations (non-normal)

returns.Low_Vol.value_counts()

# Bonus Question: What´s the correct/true optimal degree of leverage for "Low_Vol"?



# ------------------------

# ## Coding Challenge

# __Calculate and compare__ the __Kelly Criterion__ for __30 large US stocks__ that currently form the Dow Jones Industrial Average Index ("Dow Jones") for the time period between April 2019 and June 2021.

# __Hint:__ You can __import__ the price data from __"Dow_Jones.csv"__.

# Determine the Stock with the __highest and lowest Kelly Criterion__.

# (Remark: Dividends are ignored here. Hence, for simplicity reasons, the Kelly Criterion is based on Price Returns only. As a consequence, dividend-paying stocks are getting penalized.) 

# ## +++ Please stop here in case you don´t want to see the solution!!! +++++







import pandas as pd
import numpy as np

df = pd.read_csv("Dow_Jones.csv", index_col = "Date", parse_dates = ["Date"])
df

returns = np.log(df / df.shift()) # log returns
returns


def kelly_criterion(series):
    
    series = np.exp(series) - 1
    if series.var() == 0:
        return np.nan
    else:
        return series.mean() / series.var()


returns.apply(kelly_criterion).sort_values(ascending = False)

# -> highest Kelly: AAPL, Lowest Kelly: BA


