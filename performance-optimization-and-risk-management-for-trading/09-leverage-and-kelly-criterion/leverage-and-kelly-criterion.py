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

# # Trading with leverage and the Kelly criterion

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns

returns.cumsum().apply(np.exp).plot()

simple = np.exp(returns) - 1# simple returns for leverage calculation
simple

# ## Leverage & Margin trading

symbol = "USD_GBP"

leverage = 2 # equivalent to a margin of 50%

instr = simple[symbol].to_frame().copy()
instr

instr["Lev_Returns"] = instr[symbol].mul(leverage) # multiply simple returns with leverage
instr

instr["Lev_Returns"] = np.where(instr["Lev_Returns"] < -1, -1, instr["Lev_Returns"]) # loss limited to 100%
instr

instr.add(1).cumprod().plot()

# ## Finding the optimal degree of leverage

# We can either use the kelly criterion or we just run the backtest for many different leverage settings

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

results.plot()
plt.scatter(x = optimal_lev, y = max_multiple, color = "r", s= 50)
plt.xlabel("Leverage")
plt.ylabel("Multiple")
plt.title("Optimal degree of leverage")

# ## Kelly Criterion

# The Kelly Criterion closely approaches the true/correct value **if**:
# - simple returns are used (Yes)
# - dataset is sufficiently large (OK)

instr = simple[symbol].to_frame().copy()
instr

mu = instr.mean() # mean return (simple)
mu

var = instr.var() # variance of returns (simple)

kelly = mu / var
kelly

# Good approximation by **Kelly Criterion**

# ## Impact of Leverage on Reward & Risk

# ### Reward 1: Mean of simple returns

leverage = np.arange(1, 5, 0.01)

mu = []
sigma = []
sharpe = []
for lever in leverage:
    levered_returns = simple[symbol].mul(lever)
    levered_returns = pd.Series(np.where(levered_returns < -1, -1, levered_returns))
    mu.append(levered_returns.mean())
    sigma.append(levered_returns.std())
    sharpe.append(levered_returns.mean() / levered_returns.std())
results = pd.DataFrame(data = {"Leverage":list(leverage), "Mean":mu, "Std": sigma,"Sharpe": sharpe})

results.set_index("Leverage", inplace = True)
results

results.plot(subplots = True)

# #### Mean of simple returns is steadily increasing with higher leverage (misleading)
# #### Sharpe Ratio remains constant (misleading)

# ### Reard 2: Mean of log returns

leverage = np.arange(1, 5, 0.01)

mu = []
sigma = []
sharpe = []
for lever in leverage:
    levered_returns = simple[symbol].mul(lever)
    levered_returns = pd.Series(np.where(levered_returns < -1, -1, levered_returns))
    levered_returns = np.log(levered_returns + 1) # convert to log returns
    mu.append(levered_returns.mean())
    sigma.append(levered_returns.std())
    sharpe.append(levered_returns.mean() / levered_returns.std())
results = pd.DataFrame(data = {"Leverage":list(leverage), "Mean":mu, "Std": sigma,"Sharpe": sharpe})

results.set_index("Leverage", inplace = True)
results

results.plot(subplots = True)


# - Maximum Return @ Kelly
# - Sharpe Ratio steadily decreasing with higher leverage
# - Leverage amplifies losses more than it amplifies gains
# - Don't use leverage if your goal is to maximise risk-adjusted returns
# - If you want to increase return/income with leverage it has trade offs
# **Rule of thumb: leverage shouldn't be higher than Half Kelly.**

# ## Putting everything together

def kelly_criterion(series): # assuming log returns
    series = np.exp(series) - 1
    if series.var() == 0:
        return np.nan
    else:
        return series.mean() / series.var()


returns.apply(kelly_criterion).sort_values(ascending=False)

kelly_criterion(returns.Low_Vol)

# Side note: for "Low_Vol" it's not a good aproximation because:
# - majority of daily returns is zero (neutral)
# - only very few "real" observations (non-normal)

returns.Low_Vol.value_counts()


