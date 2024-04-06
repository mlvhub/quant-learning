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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns

returns.cumsum().apply(np.exp).plot()

td_year = returns.count() / ((returns.index[-1] - returns.index[0]).days / 365.25)
td_year

# ## Downside Deviation (Semi-Deviation)

symbol = "USD_GBP"

TMR = 0 # target minimum return

excess_returns = returns[symbol] - TMR # excess return over TMR
excess_returns

excess_returns = np.where(excess_returns < 0, excess_returns, 0) # setting positive excess return to zero
excess_returns

downside_deviation = np.sqrt(np.mean(excess_returns**2)) # daily downside deviation
downside_deviation

# TODO: calculate by me, is it right?
ann_downside_deviation = downside_deviation * np.sqrt(td_year.iloc[0]) # anual downside deviation
ann_downside_deviation

# ## Sortino Ratio

mean = returns[symbol].mean()
mean

sortino = (mean - TMR) / downside_deviation * np.sqrt(td_year)
sortino


# ## Putting everything together

def sortino(series, TMR = 0):
    excess_returns = (series - TMR)
    downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0) ** 2))
    
    if downside_deviation == 0:
        return np.nan
    else:
        sortino = (series.mean() - TMR) / downside_deviation * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))
        return sortino


returns.apply(sortino).sort_values(ascending = False)

sortino(series = returns.USD_GBP, TMR = 0)
