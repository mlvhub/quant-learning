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

# # Maximum Drawdown and Calmar Ratio

# ## Getting ready

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns

returns.info()

returns.cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.show()



# ## Maximum Drawdown

symbol = "USD_GBP"

returns[symbol].cumsum().apply(np.exp).plot(figsize = (12, 8))
plt.show()

instr = returns[symbol].to_frame().copy()
instr

instr["creturns"] = instr.cumsum().apply(np.exp) # cumulative returns (normalized prices with Base == 1)

instr["cummax"] = instr.creturns.cummax() # cumulative maximum of creturns

instr

instr[["creturns", "cummax"]].plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

instr["drawdown"] = -(instr["creturns"] - instr["cummax"]) / instr["cummax"] # (pos.) drawdown (in %)
instr

instr[["creturns", "cummax", "drawdown"]].plot(figsize = (15, 8), fontsize = 13, secondary_y = "drawdown")
plt.legend(fontsize = 13)
plt.show()

max_drawdown = instr.drawdown.max() # maximum drawdown
max_drawdown

instr.drawdown.idxmax() # maximum drawdown date 

instr.loc[instr.drawdown.idxmax()]

(0.941169 - 1.127116) / 1.127116



# ## Calmar Ratio

max_drawdown

cagr = np.exp(instr[symbol].sum())**(1/((instr.index[-1] - instr.index[0]).days / 365.25)) - 1 
cagr

calmar = cagr / max_drawdown
calmar



# ## Max Drawdown Duration

instr

instr[["creturns", "cummax", "drawdown"]].plot(figsize = (15, 8), fontsize = 13, secondary_y = "drawdown")
plt.legend(fontsize = 13)
plt.show()

drawdown = instr.drawdown.copy()
drawdown

# - Drawdown Period: Time Period between peaks 
# - recall: whenever drawdown == 0, a new peak has been reached

begin = drawdown[drawdown == 0].index # get all peak dates (beginning of Drawdown periods)
begin

end = begin[1:] # get the corresponding end dates for all Drawdown periods
end = end.append(pd.DatetimeIndex([drawdown.index[-1]])) # add last available date
end

periods = end - begin # time difference between peaks
periods

max_ddd = periods.max() # max drawdown duration
max_ddd

max_ddd.days



# ## Putting everything together

import pandas as pd
import numpy as np

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns


def max_drawdown(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns)/cummax
    max_dd = drawdown.max()
    return max_dd


returns.apply(max_drawdown).sort_values()


def calculate_cagr(series):
    return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1


def calmar(series):
    
    max_dd = max_drawdown(series)
    if max_dd == 0:
        return np.nan
    else:
        cagr = calculate_cagr(series)
        calmar = cagr / max_dd
        return calmar


returns.apply(calmar).sort_values(ascending = False)


def max_dd_duration(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns)/cummax
    
    begin = drawdown[drawdown == 0].index
    end = begin[1:]
    end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
    periods = end - begin
    max_ddd = periods.max()
    return max_ddd.days   


returns.apply(max_dd_duration).sort_values()



# -----------------------

# ## Coding Challenge

# __Calculate and compare__ <br>
# - __Maximum Drawdown__
# - __Calmar Ratio__
# - __Maximum Drawdown Duration__ <br>

# for __30 large US stocks__ that currently form the Dow Jones Industrial Average Index ("Dow Jones") for the time period between April 2019 and June 2021.

# __Hint:__ You can __import__ the price data from __"Dow_Jones.csv"__.
#  

# Determine the __best performing stock__ and the __worst performing stock__ according to the Calmar Ratio.

# __Compare__ Calmar Ratio and Sharpe Ratio. Does the __ranking change__?

# (Remark: Dividends are ignored here. Hence, for simplicity reasons, the Calmar Ratio is based on Price Returns only. As a consequence, dividend-paying stocks are getting penalized.) 

# ## +++ Please stop here in case you don´t want to see the solution!!! +++++







# ## Coding Challenge Solution

import pandas as pd
import numpy as np

df = pd.read_csv("Dow_Jones.csv", index_col = "Date", parse_dates = ["Date"])
df

returns = np.log(df / df.shift()) # log returns
returns


# __Maximum Drawdown__

def max_drawdown(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns)/cummax
    max_dd = drawdown.max()
    return max_dd


returns.apply(max_drawdown).sort_values()


# __Calmar Ratio__

def calculate_cagr(series):
    return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1


def calmar(series):
    
    max_dd = max_drawdown(series)
    if max_dd == 0:
        return np.nan
    else:
        cagr = calculate_cagr(series)
        calmar = cagr / max_dd
        return calmar


calm = returns.apply(calmar).sort_values(ascending = False)
calm


# Best Performing Stock: __Apple__ (AAPL) <br>
# Worst Performing Stock: __Non-determinable__ (note: you can´t compare negative Calmar Ratios)

# __Maximum Drawdown Duration__

def max_dd_duration(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns)/cummax
    
    begin = drawdown[drawdown == 0].index
    end = begin[1:]
    end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
    periods = end - begin
    max_ddd = periods.max()
    return max_ddd.days 


returns.apply(max_dd_duration).sort_values()


def sharpe(series, rf = 0):
    
    if series.std() == 0:
        return np.nan
    else:
        return (series.mean() - rf) / series.std() * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))


sha = returns.apply(sharpe).sort_values(ascending = False)
sha

merged = pd.concat([calm, sha], axis = 1)
merged

merged.columns = ["Calmar", "Sharpe"]

merged.rank(ascending = False)

# -> Some Differences. __Salesforce (CRM) gets better ranked__ with Calmar (-4) while __The Nike gets penalized__ by Calmar (+5).


