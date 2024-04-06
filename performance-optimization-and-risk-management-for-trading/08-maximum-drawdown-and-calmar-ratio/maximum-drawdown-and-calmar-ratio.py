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

# **Maximum Drawdown (MDD)**: it's a measure of an asset's largest price drop from a peak to a trough. (Investopedia.com) 
#
#
# **Calmar Ratio**: reward per Unit of Tail Risk.
# $$
# Calmar Ratio = \frac{Reward}{Tail Risk} = \frac{CAGR}{Maximum Drawdown}
# $$
#
# **Maximum Drawdown Duration**: the worst (maximum/longest) amount of time an investment has seen between peaks (equity highs). (Wikipedia.com)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns

returns.info()

returns.cumsum().apply(np.exp).plot()

# ## Maximum Drawdown

symbol = "USD_GBP"

returns[symbol].cumsum().apply(np.exp).plot()

instr = returns[symbol].to_frame().copy()
instr

instr["creturns"] = instr.cumsum().apply(np.exp) # cumulative returns (normalised prices with Base == 1)

instr["cummax"] = instr.creturns.cummax() # cumulative maximum of creturns

instr

instr[["creturns", "cummax"]].plot()

instr["drawdown"] = abs(instr["creturns"] - instr["cummax"]) / instr["cummax"] # (pos.) drawdown in %
instr

instr[["creturns", "cummax", "drawdown"]].plot(secondary_y = "drawdown", figsize = (15, 8))

max_drawdown = instr.drawdown.max() # maximum drawdown
max_drawdown

instr.drawdown.idxmax() # maximum drawdown date

instr.loc[instr.drawdown.idxmax()]

# ## Calmar Ratio

cagr = np.exp(instr[symbol].sum())**(1/((instr.index[-1] - instr.index[0]).days / 365.25)) - 1
cagr

calmar = cagr / max_drawdown
calmar

# ## Maximum Drawdown Duration

drawdown = instr.drawdown.copy()
drawdown

# **Drawdawn Period**: Time period between peaks. Whenever drawdown == 0, a new peak has been reached.

begin = drawdown[drawdown == 0].index # get all peak dates (beginning of drawdown periods)
begin

end = begin[1:] # get the corresponding end dates for all drawdown periods
end = end.append(pd.DatetimeIndex([drawdown.index[-1]])) # add last available date
end

periods = end - begin # time difference between peaks
periods

max_ddd = periods.max()
max_ddd

max_ddd.days


# ## Putting everything together

def max_drawdown(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns) / cummax
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


returns.apply(calmar).sort_values(ascending=False)


def max_dd_duration(series):
    creturns = series.cumsum().apply(np.exp)
    cummax = creturns.cummax()
    drawdown = (cummax - creturns) / cummax
    
    begin = drawdown[drawdown == 0].index
    end = begin[1:]
    end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
    periods = end - begin
    max_ddd = periods.max()
    return max_ddd.days


returns.apply(max_dd_duration).sort_values()
