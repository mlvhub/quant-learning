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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

returns = pd.read_csv("returns.csv", index_col = "Date", parse_dates = ["Date"])
returns

# **GBP_USD**: Long Position in GBP (denominated in USD) <br>
# **USD_GBP**: Short Position in GBP (== Long Position in USD; denominated in GBP) <br>
# **Levered**: USD_GBP with Leverage ("Trading USD_GBP on Margin") <br>
# **Neutral**: Neutral Positions only (no Investments / Trades)  <br>
# **Low_Vol**: Active Strategy for USD_GBP with Long, Short and Neutral Positions <br>
# **Random**: Random "Strategy" for USD_GBP with random Long, Short and Neutral Positions

returns.info()

returns.cumsum().apply(np.exp).plot()

returns.Low_Vol.cumsum().apply(np.exp).plot()

returns.Low_Vol.value_counts()

returns[["Low_Vol", "Levered"]].cumsum().apply(np.exp).plot()

# Low_Vol seems preferable, but let's create a risk-adjusted return metric to better analyse the situation.

# ## Mean Return (Reward)

returns

returns.mean()

# **Annualised mean return**

td_year = returns.count() / ((returns.index[-1] - returns.index[0]).days / 365.25)
td_year

ann_mean = returns.mean() * td_year
ann_mean

np.exp(ann_mean) - 1 # CAGR

summary = pd.DataFrame(data = {"ann_mean": ann_mean})
summary

summary.rank(ascending = False)

# #### Standard Deviation of Returns

returns.std()

# +
#### Annualised Standard Deviation
# -

td_year

ann_std = returns.std() * np.sqrt(td_year)
ann_std

summary["ann_std"] = returns.std() * np.sqrt(td_year)
summary

summary.sort_values(by = "ann_std")

# ## Risk-adjusted Return ("Sharpe Ratio")

summary

# #### Graphical Solution

summary.plot(kind = "scatter", x = "ann_std", y = "ann_mean", figsize = (15,12), s = 50, fontsize = 15)
for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i, "ann_std"]+0.001, summary.loc[i, "ann_mean"]+0.001), size = 15)
plt.xlim(-0.01, 0.23)
plt.ylim(-0.02, 0.03)
plt.xlabel("Risk(std)", fontsize = 15)
plt.ylabel("Return", fontsize = 15)
plt.title("Risk/Return", fontsize = 20)
plt.show()

# **Risk-adjusted Return Metric** ("Sharpe Ratio light")

rf = 0 # simplification, don't use this assumption for portfolio management!

summary["Sharpe"] = (summary.ann_mean - rf) / summary.ann_std
summary

summary.sort_values(by = "Sharpe", ascending=False)

td_year

returns.mean() / returns.std() * np.sqrt(td_year) # annualising daily Sharpe

# ## Putting everything together

returns


def sharpe(series, rf = 0):
    if series.std() == 0:
        return np.nan
    else:
        return (series.mean() - rf) / series.std() * np.sqrt(series.count() / ((series.index[-1] - series.index[0]).days / 365.25))


returns.apply(sharpe, rf = 0)

sharpe(series = returns.Levered, rf = 0)

# ### Limitations of "Sharpe Ratio"
#
# - only takes into account mean & variance (std)
# - assumes normally distributed returns
# - overestimates risk-adjusted returns when fat tails are present
# - can be manipulated with smoothed data (e.g. monthly returns)
# - can't compare/rank instruments with negative Sharpe Ratios
# - penalises upside and downside volatility equally
