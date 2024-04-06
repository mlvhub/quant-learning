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

# # Simple Returns vs Logarithmic Returns (log returns)

# Logarithmic returns have favourable characteristics, and simple returns have drawbacks.

# ## Discrete Compounding

# **Annual compounding**: interests accrue once a year, at the end of the year.

# **8% p.a.** with annual compounding on savings ($100) after one year.

PV = 100
r = 0.08
n = 1

100 * 1.08

FV = PV * (1 + r) ** n
FV

effective_annual_rate = (FV / PV) ** (1 / n) - 1
effective_annual_rate

# **Quarterly compounding**: interests accrue once a quarter, at the end of the quarter.

# **8% p.a.** with quarterly compounding on savings ($100) after one year.

PV = 100
r = 0.08
n = 1
m = 4

100 * 1.02 * 1.02 * 1.02 * 1.02

FV = PV * (1 + r / m) ** (n * m)
FV

effective_annual_rate = (FV / PV) ** (1 / n) - 1
effective_annual_rate

# **Take home**: the more frequent the compounding happens the better.

# **Monthly compounding**: interests accrue once a month, at the end of the month.

**8% p.a.** with monthly compounding on savings ($100) after one year.

PV = 100
r = 0.08
n = 1
m = 12

FV = PV * (1 + r / m) ** (n * m)
FV

effective_annual_rate = (FV / PV) ** (1 / n) - 1
effective_annual_rate

# ## Continuous Compounding

# **8% p.a.** with **continuous compounding** on savings ($100).

import numpy as np

PV = 100
r = 0.08
n = 1
m = 100000 # approx.infinity

FV = PV * (1 + r / m) ** (n * m)
FV

FV = PV * np.exp(n * r) # exact math with e (euler number)
FV

euler = np.exp(1)
euler

PV * euler ** (n * r)

effective_annual_rate = (FV / PV) ** (1 / n) - 1
effective_annual_rate

effective_annual_rate = np.exp(r) - 1
effective_annual_rate

r = np.log(FV / PV) # inverse calculation -> use log
r

# **Take home**: prices of traded financial instruments change (approximately) continuously.
# <br>
# Intuitively, it makes sense to work with log returns

# ## Log Returns

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft["log_return"] = np.log(msft.Price / msft.Price.shift()) # daily log returns (log of current price divided by the previous price)
msft

msft.describe()

mu = msft.log_return.mean() # mean log return = Reward
mu

sigma = msft.log_return.std() # standard deviation of lug returns = Risk/Volatility
sigma

# **Investment Multiple**

msft.Returns.add(1).prod() # compounding simple returns ("compound returns")

np.exp(msft.log_return.sum()) # adding log returns ("cumulative returns")

# **Normalised Prices with Base 1**

msft.Returns.add(1).cumprod() # compounding simple returns ("compound returns")

np.exp(msft.log_return.cumsum()) # adding log returns ("cumulative returns")

msft.log_return.cumsum().apply(np.exp) # adding log returns ("cumulative returns")

# **CAGR**

(msft.Price[-1] / msft.Price[0]) ** (1 / ((msft.index[-1] - msft.index[0]).days / 365.25)) - 1

trading_days_year = msft.Returns.count() / ((msft.index[-1] - msft.index[0]).days / 365.25)
trading_days_year

np.exp(msft.log_return.mean() * trading_days_year) - 1 # correct with mean of daily log returns

msft.Returns.mean() * trading_days_year # incorrect with mean of daily simple returns

np.exp(msft.log_return.mean() * 252) - 1 # good approximation (for US stocks)


