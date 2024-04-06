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

# # Rolling Statistics

# ### (Another) General rule in Finance/Investing: Past performance is not an indicator of future performance.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft["log_return"] = np.log(msft.Price / msft.Price.shift()) # daily log returns (log of current price divided by the previous price)
msft

ann_mu = msft.log_return.mean() * 252 # annualised mean return
ann_mu

ann_std = msft.log_return.std() * np.sqrt(252) # annualised std of returns
ann_std

# ### Are Return and Risk constant over time? No, of course not! They change over time.

# ### Let's measure/quantify  this with rolling statistics

window = 252 # rolling window 252 trading years (~1 (calendar) year)

msft.log_return.rolling(window = 252)

msft.log_return.rolling(window = 252).sum()

roll_mean = msft.log_return.rolling(window = 252).mean() * 252
roll_mean

roll_mean.iloc[252:]

roll_mean.plot(figsize = (12, 8))
plt.show()

roll_std = msft.log_return.rolling(window = 252).std() * np.sqrt(252)
roll_std

roll_std.plot(figsize = (12, 8))
plt.show()

roll_mean.plot(figsize = (12, 8))
roll_std.plot(figsize = (12, 8))
plt.show()

# **Take home**: be careful, you'll always find (sub)periods with low returns and high risk, and viceversa.
#
# - analysis period must be sufficiently long enough to reduce impact of random noise.
# - analysis period should be as short as possible and should only include the latest trends/regimes.
# - commonly used reporting period: 3 years/36 months

# +
### Another Example: Simple Moving Average (prices) - SMA
# -

sma_window = 50

msft.Price.plot(figsize = (12, 8))
msft.Price.rolling(sma_window).mean().plot()
plt.show()
