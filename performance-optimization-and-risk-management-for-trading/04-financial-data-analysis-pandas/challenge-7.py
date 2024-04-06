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

# # Coding Challenge #7

# 1. Calculate levered returns for Bitcoin (leverage = 4).
#
# 2. Visualize and compare with unlevered Investment.
#
# 3. Some Traders trade Bitcoin with extremely high leverage (> 100). Do you think this is a good idea (assuming no additional/advanced Risk Management Tools)? 

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

btc = close["BTC-USD"].dropna().copy().to_frame().rename(columns = {"BTC-USD": "Price"})
btc["Returns"] = btc.Price.pct_change()
btc

btc["Lev_Returns"] = btc.Returns.mul(4)
btc

btc["Mad_Lev_Returns"] = btc.Returns.mul(100)
btc

btc[["Returns", "Lev_Returns"]].add(1).cumprod().plot()

(btc.Returns.max(), btc.Lev_Returns.max(), btc.Mad_Lev_Returns.max())

(btc.Returns.min(), btc.Lev_Returns.min(), btc.Mad_Lev_Returns.min())

# 3. Some Traders trade Bitcoin with extremely high leverage (> 100). Do you think this is a good idea (assuming no additional/advanced Risk Management Tools)?

# A/ No

# ## Practical Test 2

# +
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

start = "2015-01-02"
end = "2022-12-31"

symbols = ["AMZN", "MSFT", "GE", "DIS", "META", "AAPL"]

df = yf.download(symbols, start, end)
df
# -

close = df.Close.copy()
close

returns = close.apply(lambda x: x.pct_change())
returns

returns.add(1).cumprod().plot()




