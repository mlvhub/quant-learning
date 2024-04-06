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

# +
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

start = "2014-10-01"
end = "2021-05-31"

symbols = ["BTC-USD"]

df = yf.download(symbols, start, end)
df
# -

btc = df.Close.dropna().copy().to_frame()
btc

btc.plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

btc["Returns"] = btc.Close.pct_change(periods = 1)
btc

mu = btc.Returns.mean() # arithmetic mean return -> Reward
mu

sigma = btc.Returns.std() # standard deviation of returns -> Risk/Volatility
sigma

# Does the rule "Higher Risk -> Higher Reward" hold? -> Yes
