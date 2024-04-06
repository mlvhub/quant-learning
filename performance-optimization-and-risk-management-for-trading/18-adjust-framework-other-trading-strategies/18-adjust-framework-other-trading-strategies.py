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

# # Creating and Backtesting Mean-Reversion Strategies (Bollinger Bands)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

data = pd.read_csv("intraday.csv", parse_dates = ["time"], index_col = "time")
data.columns = ["price"]
data

data.info()

data.plot(figsize = (12, 8))

data.loc["2019-08"].plot(figsize = (12, 8))

data["returns"] = np.log(data.div(data.shift(1)))
data

# ## Defining a Mean-Reversion Strategy (Bollinger Bands) 

# __Mean Reversion__: Financial Instruments are from time to time overbought / oversold and revert back to mean prices. 
#
# __Bollinger Bands__: Consists of a SMA (e.g. 30) and Upper and Lower Bands +- (2) Std Dev away from SMA.

SMA = 30
dev = 2

data["SMA"] = data["price"].rolling(SMA).mean()

data[["price", "SMA"]].plot(figsize = (12, 8))

data.loc["2019-08", ["price", "SMA"]].plot(figsize = (12, 8))

data.loc["2019-08", ["price", "SMA"]].plot(figsize = (12, 8))

data["price"].rolling(SMA).std()

data["price"].rolling(SMA).std().plot(figsize = (12, 8 ))

data["Lower"] = data["SMA"] - data["price"].rolling(SMA).std() * dev # Lower Band -2 Std Dev

data["Upper"] = data["SMA"] + data["price"].rolling(SMA).std() * dev # Upper Band -2 Std Dev

data.drop(columns = "returns").plot(figsize = (12, 8))

data.drop(columns = "returns").loc["2019-08"].plot(figsize = (12, 8))

data.dropna(inplace = True)

# ### Determining positions

data["distance"] = data.price - data.SMA # helper Column

data["position"] = np.where(data.price < data.Lower, 1, np.nan) # 1. oversold -> go long

data

data["position"] = np.where(data.price > data.Upper, -1, data["position"]) # 2. overbought -> go short

data

# 3. crossing SMA ("Middle Band") -> go neutral
data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])

data

data["position"] = data.position.ffill().fillna(0) # where 1-3 isn´t applicable -> hold previous position

data

data.position.value_counts()

data.drop(columns = ["returns", "distance"]).loc["2019-08"].plot(figsize = (12, 8), secondary_y = "position")

data.position.plot(figsize = (12, 8))

# ## Vectorized Strategy Backtesting

data

data["strategy"] = data.position.shift(1) * data["returns"]
data.dropna(inplace = True)
data

data["creturns"] = data["returns"].cumsum().apply(np.exp)
data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
data[["creturns", "cstrategy"]].plot(figsize = (12 , 8))

data

ptc = 0.00007

data["trades"] = data.position.diff().fillna(0).abs()
data

data.trades.value_counts()

data["strategy_net"] = data.strategy - data.trades * ptc

data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)

data

data[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8))

data[["returns", "strategy_net"]].mean() * (4 * 252) # annualized return

data[["returns", "strategy_net"]].std() * np.sqrt(4 * 252) # annualized risk

# ## Adjusting the framework and creating a Backtester Class

# **The Class live in action**

filepath = "one_minute.csv"
symbol = "EURUSD"
start = "2018-01-01"
end = "2019-12-31"
ptc = 0.00007

# +
from boll_backtester import BollBacktester

tester = BollBacktester(filepath = filepath, symbol = symbol, start = start, end = end, tc = ptc)
tester
# -

tester.test_strategy(freq = 180, window = 100, dev = 2)

tester.plot_results()


