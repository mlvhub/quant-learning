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
import yfinance as yf

start = "2014-10-01"
end = "2021-05-31"

symbol = "BA"

df = yf.download(symbol, start, end)
df

df.info()

symbol = ["BA", "MSFT", "^DJI", "EURUSD=X", "GC=F", "BTC-USD"]

# Ticker Symbols:
# - **BA**: Boeing (US Stock)
# - **MSFT**: Microsoft Corp (US Stock)
# - **^DJI**: Dow Jones Industrial Average (US Stock)
# - **EURUSD-X**: Exchange Rate for Currency Pair EUR/USD (Forex)
# - **GC=F**: Gold Price (Precious Metal/Commodity)
# - **BTC-USD**: Bitcoin in USD (Cryptocurrency)

df = yf.download(symbol, start, end)
df

df.info()

df.to_csv("multi_assets.csv")

import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

df = pd.read_csv('multi_assets.csv', header = [0, 1], index_col = 0, parse_dates = [0])
df

df.info()

df.Close

df.Close.BA

df.loc[:, ('Close', 'BA')]

df.loc['2015-01-07']

df.loc['2015']

df.loc['2020-06', ('Close', 'BA')]

swapped_df = df.swaplevel(axis = 'columns').sort_index(axis = 'columns')
swapped_df

swapped_df['EURUSD=X']

swapped_df['BTC-USD']

close = df.Close.copy()
close

close.describe()

close.BA.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

close.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

# **Take Home**: prices are on a different scale, so it's not meaningful to compare them.

# ## Normalizing Financial Time Series to a Base Value (100)

close

close.iloc[0, 0] # first price of BA

close.BA.div(close.iloc[0, 0]).mul(100)

close.iloc[0] # first price of tickers

norm = close.div(close.iloc[0]).mul(100)
norm

norm.dropna().plot(figsize = (15, 8), fontsize = 13, logy = False)
plt.legend(fontsize = 13)
plt.show()

norm.dropna().plot(figsize = (15, 8), fontsize = 13, logy = True)
plt.legend(fontsize = 13)
plt.show()

# **Take Home**: normalised prices help to compare different financial instruments, but they are limited when it comes to measuring/comparing performance in detail.

close.to_csv('close.csv')

# ## Price Changes and Financial Returns

# More meaningful/useful than prices: price changes

import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

close = pd.read_csv('close.csv', index_col = "Date", parse_dates = ["Date"])
close

msft = close.MSFT.dropna().to_frame().copy()
msft

msft.rename(columns = {"MSFT": "Price"}, inplace = True)
msft

msft.shift(periods = 1) # move price one day forward

msft["P_lag1"] = msft.shift(periods = 1)
msft

msft["P_diff"] = msft.Price.sub(msft.P_lag1) # calculate price diff (1)
msft

msft["P_diff2"] = msft.Price.diff(periods = 1) # calculate price diff (2)
msft

msft.P_diff.equals(msft.P_diff2)

# **Absolute Price Changes are not meaningul**

# **Relative/Percentage Price Changes** (Returns)

msft.Price.div(msft.P_lag1) - 1 # alternative 1

msft["Returns"] = msft.Price.pct_change(periods = 1) # alternative 2
msft

# **Take Home: Relative Price Changes (Returns) are meaningful and comparable across instruments**

clean_msft = msft.drop(columns = ["P_lag1", "P_diff", "P_diff2"])
clean_msft

clean_msft.to_csv("msft.csv")
