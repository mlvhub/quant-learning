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

start = "2015-01-02"
end = "2020-12-31"

symbols = ["GE", "BRK-A"]

df = yf.download(symbols, start, end)
df
# -

close = df.Close.copy()
close

close.GE.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

norm = close.div(close.iloc[0])
norm

norm.dropna().plot(figsize = (15, 8), fontsize = 13, logy = False)
plt.legend(fontsize = 13)
plt.show()

norm.GE[-1] # last value of GE (in the dataset)


