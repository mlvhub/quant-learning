# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# run the Jupyter notebook inside the chapter folder for local imports to work
import backtrader as bt
from LuxorStrategy import LuxorStrategy
from datetime import datetime
import pandas as pd


# %%
# autoreload the modules
# %load_ext autoreload
# %autoreload 2

# %%
# Create a cerebro instance
cerebro = bt.Cerebro()

# Add the strategy
cerebro.addstrategy(LuxorStrategy)

# %%
# Load data

# generated with: 

# curl \                                                                              main ✖ ◼
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer API_KEY" \
#   "https://api-fxpractice.oanda.com/v3/instruments/GBP_USD/candles?from=2012-10-21&count=1000&price=M&granularity=M30" | jq -r '.candles[] | "\(.time), \(.mid.o), \(.mid.h), \(.mid.l), \(.mid.c)"' > data.csv
# NOTE: add headers manually: datetime, open, high, low, close

# Read the CSV file
df = pd.read_csv('data.csv', parse_dates=[0])  # Replace with your CSV file path
df.info()

# %%
# Convert to Backtrader data feed
data = bt.feeds.PandasData(
    dataname=df,
    datetime=0,  # Assuming first column is datetime
    open=1,      # Column position for Open
    high=2,      # Column position for High
    low=3,       # Column position for Low
    close=4,     # Column position for Close
    volume=5,    # Column position for Volume
    openinterest=6
)

# %%
cerebro.adddata(data)

# Set starting cash
cerebro.broker.setcash(100000.0)


# %%
# Run the strategy
cerebro.run()


# %%
# Print final portfolio value
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

# %%
# Plot the results
cerebro.plot(iplot=False)


# %%
