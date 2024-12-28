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
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol
import pandas as pd
import matplotlib.pyplot as plt

# %%
# If using UTC, we get the following error: `AttributeError: ‘datetime.timezone’ object has no attribute ‘key’`
start_date = pd.Timestamp('2003-01-01')
end_date = pd.Timestamp("2017-08-31")


# %%
def initialize(context):
    dji = [
        "AAPL",
        "AXP",
        "BA",
        "CAT",
        "CSCO",
        "CVX",
        "DIS",
        # Symbol 'DWDP' is not available in Quandl, it might be DD instead
        # "DWDP",
        "DD",
        "GS",
        "HD",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PFE",
        "PG",
        "TRV",
        "UNH",
        "UTX",
        "V",
        "VZ",
        "WBA",
        "WMT",
        "XOM"
    ]
    context.dji_symbols = [symbol(s) for s in dji]
    context.index_average_window = 100


# %%
def handle_data(context, data):
    stock_hist = data.history(context.dji_symbols, 'close', context.index_average_window, '1d')

    stock_analytics = pd.DataFrame()

    # add column for above or below average
    stock_analytics['above_mean'] = stock_hist.iloc[-1] > stock_hist.mean()

    # set weight for stocks to buy
    stock_analytics.loc[stock_analytics['above_mean'] == True, 'weight'] = 1 / len(context.dji_symbols)
    # set weight to 0 for the rest
    stock_analytics.loc[stock_analytics['above_mean'] == False, 'weight'] = 0.0

    for stock, analytics in stock_analytics.iterrows():
        if data.can_trade(stock):
            order_target_percent(stock, analytics['weight'])


# %%
def analyze(context, perf):
    fig = plt.figure(figsize=(12, 8))

    # First chart
    ax = fig.add_subplot(311)
    ax.set_title("Strategy Results")
    ax.semilogy(
        perf["portfolio_value"], linestyle="-", label="Equity Curve", linewidth=3.0
    )
    ax.legend()
    ax.grid(False)

    # Second chart
    ax = fig.add_subplot(312)
    ax.plot(perf["gross_leverage"], label="Exposure", linestyle="-", linewidth=1.0)
    ax.legend()
    ax.grid(True)

    # Third chart
    ax = fig.add_subplot(313)
    ax.plot(perf["returns"], label="Returns", linestyle="-.", linewidth=1.0)
    ax.legend()
    ax.grid(True)


# %%
result = run_algorithm(
    start=start_date,
    end=end_date,
    initialize=initialize,
    handle_data=handle_data,
    analyze=analyze,
    capital_base=10000,
    data_frequency='daily',
    bundle='quandl'
)

# %% [markdown]
# There is a major issue with the backtest. We're using the current (at the time the book was written) constituents of the DJI, and the index did not have the same constituents back in 2003 when this simulation starts.
#
# Stocks end up in an index when they perform well in the past, therefore we already know the strategy will perform well in the past.
