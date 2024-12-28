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
#start_date = pd.Timestamp('1997-01-01', tz='utc')
#end_date = pd.Timestamp('2021-03-30', tz='utc')
start_date = pd.Timestamp('1997-01-01')
# Quandl data is only available up to 2018-03-27
#end_date = pd.Timestamp('2021-03-30')
end_date = pd.Timestamp("2018-03-27")


# %%
def initialize(context):
    context.stock = symbol('AAPL')
    context.index_average_window = 100


# %%
def handle_data(context, data):
    equities_history = data.history(context.stock, 'close', context.index_average_window, '1d')
    if equities_history.iloc[-1] > equities_history.mean():
        stock_weight = 1.0
    else:
        stock_weight = 0.0
    order_target_percent(context.stock, stock_weight)


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
# This notebook is a simple backtest of a moving average crossover strategy with a single asset in a single market.
#
# This is far from a complete trading strategy, as there is no diversification possible with single market strategies, they become pure market timing models and this rarely works out in real life.
# Another issue is the selection of the market itself, as this is discretionary.
#
# A long-only trend following logic on a stock which has had a tremendous bull run makes for a great looking backtest, but it does not have any predictive value.
# If we were to run a backtest on stocks starting from ten years ago, the instrument selection must resemble what we would have chosen back then and not today to have a fighting chance of having any predictive value.
