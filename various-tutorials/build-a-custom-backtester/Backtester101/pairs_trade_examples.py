# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: back
#     language: python
#     name: python3
# ---

# # Setup

# Setup a simple pairs trading strategy for Roku (ROKU) and Netflix (NFLX).
#
# We will enter a position (buy) if one stock has moved 5% or more than the other one over the course of the last five days. 
# We will sell the top one and buy the bottom one until it reverses.

# +
from backtester.data_handler import DataHandler
from backtester.backtester import Backtester
from backtester.strategies import Strategy

symbol = "NFLX,ROKU"
start_date = "2023-01-01"
# -

# # Backtesting

# +
import pandas as pd

data = DataHandler(
    symbol=symbol,
    start_date=start_date,
).load_data()
data = pd.merge(
    data["NFLX"].reset_index(),
    data["ROKU"].reset_index(),
    left_index=True,
    right_index=True,
    suffixes=("_NFLX", "_ROKU"),
)
# We want to trade the ROKU stock so we rename the close_ROKU column to close
data = data.rename(columns={"close_ROKU": "close"})
data.head()

# +
strategy = Strategy(
    indicators={
        "day_5_lookback_NFLX": lambda row: row["close_NFLX"].shift(5),
        "day_5_lookback_ROKU": lambda row: row["close"].shift(5),
    },
    signal_logic=lambda row: (
        1
        if row["close_NFLX"] > row["day_5_lookback_NFLX"] * 1.05
        else -1 if row["close_NFLX"] < row["day_5_lookback_NFLX"] * 0.95 else 0
    ),
)
data = strategy.generate_signals(data)

backtester = Backtester()
backtester.backtest(data)
backtester.calculate_performance()
# -

#
