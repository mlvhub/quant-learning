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

from backtester.data_handler import DataHandler
from backtester.backtester import Backtester
from backtester.strategies import Strategy

data = DataHandler(symbol="HE").load_data_from_csv("example_data.csv")
data.head()

# # Backtesting

# +
# Define your strategy, indicators, and signal logic here
strategy = Strategy(
    indicators={},
    signal_logic=lambda row: (1 if row["trade_signal_sentiment"] > 0 else -1),
)
data = strategy.generate_signals(data)

backtester = Backtester()
backtester.backtest(data)
backtester.calculate_performance()
