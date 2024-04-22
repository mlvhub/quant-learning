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

# Mean reversion strategy with the goal to sell the asset if it is trading more than 3 standard deviations above the rolling mean and to buy the asset if it is trading more than 3 standard deviations below the rolling mean.

# +
from backtester.data_handler import DataHandler
from backtester.backtester import Backtester
from backtester.strategies import Strategy

symbol = "HE"
start_date = "2022-01-01"
end_date = "2022-12-31"
# -

# # Backtesting

# +
data = DataHandler(symbol=symbol, start_date=start_date, end_date=end_date).load_data()

# Define your strategy, indicators, and signal logic here
strategy = Strategy(
    indicators={
        "sma_50": lambda row: row["close"].rolling(window=50).mean(),
        "std_3": lambda row: row["close"].rolling(window=50).std() * 3,
        "std_3_upper": lambda row: row["sma_50"] + row["std_3"],
        "std_3_lower": lambda row: row["sma_50"] - row["std_3"],
    },
    signal_logic=lambda row: (
        1
        if row["close"] < row["std_3_lower"]
        else -1 if row["close"] > row["std_3_upper"] else 0
    ),
)
data = strategy.generate_signals(data)

backtester = Backtester()
backtester.backtest(data)
backtester.calculate_performance()

# +
data = DataHandler(symbol=symbol, start_date=start_date, end_date=end_date).load_data()

# Define your strategy, indicators, and signal logic here
strategy = Strategy(
    indicators={
        "sma_50": lambda row: row["close"].rolling(window=50).mean(),
        "std_3": lambda row: row["sma_50"].std() * 3,
        "std_3_upper": lambda row: row["sma_50"] + row["std_3"],
        "std_3_lower": lambda row: row["sma_50"] - row["std_3"],
    },
    signal_logic=lambda row: (
        1
        if row["close"] < row["std_3_lower"]
        else -1 if row["close"] > row["std_3_upper"] else 0
    ),
)
data = strategy.generate_signals(data)

backtester = Backtester()
backtester.backtest(data)
backtester.calculate_performance()
# -

#
