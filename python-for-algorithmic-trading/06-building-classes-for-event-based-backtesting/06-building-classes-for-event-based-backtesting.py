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

# When it comes to vectorized backtesting, potential shortcomings of the approach are the following:
#
# - Look-ahead bias: vectorized backtesting is based on the complete data set available and does not take into account that new data arrives incrementally
# - Simplification: for example, fixed transaction costs cannot be modeled by vectorization, which is mainly based on relative returns. Also, fixed amounts per trade or the non-divisibility of single financial instruments (for example, a share of a stock) cannot be modeled properly
# - Non-recursiveness: algorithms, embodying trading strategies, might take recurse to state variables over time, like profit and loss up to a certain point in time or similar path-dependent statistics. Vectorization cannot cope with such features

# Event-based backtesting allows one to address these issues by a more realistic approach to model trading realities, some of its advantages are:
#
# - Incremental approach: as in the trading reality, backtesting takes place on the premise that new data arrives incrementally, tick-by-tick and quote-by-quote.
# - Realistic modeling: one has complete freedom to model those processes that are triggered by a new and specific event.
# - Path dependency: it is straightforward to keep track of conditional, recursive, or otherwise path-dependent statistics, such as the maximum or minimum price seen so far, and to include them in the trading algorithm.
# - Reusability: backtesting different types of trading strategies requires a similar base functionality that can be implemented and unified through object-oriented programming.
# - Close to trading: certain elements of an event-based backtesting system can sometimes also be used for the automated implementation of the trading strategy.
#

# ## Backtesting Base Class

# +
from backtest_base import BacktestBase

bb = BacktestBase('AAPL.O', '2010-1-1', '2019-12-31', 10000)
bb.data.info()
# -

bb.plot_data()

# ## Long-Only Backtesting Class

# +
from backtest_long_only import BacktestLongOnly

lobt = BacktestLongOnly('AAPL.O', '2010-1-1', '2019-12-31', 10000, verbose=False)


# -

def run_strategies():
    lobt.run_sma_strategy(42, 252)
    lobt.run_momentum_strategy(60)
    lobt.run_mean_reversion_strategy(50, 5)


run_strategies()

# transaction costs: 10 USD fix, 1% variable
lobt = BacktestLongOnly('AAPL.O', '2010-1-1', '2019-12-31', 10000, 10.0, 0.01, False)
run_strategies()

# ## Long-Short Backtesting Class

# +
from backtest_long_short import BacktestLongShort

lsbt = BacktestLongShort('EUR=', '2010-1-1', '2019-12-31', 10000, verbose=False)


# -

def run_strategies():
    lsbt.run_sma_strategy(42, 252)
    lsbt.run_momentum_strategy(60)
    lsbt.run_mean_reversion_strategy(50, 5)


run_strategies()

lsbt = BacktestLongShort('AAPL.O', '2010-1-1', '2019-12-31', 10000, 10.0, 0.01, False)
run_strategies()
