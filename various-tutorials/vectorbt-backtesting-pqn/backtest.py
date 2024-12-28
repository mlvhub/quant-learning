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
import numpy as np
import scipy.stats as stats

import vectorbt as vbt

# %%
# Create an array of moving average windows to test and download price data for.
windows = np.arange(10, 50)

price = vbt.YFData.download('AAPL').get('Close')

# %%
# Create data splits for the walk-forward optimisation.

# This segments the prices into 30 splits, each two years long, 
# and reserves 180 days for the test.
(in_price, in_indexes), (out_price, out_indexes) = price.vbt.rolling_split(
    n=30, 
    window_len=365 * 2,
    set_lens=(180,),
    left_to_right=False,
)


# %%
# Build two moving averages for each window we pass in.
# It creates DataFrames showing where the fast-moving average 
# crosses above the slow-moving average. These are the trade entries. 
# It does the opposite for the trade exits.
def simulate_all_params(price, windows, **kwargs):
    fast_ma, slow_ma = vbt.MA.run_combs(
        price, windows, r=2, short_names=["fast", "slow"]
    )
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(price, entries, exits, **kwargs)
    return pf.sharpe_ratio()


# %%
# Figure out the combination of windows that maximises the Sharpe ratio.

# Returns the indexes in the DataFrame for the windows in each data split 
# that maximises the Sharpe ratio. 
def get_best_index(performance, higher_better=True):
    if higher_better:
        return performance[performance.groupby('split_idx').idxmax()].index
    return performance[performance.groupby('split_idx').idxmin()].index

# Returns the window values.
def get_best_params(best_index, level_name):
    return best_index.get_level_values(level_name).to_numpy()


# %%
# Runs the backtest with the windows that maximise the Sharpe ratio. 
# Creates the moving average values that maximise the Sharpe ratio, 
# runs the backtest, and returns the Sharpe ratio.
def simulate_best_params(price, best_fast_windows, best_slow_windows, **kwargs):
    fast_ma = vbt.MA.run(price, window=best_fast_windows, per_column=True)
    slow_ma = vbt.MA.run(price, window=best_slow_windows, per_column=True)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(price, entries, exits, **kwargs)
    return pf.sharpe_ratio()


# %%
# Optimise the moving average windows on the in-sample data.
in_sharpe = simulate_all_params(
    in_price, 
    windows, 
    direction="both", 
    freq="d"
)

# %%
# Get the optimized windows and test them with out-of-sample data.
in_best_index = get_best_index(in_sharpe)

in_best_fast_windows = get_best_params(
    in_best_index,
    'fast_window'
)
in_best_slow_windows = get_best_params(
    in_best_index,
    'slow_window'
)
in_best_window_pairs = np.array(
    list(
        zip(
            in_best_fast_windows, 
            in_best_slow_windows
        )
    )
)

# %%
# Create a DataFrame that has the Sharpe ratio for the backtest using out-of-sample 
# test data and the window values that optimize the Sharpe ratio from the in-sample data.
out_test_sharpe = simulate_best_params(
    out_price, 
    in_best_fast_windows, 
    in_best_slow_windows, 
    direction="both", 
    freq="d"
)

# %% [markdown]
# The whole point of this analysis is to understand if the parameters you fit on the in-sample data can be used in real life to make money.
#
# The most common issue in backtesting is overfitting to random data. (Especially when using technical analysis.)

# %%
in_sample_best = in_sharpe[in_best_index].values
out_sample_test = out_test_sharpe.values

t, p = stats.ttest_ind(
    a=out_sample_test,
    b=in_sample_best,
    alternative="greater"
)

# %%
print(f"t-statistic: {t:.4f}, p-value: {p:.4f}")

# %% [markdown]
# In this case, the p-value is close to 1 which means you cannot reject the null hypothesis that the out-of-sample Sharpe ratios are greater than the in-sample Sharpe ratios.
#
# In other words, you are overfitted to noise.
#
# The moving crossover is a toy example that is known not to make money. But the technique of optimizing parameters using walk-forward optimization is the state-of-the-art way of removing as much bias as possible.

# %% [markdown]
#
