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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from zipline import run_algorithm
from zipline.api import order_target_percent, symbol, schedule_function, date_rules, time_rules
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf


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
    context.universe = [symbol(s) for s in dji]
    context.history_window = 20
    context.stocks_to_hold = 10
    schedule_function(handle_data, date_rules.month_start(), time_rules.market_close())


# %%
def month_perf(ts):
    perf = (ts[-1]/ts[0]) - 1
    return perf


# %%
def handle_data(context, data):
    stock_hist = data.history(context.universe, 'close', context.history_window, '1d')
    perf_table = stock_hist.apply(month_perf).sort_values(ascending=False)

    buy_list = perf_table[:context.stocks_to_hold]
    the_rest = perf_table[context.stocks_to_hold:]

    # Make sure we are flat the rest
    for stock in the_rest.index:
        if data.can_trade(stock):
            order_target_percent(stock, 0.0)

    for stock, perf in buy_list.items():
        stock_weight = 1 / context.stocks_to_hold
        if data.can_trade(stock):
            order_target_percent(stock, stock_weight)


# %%
def analyze(context, perf):
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)
    pf.create_returns_tear_sheet(returns, benchmark_rets=None)


# %%
start_date = pd.Timestamp('2003-01-01')
end_date = pd.Timestamp("2017-08-31")

# %%
result = run_algorithm(
    start=start_date,
    end=end_date,
    initialize=initialize,
    analyze=analyze,
    capital_base=10000,
    data_frequency='daily',
    bundle='quandl'
)

# %% [markdown]
# Returns alone won't tell us the whole story. We need to look at metrics such as the maximum drawdown, the Sharpe ratio, and the annualised volatility.
#
# We should be looking to get good numbers, but more importantly, realistic numbers. If our backtest numbers look too good to be true, they probably are.
# We are unlikely to compound over 15% per year over a long period of time. We are unlikely to have a Sharpe ratio of 1 or more and we will probably see maximum drawdowns of three times our annualised return. (These are just guidelines)

# %%
for column in result:
    print( column)


# %%
result.loc[' 2010-11-17']


# %%
