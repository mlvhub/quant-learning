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
# # %matplotlib inline
import pandas as pd


# %% [markdown]
# When it comes to calculating correlations, we will notice most practitioners prefer the use of log returns. Mainly because it can be convenient to work with log returns when processing and analysing time series data, but it has no discernible difference to the end result.
#
# What's important to understand however, is that for correlations to make sense, we must use either percent returns or log returns.

# %%
def get_returns(file):
    return pd.read_csv(file+'.csv', index_col=0, parse_dates=True).pct_change()



# %%
df = get_returns('./06-pandas/sp500')
df

# %%
df['NDX'] = get_returns('./06-pandas/ndx')
df


# %%
# Plot the correlation between the SP500 and NASDAQ over the last 200 days
# slice syntax: [start:stop:step]
df['SP500'].rolling(50).corr(df['NDX'])[-200:].plot()

# %%
