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
# %matplotlib inline
import pandas as pd

data = pd.read_csv('./06-pandas/sp500.csv', index_col='Date', parse_dates=['Date'])
data['SMA'] = data['SP500'].rolling(50).mean()
data.plot()



# %%
help(pd.read_csv)

# %%
