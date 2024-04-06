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
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('../DIS.csv', index_col = 0, parse_dates = [0])
df

# %%
df['daily_return'] = df['Adj Close'].pct_change()
df

# %%
df['cumulative_return'] = (df['daily_return'] + 1).cumprod() - 1
df

# %%
#draw cumulative return
ax = df['cumulative_return'].plot(figsize = (12, 7), title = 'Cumulative')
ax.set_xlabel("Year")
ax.set_ylabel('Cumulative Return')

# %%
df['cumulative_return'][-1] ** (1/58) - 1

# %%
# Data after 2000-01-01
disney_filtered = df.loc["2000-01-01":]
disney_filtered

# %%
