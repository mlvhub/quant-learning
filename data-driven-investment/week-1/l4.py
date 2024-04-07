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
df = pd.read_csv("../SP500.csv", index_col='year')
df

# %%
df['annual_return'] = df['SP500level'].pct_change()
df

# %%
ax = df['annual_return'].plot(figsize = (12, 7))
ax.set_title("SP500 SP500 Annual Return")
ax.set(xlabel="Year", ylabel="Annual Return")

# %%
columns_to_plot = ['annual_return', 'dividendYield']
ax = df[columns_to_plot].plot(figsize = (12, 7), secondary_y = 'dividendYield')
ax.set_title("SP500 Divident Yield & Annual Return")

# %%
return_shifted = df.copy()
return_shifted['annual_return'] = return_shifted['annual_return'].shift(-1)
return_shifted

# %%
return_shifted['annual_return'].corr(return_shifted['dividendYield'])

# %%
return_shifted['annual_return'].corr(return_shifted['Peratio'])

# %%
return_shifted['annual_return'].corr(return_shifted['ShillerPEratio'])

# %%
yield_shifted = return_shifted.copy()
yield_shifted['10yearTyield'] = yield_shifted['10yearTyield'].shift(-1)
yield_shifted['annual_return'].corr(yield_shifted['10yearTyield'])

# %%
df.to_csv("../SP500_data.csv")
