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


# %%
def get_returns(file):
    return pd.read_csv(file+'.csv', index_col='Date', parse_dates=['Date'])



# %%
def calc_corr(ser1, ser2, window):
    ret1 = ser1.pct_change()
    ret2 = ser2.pct_change()
    return ret1.rolling(window).corr(ret2)


# %%
points_to_plot = 300

data = get_returns('./06-pandas/indexes')
data

# %%
# rebase the two series to the same starting point 
for ind in data:
    data[ind + '_rebased'] = (data[-points_to_plot:][ind].pct_change() + 1).cumprod()
data

# %%
data[['SP500_rebased', 'NDX_rebased']].plot(figsize=(10, 5))

# %%
# relative strength, NDX to SP500
data['rel_str'] = data['NDX'] / data['SP500']
data

# %%
data['rel_str'].plot(figsize=(10, 5))

# %%
# 100 day rolling correlation
data['corr'] = calc_corr(data['NDX'], data['SP500'], 100)
data


# %%
data['corr'].plot(figsize=(10, 5))

# %%
import matplotlib.pyplot as plt

plot_data = data[-points_to_plot:]

fig = plt.figure(figsize=(12, 12))

# The first subplot, planning for 3 plots high, 1 plot wide, this being the first. 
ax = fig.add_subplot(311)
ax.set_title('Index Comparison')
ax.semilogy(plot_data['SP500_rebased'], linestyle ='-', label='S&P 500', linewidth=3.0)
ax.semilogy(plot_data['NDX_rebased'], linestyle ='--', label='Nasdaq', linewidth=3.0)
ax.legend()
ax.grid(False)

ax = fig.add_subplot(312)
ax.set_title('Relative Strength')
ax.plot(plot_data['rel_str'], linestyle='--', label='NDX/SP500', linewidth=3.0)
ax.legend()
ax.grid(True)

ax = fig.add_subplot(313)
ax.set_title('50 Day Rolling Correlation')
ax.plot(plot_data['corr'], linestyle='--', label='Correlation', linewidth=3.0)
ax.legend()
ax.grid(True)

plt.show()

# %%
