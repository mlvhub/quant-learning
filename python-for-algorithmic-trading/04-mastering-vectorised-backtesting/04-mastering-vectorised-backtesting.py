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

# ## Trading Strategies
#
# ### Simple moving averages (SMA) based strategies
#
# A signal is derived, for example, when an SMA defined on a shorter time window—say 42 days—crosses an SMA defined on a longer time window—say 252 days.
#
# ### Momentum strategies
#
# These are strategies that arex based on the hypothesis that recent performance will persist for some additional time.
#
# ### Mean-reversion strategies
#
# The reasoning behind mean-reversion strategies is that stock prices or prices of other financial instruments tend to revert to some mean level or to some trend level when they have deviated too much from such levels.

# ## Making Use of Vectorization
#
#
#

v = [1, 2, 3, 4, 5]

sm = [2 * i for i in v]
sm

# doesn't work as expected with the stdlib
2 * v

# ### Vectorization with NumPy

import numpy as np

a = np.array(v)
a

type(a)

# works as expected with NumPy
2 * a

# +
# multi-dimensional arrays:

a = np.arange(12).reshape((4, 3))
a
# -

2 * a

a.mean()

a.mean(axis=0)

# ### Vectorization with pandas

a = np.arange(15).reshape(5, 3)
a

import pandas as pd

columns = list('abc')
columns

index = pd.date_range('2021-7-1', periods=5, freq='B')
index

df = pd.DataFrame(a, columns=columns, index=index)
df

# works similarly to NumPy, with the difference being aggregation is done column-wise
2 * df

df.sum()

np.mean(df)

df['a'] + df['c']

# boolean conditions
df['a'] > 5

df[df['a'] > 5]

df['c'] > df['b']

# ## Strategies Based on Simple Moving Averages

# ### Getting into the Basics

raw = pd.read_csv('../data/pyalgo_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
raw.info()

data = pd.DataFrame(raw["EUR="])

data.rename(columns={'EUR=': 'price'}, inplace=True)

data.info()

data['SMA1'] = data['price'].rolling(42).mean()

data['SMA2'] = data['price'].rolling(252).mean()

data.tail()

# %matplotlib inline
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

data.plot()

data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
data.tail()

data.dropna(inplace=True)

data['position'].plot(ylim=[-1.1, 1.1])

data['returns'] = np.log(data['price'] / data['price'].shift(1))

data['returns'].hist(bins=35)

data['strategy'] = data['position'].shift(1) * data['returns']

data[['returns', 'strategy']].sum()

data[['returns', 'strategy']].sum().apply(np.exp)

data[['returns', 'strategy']].cumsum().apply(np.exp).plot()

data[['returns', 'strategy']].mean() * 252

np.exp(data[['returns', 'strategy']].mean() * 252) - 1

data[['returns', 'strategy']].std() * 252 ** 0.5

(data[['returns', 'strategy']].apply(np.exp) - 1).std() * 252 ** 0.5

data['cumret'] = data['strategy'].cumsum().apply(np.exp)

data['cummax'] = data['cumret'].cummax()

data[['cumret', 'cummax']].dropna().plot(figsize=(10, 6))

drawdown = data['cummax'] - data['cumret']
drawdown.max()

temp = drawdown[drawdown == 0]
periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
periods[12:15]

periods.max()

# ## Generalizing the Approach

# +
import sma_vector_backtester as SMA

smabt = SMA.SMAVectorBacktester('EUR=', 42, 252, '2010-1-1', '2019-12-31')
# -

smabt.run_strategy()

# +
# %time

smabt.optimize_parameters((30, 50, 2), (200, 300, 2))
# -

smabt.plot_results()

# ## Strategies Based on Momentum

# There are two basic types of momentum strategies:
#
# The first type is cross-sectional momentum strategies. Selecting from a larger pool of instruments, these strategies buy those instruments that have recently outperformed relative to their peers (or a benchmark) and sell those instruments that have underperformed. The basic idea is that the instruments continue to outperform and underperform, respectively—at least for a certain period of time. 
#
# The second type is time series momentum strategies. These strategies buy those instruments that have recently performed well and sell those instruments that have recently performed poorly. In this case, the benchmark is the past returns of the instrument itself.

# ### Getting into the Basics

data = pd.DataFrame(raw['XAU='])

data.rename(columns={'XAU=': 'price'}, inplace=True)

data['returns'] = np.log(data['price'] / data['price'].shift(1))

data['position'] = np.sign(data['returns'])

data['strategy'] = data['position'].shift(1) * data['returns']

data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))

data['position'] = np.sign(data['returns'].rolling(3).mean())

data['strategy'] = data['position'].shift(1) * data['returns']

data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))

fn = '../data/AAPL_1min_05052020.csv'

data = pd.read_csv(fn, index_col=0, parse_dates=True)

data.info()

data['returns'] = np.log(data['CLOSE'] / data['CLOSE'].shift(1))

# +
to_plot = ['returns']

for m in [1, 3, 5, 7, 9]:
    data['position_%d' % m] = np.sign(data['returns'].rolling(m).mean())
    data['strategy_%d' % m] = (data['position_%d' % m].shift(1) * data['returns'])
    to_plot.append('strategy_%d' % m)
# -

data[to_plot].dropna().cumsum().apply(np.exp).plot(title='AAPL intraday 05. May 2020', figsize=(10, 6), style=['-', '--', '--', '--', '--', '--'])

# ### Generalizing the Approach

# +
import mom_vector_backtester as Mom

mombt = Mom.MomVectorBacktester('XAU=', '2010-1-1', '2019-12-31', 10000, 0.0)
# -

mombt.run_strategy(momentum=3)

mombt.plot_results()

mombt = Mom.MomVectorBacktester('XAU=', '2010-1-1', '2019-12-31', 10000, 0.001)

mombt.run_strategy(momentum=3)

mombt.plot_results()

# ## Strategies Based on Mean Reversion

# Roughly speaking, mean-reversion strategies rely on a reasoning that is the opposite of momentum strategies. If a financial instrument has performed “too well” relative to its trend, it is shorted, and vice versa.
#
# The idea is to define a threshold for the distance between the current stock price and the SMA, which signals a long or short position.

# ### Getting into the Basics

data = pd.DataFrame(raw['GDX'])

data.rename(columns={'GDX': 'price'}, inplace=True)

data['returns'] = np.log(data['price'] / data['price'].shift(1))

data["returns"].plot()

SMA = 25

data['SMA'] = data['price'].rolling(SMA).mean()

threshold = 3.5

data['distance'] = data['price'] - data['SMA']

data['distance'].dropna().plot(figsize=(10, 6), legend=True)
plt.axhline(threshold, color='r')
plt.axhline(-threshold, color='r')
plt.axhline(0, color='r')

data['position'] = np.where(data['distance'] > threshold, -1, np.nan)

data['position'] = np.where(data['distance'] < -threshold, 1, data['position'])

data['position'] = np.where(data['distance'] * data['distance'].shift(1) < 0, 0, data['position'])

data['position'] = data['position'].ffill().fillna(0)

data['position'].iloc[SMA:].plot(ylim=[-1.1, 1.1], figsize=(10, 6))

data['strategy'] = data['position'].shift(1) * data['returns']

data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))

# ### Generalizing the Approach

import mr_vector_backtester as MR

mrbt = MR.MRVectorBacktester('GLD', '2010-1-1', '2019-12-31', 10000, 0.001)

mrbt.run_strategy(SMA=43, threshold=7.5)

mrbt.plot_results()


