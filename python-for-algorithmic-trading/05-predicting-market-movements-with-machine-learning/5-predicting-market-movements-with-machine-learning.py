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

# ## Using Linear Regression for Market Movement Prediction

import os
import random
import numpy as np
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

x = np.linspace(0, 10)
x


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
set_seeds()

y = x + np.random.standard_normal(len(x))
y

reg = np.polyfit(x, y, deg=1)
reg

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='data')
plt.plot(x, np.polyval(reg, x), 'r', lw=2.5, label='linear regression')
plt.legend(loc=0)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='data')
xn = np.linspace(0, 20)
plt.plot(xn, np.polyval(reg, xn), 'r', lw=2.5, label='linear regression')
plt.legend(loc=0)

# ### The Basic Idea for Price Prediction

# The number of days used as input is generally called lags. Using today’s index level and the two more from before therefore translates into three lags.

x = np.arange(12)
x

lags = 3

m = np.zeros((lags + 1, len(x) - lags))

m[lags] = x[lags:]
for i in range(lags):
    m[i] = x[i:i - lags]

m

m.T

reg = np.linalg.lstsq(m[:lags].T, m[lags], rcond=None)[0]
reg

np.dot(m[:lags].T, reg)

# ### Predicting Index Levels

import pandas as pd

raw = pd.read_csv("pyalgo_eikon_eod_data.csv", index_col=0, parse_dates=True).dropna()

raw.info()

symbol = 'EUR='

data = pd.DataFrame(raw[symbol])

data.rename(columns={symbol: 'price'}, inplace=True)

data

lags = 5

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['price'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

reg = np.linalg.lstsq(data[cols], data['price'], rcond=None)[0]
reg

data['prediction'] = np.dot(data[cols], reg)

data

data[['price', 'prediction']].plot(figsize=(10, 6))

data[['price', 'prediction']].loc['2019-10-1':].plot(figsize=(10, 6))

# ### Predicting Future Returns

#
