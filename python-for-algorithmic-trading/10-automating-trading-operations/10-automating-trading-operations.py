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

# # Automating Trading Operations

# ## Capital Management

# ### Kelly Criterion in Binomial Setting

# +
import math
import time
import numpy as np
import pandas as pd
import datetime as dt
from pylab import plt, mpl

np.random.seed(1000)
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

# +
p = 0.55

f = p - (1 - p)

I = 50
n = 100


# -

def run_simulation(f):
    c = np.zeros((n, I))
    c[0] = 100
    for i in range(I):
        for t in range(1, n):
            o = np.random.binomial(1, p)
            if o > 0:
                c[t, i] = (1 + f) * c[t - 1, i]
            else:
                c[t, i] = (1 - f) * c[t - 1, i]
    return c


c_1 = run_simulation(f)
c_1

c_1.round(2)

plt.figure(figsize=(10, 6))
plt.plot(c_1, 'b', lw=0.5)
plt.plot(c_1.mean(axis=1), 'r', lw=2.5)

# +
c_2 = run_simulation(0.05)
c_3 = run_simulation(0.25)
c_4 = run_simulation(0.5)

plt.figure(figsize=(10, 6))
plt.plot(c_1.mean(axis=1), 'r', label='$f^*=0.1$')
plt.plot(c_2.mean(axis=1), 'b', label='$f=0.05$')
plt.plot(c_3.mean(axis=1), 'y', label='$f=0.25$')
plt.plot(c_4.mean(axis=1), 'm', label='$f=0.5$')
plt.legend(loc=0)
# -

# ### Kelly Criterion for Stocks and Indices

raw = pd.read_csv('./pyalgo_eikon_eod_data.csv', index_col=0, parse_dates=True)
raw.head()

symbol = '.SPX'
data = pd.DataFrame(raw[symbol])
data.head()

data['return'] = np.log(data / data.shift(1))
data.dropna(inplace=True)
data.tail()

mu = data['return'].mean() * 252
mu

sigma = data['return'].std() * 252 ** 0.5
sigma

# +
r = 0.0

kelly = (mu - r) / sigma ** 2
kelly

# +
equs = []

def kelly_strategy(f):
    global equs
    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)
    cap = 'capital_{:.2f}'.format(f)
    data[equ] = 1
    data[cap] = data[equ] * f
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]
        data.loc[t, cap] = data[cap].loc[t_1] * \
                            math.exp(data['return'].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - \
                            data[cap].loc[t_1] + \
                            data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f


# -

kelly_strategy(f * 0.5)

kelly_strategy(f * 0.66)

kelly_strategy(f)

data[equs].tail()

ax = data['return'].cumsum().apply(np.exp).plot(figsize=(10, 6))
data[equs].plot(ax=ax, legend=True)

# ## ML-Based Trading Strategy

# ### Vectorized Backtesting

# +
import tpqoa

# %time api = tpqoa.tpqoa('../pyalgo.cfg')
# -

instrument = 'EUR_USD'

raw = api.get_history(instrument,
    start='2020-06-08',
    end='2020-06-13',
    granularity='M10',
    price='M')
raw.tail()

raw.info()

spread = 0.00012

mean = raw['c'].mean()
mean

ptc = spread / mean
ptc

raw['c'].plot(figsize=(10, 6), legend=True)

# +
data = pd.DataFrame(raw['c'])
data.columns = [instrument,]

data.tail()

# +
window = 20

data['return'] = np.log(data / data.shift(1))
data['vol'] = data['return'].rolling(window).std()
data['mom'] = np.sign(data['return'].rolling(window).mean())
data['sma'] = data[instrument].rolling(window).mean()
data['min'] = data[instrument].rolling(window).min()
data['max'] = data[instrument].rolling(window).max()

data.dropna(inplace=True)
data.tail()

# +
lags = 6
features = ['return', 'vol', 'mom', 'sma', 'min', 'max']

cols = []
for f in features:
    for lag in range(1, lags + 1):
        col = f'{f}_lag_{lag}'
        data[col] = data[f].shift(lag)
        cols.append(col)

data.dropna(inplace=True)
data.tail()
# -

data['direction'] = np.where(data['return'] > 0, 1, -1)
data[cols].iloc[:lags, :lags]

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

n_estimators=15
random_state=100
max_depth=2
min_samples_leaf=15
subsample=0.33

dtc = DecisionTreeClassifier(random_state=random_state,
                                      max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf)

# The idea of boosting in the context of classification is to use an ensemble of 
# base classifiers to arrive at a superior predictor that is supposed to be less prone to overfitting
model = AdaBoostClassifier(base_estimator=dtc,
                                   n_estimators=n_estimators,
                                   random_state=random_state)

split = int(len(data) * 0.7)
split

train = data.iloc[:split].copy()
train.tail()

mu, std = train.mean(), train.std()

train_ = (train - mu) / std

model.fit(train_[cols], train['direction'])

# in-sample accuracy
accuracy_score(train['direction'], model.predict(train_[cols]))

test = data.iloc[split:].copy()

test_ = (test - mu) / std

test['position'] = model.predict(test_[cols])

# out-of-sample accuracy
accuracy_score(test['direction'], test['position'])

# taking costs into account
test['strategy'] = test['position'] * test['return']
sum(test['position'].diff() != 0)

test['strategy_tc'] = np.where(test['position'].diff() != 0,
                                        test['strategy'] - ptc,
                                        test['strategy'])

test[['return', 'strategy', 'strategy_tc']].sum(
                 ).apply(np.exp)

test[['return', 'strategy', 'strategy_tc']].cumsum(
                 ).apply(np.exp).plot(figsize=(10, 6))

# ### Optimal Leverage

mean = test[['return', 'strategy_tc']].mean() * len(data) * 52
mean

var = test[['return', 'strategy_tc']].var() * len(data) * 52
var

vol = var ** 0.5
vol

mean / var

mean / var * 0.5

# +
to_plot = ['return', 'strategy_tc']

for lev in [10, 20, 30, 40, 50]:
    label = 'lstrategy_tc_%d' % lev
    test[label] = test['strategy_tc'] * lev
    to_plot.append(label)

test[to_plot].cumsum().apply(np.exp).plot(figsize=(10, 6))
# -

# ### Risk Analysis

# +
# The risk analysis that follows assumes a leverage ratio of 30.
# Maximum drawdown is the largest loss (dip) after a recent high.
# Longest drawdown period is the longest period the trading strategy needs to get back to a recent high

"""
The analysis assumes that the initial equity position is 3,333 EUR leading to an initial 
position size of 100,000 EUR for a leverage ratio of 30.
"""

equity = 3333

risk = pd.DataFrame(test['lstrategy_tc_30'])
risk['equity'] = risk['lstrategy_tc_30'].cumsum().apply(np.exp) * equity
risk['cummax'] = risk['equity'].cummax()
risk['drawdown'] = risk['cummax'] - risk['equity']
risk['drawdown'].max()
# -

t_max = risk['drawdown'].idxmax()
t_max

# Technically, a new high is characterized by a drawdown value of 0.
temp = risk['drawdown'][risk['drawdown'] == 0]
periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
periods[20:30]

t_per = periods.max()
t_per

t_per.seconds / 60 / 60

risk[['equity', 'cummax']].plot(figsize=(10, 6))
plt.axvline(t_max, c='r', alpha=0.5)

# +
"""
Another important risk measure is value-at-risk (VaR). 
It is quoted as a currency amount and represents the maximum loss to be expected 
given both a certain time horizon and a confidence level.
"""

import scipy.stats as scs

percentiles = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
risk['return'] = np.log(risk['equity'] /
                                  risk['equity'].shift(1))
VaR = scs.scoreatpercentile(equity * risk['return'], percentiles)

def print_var(VaR):
    print('{}    {}'.format('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percentiles, VaR):
        print('{:16.2f} {:16.3f}'.format(100 - pair[0], -pair[1]))

print_var(VaR)

# +
"""
Finally, the following code calculates the VaR values for a time horizon of 
one hour by resampling the original DataFrame object.
"""

hourly = risk.resample('1H', label='right').last()
hourly['return'] = np.log(hourly['equity'] /
                                   hourly['equity'].shift(1))
VaR = scs.scoreatpercentile(equity * hourly['return'], percentiles)

print_var(VaR)
# -

# ### Persisting the Model Object

# +
import pickle

algorithm = {'model': model, 'mu': mu, 'std': std}
pickle.dump(algorithm, open('algorithm.pkl', 'wb'))
# -

# ## Online Algorithm

algorithm = pickle.load(open('algorithm.pkl', 'rb'))
algorithm['model']


class MLTrader(tpqoa.tpqoa):
    def __init__(self, config_file, algorithm):
        super(MLTrader, self).__init__(config_file)
        self.model = algorithm['model']
        self.mu = algorithm['mu']
        self.std = algorithm['std']
        self.units = 100000
        self.position = 0
        self.bar = '5s'
        self.window = 2
        self.lags = 6
        self.min_length = self.lags + self.window + 1
        self.features = ['return', 'sma', 'min', 'max', 'vol', 'mom']
        self.raw_data = pd.DataFrame()

    def prepare_features(self):
        self.data['return'] = np.log(self.data['mid'] /
                                    self.data['mid'].shift(1))
        self.data['sma'] = self.data['mid'].rolling(self.window).mean()
        self.data['min'] = self.data['mid'].rolling(self.window).min()
        self.data['mom'] = np.sign(
            self.data['return'].rolling(self.window).mean())
        self.data['max'] = self.data['mid'].rolling(self.window).max()
        self.data['vol'] = self.data['return'].rolling(
            self.window).std()
        self.data.dropna(inplace=True)
        self.data[self.features] -= self.mu
        self.data[self.features] /= self.std
        self.cols = []
        for f in self.features:
            for lag in range(1, self.lags + 1):
                col = f'{f}_lag_{lag}'
                self.data[col] = self.data[f].shift(lag)
                self.cols.append(col)

    def on_success(self, time, bid, ask):
        df = pd.DataFrame({'bid': float(bid), 'ask': float(ask)},
                        index=[pd.Timestamp(time).tz_localize(None)])
        self.raw_data = self.raw_data.append(df)
        self.data = self.raw_data.resample(self.bar,
                                label='right').last().ffill()
        self.data = self.data.iloc[:-1]
        if len(self.data) > self.min_length:
            self.min_length +=1
            self.data['mid'] = (self.data['bid'] +
                                self.data['ask']) / 2
            self.prepare_features()
            features = self.data[
                self.cols].iloc[-1].values.reshape(1, -1)
            signal = self.model.predict(features)[0]
            print(f'NEW SIGNAL: {signal}', end='\r')
            if self.position in [0, -1] and signal == 1:
                print('*** GOING LONG ***')
                self.create_order(self.stream_instrument,
                            units=(1 - self.position) * self.units)
                self.position = 1
            elif self.position in [0, 1] and signal == -1: 
                print('*** GOING SHORT ***')
                self.create_order(self.stream_instrument,
                            units=-(1 + self.position) * self.units)
                self.position = -1


mlt = MLTrader('../pyalgo.cfg', algorithm)
mlt.stream_data(instrument, stop=5)
