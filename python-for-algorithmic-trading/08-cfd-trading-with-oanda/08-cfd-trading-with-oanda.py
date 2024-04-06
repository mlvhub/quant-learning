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

import tpqoa

api = tpqoa.tpqoa('../pyalgo.cfg')

# ## Retrieving Historical Data
#
# ### Looking Up Instruments Available for Trading

api.get_instruments()[:15]

# ### Backtesting a Momentum Strategy on Minute Bars

instrument = 'EUR_USD'
start = '2020-08-10'
end = '2020-08-12'
granularity = 'M1'
price = 'M'

data = api.get_history(instrument, start, end, granularity, price)
data

data.info()

data[['c', 'volume']].head()

import numpy as np

data['returns'] = np.log(data['c'] / data['c'].shift(1))

cols = []
for momentum in [15, 30, 60, 120]:
    col = f'position_{momentum}'
    data[col] = np.sign(data['returns'].rolling(momentum).mean())
    cols.append(col)

from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

strats = ['returns']

for col in cols:
    strat = 'strategy_{}'.format(col.split('_')[1])
    data[strat] = data[col].shift(1) * data['returns']
    strats.append(strat)

data[strats].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))

# ### Factoring In Leverage and Margin

# leverage of 20:1
data[strats].dropna().cumsum().apply(lambda x: x * 20).apply(np.exp).plot(figsize=(10, 6))

# ## Working with Streaming Data

instrument = 'EUR_USD'

api.stream_data(instrument, stop=10)

# ## Placing Market Orders

api.create_order(instrument, 1000)

api.create_order(instrument, -1500)

api.create_order(instrument, 500)

# ## Implementing Trading Strategies in Real Time

import MomentumTrader as MT

mt = MT.MomentumTrader(
    './pyalgo.cfg',
    instrument=instrument,
    bar_length='10s',
    momentum=6,
    units=10000
)

mt.stream_data(mt.instrument, stop=500)

# close out the final position
oo = mt.create_order(instrument, units=-mt.position * mt.units,
                              ret=True, suppress=True)

# ## Retrieving Account Information 

api.get_account_summary()

api.get_transactions(tid=int(oo['id']) - 2)

api.get_transactions()
