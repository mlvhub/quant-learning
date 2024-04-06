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

# # Measuring Reward and Risk of an Investment

# **General Rule in Finance/Investing**: Higher risk must be rewarded with higher returns.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

msft = pd.read_csv('msft.csv', index_col = 'Date', parse_dates = ['Date'])
msft

msft.Price.plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

# - Reward: positive returns
# - Risk: volatility of returns

msft.describe()

mu = msft.Returns.mean() # arithmetic mean return -> Reward
mu

sigma = msft.Returns.std() # standard deviation of returns -> Risk/Volatility
sigma

np.sqrt(msft.Returns.var()) # standard deviation formula (variance)**1/2


