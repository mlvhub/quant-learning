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

# # Investment Multiple and CAGR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

msft = pd.read_csv('msft.csv', index_col = 'Date', parse_dates = ['Date'])
msft

# **Investment Multiple**: ending value of 1 [dollar] initially invested
# <br>
# Multiple = Ending value / Initial investment

multiple = (msft.Price[-1] / msft.Price[0])
multiple

# **Price increase (in %)**

(multiple - 1) * 100

msft.Price / msft.Price[0] # similar/identical concept: normalised price with base value 1

# **Drawback of Investment Multiple**: doesn't take into account the investment period. I'ts only meaningul paired with the investment period.

# **Compound Annual Growth Rate (CAGR)**: the (constant annual) rate of return that would be required for an investment to grow from its beginning balance to its ending balance, assuming the profits were reinvested at the end of each year of the investment's lifespan. (Wikipedia)

start = msft.index[0]
start

end = msft.index[-1]
end

td = end - start
td

td_years = td.days / 365.25
td_years

cagr = multiple ** (1 / td_years) - 1
cagr

cagr = (msft.Price[-1] / msft.Price[0]) ** (1 / ((msft.index[-1] - msft.index[0]).days / 365.25)) -1
cagr

(1 + cagr) ** td_years # alternative to calculate the multiple with CAGR

# **CAGR can be used to compare investments with different investment horizons.**

multiple = (1 + msft.Returns).prod() # another alternative to calculate the multiple
multiple

n = msft.Returns.count()
n

geo_mean = multiple ** (1 / n) - 1 # geometric mean return (daily)
geo_mean

(1 + geo_mean) ** n # yet another alternative to calculate the multiple (geometric mean)

# **Compound returns, CAGR and geometric mean return are closely related concepts.**

mu = msft.Returns.mean() # arithmetic mean return
mu

# **The arithmetic mean returns is always greater than the geometric mean return and less useful.**

(1 + mu) ** n # NOT possible to calculate the multiple with the arithmetic mean


