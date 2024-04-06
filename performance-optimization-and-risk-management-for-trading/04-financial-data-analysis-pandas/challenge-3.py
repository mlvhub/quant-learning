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

# # Coding Challenge #3
#
# 1. Calculate Boeing´s Investment Multiple
#
# 2. Calculate Boeing´s CAGR
#
# 3. Calculate Boeing´s Geometric Mean Return
#
# 4. Calculate Boeing´s Investment Multiple with compound daily returns
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

close = pd.read_csv('close.csv', index_col = 'Date', parse_dates = ['Date'])
close

boeing = close.BA.copy().dropna().to_frame().rename(columns = {'BA': 'Price'})
boeing["Returns"] = boeing.Price.pct_change(periods = 1)
boeing

multiple = (boeing.Price[-1] / boeing.Price[0])
multiple

cagr = (boeing.Price[-1] / boeing.Price[0]) ** (1 / ((boeing.index[-1] - boeing.index[0]).days / 365.25)) -1
cagr

n = boeing.Returns.count()
n

geo_mean = multiple ** (1 / n) - 1 # geometric mean return (daily)
geo_mean

multiple_daily = (1 + geo_mean) ** n
multiple_daily


