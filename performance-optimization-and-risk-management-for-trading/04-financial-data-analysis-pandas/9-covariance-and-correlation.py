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

# # Covariance and Correlation

# Do instruments/assets move together (if so, to what extend?)
#
# Three cases:
# - unrelated (no relationship/correlation)
# - moving together (positive relationship/correlation)
# - move in opposite directions (negative relationship/correlation)
#
# **Correlation between instruments/assets plays an important role in portfolio management.**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

close["USDEUR=X"] = 1/close["EURUSD=X"]
close

returns = close.apply(lambda x: np.log(x.dropna() / x.dropna().shift()))
returns

returns.cov() # covariance (hard to interpret)

returns.corr() # correlation coefficient (easy to interpret)

# Three cases:
# - no correlation: coefficient == 0
# - moving together: 0 < correlation coefficient <= 1 (positive)
# - moving in opposite directions -1 <= correlation coefficient < 0 (negative)

import seaborn as sns

plt.figure(figsize = (12, 8))
sns.set(font_scale = 1.4)
sns.heatmap(returns.corr(), cmap = "RdYlBu_r", annot = True, annot_kws = {"size": 15}, vmin = -1, vmax = 1)

# **Take home: similar assets are (highly) positive correlated. Different assets exhibit low/no/negative correlation.**
# In portfolio management it's beneficial to have assets with low/no/negative correlation (portfolio diversification).
