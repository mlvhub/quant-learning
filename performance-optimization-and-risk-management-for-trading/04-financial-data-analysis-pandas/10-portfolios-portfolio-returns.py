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

# # Portfolio of Assets and Portfolio Returns

import pandas as pd
import numpy as np

prices = pd.DataFrame(data = {"Asset_A": [100, 112], "Asset_B": [100, 104]}, index = [0, 1])
prices

prices["Total"] = prices.Asset_A + prices.Asset_B
prices

returns = prices.pct_change() # simple returns
returns

0.5 * 0.12 + 0.5 * 0.04 # correct (portfolio return == weighted average of simple returns)

log_returns = np.log(prices / prices.shift()) # log returns
log_returns

0.5 * log_returns.iloc[1, 0] + 0.5 * log_returns.iloc[1, 1] # incorrect (portfolio return != weighted average of log returns)

# **Take home:** while log returns **are time-additive**, they **are not asset-additive**.
# (while simple returns are not time-additive, they are asset-additive)
