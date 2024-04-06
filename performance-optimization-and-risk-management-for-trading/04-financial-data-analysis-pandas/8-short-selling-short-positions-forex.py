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

# # Short Selling/Short Positions

# What's the rational behind short selling an instrument?
#
# **Making profits/positive returns when prices fall.**

# **Stocks example:**
#
# Today an investor buys the ABC stock for \\$100. One day later he sells the stock for \\$110.
# <br>
# **Profit: \\$10**
# <br>
# **Long Position (benefits from rising prices)**
#
# Today an investor borrows the ABC stock from another investor and sells it for \\$100. One day later he buys the stock for \\$90 and retuns it to the lender.
# <br>
# **Profit: \\$10**
# <br>
# **Short Position (benefits from failling prices)**
#
# In some countries (and for some instruments like stocks) short selling is prohibited.
# <br>
# Most intuitive/popular use case for short selling: **Currencies (Forex)**

# **EUR/USD** ("Long Euro" == "Short USD")

t0 = 1.10
t1 = 1.25

# Today an investor buys €1 and pays \\$1.10. One day later he sells €1 for \\$1.25.
# <br>
# **Profit: \$10**
# <br>
# **Long Position (benefits from rising prices)**

t1 / t0 - 1 # the EUR appreciates by 13.64% relative to USD (simple return)

# EUR Long Position returns **13.64%** (simple return)

# What return would you expected for the corresponding EUR short position?

t0 = 1 / 1.10
t1 = 1 / 1.25

t1 / t0 - 1 # the USD depreciates by 12% relative to EUR

# EUR short position returns **-12%** (simple return)

# ## Real Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.6f}'.format
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

close["USDEUR=X"] = 1/close["EURUSD=X"]
close

fx = close[["EURUSD=X", "USDEUR=X"]].dropna().copy()
fx

fx.plot()

simple_ret = fx.pct_change() # simple returns
simple_ret

simple_ret.add(1).prod() - 1 # compound simple returns

# **For simple returns: long position retuns != short position returns * (-1)**

log_ret = np.log(fx / fx.shift()) # log returns
log_ret

log_ret.sum() # cumulative log returns

# **For log returns: long position returns == short position returns * (-1)**

norm_fx = log_ret.cumsum().apply(np.exp) # normalised prices (base 1)
norm_fx

norm_fx.iloc[0] = [1, 1]
norm_fx

norm_fx.plot()
