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

# # Margin Trading & Levered Returns

# Definition: "Margin trading refers to the practice of using borrowed funds from a broker to trade a financial asset, which forms the collateral for the loan from the broker." (Investopedia.com)
#
# In simple words: investors don't pay the full price but they get the full benefit (minus borrowing costs)
#
# It's a two edged sword: **leverage amplifies both gains and losses.**
# <br>
# In the event of a loss, the collateral gets reduced and the investor either posts additional margin or the broker closes the position (margin call).
#
# **Example**
#
# A trader buys a stock (\\$100) on margin (50%). After one day the price increases to \\$110.
# <br>
# Calculate the unlevered return and levered return.

import numpy as np

p0 = 100
p1 = 110
leverage = 2
margin = p0 / 2

margin

unlev_return = (p1 - p0) / p0 # simple return
unlev_return

lev_return = (p1 - p0) / margin # simple return
lev_return

lev_return == unlev_return * leverage # this relationship is true for simple returns

unlev_return = np.log((p1 - p0) / p0 + 1) # log returns
unlev_return

lev_return = np.log((p1 - p0) / margin + 1) # log returns
lev_return

lev_return == unlev_return * leverage # this relationship is NOT true for log returns

# **Take home: to calculate levered returns, don't multiply leverage with log returns.**

# ## Levered Returns

# **Hypothesis: for (highly) profitable investment, the more leverage the better.**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.6f}'.format
plt.style.use("seaborn-v0_8")

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft["Simple_Ret"] = msft.Price.pct_change() # simple returns
msft

leverage = 7

# (Simplified) Assumptions:
# - restore leverage on a daily basis (by buying/selling shares)
# - no trading costs
# - no borrowing costs

msft["Lev_Returns"] = msft.Simple_Ret.mul(leverage) # levered simple returns
msft

msft["Lev_Returns"] = np.where(msft["Lev_Returns"] < -1, -1, msft["Lev_Returns"]) # limit loss to 100%

msft[["Returns", "Lev_Returns"]].add(1).cumprod().plot()

msft.Simple_Ret.max()

msft.Lev_Returns.max()

msft.Simple_Ret.min()

msft.Lev_Returns.min()

# **What happens when leverage greater than ...?**

-1 / msft.Simple_Ret.min()

# **Take home:**
# 1. with leverage you can (theoretically) lose more than the initial margin (in practice: margin call/margin closeout before)
# 2. even for (highly) profitable instruments the hypothesis **"the more leverage the better**" does not hold
# 3. it's a two edged (**non-symmetrical**) sword: leverage amplifies losses more than it amplifies gains
