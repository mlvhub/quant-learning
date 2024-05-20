# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
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

# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('../SP500_data.csv')
df

#df = df[['year','annual_return','dividendYield']][df['year'] > 1990]
df = df[['year','annual_return','dividendYield']][df['year'] < 1990]
df

df.describe()

df.loc[df['annual_return'] > 0]['annual_return'].describe()

df.loc[df['annual_return'] < 0]['annual_return'].describe()

df['annual_return'].plot(kind = 'hist', figsize = (12, 7), title = 'S&P500')

fig, ax = plt.subplots(figsize = (7, 12))
fig = sm.qqplot(df['annual_return'], line = 'q', ax = ax)
ax.set_title('S&P500 annual return normal probability plot')
ax.set(xlabel="Normal Scores", ylabel="S&P500 Annual Return")

return_shifted = df.copy()
return_shifted['annual_return'] = return_shifted['annual_return'].shift(-1)
return_shifted

result = smf.ols("annual_return ~ dividendYield", data = return_shifted).fit()
result.summary()

result.summary2()

table = result.summary2().tables[1].iloc[1:,]
table

table["adj_rsquared"] =result.rsquared_adj
table


