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

# ### Getting and Preparing the Data (Part 1) 

import pandas as pd

movie = pd.read_csv("movies_metadata.csv", low_memory= False)

movie

movie.info()

pd.to_datetime(movie.release_date, errors = "coerce")

movie = movie.set_index(pd.to_datetime(movie.release_date, errors = "coerce")).drop(columns = ["release_date"])

movie.sort_index(inplace = True)

movie

df = movie.loc[:, ["title", "budget", "revenue"]].copy()

df

df.info()

df.budget = pd.to_numeric(df.budget, errors = "coerce")



# ### Getting and preparing the Data (Part 2) 

df

df.info()

df.describe()

df.iloc[:, -2:]  = df.iloc[:, -2:] / 1000000

df

df.loc[df.title.isna()]

df.dropna(inplace = True)

df.info()

df.budget.value_counts()

df.revenue.value_counts()

df = df.loc[(df.revenue > 0) & (df.budget > 0)]

df

df.info()

df.describe()

df.sort_values("budget", ascending = False)

df.sort_values("revenue", ascending = False)

df.to_csv("bud_vs_rev.csv")



# ### How to calculate Covariance and Correlation 

import pandas as pd
import numpy as np

df = pd.read_csv("bud_vs_rev.csv", parse_dates = ["release_date"], index_col = "release_date")

df

df = df.loc["2016"]

df

df.info()

df.describe()

df.mean(numeric_only=True)

df.var(numeric_only=True)

df.cov()

df.budget.cov(df.revenue)

df.corr()

df.budget.corr(df.revenue)

df.budget.cov(df.revenue) / (df.budget.std() * df.revenue.std())

np.cov(df.budget, df.revenue)

np.corrcoef(df.budget, df.revenue)



# ### Correlation and Scatterplots – visual Interpretation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bud_vs_rev.csv", parse_dates = ["release_date"], index_col = "release_date")

df = df.loc["2016"]

df

df.plot(kind = "scatter", x = "budget", y = "revenue", figsize = (15, 10), fontsize = 15)
plt.xlabel("Budget (in MUSD)", fontsize = 13)
plt.ylabel("Revenue (in MUSD)", fontsize = 13)
plt.show()

sns.set(font_scale=1.5)
sns.lmplot(data = df, x = "budget", y = "revenue", height = 8) # new instead of jointplot
plt.show()


