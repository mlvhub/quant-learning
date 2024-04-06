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

# # Instrument Performance Comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

close = pd.read_csv('close.csv', index_col = "Date", parse_dates = ["Date"])
close

close.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

close.info()

np.log(close / close.shift()).info() # keep NaN

close.apply(lambda x: np.log(x.dropna() / x.dropna().shift())).info() # remove NaN

returns = close.apply(lambda x: np.log(x.dropna() / x.dropna().shift()))
returns

returns.info()

returns.describe()

summary = returns.agg(["mean", "std"]).T
summary

summary.columns = ["Mean", "Std"]
summary

summary.plot(kind = "scatter", x = "Std", y = "Mean", figsize = (15,12), s = 50, fontsize = 15)
for i in summary.index:
    plt.annotate(i, xy = (summary.loc[i, "Std"] + 0.00005, summary.loc[i, "Mean"] + 0.00005), size = 15)
plt.xlabel("Risk (std)", fontsize = 15)
plt.ylabel("Mean Return", fontsize = 15)
plt.title("Mean-Variance Analysis", fontsize = 20)

# **Take Home**: 
# - there's no clear best performer without further analysis.
# - higher risk means higher returns.
# - BA underperformed.
#
# Mean-Variance analysis has one major shortcoming: it assumes that financial returns follow a normal distribution, and that's (typically) not true.
#
# Standard Deviation of Returns underestimates the true/full risk of an investment as it fails to measure "Tail Risks".

# ## Normality of Financial Returns

msft = pd.read_csv('msft.csv', index_col = 'Date', parse_dates = ['Date'])
msft

msft["log_return"] = np.log(msft.Price / msft.Price.shift()) 
msft

msft.describe()

msft.log_return.plot(kind = "hist", bins = 100, figsize = (15, 8), density = True, fontsize = 15)
plt.xlabel("Daily Returns", fontsize = 15)
plt.ylabel("Frequency", fontsize = 15)
plt.title("Frequency Distribution of Returns", fontsize = 20)

# **Do MSFT Returns follow a normal distribution?**
#
# A normally distributed random variable can fully described by its:
# - mean
# - standard deviation

# Higher Central Moments are zero:
# - skew = 0 (measures symmetry around the mean)
# - (excess) kurtosis = 0 (positive excess kurtosis = more observations in the "tails")

mu = msft.log_return.mean()
mu

sigma = msft.log_return.std()
sigma

import scipy.stats as stats

stats.skew(msft.log_return.dropna()) # in a normal distribution: skew = 0

stats.kurtosis(msft.log_return.dropna(), fisher = True) # in a normal distribution: (fisher) kurtosis == 0

# **Take home**: MSFT returns exhibit "fat tails" (extreme positive/negative outcomes)

x = np.linspace(msft.log_return.min(), msft.log_return.max(), 10000)
x

y = stats.norm.pdf(x, loc = mu, scale = sigma) # creating values for a distribution with mu,sigma
y

plt.figure(figsize = (20, 8))
plt.hist(msft.log_return, bins = 500, density = True, label = "Frequency Distribution of daily Returns (MSFT)")
plt.plot(x, y, linewidth = 3, color = "red", label = "Normal Distribution")
plt.title("Normal Distribution", fontsize = 20)
plt.xlabel("Daily Returns", fontsize = 15)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

# **Take home**: MSFT returns exhibit "fat tails" (extreme positive/negative outcomes).

# Testing the normality of MSFT Returns based on the sample (Oct 2014 to May 2021):
#     
# Hypothesis Test with H0 hypothesis: MSFT Returns (full population) follow a normal distribution.

z_stat, p_value = stats.normaltest(msft.log_return.dropna())

z_stat

p_value

round(p_value, 10)

# Assuming that MSFT Returns (generally) follow a normal distribution, there is 0% probability that we get those extreme outcomes in a sample.
#
# **Take home**: MSFT Returns don't follow a normal distribution as they exhibit "fat tails". Extreme events/outcomes are not reflected in the Mean-Variance analysis. The standard deviation of returns underestimates true risk.
