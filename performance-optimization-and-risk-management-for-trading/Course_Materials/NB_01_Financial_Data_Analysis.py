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

# + active=""
# # Financial Data Analysis with Python - a Deep Dive
# -

# ## Installing required Libraries/Packages

# Install yfinance with:
# - pip install yfinance 

# (first: conda update anaconda) 



# ## Loading Financial Data from the Web

import pandas as pd
import yfinance as yf

start = "2014-10-01"
end = "2021-05-31"

symbol = "BA"

df = yf.download(symbol, start, end)
df

df.info()

symbol = ["BA", "MSFT", "^DJI", "EURUSD=X", "GC=F", "BTC-USD"]

# Ticker Symbols: <br>
# - __BA__: Boeing (US Stock) 
# - __MSFT__: Microsoft Corp (US Stock)
# - __^DJI__: Dow Jones Industrial Average (US Stock Index)
# - __EURUSD=X__: Exchange Rate for Currency Pair EUR/USD (Forex)
# - __GC=F__: Gold Price (Precious Metal / Commodity)
# - __BTC-USD__: Bitcoin in USD (Cryptocurrency)

df = yf.download(symbol, start, end)
df

df.info()

df.to_csv("multi_assets.csv")



# ## Initial Inspection and Visualization

import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn-v0_8")

df = pd.read_csv("multi_assets.csv", header = [0, 1], index_col = 0, parse_dates = [0])
df

df.info()

df.Close # outer index level

df.Close.BA # one column (1)

df.loc[:, ("Close", "BA")] # one column (2)

df.loc["2015-01-07"] # one row

df.loc["2015"] # one year

df.loc["2020-06":, ("Close", "BA")] # one month for one column

df = df.swaplevel(axis = "columns").sort_index(axis = "columns") # swap outer and inner index
df

df["EURUSD=X"]

df["BTC-USD"]

df = df.swaplevel(axis = "columns").sort_index(axis = "columns") # swap outer and inner index
df

close = df.Close.copy() # select daily close prices only
close

close.describe()

close.BA.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

close.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

# __Take Home: Absolute Prices are absolutely meaningless/useless (in most cases)__ <br>
# - Prices that are on a different scale are hard to compare 
# - A higher Price does not imply a higher value or a better performance



# ## Normalizing Financial Time Series to a Base Value (100)

# __-> all instruments start at the very same Level (Base Value)__

close

close.iloc[0,0] # first price BA

close.BA.div(close.iloc[0,0]).mul(100)

close.iloc[0] # first Price all tickers

norm = close.div(close.iloc[0]).mul(100)
norm

norm.dropna().plot(figsize = (15, 8), fontsize = 13, logy = True)
plt.legend(fontsize = 13)
plt.show()

# __Take Home: Normalized Prices help to compare Financial Instruments...<br>
# ...but they are limited when it comes to measuring/comparing Performance in more detail.__

close.to_csv("close.csv")



# ---------------------------------------------

# __Coding Challenge #1__

# 1. Load Stock Price Data for General Electric (GE) and another ticker symbol of your choice from 2015-01-02 until 2020-12-31.<br>
# Go to https://finance.yahoo.com/ and get the right ticker symbol. For instruments traded outside the US, you have to add a country/exchange suffix. <br>
# Check the suffix list here https://help.yahoo.com/kb/exchanges-data-providers-yahoo-finance-sln2310.html As an example, the suffix for the National Indian Stock Exchange is .NS -> Ticker Symbol for Reliance is Reliance.NS

# 2. Select Close prices only and create a price chart for GE.

# 3. Normalize the stock prices for GE and the Ticker Symbol of your choice (Base Value: 1) and visualize! What´s the final normalized price for GE on 2020-12-30? 

# _You can find the solution for the Coding Challenges at the end of this notebook_.

# -----------------------------------------------------



# ## Price Changes and Financial Returns

# __More meaningful/useful than Prices: Price changes__

import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

msft = close.MSFT.dropna().to_frame().copy()

msft

msft.rename(columns = {"MSFT":"Price"}, inplace = True)

msft

msft.shift(periods = 1)

msft["P_lag1"] = msft.shift(periods = 1)
msft

# __Absolute Price Changes__ (Difference)

msft["P_diff"] = msft.Price.sub(msft.P_lag1) # Alternative 1

msft

msft["P_diff2"] = msft.Price.diff(periods = 1)  # Alternative 2

msft

msft.P_diff.equals(msft.P_diff2)

# __-> Absolute Price Changes are not meaningful__

# __Relative/Percentage Price Changes__ (Returns)

msft.Price.div(msft.P_lag1) - 1 # Alternative 1

msft["Returns"] = msft.Price.pct_change(periods = 1) # Alternative 2
msft

46.0900 / 45.7600 - 1

(46.0900 / 45.7600 - 1) * 100

# __Take Home: Relative Price Changes (Returns) are meaningful and comparable across instruments__

msft.drop(columns = ["P_lag1", "P_diff", "P_diff2"], inplace = True)

msft

msft.to_csv("msft.csv")



# ## Measuring Reward and Risk of an Investment

# __General Rule in Finance/Investing: Higher Risk must be rewarded with higher Returns__.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn-v0_8")

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft.Price.plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

# - Reward: Positive Returns
# - Risk: Volatility of Returns

msft.describe()

mu = msft.Returns.mean() # arithmetic mean return -> Reward
mu

sigma = msft.Returns.std() # standard deviation of returns -> Risk/Volatility
sigma

np.sqrt(msft.Returns.var())



# ----------------------------------------

# __Coding Challenge #2__

# 1. Calculate daily returns for Bitcoin.

# 2. Calculate the arithmetic mean return and the standard deviation of returns for Bitcoin. 

# 3. Compare Bitcoin with Microsoft (mu = 0.00116, sigma = 0.01726). Does the rule "Higher Risk -> Higher Reward" hold?

# -----------------------------------



# ## Investment Multiple and CAGR 

# __Two alternative reward metrics that are more intuitive and easier to interpret.__

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn-v0_8")

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

# __Investment Multiple__: Ending Value of 1 [Dollar] invested. <br>
# Multiple = Ending Value / Initial Investment

multiple = (msft.Price[-1] / msft.Price[0])
multiple

# __Price Increase (in %)__

(multiple - 1) * 100

msft.Price / msft.Price[0] # similar/identical concept: Normalized Price with Base Value 1

# __Drawback of Investment Multiple: Doesn´t take into account investment Period. Meaningful only in conjunction with Investment Period.__

# __Compound Annual Growth Rate (CAGR)__: The (constant annual) rate of return that would be required for an investment to grow from its beginning balance to its ending balance, assuming the profits were reinvested at the end of each year of the investment's lifespan. (Wikipedia)

start = msft.index[0]
start

end = msft.index[-1]
end

td = end - start
td

td_years = td.days / 365.25
td_years

cagr = multiple**(1 / td_years) - 1 # short version
cagr

cagr = (msft.Price[-1]/msft.Price[0])**(1/((msft.index[-1] - msft.index[0]).days / 365.25)) - 1 # long
cagr

(1 + cagr)**(td_years) # alternative #2 to calculate multiple (cagr)

# __-> CAGR can be used to compare Investments with different investment horizons.__



# ## Compound Returns & Geometric Mean Return

msft

multiple = (1 + msft.Returns).prod() # alternative #3 to calculate multiple (compounding daily returns)
multiple

n = msft.Returns.count()
n

geo_mean = multiple**(1/n) - 1 # Geometric mean return (daily)
geo_mean

(1 + geo_mean)**n # alternative #4 to calculate multiple (geometric mean)

# __-> Compound returns, CAGR & geometric mean return are closely related concepts__.

mu = msft.Returns.mean() # arithmetic mean return
mu

# __The arithmetic mean return is always greater than the geometric mean return... and less useful__. 

(1 + mu)**n # calculate multiple? not possible with arithmetic mean!



# ----------------------------

# __Coding Challenge #3__

# 1. Calculate Boeing´s Investment Multiple 

# 2. Calculate Boeing´s CAGR

# 3. Calculate Boeing´s Geometric Mean Return

# 4. Calculate Boeing´s Investment Multiple with compound daily returns

# ----------------------------------



# ## Preview: Simple Returns vs. Logarithmic Returns (log returns)

# Very often log returns are used instead of simple returns.<br>
# - favourable characteristics of log returns
# - drawbacks of simple returns

# Problem: Many Students / Practitioners feel uncomfortable with log returns. <br>
# -> more detailed background on log returns in the next two Lectures (Discrete vs. Continuous Compounding)



# ## Discrete Compounding

# __Annual Compounding__ -> Interests accrue once a year at the end of the year

# Your Savings Bank offers an interest rate of __8% p.a. (stated rate)__ with __annual compounding__ on your savings (__USD 100__).<br>
# Calculate the __value__ of your savings account __after one year__ and the corresponding __effective annual interest rate__. 

# __-> Interests are calculated and added to your savings account once at the end of each year.__

PV = 100
r = 0.08
n = 1

100 * 1.08

FV = PV * (1 + r)**n
FV

effective_annual_rate = (FV / PV)**(1/n) - 1 
effective_annual_rate



# __Quarterly Compounding__ -> Interests accrue once a quarter at the end of the quarter

# Your Savings Bank offers an interest rate of __8% p.a. (stated rate)__ with __quarterly compounding__ on your savings (__USD 100__).<br>
# Calculate the __value__ of your savings account __after one year__ and the corresponding __effective annual interest rate__. 

# __-> Interests are calculated and added to your savings account at the end of each quarter.__

PV = 100
r = 0.08
n = 1
m = 4

100 * 1.02 * 1.02 * 1.02 * 1.02

FV = PV * (1 + r/m)**(n*m)
FV

effective_annual_rate = (FV / PV)**(1/n) - 1 
effective_annual_rate

# __-> Quarterly compounding is favourable (everything else equal) as we earn compound interest (interest on interest).__



# __Monthly Compounding__ -> Interests accrue once a month at the end of the month

# Your Savings Bank offers an interest rate of __8% p.a. (stated rate)__ with __monthly compounding__ on your savings (__USD 100__).<br>
# Calculate the __value__ of your savings account __after one year__ and the corresponding __effective annual interest rate__. 

# __-> Interests are calculated and added to your savings account at the end of each month.__

PV = 100
r = 0.08
n = 1
m = 12

FV = PV * (1 + r/m)**(n*m)
FV

effective_annual_rate = ((FV / PV)**(1/n) - 1) 
effective_annual_rate



# ## Continuous Compounding 

import numpy as np

# Your Savings Bank offers an interest rate of __8% p.a. (stated rate)__ with __continuous compounding__ on your savings (__USD 100__).<br>
# Calculate the __value__ of your savings account __after one year__ and the corresponding __effective annual interest rate__. 

# __-> Interests are calculated and added to your savings account continuously (infinitely large number of compounding events).__ -> continuous exponential growth that can be observed in nature

PV = 100
r = 0.08
n = 1
m = 100000 # approx.infinity

FV = PV * (1 + r/m)**(n*m) # approx. with large m
FV

FV = PV * np.exp(n * r) # exact math with e (euler number)
FV

euler = np.exp(1)
euler

PV * euler**(n * r)

effective_annual_rate = ((FV / PV)**(1/n) - 1) # Alt 1
effective_annual_rate

effective_annual_rate = np.exp(r) - 1 # Alt 2
effective_annual_rate

# Let´s assume we only observe PV and FV, how to calculate the stated rate/continuously compounded rate of 8%?

r = np.log(FV / PV) # inverse calculation -> use log (Alt 1)
r

r = np.log(effective_annual_rate + 1) # inverse calculation -> use log (Alt 2)
r

# __Take Home: Prices of traded Financial Instruments change (approx.) continuously. <br>
# -> Intuitively, it makes a lot of sense to work with log returns.__ 



# ## Log Returns

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft["log_ret"] = np.log(msft.Price / msft.Price.shift()) # daily log returns

msft

msft.describe()

mu = msft.log_ret.mean() # mean log return -> Reward
mu

sigma = msft.log_ret.std() # standard deviation of log returns -> Risk/Volatility
sigma



# ## Simple Returns vs Log Returns ( Part 1)

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

df = pd.DataFrame(data = [100, 50, 90], columns = ["Price"])
df

df["SR"] = df.Price.pct_change() # simple returns

df["LR"] = np.log(df.Price / df.Price.shift()) # log returns

df

periods = df.SR.count()
periods

# __The arithmetic mean of simple returns can be misleading!__

mean_sr = df.SR.mean()
mean_sr

100 * (1 + mean_sr)**periods # wrong!!!

# __We should use Compound Simple Returns / Geometric Mean, or even better...__

geo_mean = (1 + df.SR).prod()**(1 / periods) - 1
geo_mean

100 * (1 + geo_mean)**periods # correct!!!

# __...Log Returns which are additive over time!__

sum_lr = df.LR.sum()
sum_lr

100 * np.exp(sum_lr) # correct!!!

mean_lr = df.LR.mean()
mean_lr

100 * np.exp(mean_lr * periods) # correct!!!

# __Take Home: Log Returns are additive over time. Simple Returns are not additive over time (but they can be multiplied/compounded)__



# ## Simple Returns vs. Log Returns (Part 2)

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.6f}'.format

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])

msft["log_ret"] = np.log(msft.Price / msft.Price.shift())

msft

# __Investment Multiple__

msft.Returns.add(1).prod() # compounding simple returns ("compound returns")

np.exp(msft.log_ret.sum())  # adding log returns ("cumulative returns")

# __Normalized Prices with Base 1__

msft.Returns.add(1).cumprod() # compounding simple returns ("compound returns")

np.exp(msft.log_ret.cumsum()) # adding log returns ("cumulative returns")

msft.log_ret.cumsum().apply(np.exp) # adding log returns ("cumulative returns")

# __CAGR__

(msft.Price[-1]/msft.Price[0])**(1/((msft.index[-1] - msft.index[0]).days / 365.25)) - 1

trading_days_year = msft.Returns.count() / ((msft.index[-1] - msft.index[0]).days / 365.25)
trading_days_year

np.exp(msft.log_ret.mean() * trading_days_year) - 1 # correct with mean of daily log returns!

msft.Returns.mean() * trading_days_year # incorrect with mean of daily simple returns!

np.exp(msft.log_ret.mean() * 252) - 1 # good approximation (for us stocks)



# --------------------------------------------

# __Coding Challenge #4__

# 1. Calculate daily log returns for Boeing.

# 2. Use Boeing´s log returns to calculate 
# - Investment Multiple
# - CAGR (assuming 252 trading days)
# - Normalized Prices (Base = 1)

# ---------------------------------------------



# ## Performance Comparison

# __General Rule in Finance/Investing: Higher Risk must be rewarded with higher Returns__.

# __Which instrument(s) performed best/worst in the past in terms of risk & return?__

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

close.info()

close.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

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
    plt.annotate(i, xy=(summary.loc[i, "Std"]+0.00005, summary.loc[i, "Mean"]+0.00005), size = 15)
plt.xlabel("Risk (std)", fontsize = 15)
plt.ylabel("Mean Return", fontsize = 15)
plt.title("Mean-Variance Analysis", fontsize = 20)
plt.show()

# -> There is __no clear "best-performer"__ among ["EURUSD=X", "GC=F", "^DJI", "MSFT", "BTC-USD"] (without further analysis). __Higher risk__ is getting rewarded with __higher returns__. __BA underperformed__.



# __Take Home: Mean-Variance Analysis has one major shortcoming: It assumes that financial returns follow a Normal Distribution. That´s (typically) not True.<br> -> Standard Deviation of Returns underestimates the true/full risk of an Investment as it fails to measure "Tail Risks".__ 



# ## Normality of Financial Returns 

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft["log_ret"] = np.log(msft.Price / msft.Price.shift()) 
msft

msft.describe()

msft.log_ret.plot(kind = "hist", figsize = (15 ,8), bins = 100, fontsize = 15, density = False) # Frequency Distribution of log returns
plt.xlabel("Daily Returns", fontsize = 15)
plt.ylabel("Frequency", fontsize = 15)
plt.title("Frequency Distribution of Returns", fontsize = 20)
plt.show()

# __Do MSFT Returns follow a Normal Distribution?__ <br><br>
# A normally distributed random variable can be fully described by its 
# - mean
# - standard deviation

# Higher Central Moments are zero:
# - Skew = 0 (measures symmetrie around the mean)
# - (Excess) Kurtosis = 0 (positve excess Kurtosis -> more observations in the "tails")

mu = msft.log_ret.mean()
mu

sigma = msft.log_ret.std()
sigma

import scipy.stats as stats

stats.skew(msft.log_ret.dropna()) # in a Normal Distribution: skew == 0

stats.kurtosis(msft.log_ret.dropna(), fisher = True) # in a Normal Distribution: (fisher) kurtosis == 0

# __-> MSFT Returns exhibit "Fat Tails" (extreme positive/negative outcomes).__

x = np.linspace(msft.log_ret.min(), msft.log_ret.max(), 10000)
x

y = stats.norm.pdf(x, loc = mu, scale = sigma) # creating y values a for normal distribution with mu, sigma
y

plt.figure(figsize = (20, 8))
plt.hist(msft.log_ret, bins = 500, density = True, label = "Frequency Distribution of daily Returns (MSFT)")
plt.plot(x, y, linewidth = 3, color = "red", label = "Normal Distribution")
plt.title("Normal Distribution", fontsize = 20)
plt.xlabel("Daily Returns", fontsize = 15)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

# __-> MSFT Returns exhibit "Fat Tails" (extreme positive/negative outcomes).__ 



# __Testing the normality of MSFT Returns based on the sample (Oct 2014 to May 2021):__ <br>
# __-> Hypothesis Test with H0 Hypothesis: MSFT Returns (full population) follow a normal Distribution.__ 

z_stat, p_value = stats.normaltest(msft.log_ret.dropna())

z_stat # high values -> reject H0

p_value # low values (close to zero) -> reject H0

round(p_value, 10)

# __-> Assuming that MSFT Returns (generally) follow a Normal Distribution, there is 0% probability that we get that extreme outcomes in a sample.__ 

# __Take Home: MSFT Returns don´t follow a Normal Distribution as they exhibit "Fat Tails". Extreme Events/Outcomes are not reflected in the Mean-Variance Analysis. The Standard Deviation of Returns underestimates true Risk.__



# ## Annualizing Mean Return and Std of Returns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.6f}'.format
plt.style.use("seaborn-v0_8")

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"], usecols = ["Date", "Price"])

msft

msft["log_ret"] = np.log(msft.Price / msft.Price.shift())

msft.log_ret.agg(["mean", "std"]) # mean and std based on daily returns

ann_mu = msft.log_ret.mean() * 252 
ann_mu

cagr = np.exp(ann_mu) - 1 # don´t mix up with cagr
cagr

ann_std = msft.log_ret.std() * np.sqrt(252) 
ann_std

ann_std = np.sqrt(msft.log_ret.var() * 252) # annualized std of returns (Alt 2)
ann_std



# ## Resampling / Smoothing

msft.Price.plot(figsize = (12, 8))
plt.legend()
plt.show()

monthly = msft.Price.resample("M").last() ## resample to monthly (month end)
monthly

monthly.plot(figsize = (12, 8))
plt.legend()
plt.show()

# __How will the Mean-Variance Analysis change with smoothed data?__

freqs = ["A", "Q", "M", "W-Fri", "D"]
periods = [1, 4, 12, 52, 252]
ann_mean = []
ann_std = []

for i in range(5):
    resamp = msft.Price.resample(freqs[i]).last() # resample
    ann_mean.append(np.log(resamp / resamp.shift()).mean() * periods[i]) # calc. annualized mean
    ann_std.append(np.log(resamp / resamp.shift()).std() * np.sqrt(periods[i])) # calc. annualized std

ann_mean

summary = pd.DataFrame(data = {"ann_std":ann_std, "ann_mean":ann_mean}, index = freqs)
summary

summary.plot(kind = "scatter", x = "ann_std", y = "ann_mean", figsize = (15,12), s = 50, fontsize = 15)
for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i, "ann_std"]+0.001, summary.loc[i, "ann_mean"]+0.001), size = 15)
plt.ylim(0, 0.3)
plt.xlabel("ann. Risk(std)", fontsize = 15)
plt.ylabel("ann. Return", fontsize = 15)
plt.title("Risk/Return", fontsize = 20)
plt.show()

# __-> Smoothing reduces (observed) Risk__. 

# Dubious practices:
# - Managing (Manipulating) Performance in Performance Reportings.
# - Comparing assets with different granularity and pricing mechanisms -> e.g. non-listed (alternative assets) vs. listed assets 
# - Adjusting granularity to investor´s (average) holding period -> Volatility is still there.



# ## Rolling Statistics

# __(Another) general Rule in Finance/Investing: Past performance is not an indicator of future performance__.

msft

ann_mu = msft.log_ret.mean() * 252 # annualized mean return
ann_mu

ann_std = msft.log_ret.std() * np.sqrt(252) # annualized std of returns (Alt 1)
ann_std

# __Are Return and Risk constant over time? No, of course not! They change over time.__

# __Let´s measure/quantify this with rolling statistics!__

window = 252 # rolling window 252 trading days (~ 1 Year)

msft.log_ret.rolling(window = 252)

msft.log_ret.rolling(window = 252).sum() # Alt 1

roll_mean = msft.log_ret.rolling(window = 252).mean() * 252 # Alt 2
roll_mean

roll_mean.iloc[250:]

roll_mean.plot(figsize = (12, 8))
plt.show()

roll_std = msft.log_ret.rolling(window = 252).std() * np.sqrt(252)
roll_std

roll_std.plot(figsize = (12, 8))
plt.show()

roll_mean.plot(figsize = (12, 8))
roll_std.plot()
plt.show()

# __Take Home__: Be careful, you´ll always find (sub-)periods with __low returns & high risk__ and __high returns & low risk__. 

# - Analysis Period must be __sufficiently long__ to reduce impact of random noise. <br>
# - Analysis Period should be __as short as possible__ and should only include the __latest trends / regimes__.
# - Commonly used reporting period: __3 Years / 36 Months__

# __Another Example: Simple Moving Average (Prices) - SMA__

sma_window = 50

msft.Price.plot(figsize = (12, 8))
msft.Price.rolling(sma_window).mean().plot()
plt.show()



# --------------------------------------------

# __Coding Challenge #5__

# 1. Calculate daily log returns for Boeing.
#
# 2. Use Boeing´s daily log returns to calculate the annualized mean and annualized std (assume 252 trading days per year).
#
# 3. Resample to monthly prices and compare the annualized std (monthly) with the annualized std (daily). Any differences?
#
# 4. Keep working with monthly data and calculate/visualize the rolling 36 months mean return (annualized).

# ---------------------------------------------



# ## Short Selling / Short Positions (Part 1)

# What´s the rational behind short selling an instrument? <br>
# __-> making profits/positive returns when prices fall.__

# __Stocks Example:__

# Today an Investor __buys__ the ABC Stock for USD 100. One day later he __sells__ the stock for USD 110. <br> 
# __-> Profit: USD 10__ <br>
# ->__Long Position__ (benefit from rising prices):

# Today an Investor __borrows__ the ABC Stock from another Investor and __sells__ it for USD 100. One day later he __buys__ the stock for USD 90 and __returns__ it to the lender.<br>
# __-> Profit: USD 10__  <br>
# ->__Short Position__ (benefit from falling prices):

# In some countries (and for some instruments like stocks) short selling is prohibited. <br>
# Most intuitive/popular use case for short selling: __Currencies (Forex)__



# ## Short Selling / Short Positions (Part 2)

# __EUR/USD__ ("Long Euro" == "Short USD")

t0 = 1.10
t1 = 1.25

# Today an Investor __buys__ EUR 1 and pays USD 1.10. One day later he __sells__ EUR 1 for USD 1.25 <br>
# __-> Profit: USD 0.15__  <br>
# ->__Long Position Euro__ (benefit from rising EUR prices):

t1 / t0 - 1 # The EUR appreciates by 13.64% relative to USD (simple return)

# -> EUR __Long__ Position returns __+13.64%__ (simple return) 

# What return would you expect for the corresponding EUR __Short__ position? That´s a "no brainer": __-13.64%__, right? 

# __Surprisingly, that´s incorrect!!!__



# Inverse Rate: __USD/EUR__ ("Short Euro" == "Long USD")

t0 = 1 / 1.10
t1 = 1 / 1.25

print(t0, t1)

# Today an Investor __buys__ USD 1 and pays 0.9091 Euro. One day later he __sells__ USD 1 for EUR 0.8 __<br>
# -> Loss: EUR 0.1091__  <br>

t1 / t0 - 1 # The USD depreciates by 12.0% relative to EUR

# -> EUR __Short__ Position returns __-12.0%__ (simple return)

# __Take Home: When using simple returns, long position return != short position return * (-1)__ <br>
# __-> Use log returns!__



# ## Short Selling / Short Positions (Part 3)

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

fx.plot(figsize = (12,8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

simple_ret = fx.pct_change() # simple returns
simple_ret

simple_ret.add(1).prod() - 1 # compound simple returns

# __-> For simple returns: long position returns != short position returns * (-1)__

log_ret = np.log(fx / fx.shift()) # log returns
log_ret

log_ret.sum() # cumulative log returns

# __-> For log returns: long position returns == short position returns * (-1)__

norm_fx = log_ret.cumsum().apply(np.exp) # normalized prices (Base 1)
norm_fx

norm_fx.iloc[0] = [1, 1]

norm_fx

norm_fx.plot(figsize = (12,8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()



# --------------------------------------------

# __Coding Challenge #6__

# 1. Calculate daily log returns for Boeing.
#
# 2. Calculate the annualized mean and annualized std (assume 252 trading days per year) for a short position in Boeing (ignore Trading and Borrowing Costs).

# ---------------------------------------------



# ## Covariance and Correlation

# Do instruments/assets __move together__ (and to what extent)? <br>
#
# Three cases:
# - unrelated (__no__ relationship/correlation)
# - moving together (__positive__ relationship/correlation)
# - moving in opposite directions (__negative__ relationship/correlation) 

# __-> Correlation between instruments/assets play an important role in portfolio management.__

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
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
# - no correlation: __correlation coefficient == 0__
# - moving together: __0 < correlation coefficient <= 1__ (positive)
# - moving in opposite directions: __-1 <= correlation coefficient < 0__ (negative)

import seaborn as sns

plt.figure(figsize=(12,8))
sns.set(font_scale=1.4)
sns.heatmap(returns.corr(), cmap = "RdYlBu_r", annot = True, annot_kws={"size":15}, vmin = -1, vmax = 1)
plt.show()

# __Take Home: Similar assets are (highly) positive correlated. Different assets exhibit low/no/negative correlation.__ <br>
# -> In portfolio management it´s beneficial to have assets with low/no/negative correlation (portfolio diversification effect).



# ## Portfolio of Assets and Portfolio Returns

import pandas as pd
import numpy as np

prices = pd.DataFrame(data = {"Asset_A": [100, 112], "Asset_B":[100, 104]}, index = [0, 1])
prices

prices["Total"] = prices.Asset_A + prices.Asset_B

prices

returns = prices.pct_change() # simple returns
returns

0.5 * 0.12 + 0.5 * 0.04 # correct (portfolio return == weighted average of simple returns)

log_returns = np.log(prices / prices.shift()) # log returns
log_returns

0.5 * log_returns.iloc[1,0] + 0.5 * log_returns.iloc[1,1] # incorrect (portfolio return != weighted average of log returns)

# __Take Home: While log returns are time-additive, they are not asset-additive.__ <br>
# (While simple returns are not time-additive, they are asset-additive.)



# ## Margin Trading & Levered Returns (Part 1)

# __Definition__: "Margin trading refers to the practice of using __borrowed funds__ from a broker to trade a financial asset, which forms the collateral for the loan from the broker." (Investopedia.com) 

# In Simple Words: Investors __don´t pay the full price__ but they get the full benefit (less borrowing costs).

# It´s a two edged sword: Leverage __amplifies both gains and losses__. <br> In the event of a loss, the collateral gets reduced and the Investor either posts additional margin or the brokers closes the position.

# __Example__

# A Trader buys a stock (stock price: 100) __on margin (50%)__. After one day the price increases to 110.<br>
# Calculate __unlevered return__ and __levered return__.

import numpy as np

P0 = 100
P1 = 110
leverage = 2
margin = P0/2

margin

unlev_return = (P1 - P0) / P0 # simple return
unlev_return

lev_return = (P1 - P0) / margin # simple return 
lev_return

lev_return == unlev_return * leverage # this relationship is true for simple returns...

unlev_return = np.log((P1 - P0) / P0 + 1) # log return
unlev_return

lev_return = np.log((P1 - P0) / margin + 1) # log return
lev_return

lev_return == unlev_return * leverage # this relationship does not hold for log returns...

# __Take Home: To calculate levered returns, don´t multiply leverage with log returns!__



# ## Margin Trading & Levered Returns (Part 2)

# __Hypothesis: For (highly) profitable Investment: The more leverage, the better?__

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
msft

msft["Simple_Ret"] = msft.Price.pct_change() # simple returns

leverage = 2

# (Simplified) __Assumptions__:
# - Restore leverage on a daily basis (by buying/selling shares)
# - no trading costs
# - no borrowing costs

msft["Lev_Returns"] = msft.Returns.mul(leverage) # levered simple returns
msft

msft["Lev_Returns"] = np.where(msft["Lev_Returns"] < -1, -1, msft["Lev_Returns"])

msft

msft[["Returns", "Lev_Returns"]].add(1).cumprod().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

msft.Simple_Ret.max()

msft.Lev_Returns.max()

msft.Simple_Ret.min()

msft.Lev_Returns.min()

# __What happens when leverage greater than...?__

-1 / msft.Simple_Ret.min()



# __Take Home:__
# 1. With Leverage you can (theoretically) __lose more than the initial Margin__ (in practice: margin call / margin closeout before)
# 2. Even for (highly) profitable instruments: __"The more leverage the better" does not hold__.
# 3. It´s a two edged (__non-symmetrical__) sword: __Leverage amplifies losses more than it amplifies gains__.



# --------------------------------

# __Coding Challenge #7__

# 1. Calculate levered returns for Bitcoin (leverage = 4). 
#
# 2. Visualize and compare with unlevered Investment.
#
# 3. Some Traders trade Bitcoin with extremely high leverage (> 100). Do you think this is a good idea (assuming no additional/advanced Risk Management Tools)?

# ---------------------------------------------



# --------------------------------------

# ## Coding Challenge Solutions

# __Coding Challenge #1__

# 1. Load Stock Price Data for General Electric (GE) and another ticker symbol of your choice from 2015-01-02 until 2020-12-31.<br>
# Go to https://finance.yahoo.com/ and get the right ticker symbol. For instruments traded outside the US, you have to add a country/exchange suffix. <br>
# Check the suffix list here https://help.yahoo.com/kb/exchanges-data-providers-yahoo-finance-sln2310.html As an example, the suffix for the National Indian Stock Exchange is .NS -> Ticker Symbol for Reliance is Reliance.NS

# 2. Select Close prices only and create a price chart for GE.

# 3. Normalize the stock prices for GE and the Ticker Symbol of your choice (Base Value: 1) and visualize! What´s the final normalized price for GE on 2020-12-30? 

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn-v0_8")

start = "2015-01-02"
end = "2020-12-31"
symbol = ["GE", "Reliance.NS"]

df = yf.download(symbol, start, end)
df

close = df.Close.copy()
close

close.GE.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

norm = close.div(close.iloc[0]).mul(1)
norm

# -> The final normalized Price is 0.4445.

norm.dropna().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()



# __Coding Challenge #2__

# 1. Calculate daily returns for Bitcoin.

# 2. Calculate the arithmetic mean return and the standard deviation of returns for Bitcoin. 

# 3. Compare Bitcoin with Microsoft (mu = 0.00116, sigma = 0.01726). Does the rule "Higher Risk -> Higher Reward" hold?

import pandas as pd
pd.options.display.float_format = '{:.4f}'.format

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

btc = close["BTC-USD"].dropna().to_frame().copy()
btc

btc["Returns"] = btc.pct_change(periods = 1)
btc

btc

mu = btc.Returns.mean() 
mu

sigma = btc.Returns.std()
sigma

mu > 0.00116

sigma > 0.01726

# Does the rule "Higher Risk -> Higher Reward" hold? -> Yes



# __Coding Challenge #3__

# 1. Calculate Boeing´s Investment Multiple 

# 2. Calculate Boeing´s CAGR

# 3. Calculate Boeing´s Geometric Mean Return

# 4. Calculate Boeing´s Investment Multiple with compound daily returns

import pandas as pd
pd.options.display.float_format = '{:.4f}'.format

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

ba = close["BA"].dropna().to_frame().copy()
ba

ba["Returns"] = ba.pct_change(periods = 1)
ba

multiple = ba.BA[-1] / ba.BA[0]
multiple

cagr = (ba.BA[-1]/ba.BA[0])**(1/((ba.index[-1] - ba.index[0]).days / 365.25)) - 1 
cagr

n = ba.Returns.count()
n

geo_mean = (1 + ba.Returns).prod()**(1/n) - 1 
geo_mean

multiple = ba.Returns.add(1).prod()
multiple

# __Coding Challenge #4__

# 1. Calculate daily log returns for Boeing.

# 2. Use Boeing´s log returns to calculate 
# - Investment Multiple
# - CAGR (assuming 252 trading days)
# - Normalized Prices (Base = 1)

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

ba = close["BA"].dropna().to_frame().copy()
ba

ba["log_ret"] = np.log(ba / ba.shift())
ba

multiple = np.exp(ba.log_ret.sum())
multiple

cagr = np.exp(ba.log_ret.mean() * 252) - 1
cagr

norm = ba.log_ret.cumsum().apply(np.exp)
norm



# __Coding Challenge #5__

# 1. Calculate daily log returns for Boeing.
#
# 2. Use Boeing´s daily log returns to calculate the annualized mean and annualized std (assume 252 trading days per year).
#
# 3. Resample to monthly prices and compare the annualized std (monthly) with the annualized std (daily). Any differences?
#
# 4. Keep working with monthly data and calculate/visualize the rolling 36 months mean return (annualized).

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

ba = close["BA"].dropna().to_frame().copy()
ba

ba["log_ret"] = np.log(ba / ba.shift())
ba

ann_mu = ba.log_ret.mean() * 252 
ann_mu

ann_std = ba.log_ret.std() * np.sqrt(252) 
ann_std

monthly = ba.BA.resample("M").last().to_frame()
monthly

monthly["Returns"] = np.log(monthly / monthly.shift())
monthly

ann_std = monthly.Returns.std() * np.sqrt(12) 
ann_std

# -> Risk (monthly) slighly lower than Risk (daily) 

window = 36

roll_mean = monthly.Returns.rolling(window = window).mean() * 12
roll_mean

roll_mean.plot(figsize = (12, 8))
plt.show()



# __Coding Challenge #6__

# 1. Calculate daily log returns for Boeing.
#
# 2. Calculate the annualized mean and annualized std (assume 252 trading days per year) for a short position in Boeing (ignore Trading and Borrowing Costs).

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

ba = close["BA"].dropna().to_frame().copy()
ba

ba["log_ret"] = np.log(ba / ba.shift())
ba

ba["short"] = ba.log_ret * (-1)
ba

ann_mean = ba.short.mean() * 252 # equal to ann_mean of long position * (-1)
ann_mean

ann_std = ba.short.std() * np.sqrt(252) # same as ann_std of long position
ann_std



# __Coding Challenge #7__

# 1. Calculate levered returns for Bitcoin (leverage = 4). 
#
# 2. Visualize and compare with unlevered Investment.
#
# 3. Some Traders trade Bitcoin with extremely high leverage (> 100). Do you think this is a good idea (assuming no additional/advanced Risk Management Tools)?

import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.4f}'.format
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
close

btc = close["BTC-USD"].dropna().to_frame().copy()
btc

btc["Returns"] = btc.pct_change(periods = 1)
btc

leverage = 4

btc["Lev_Returns"] = btc.Returns.mul(leverage) # levered simple returns
btc

btc["Lev_Returns"] = np.where(btc["Lev_Returns"] < -1, -1, btc["Lev_Returns"])

btc[["Returns", "Lev_Returns"]].add(1).cumprod().plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

# -> Trading Bitcoin with (high) leverage requires advanced risk monitoring/management. Otherwise, a complete loss is very likely (sooner or later...).


