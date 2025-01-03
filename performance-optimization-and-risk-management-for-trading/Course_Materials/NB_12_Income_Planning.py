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

# # Sustainable Income Planning

# ## Simple Income Calculation (too simple)

# A Trader has created a levered trading strategy that (on average) generates an __annualized mean return of 105%__ (log) with a __standard deviation of 67%__. <br>
# The applicable (flat) __tax rate is 25%__ and the __inflation__ protection shall be __3% p.a.__<br>
# Calculate the Trader´s __sustainable income__ if the trader starts with __USD 10,000__ (available funds for trading).

import numpy as np

ann_mean = 1.05
ann_std = 0.67

cagr = np.exp(ann_mean) - 1
cagr

I0 = 10000
tax_rate = 0.25
infl = 0.03

ann_income = I0 * (cagr * (1-tax_rate) - infl)
ann_income

monthly_income = ann_income / 12
monthly_income

# Is that realistic? __No!__ It assumes __zero risk__ and __annual year-end distributions__.



# ## Introduction to Simulations (Part 1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

ann_mean = 1.05
ann_std = 0.67

td_per_year = 259
td_per_year

daily_mean = ann_mean / td_per_year
daily_mean

daily_std = ann_std / np.sqrt(td_per_year)
daily_std

# __Returns follow a random process__. It can get better/worse than the average/expected performance (random noise). 

# __One Simulation (over a 1-year period)__

days = 259

np.random.seed(123)
returns = np.random.normal(loc = daily_mean, scale = daily_std,
                           size = days) # normal distribution (simplified)

returns = np.insert(returns, 0, 0)

returns

len(returns)

creturns = np.exp(returns.cumsum())
creturns

plt.figure(figsize = (12, 8))
plt.plot(creturns)
plt.xlabel("Time (Days)", fontsize = 12)
plt.ylabel("Multiple",  fontsize = 12)
plt.show()

np.mean(returns) * td_per_year

np.std(returns) * np.sqrt(td_per_year)

# __How good/bad can it get? -> Many Simulations__



# ## Introduction to Simulations (Part 2)

# __Many Simulations (each over 1-year period)__

sims = 1000
sims

days

np.random.seed(123)
returns = np.random.normal(loc = daily_mean, scale = daily_std, size = days * sims).reshape(days, sims)
returns.shape

returns

returns = np.insert(returns, 0, 0, axis = 0)
returns

df = pd.DataFrame(data = returns)
df

df = df.cumsum().apply(np.exp)
df

plt.figure(figsize = (12 ,8))
plt.plot(df.values)
plt.xlabel("Days", fontsize = 12)
plt.ylabel("Normalized Price", fontsize = 12)
plt.show()

df.iloc[-1].plot(kind = "hist", bins = 100, figsize = (12, 8)) # final multiple after 1 year
plt.show()

df.iloc[-1].describe()

np.percentile(df.iloc[-1], [10, 90])

# Keep in mind: This assumes normally distributed returns -> In reality, there is (a bit) more weight in the tails.

# So far, we ignored annual __Taxes__ and monthly __Income Distributions__ -> Next Step: __path-dependent Simulation__



# ## A path-dependent Simulation with Taxes and Income - Introduction

# __Tax Payments__

# Assumptions: 
# - annual, year-end payments.
# - Full Trading Profit in a calender year is taxable @ x% flat tax rate (no loss carryforward) <br>
# __++++ This is not Tax Advice. Please consult your local tax advisor!!! ++++__

# __Income Distributions__

# Assumptions: 
# - monthly, in arrears
# - increase on a monthly basis @ inflation rate

# Tax Payments and Income Distributions are limited to Trading Profits and the Trading Capital (can´t get negative). <br> (no additional/subsequent funding)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
plt.style.use("seaborn-v0_8")


class Trade_Income():
    
    def __init__(self, start, end, I0, dist, infl, tax):
        
        self.start = start
        self.end = end
        self.I0 = I0
        self.dist = dist
        self.infl = infl
        self.tax_rate = tax
        self.prepare_data()
    
    def prepare_data(self):
        
        self.index = pd.date_range(start = self.start, end = self.end, freq= "D")
        self.td_index = pd.date_range(start = self.start, end = self.end, freq= "B")
        self.days = len(self.td_index) 
        self.td_per_year = (self.days / ((self.td_index[-1] - self.td_index[0]).days / 365.25))
        self.tax_index = pd.date_range(start = self.start, end = self.end, freq= "BA-DEC")
        self.dist_index = pd.date_range(start = self.start, end = self.end, freq= "BM")
        
        dist = [self.dist * (1 + self.infl)**(i/12) for i in range(len(self.dist_index))]
        dist = pd.Series(dist, index = self.dist_index)
        tax = pd.Series(True, index = self.tax_index)
        df = pd.DataFrame(data = {"Dist":dist, "Tax":tax}, index = self.index)
        df.Dist.fillna(0, inplace = True)
        df.Tax.fillna(False, inplace = True)
        self.raw_data = df
        
    def simulate_one(self, ann_mean, ann_std, seed = 123):
        
        daily_mean = ann_mean / self.td_per_year
        daily_std = ann_std / np.sqrt(self.td_per_year)
        
        np.random.seed(seed)
        returns = np.random.normal(loc = daily_mean, scale = daily_std, size = self.days)
        returns = pd.Series(np.exp(returns) - 1, index = self.td_index)
        data = self.raw_data.copy()
        data["returns"] = returns
        data.returns.fillna(0, inplace = True)
        self.data = data
        
        self.path_dependent()
        
    def simulate_many(self, ann_mean, ann_std, seed = 123, sims = 1000):
        
        daily_mean = ann_mean / self.td_per_year
        daily_std = ann_std / np.sqrt(self.td_per_year)
        
        np.random.seed(seed)
        matrix = np.random.normal(loc = daily_mean, scale = daily_std, size = sims * self.days).reshape(sims, self.days)
        
        results = []
        for sim in range(sims):
            returns = matrix[sim, :]
            returns = pd.Series(np.exp(returns) - 1, index = self.td_index)
            data = self.raw_data.copy()
            data["returns"] = returns
            data.returns.fillna(0, inplace = True)
            self.data = data
            self.path_dependent()
            
            results.append(round(self.data.Equity[-1], 0))
        return results
                                                                            
    def path_dependent(self):
        
        Equity = [I0]
        Year_Begin = I0
        Year_Distr = 0
        dist_list = []
        tax_list = []
        
        df = self.data.copy()

        for i in range(len(self.index)):
            equity_bd = Equity[i] * (1 + df.returns[i])
            distribution = min(df.Dist[i], equity_bd)
            dist_list.append(distribution)
    
            equity_bt = equity_bd - distribution
            Year_Distr += distribution
    
            if df.Tax[i]:
                taxable_income = max(0, Year_Distr + equity_bt - Year_Begin)
                tax_pay = self.tax_rate * taxable_income
                tax_list.append(tax_pay)
                equity_at = max(0, equity_bt - tax_pay)
                Year_Begin = equity_at
                Year_Distr = 0
    
            else:
                equity_at = equity_bt
    
            Equity.append(equity_at)
        
        df["Equity"] = Equity[1:]
        df["Dist"] = dist_list
        df["Tax"] = pd.Series(tax_list, index = self.tax_index)
        df.Tax.fillna(0, inplace = True)
        
        self.data = df   


start = "2020-01-01"
end = "2029-12-31" # 10 years
I0 = 10000 # initial trading capital
dist = 545.7 # (initial) monthly distribution
infl = 0.03 # inflation rate
tax = 0.25 # flat tax rate

TI = Trade_Income(start, end, I0, dist, infl, tax)
TI

# __One Simulation__

ann_mean = 1.05
ann_std = 0.67 

TI.simulate_one(ann_mean, ann_std, seed = 123)

TI.data

TI.data.Dist.plot(figsize = (12, 8)) # income distributions
plt.show()

TI.data.Tax.plot(figsize = (12, 8)) # tax payments
plt.show()

TI.data.Equity.plot(figsize = (12, 8)) # Equity (Trading Capital)
plt.show()

required_end_value = I0 * (1+infl)**10 # capital shall increase @ 3% p.a.
required_end_value

TI.data.Equity[-1] > required_end_value



# ## A path-dependent Simulation with Taxes and Income - many Simulations

start = "2020-01-01"
end = "2029-12-31"
I0 = 10000
distr = 545.7 # monthly distribution
infl = 0.03
tax = 0.25

ann_mean = 1.05
ann_std = 0.67 

TI = Trade_Income(start, end, I0, distr, infl, tax)
TI

results = TI.simulate_many(ann_mean, ann_std, seed = 123, sims = 1000)

results

required_end_value

(np.array(results) < required_end_value).mean() # shortfall probabilty over the next 10 years

# -> With an initial Income Distribution of USD 545.7, the shortfall probability over the next 10 years is 31.7% <br>
# -> Level of Income is not sustainable.



# ## Shortfall Risk and a Sustainable Income Level

# __Determine a sustainable income distribution level with a shortfall probability of approx. 5%.__

# (Acceptable Shortfall Probability depends on the Individual´s Risk Tolerance)

# A Trader has created a levered trading strategy that (on average) generates an __annualized mean return of 105%__ (log) with a __standard deviation of 67%__. <br>
# The applicable (flat) __tax rate is 25%__ and the __inflation__ protection shall be __3% p.a.__<br>
# Calculate the Trader´s __sustainable income__ if the trader starts with __USD 10,000__ (available funds for trading).

simple_sol = 1136
simple_sol

# __Rule of Thumb: Adjustment Factor between 20% and 50%__

adj_factor = 0.25 # 25%

distr = simple_sol * adj_factor
distr

TI = Trade_Income(start, end, I0, distr, infl, tax)
TI

results = TI.simulate_many(ann_mean, ann_std, seed = 123, sims = 1000)

results

required_end_value

(np.array(results) < required_end_value).mean() # shortfall probabilty after 10 years

# __-> 25% of simple Solution Income!__



# ## Final Remarks

# __Final Remarks__

# - (Trying to) simulate the Future is not an exact science
# - It is based on various assumptions and uncertainties
# - there are more complex/accurate models
# - very effective additional feature: making dynamic (path-dependent) adjustments to income distributions

# Key message: If Income Distributions are too high, Risk of running out of trading capital is high as well.
# Approx. __20% to 50%__ of simple solution Income!



# __Excursus: Advanced Performance Attribution__

# Simple Solution Income: __USD 1136__ <br>
# Actual (sustainable) Income: __USD 284__

# -> The Difference is attributable to __Timing__ and __Risk__. What´s the sustainable Income if we assume __zero Risk__?

start = "2020-01-01"
end = "2029-12-31"
I0 = 10000
distr = 739.08 # sustainable monthly income assuming zero risk
infl = 0.03
tax = 0.25

ann_mean = 1.05
ann_std = 0 # zero risk

TI = Trade_Income(start, end, I0, distr, infl, tax)
TI

TI.simulate_one(ann_mean, ann_std, seed = 123)

TI.data

TI.data.Dist.plot(figsize = (12, 8))
plt.show()

TI.data.Tax.plot(figsize = (12, 8))
plt.show()

TI.data.Equity.plot(figsize = (12, 8))
plt.show()

# - 284 -> 739 attributable to Risk
# - 739 -> 1136 attributable to Timing


