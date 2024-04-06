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

# # Statistics for Finance and Business with Python

# ## Descriptive Statistics

# ### Population vs. Sample (Price Returns for S&P 500 Companies in 2017)

# The __S&P 500__, or just the S&P, is a __stock market index__ that measures the stock performance of __500 large companies__ listed on stock exchanges in the United States. It is one of the most commonly followed equity indices, and many consider it to be one of the best representations of the U.S. stock market. <br>
# The S&P 500 is a __capitalization-weighted index__ and the performance of the __10 largest companies__ in the index account for __21.8% of the performance__ of the index. 

# __What is the equally weighted return in 2017?__

import numpy as np
np.set_printoptions(precision=2, suppress= True)

# __Population: 2017 Price Return for all 500 Companies__ 

pop = np.loadtxt("SP500_pop.csv", delimiter = ",", usecols = 1)

pop

pop = pop * 100

pop.size

# __Sample: 2017 Price Return for 50 Companies (randomly selected)__ 

sample = np.loadtxt("sample.csv", delimiter = ",", usecols = 1)

sample = sample * 100

sample

sample.size

for i in sample:
    print(i in pop)

np.isin(sample, pop)



# ### Visualizing Frequency Distributions with plt.hist()

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress= True)

pop

sample

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75)
plt.title("Absolute Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.xticks(np.arange(-100, 401, 25))
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(sample, bins = 15)
plt.title("Absolute Frequencies - Sample", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.show()



# ### Relative and Cumulative Frequencies with plt.hist()

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress= True)

pop.size

sample.size

(np.ones(len(pop)) / len(pop))

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75, weights = np.ones(len(pop)) / len(pop))
plt.title("Relative Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Relative Frequency", fontsize = 15)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75, density = True)
plt.title("Relative Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Relative Frequency", fontsize = 15)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75, density = False, cumulative = True)
plt.title("Cumulative Absolute Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Cumulative Absolute Frequency", fontsize = 15)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75, density = True, cumulative = True)
plt.title("Cumulative Relative Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Cumulative Relative Frequency", fontsize = 15)
plt.show()



# ### Measures of Central Tendency - Mean and Median

import numpy as np
np.set_printoptions(precision=4, suppress= True)

# __Population Mean__

pop

pop.mean()

np.mean(pop)

# __Sample Mean__

sample

sample.mean()

# __Median__

np.median(pop)

np.median(sample)

sample.sort()

(sample[24] + sample[25]) / 2



# ### Measures of Central Tendency - Geometric Mean

import numpy as np
np.set_printoptions(precision=4, suppress= True)

Price_2015_2018 = np.array([100, 107, 102, 110])

ret = Price_2015_2018[1:] / Price_2015_2018[:-1] - 1
ret

mean = ret.mean()
mean

100 * (1 + mean)**3

geo_mean = (1 + ret).prod()**(1/ret.size) - 1
geo_mean

100 * (1 + geo_mean)**3

(110 / 100)**(1/ret.size) - 1 



# ### Excursus: Log Returns

import numpy as np
np.set_printoptions(precision=4, suppress= True)

Price_2015_2018 = np.array([100, 107, 102, 110])

ret = Price_2015_2018[1:] / Price_2015_2018[:-1] - 1
ret

100 * (ret + 1).prod()

log_ret = np.log(Price_2015_2018[1:] / Price_2015_2018[:-1])
log_ret

mean_log = log_ret.mean()
mean_log

100 * np.exp(mean_log * 3)

add = log_ret.sum()
add

100 * np.exp(add)



# ### Range, Minimum and Maximum

import numpy as np
np.set_printoptions(precision=4, suppress= True)

pop = np.loadtxt("SP500_pop.csv", delimiter = ",", usecols = 1)

pop = pop * 100

pop

pop.size

pop.max()

pop.min()

range = pop.ptp()
range

pop.max() - pop.min()

sample = np.loadtxt("sample.csv", delimiter = ",", usecols = 1)

sample = sample * 100

sample

sample.size

np.max(sample)

np.min(sample)

sample.ptp()




# ### Variance and Standard Deviation

import numpy as np
np.set_printoptions(precision=4, suppress= True)

pop

sample

# __Population Variance__

pop.var()

np.var(pop)

# __Sample Variance__

np.var(sample)

np.var(sample, ddof = 1)

# __Standard Deviation__

np.sqrt(pop.var())

pop.std()

np.sqrt(np.var(sample, ddof = 1))

sample.std(ddof = 1)



# ### Percentiles

import numpy as np
np.set_printoptions(precision=4, suppress= True)

pop

np.percentile(pop, 50)

np.median(pop)

np.percentile(pop, 5)

np.percentile(pop, 95)

np.percentile(pop, [25, 75])

np.percentile(pop, [5, 95])

np.percentile(pop, [2.5, 97.5])



# ### How to calculate Skew & Kurtosis with scipy

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress= True)

pop = np.loadtxt("SP500_pop.csv", delimiter = ",", usecols = 1)

pop = pop * 100

pop

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75)
plt.title("Absolute Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.xticks(np.arange(-100, 401, 25))
plt.show()

import scipy.stats as stats

# __Skew__

stats.skew(pop)

# __Kurtosis__

stats.kurtosis(pop, fisher = True)

stats.kurtosis(pop, fisher = False)


