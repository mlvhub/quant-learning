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

# ## Common Probability Distributions and Confidence Intervals

# ### Generating Random Numbers with Numpy

import numpy as np

np.random.randint(low = 1, high = 11, size = 10)

np.random.random(size = 10)

np.random.uniform(low = 1, high = 10, size = 10)

np.random.normal(size = 10)

np.random.normal(loc = 100, scale = 10, size = 10)



# ### Reproducibility with np.random.seed()

import numpy as np

np.random.randint(low = 1, high = 11, size = 10)

np.random.randint(low = 1, high = 11, size = 10)

np.random.randint(low = 1, high = 11, size = 10)

np.random.seed(123)
np.random.randint(low = 1, high = 11, size = 10)

np.random.seed(123)
np.random.randint(low = 1, high = 11, size = 10)

np.random.seed(5)
np.random.randint(low = 1, high = 11, size = 10)

np.random.seed(5)
np.random.randint(low = 1, high = 11, size = 10)



# ### Discrete Uniform Distributions

import numpy as np
import matplotlib.pyplot as plt

np.random.randint(low = 1, high = 7, size = 10)

np.random.seed(123)
a = np.random.randint(low = 1, high = 7, size = 100000)
a

a.mean()

a.std()

100000/6

plt.figure(figsize = (12, 8))
plt.hist(a, bins = 6, ec = "black")
plt.title("Discrete Uniform Distribution", fontsize = 20)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(a, bins = 6, weights = np.ones(len(a)) / len(a), ec = "black")
plt.title("Discrete Uniform Distribution", fontsize = 20)
plt.ylabel("Relative Frequency", fontsize = 15)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(a, bins = 6, density = True, ec = "black")
plt.title("Discrete Uniform Distribution", fontsize = 20)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(a, bins = 6, density = True, cumulative= True, ec = "black")
plt.title("Discrete Uniform Distribution", fontsize = 20)
plt.ylabel("Cumulative Relative Frequency", fontsize = 15)
plt.show()



# ## Continuous Uniform Distributions

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
b = np.random.uniform(low = 0, high = 10, size = 10000000)

b

b.mean()

b.std()

plt.figure(figsize = (12, 8))
plt.hist(b, bins = 1000, density = True)
plt.title("Continuous Uniform Distribution", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.show()

plt.figure(figsize = (12, 8))
plt.hist(b, bins = 1000, density= True, cumulative= True)
plt.grid()
plt.title("Continuous Uniform Distribution", fontsize = 20)
plt.ylabel("cdf", fontsize = 15)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()



# ### Creating a normally distributed Random Variable

import numpy as np
import matplotlib.pyplot as plt

mu = 100
sigma = 2
size = 1000000

np.random.seed(123)
pop = np.random.normal(loc = mu, scale = sigma, size = size)

pop.size

pop.mean()

pop.std()

plt.figure(figsize = (20, 8))
plt.hist(pop, bins = 1000)
plt.title("Normal Distribution", fontsize = 20)
plt.xlabel("Screw Length", fontsize = 15)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.show()

import scipy.stats as stats

stats.skew(pop)

stats.kurtosis(pop)

stats.kurtosis(pop, fisher= False)

stats.describe(pop)



# ### Normal Distribution - Probability Density Function (pdf) with scipy.stats

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

mu = 100
sigma = 2

x = np.linspace(90, 110, 1000)
x

y = stats.norm.pdf(x, loc = mu, scale = sigma)
y

plt.figure(figsize = (20, 8))
plt.hist(pop, bins = 1000, density = True)
plt.plot(x, y, linewidth = 3, color = "red")
plt.grid()
plt.title("Normal Distribution", fontsize = 20)
plt.xlabel("Screw Length", fontsize = 15)
plt.ylabel("pdf", fontsize = 15)
plt.show()

pop



# ### Normal Distribution - Cumulative Distribution Function (cdf) with scipy.stats

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

mu = 100
sigma = 2

x = np.linspace(90, 110, 1000)

y = stats.norm.cdf(x, loc = mu, scale = sigma)

plt.figure(figsize = (20, 8))
plt.hist(pop, bins = 1000, density= True, cumulative= True)
plt.plot(x, y, color = "red", linewidth = 3)
plt.grid()
plt.title("Normal Distribution", fontsize = 20)
plt.xlabel("Screw Length", fontsize = 15)
plt.ylabel("cdf", fontsize = 15)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()

pop



# ### The Standard Normal Distribution and Z-Scores

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

pop

mu = pop.mean()
sigma = pop.std()

mu

sigma

pop[0]

(pop[0] - mu) / sigma

pop[1]

(pop[1] - mu) / sigma

(pop - mu) / sigma

z = stats.zscore(pop)
z

round(z.mean(), 4)

z.std()

stats.skew(z)

stats.kurtosis(z)

x = np.linspace(-4, 4, 1000)

y = stats.norm.pdf(x, loc = 0, scale = 1)

plt.figure(figsize = (20, 8))
#plt.hist(z, bins = 1000, density= True)
plt.grid()
plt.plot(x, y, linewidth = 3, color = "red")
plt.xticks(np.arange(-4, 5, 1),
           labels = ["-4σ = -4", "-3σ = -3", "-2σ = -2", "-1σ = -1", "mu = 0", "1σ = 1", "2σ = 2", "3σ = 3", "4σ = 4"],
           fontsize = 15)
plt.title("Standard Normal Distribution", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.show()

y = stats.norm.cdf(x)

plt.figure(figsize = (20, 8))
#plt.hist(z, bins = 1000, density= True, cumulative= True)
plt.plot(x, y, color = "red", linewidth = 3)
plt.grid() 
plt.xticks(np.arange(-4, 5, 1),
           labels = ["-4σ = -4", "-3σ = -3", "-2σ = -2", "-1σ = -1", "mu = 0", "1σ = 1", "2σ = 2", "3σ = 3", "4σ = 4"],
           fontsize = 15)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title("Standard Normal Distribution", fontsize = 20)
plt.ylabel("cdf", fontsize = 15)
plt.show()



# ### Probabilities and Z-Scores with scipy.stats

import numpy as np
import scipy.stats as stats

stats.norm.cdf(-1, loc = 0, scale = 1)

1 - stats.norm.cdf(-1)

stats.norm.cdf(1)

1 - stats.norm.cdf(1)

stats.norm.cdf(1) - stats.norm.cdf(-1)

stats.norm.cdf(-2)

1 - stats.norm.cdf(2)

stats.norm.cdf(2) - stats.norm.cdf(-2)

stats.norm.cdf(0)

pop

minus_two_sigma = pop.mean() - 2 * pop.std()
minus_two_sigma

(pop < minus_two_sigma).mean()

1 -stats.norm.cdf(x = 105, loc = pop.mean(), scale = pop.std())

z = (105-pop.mean()) / pop.std()
z

stats.norm.cdf(z)

stats.norm.ppf(0.5, loc = 0, scale = 1)

stats.norm.ppf(0.05)

stats.norm.ppf(0.95)

stats.norm.ppf(loc = pop.mean(), scale = pop.std(), q = 0.05)

stats.norm.ppf(loc = pop.mean(), scale = pop.std(), q = 0.95)



# ### Confidence Intervals

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# The ABC Company produces screws. The length of the screws follows a __Normal Distribution__ with __mean 100__ (millimeters) and __standard deviation 2__ (millimeters). Determine the __Confidence Interval__ around the mean where __90%__ of all observations can be found.
#

conf = 0.90

tails = (1-conf) / 2
tails

left = stats.norm.ppf(tails)
left

right = stats.norm.ppf(1-tails)
right

stats.norm.interval(conf)

left, right = stats.norm.interval(conf)

left

right

x = np.linspace(-5, 5, 1000)

y = stats.norm.pdf(x)

plt.figure(figsize = (20, 8))
plt.plot(x, y, color = "black", linewidth = 2)
plt.fill_between(x, y, where = ((x > right) | (x < left)), color = "blue", alpha = 0.2)
plt.fill_between(x, y, where = ((x < right) & (x > left)), color = "red", alpha = 0.2)
plt.grid()
plt.annotate("5%", xy = (1.75, 0.01), fontsize = 20)
plt.annotate("5%", xy = (-2.25, 0.01), fontsize = 20)
plt.annotate("90%", xy = (-0.6, 0.2), fontsize = 40)
plt.annotate("-1.645σ", xy = (-1.645, -0.015), fontsize = 10)
plt.annotate("1.645σ", xy = (1.645, -0.015), fontsize = 10)
plt.xticks(np.arange(-4, 5, 1), 
           labels = ["-4σ = -4", "-3σ = -3", "-2σ = -2", "-1σ = -1", "mu = 0", "1σ = 1", "2σ = 2", "3σ = 3", "4σ = 4"],
           fontsize = 10)
plt.title("Standard Normal Distribution", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.show()

x = np.linspace(-5, 5, 1000)

y = stats.norm.cdf(x)

plt.figure(figsize = (12, 8))
plt.margins(x = 0, y = 0)
plt.plot(x, y, color = "black", linewidth = 2)
plt.vlines(x = [left, right], ymin = 0, ymax = [stats.norm.cdf(left), stats.norm.cdf(right)], linestyle = "--")
plt.hlines(y = [stats.norm.cdf(left), stats.norm.cdf(right)], xmin = -5, xmax = [left, right], linestyle = "--")
plt.grid()
plt.xticks(np.arange(-4, 5, 1), 
           labels = ["-4σ = -4", "-3σ = -3", "-2σ = -2", "-1σ = -1", "mu = 0", "1σ = 1", "2σ = 2", "3σ = 3", "4σ = 4"],
           fontsize = 15)
plt.yticks(np.arange(0, 1.1, 0.05), fontsize = 10)
plt.annotate("-1.645σ", xy = (-1.60, 0.015), fontsize = 10)
plt.annotate("1.645σ", xy = (1.7, 0.015), fontsize = 10)
plt.title("Standard Normal Distribution", fontsize = 20)
plt.ylabel("cdf", fontsize = 15)
plt.show()

stats.norm.interval(conf, loc = 100, scale = 2)

pop

left, right = np.percentile(pop, [5, 95])

left 

right


