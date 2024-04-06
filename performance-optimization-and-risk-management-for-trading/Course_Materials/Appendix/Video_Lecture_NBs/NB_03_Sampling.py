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

# ## Sampling and Estimation

# ### Sampling with np.random.choice() 

import numpy as np

mu = 100
sigma = 2
size = 1000000

np.random.seed(123)
pop = np.random.normal(loc = mu, scale = sigma, size = size)

pop

pop.mean()

pop.std()

sample_size = 50

np.random.seed(123)
sample = np.random.choice(pop, sample_size, replace = False)

sample

sample.size

# __Sample Statistics__

sample.mean()

sample.std(ddof = 1)

# __Sampling Error__

sample.mean() - pop.mean()

np.random.choice(pop, 2, replace = False).mean()

np.random.choice(pop, 1000, replace = False).mean()



# ### Sampling Distribution

import numpy as np
import matplotlib.pyplot as plt

pop.size

sample_size = 10
samples = 10000

np.random.seed(123)
sample_means_10 = []
for i in range(samples):
    sample_means_10.append(np.random.choice(pop, sample_size, replace = False).mean()) 

sample_means_10

len(sample_means_10)

sample_size = 50

np.random.seed(123)
sample_means_50 = []
for i in range(samples):
    sample_means_50.append(np.random.choice(pop, sample_size, replace = False).mean())  

sample_means_50

plt.figure(figsize = (12, 8))
plt.hist(sample_means_10, bins = 200, alpha = 0.5, color = "blue", label = "Sample Size: 10", range = [98, 102])
plt.hist(sample_means_50, bins = 200, alpha = 0.5, color = "red",label = "Sample Size: 50", range = [98, 102])
plt.title("Sampling Distribution of the Mean", fontsize = 20)
plt.xlabel("Sample Means", fontsize = 15)
plt.ylabel("Frequency", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

np.mean(sample_means_10)

np.mean(sample_means_50)

pop.mean()



# ### Standard Error of the sample mean

import numpy as np

np.std(sample_means_10)

np.std(sample_means_50)

pop.std()

pop.std() / np.sqrt(10)

pop.std() / np.sqrt(50)

standard_error = pop.std() / np.sqrt(10)
standard_error



# ### Central Limit Theorem (Part 1)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

pop

pop.mean()

pop.std()

sample_size = 2
samples = 10000

standard_error = pop.std() / np.sqrt(sample_size)
standard_error

sampling_distr_mean = pop.mean()
sampling_distr_mean

np.random.seed(123)
sample_means_2 = []
for i in range(samples):
    sample_means_2.append(np.random.choice(pop, sample_size, replace = False).mean())  

np.std(sample_means_2)

np.mean(sample_means_2)

x = np.linspace(96, 104, 1000)
y = stats.norm.pdf(x, loc = sampling_distr_mean, scale = standard_error)

plt.figure(figsize = (12, 8))
plt.hist(sample_means_2, bins = 100, alpha = 0.5, label = "Sample Size: 2", density= True)
plt.plot(x, y, color = "red", linewidth = 3, label = "Normal Distribution")
plt.title("Sampling Distribution of the Mean", fontsize = 20)
plt.xlabel("Sample Means", fontsize = 15)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()



# ### Central Limit Theorem (Part 2)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(123)
pop = np.random.uniform(low = 0, high = 10, size = 100000)

pop

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 100, density = True)
plt.title("Uniform Distribution", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.show()

pop.mean()

pop.std()

sample_size = 50
samples = 100000

sample_means = []
for i in range(samples):
    sample_means.append(np.random.choice(pop, sample_size, replace = False).mean())  

np.std(sample_means)

standard_error = pop.std() / np.sqrt(sample_size)
standard_error

np.mean(sample_means)

sampling_distr_mean = pop.mean()
sampling_distr_mean

x = np.linspace(3, 7, 1000)
y = stats.norm.pdf(x, loc = sampling_distr_mean, scale = standard_error)

plt.figure(figsize = (12, 8))
plt.hist(sample_means, bins = 500, density = True, label = "Sample Size: {}".format(sample_size))
plt.plot(x, y, color = "red", linewidth = 3, label = "Normal Distribution")
plt.legend(fontsize = 15)
plt.show()



# ### Point Estimates vs. Confidence Interval Estimates (known Population Variance)

import numpy as np
import scipy.stats as stats

# The ABC Company produces screws. The length of the screws follows a Normal Distribution with a (population) __standard deviation of 2 millimeters__ (mm). Based on a recently drawn sample (__sample size = 50__), construct a __90% Confidence Interval__ for the __true population mean__ length. The __sample mean is 99.917 mm__ and the sample standard deviation is 2.448 mm.  

sample = np.loadtxt("sample2.csv", delimiter = ",", usecols = 1)

sample

sample_size = sample.size
sample_size

pop_std = 2

# __Point Estimate__

point_est = sample.mean()
point_est

# __Standard Error (known Population Variance)__

standard_error = pop_std / np.sqrt(sample_size)
standard_error

conf = 0.90

left_z, right_z = stats.norm.interval(conf)

left_z

right_z

# __Confidence Interval Estimate__

conf_int = (point_est + left_z * standard_error, point_est + right_z * standard_error)

conf_int

stats.norm.interval(conf, loc = point_est, scale = standard_error)



# ### Unknown Population Variance - The Standard Case (Example 1)

# The ABC Company produces screws. The length of the screws follows a Normal Distribution. Based on a recently drawn sample (__sample size = 50__), construct a __90% Confidence Interval__ for the __true population mean__ length. The __sample mean is 99.917 mm__ and the __sample standard deviation is 2.448 mm__. 

import numpy as np
import scipy.stats as stats

sample = np.loadtxt("sample2.csv", delimiter = ",", usecols = 1)

sample_size = sample.size
sample_size

point_est_mean = sample.mean()
point_est_mean

point_est_std = sample.std(ddof = 1)
point_est_std

standard_error = point_est_std / np.sqrt(sample_size)
standard_error

conf = 0.90

left_t, right_t = stats.t.interval(conf, df = sample_size - 1)

left_t

right_t

# __Confidence Interval Estimate__

conf_int = (point_est_mean + left_t * standard_error, point_est_mean + right_t * standard_error)

conf_int

stats.t.interval(conf, loc = point_est_mean, scale = standard_error, df = sample_size - 1)



# ### Unknown Population Variance - The Standard Case (Example 2)

# The __S&P 500__, or just the S&P, is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. The S&P 500 is a __capitalization-weighted__ index and the performance of the 10 largest companies in the index account for 21.8% of the performance of the index. <br><br>
# You have a random sample with 50 stocks/companies and their annual returns for the year 2017 (__sample size = 50__). Estimate the (__equally-weighted__) mean return for the whole S&P 500 population for the year 2017 by constructing a __95% Confidence Interval__. Assume a __sample mean of 25.32%__ and a __sample standard deviation of 30.50%__.  

import numpy as np
import scipy.stats as stats

sample = np.loadtxt("sample.csv", delimiter = ",", usecols = 1)

sample

sample_size = sample.size
sample_size

point_est_mean = sample.mean()
point_est_mean

point_est_std = sample.std(ddof = 1)
point_est_std

standard_error = point_est_std / np.sqrt(sample_size)
standard_error

conf = 0.95

stats.t.interval(conf, loc = point_est_mean, scale = standard_error, df = sample_size - 1)



# ### Student´s t-Distribution vs. Normal Distribution with scipy.stats

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

t_2 = stats.t.pdf(x, df = 2 - 1) 

t_5 = stats.t.pdf(x, df = 5 - 1) 

t_30 = stats.t.pdf(x, df = 30 - 1) 

t_1000 = stats.t.pdf(x, df = 1000 - 1) 

N = stats.norm.pdf(x)

plt.figure(figsize = (12, 8))
#plt.plot(x, t_2, linewidth = 2, label = "Student´s t (Sample: 2)")
#plt.plot(x, t_5, linewidth = 2, label = "Student´s t (Sample: 5)")
#plt.plot(x, t_30, linewidth = 2, label = "Student´s t (Sample: 30)")
plt.plot(x, t_1000, linewidth = 2, label = "Student´s t (Sample: 1000)")
plt.plot(x, N, linewidth = 2, label = "Standard Normal")
plt.title("Student´s t-Distribution", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()



# ### Bootstrapping: an alternative method without Statistics

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

sample = np.loadtxt("sample.csv", delimiter = ",", usecols = 1)

sample

sample_size = sample.size
sample_size

sample.mean()

sims = 1000000

np.random.seed(123)
bootstrap = []
for i in range(sims):
    bootstrap.append(np.random.choice(sample, size = sample_size, replace = True).mean())

len(bootstrap)

plt.figure(figsize = (12, 8))
plt.hist(bootstrap, bins = 1000)
plt.grid()
plt.xticks(np.arange(0, 0.5, 0.05))
plt.ylabel("Absolute Frequency", fontsize = 13)
plt.xlabel("Mean Return", fontsize = 13)
plt.show()

np.percentile(bootstrap, [2.5, 97.5])


