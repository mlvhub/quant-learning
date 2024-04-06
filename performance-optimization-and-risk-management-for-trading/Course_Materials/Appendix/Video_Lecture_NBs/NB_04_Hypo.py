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

# ## Hypothesis Testing / Inferential Statistics

# ### Two-tailed Z-Test with known Population Variance

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# The ABC Company produces screws with a __target length of 100__ millimeters (mm).
# The length of the screws follows a __Normal Distribution__ with a (population) __standard deviation of 2 mm__.<br><br>
# The machines need to be cleaned and recalibrated once a week. After the cleaning/recalibration process, ABC produces a sample of 20 screws to check whether the machines are correctly calibrated (__mean length = 100 mm__).<br> <br>
# After the most recent calibration you suspect that the machines are incorrectly calibrated. Based on the drawn sample (__sample size = 20__) with __sample mean 100.929__ mm, test on a __2% level of significance__, whether the machine is correctly calibrated or corrupted (two-tailed).
# <br><br>Calculate the __z-statistic__ and the __p-value__ of your test.

# __Corrupted Machine__

mu = 100.7 # unknown
pop_std = 2 # known

sample_size = 20

np.random.seed(123)
sample = np.random.normal(loc = mu, scale = pop_std, size = sample_size)

sample

point_est_mean = sample.mean()
point_est_mean

standard_error = pop_std / np.sqrt(sample_size)
standard_error

# __H0:__ mean == 100 <br> 
# __Ha:__ mean != 100

H0 = 100

# __10% Significance Level__

conf = 0.90

stats.norm.interval(conf, loc = H0, scale = standard_error)

left, right = stats.norm.interval(conf, loc = H0, scale = standard_error)

left

right

point_est_mean

x = np.linspace(97, 103 , 1000)

y = stats.norm.pdf(x, loc = H0, scale = standard_error) 

plt.figure(figsize = (20, 8))
plt.plot(x, y, linewidth = 3, label = "Normal Distribution")
plt.vlines(x = H0, ymin = 0, ymax = 0.90)
plt.vlines(x = point_est_mean, ymin = 0, ymax = 0.1, color = "red")
plt.fill_between(x, y, where = ((x > right) | (x < left)), color = "blue", alpha = 0.2)
plt.annotate("5%", xy = (99.05, 0.02), fontsize = 15)
plt.annotate("5%", xy = (101, 0.02), fontsize = 15)
plt.annotate("Reject H0!\nMachine is corrupted!", xy = (96, 0.5), fontsize = 30)
plt.annotate("H0: Mean = 100", xy = (100.05, 0.3), fontsize = 15)
plt.annotate("Point Estimate", xy = (101, 0.14), color = "red",fontsize = 15)
plt.grid()
plt.title("Z-Test (two-sided)", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

plt.figure(figsize = (20, 8))
plt.plot(x, y, linewidth = 3, label = "Normal Distribution")
plt.vlines(x = H0, ymin = 0, ymax = 0.90)
plt.vlines(x = point_est_mean, ymin = 0, ymax = 0.1, color = "red")
plt.vlines(x = H0-(point_est_mean-H0), ymin = 0, ymax = 0.1, color = "red")
plt.fill_between(x, y, where = ((x > point_est_mean) | (x < (H0-(point_est_mean-H0)))), color = "blue", alpha = 0.2)
#plt.annotate("0.5 p-value", xy = (98.5, 0.04), fontsize = 15)
#plt.annotate("0.5 p-value", xy = (101, 0.02), fontsize = 15)
#plt.annotate("Reject H0!\nMachine is corrupted!", xy = (96, 0.5), fontsize = 30)
plt.annotate("H0: Mean = 100", xy = (100.05, 0.1), fontsize = 15)
plt.annotate("Point Estimate", xy = (101, 0.14), color = "red",fontsize = 15)
plt.grid()
plt.title("Z-Test (two-sided)", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

# __Lower Level of Significance (2%)__

conf = 0.98

stats.norm.interval(conf, loc = H0, scale = standard_error)

left, right = stats.norm.interval(conf, loc = H0, scale = standard_error)

x = np.linspace(96, 104 , 1000)

y = stats.norm.pdf(x, loc = H0, scale = standard_error) 

plt.figure(figsize = (20, 8))
plt.plot(x, y, linewidth = 3, label = "Normal Distribution")
plt.vlines(x = H0, ymin = 0, ymax = 0.90)
plt.vlines(x = point_est_mean, ymin = 0, ymax = 0.1, color = "red")
plt.fill_between(x, y, where = ((x > right) | (x < left)), color = "blue", alpha = 0.2)
plt.annotate("1%", xy = (99.05, 0.02), fontsize = 15)
plt.annotate("1%", xy = (101.4, 0.02), fontsize = 15)
plt.annotate("Do not reject H0!", xy = (96, 0.5), fontsize = 30)
plt.annotate("H0: Mean = 100", xy = (100.05, 0.3), fontsize = 15)
plt.annotate("Point Estimate", xy = (101, 0.14), color = "red",fontsize = 15)
plt.grid()
plt.title("Z-Test (two-sided)", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()



# ### Calculating and interpreting z-statistic and p-value

z_stat = (point_est_mean - H0) / standard_error
z_stat

stats.norm.cdf(-abs(z_stat))

p_value = 2 * stats.norm.cdf(-abs(z_stat))

p_value

x = np.linspace(-4, 4, 1000)

y = stats.norm.pdf(x)

plt.figure(figsize = (20, 8))
plt.plot(x, y, linewidth = 3, label = "Standard Normal Distribution")
plt.vlines(x = 0, ymin = 0, ymax = 0.40)
plt.vlines(x = z_stat, ymin = 0, ymax = 0.05, color = "red")
plt.vlines(x = -z_stat, ymin = 0, ymax = 0.05, color = "red")
plt.fill_between(x, y, where = ((x > abs(z_stat)) | (x < -abs(z_stat))), color = "blue", alpha = 0.2)
plt.annotate("0.5 p-value", xy = (-2.9, 0.05), fontsize = 15)
plt.annotate("0.5 p-value", xy = (2.3, 0.05), fontsize = 15)
plt.annotate("-z-Statistic", xy = (-2, 0.02), color = "red", fontsize = 15)
plt.annotate("z-statistic", xy = (1.4, 0.02), color = "red",fontsize = 15)
plt.grid()
plt.title("Z-Test (two-sided)", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()



# ### One-tailed Z-Test with known Population Variance

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# After the most recent calibration you suspect that the machines are incorrectly calibrated and produce screws with a __mean length greater than 100 mm__.  Based on the drawn sample (__sample size = 20__) with __sample mean 100.929__ mm, perform a one-tailed test with __5% level of significance__. <br><br>Calculate the z-statistic and the p-value of your test.

# __H0:__ mean <= 100 <br> 
# __Ha:__ mean > 100

# __Corrupted Machine__

mu = 100.7 # unknown
pop_std = 2 # known

sample_size = 20

np.random.seed(123)
sample = np.random.normal(loc = mu, scale = pop_std, size = sample_size)

sample

point_est_mean = sample.mean()
point_est_mean

standard_error = pop_std / np.sqrt(sample_size)
standard_error

H0 = 100

conf = 0.95

stats.norm.ppf(0.95, loc = H0, scale = standard_error)

right = stats.norm.ppf(conf, loc = H0, scale = standard_error)
right

x = np.linspace(96, 104 , 1000)

y = stats.norm.pdf(x, loc = H0, scale = standard_error) 

plt.figure(figsize = (20, 8))
plt.plot(x, y, linewidth = 3, label = "Normal Distribution")
plt.vlines(x = H0, ymin = 0, ymax = 0.90)
plt.vlines(x = point_est_mean, ymin = 0, ymax = 0.1, color = "red")
plt.fill_between(x, y, where = ((x > right)), color = "blue", alpha = 0.2)
plt.annotate("5%", xy = (101, 0.02), fontsize = 15)
plt.annotate("Reject H0!\nMean is greater than 100!", xy = (96, 0.5), fontsize = 30)
plt.annotate("H0: Mean <= 100", xy = (100.05, 0.3), fontsize = 15)
plt.annotate("Point Estimate", xy = (101, 0.14), color = "red",fontsize = 15)
plt.grid()
plt.title("Z-Test (one-sided)", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

z_stat = (point_est_mean - H0) / standard_error
z_stat

p_value = stats.norm.cdf(-abs(z_stat))

p_value



# ### Two-sided t-Test (unknown Population Variance)

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# The S&P 500, or just the S&P, is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. The S&P 500 is a capitalization-weighted index and the performance of the 10 largest companies in the index account for 21.8% of the performance of the index. <br> <br>
# You have a random sample with 50 stocks/companies and their annual returns for the year 2017 (__sample size = 50__). <br>Test on a __5% level of significance__, whether the (equally-weighted) mean return for the whole S&P 500 population for the year 2017 is __equal to 0%__ or not. Calculate the t-statistic and the p-value of your test.<br> <br>  Assume a __sample mean of 25.32%__ and a __sample standard deviation of 30.50%__.  

# __H0:__ The mean return is equal to 0%. <br> 
# __Ha:__ The mean return is unequal to 0%.

sample = np.loadtxt("sample.csv", delimiter = ",", usecols = 1)

sample

sample_size = sample.size
sample.size

point_est_mean = sample.mean()
point_est_mean

standard_error = sample.std(ddof = 1) / np.sqrt(sample_size)
standard_error

# __5% Significance Level__

H0 = 0

conf = 0.95

stats.t.interval(conf, loc = H0, scale = standard_error, df = sample_size - 1)

left, right = stats.t.interval(conf, loc = H0, scale = standard_error, df = sample_size - 1)

left

right

point_est_mean

x = np.linspace(-0.2, 0.2 , 1000)

y = stats.t.pdf(x, loc = H0, scale = standard_error, df = sample_size - 1) 

plt.figure(figsize = (20, 8))
plt.plot(x, y, linewidth = 3, label = "t-Distribution")
plt.vlines(x = H0, ymin = 0, ymax = 9)
plt.vlines(x = point_est_mean, ymin = 0, ymax = 10, color = "red")
plt.fill_between(x, y, where = ((x > right) | (x < left)), color = "blue", alpha = 0.2)
plt.annotate("2.5%", xy = (-0.1, 0.2), fontsize = 15)
plt.annotate("2.5%", xy = (0.09, 0.2), fontsize = 15)
plt.annotate("Reject H0!\n2017 Returns are not equal to 0%!", xy = (-0.2, 6.5), fontsize = 20)
plt.annotate("H0: Mean = 0%", xy = (0.005, 3), fontsize = 15)
plt.annotate("Point Estimate", xy = (0.2, 5), color = "red",fontsize = 15)
plt.grid()
plt.title("t-Test (two-sided)", fontsize = 20)
plt.ylabel("pdf", fontsize = 15)
plt.legend(fontsize = 15)
plt.show()

t_stat = (point_est_mean - H0) / standard_error

t_stat

p_value = 2 * stats.t.cdf(-abs(t_stat), df = sample_size - 1)

p_value

format(p_value, ".10f")

stats.ttest_1samp(sample, H0)

t_stat, p_value = stats.ttest_1samp(sample, H0)

t_stat

p_value



# ### One-sided t-test

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# You have a random sample with 50 stocks/companies and their annual returns for the year 2017 (__sample size = 50__). Test on a __5% level of significance__, whether the (equally-weighted) mean return for the whole S&P 500 population for the year 2017 is __equal to or less than 15%__ (H0). Assume a sample mean of 25.32% and a sample standard deviation of 30.50%. Calculate the t-statistic and the p-value of your test.

# __H0:__ The mean return is equal to or less than 15%. <br> 
# __Ha:__ The mean return is greater than 15%.

sample = np.loadtxt("sample.csv", delimiter = ",", usecols = 1)

sample

sample_size = sample.size
sample.size

point_est_mean = sample.mean()
point_est_mean

standard_error = sample.std(ddof = 1) / np.sqrt(sample_size)
standard_error

H0 = 0.15

t_stat = (point_est_mean - H0) / standard_error

t_stat

p_value = stats.t.cdf(-abs(t_stat), df = sample_size - 1)

p_value

stats.ttest_1samp(sample, H0)



# ### Hypothesis Testing with Bootstrapping

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# The S&P 500, or just the S&P, is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. The S&P 500 is a capitalization-weighted index and the performance of the 10 largest companies in the index account for 21.8% of the performance of the index. <br> <br>
# You have a random sample with 50 stocks/companies and their annual returns for the year 2017 (__sample size = 50__). <br>Test on a __5% level of significance__, whether the (equally-weighted) mean return for the whole S&P 500 population for the year 2017 is __equal to 15%__ or not. Calculate the t-statistic and the p-value of your test.<br> <br>  Assume a __sample mean of 25.32%__ and a __sample standard deviation of 30.50%__.  

# __H0:__ The mean return is equal to 15%. <br> 
# __Ha:__ The mean return is unequal to 15%.

H0 = 0.15

sample = np.loadtxt("sample.csv", delimiter = ",", usecols = 1)

sample

sample_size = sample.size
sample.size

point_est_mean = sample.mean()
point_est_mean

sims = 1000000

np.random.seed(111)
bootstrap = []
for i in range(sims):
    bootstrap.append(np.random.choice(sample, size = sample_size, replace = True).mean())

bootstrap

len(bootstrap)

plt.figure(figsize = (12, 8))
plt.hist(bootstrap, bins = 1000)
plt.grid()
plt.ylabel("Absolute Frequency", fontsize = 13)
plt.xlabel("Mean Return", fontsize = 13)
plt.show()

bootstrap_a = np.array(bootstrap)

bootstrap_mean = bootstrap_a.mean()
bootstrap_mean

bootstrap_shifted = bootstrap_a - bootstrap_mean + H0

bootstrap_shifted

bootstrap_shifted.mean()

plt.figure(figsize = (12, 8))
plt.hist(bootstrap_shifted, bins = 1000)
plt.grid()
plt.ylabel("Absolute Frequency", fontsize = 13)
plt.xlabel("Mean Return", fontsize = 13)
plt.show()

(bootstrap_shifted >= point_est_mean).mean()

p_value = 2 * (bootstrap_shifted >= point_est_mean).mean()
p_value

import scipy.stats as stats

stats.ttest_1samp(sample, H0)



# ### Tests for Normality of Returns

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt 
np.set_printoptions(precision=2, suppress= True)

pop = np.loadtxt("SP500_pop.csv", delimiter = ",", usecols = 1)

pop

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75)
plt.title("Absolute Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2017 (in %)", fontsize = 15)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.show()

pop.mean()

pop.std()

stats.skew(pop)

stats.kurtosis(pop)

sims = 1000000

kurtosis = []
np.random.seed(123)
for i in range(sims):
    kurtosis.append(stats.kurtosis(np.random.normal(size = 500)))

kurtosis

plt.figure(figsize = (12, 8))
plt.hist(kurtosis, bins = 1000)
plt.show()

stats.kurtosistest(pop)

z_stat, p_value = stats.kurtosistest(pop)

format(p_value, ".41f")

stats.skewtest(pop)

stats.normaltest(pop)


