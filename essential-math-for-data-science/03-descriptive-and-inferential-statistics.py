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

# # Descriptive and Inferential Statistics

# ### Population
#
# A _population_ is a particular group of interest we want to study, such as “all seniors over the age of 65 in the North America,” “all golden retrievers in Scotland,” or “current high school sophomores at Los Altos High School.”

# ### Sample
#
# A _sample_ is a subset of the population that is ideally random and unbiased, which we use to infer attributes about the population. We often have to study samples because polling the entire population is not always possible. 

# ### Bias
#
# The way to overcome bias is to truly at random select students (in the example study) from the entire population, and they cannot elect themselves into or out of the sample voluntarily.
#
# There are many types of bias, but they all have the same effect of distorting findings.
#
# _Confirmation bias_ is gathering only data that supports your belief, which can even be done unknowingly. An example of this is following only social media accounts you politically agree with, reinforcing your beliefs rather than challenging them.
#
# _Self-selection bias_ is when certain types of subjects are more likely to include themselves in the experiment. Walking onto a flight and polling the customers if they like the airline over other airlines, and using that to rank customer satisfaction among all airlines, is silly. Why? Many of those customers are likely repeat customers, and they have created self-selection bias.
#
# _Survival bias_ captures only living and survived subjects, while the deceased ones are never accounted for.
#

# ## Descriptive Statistics

# ### Mean and Weighted Mean
#
#

# +
# Number of pets each person owns
sample = [1, 3, 2, 5, 7, 0, 2, 3]

mean = sum(sample) / len(sample)
mean
# -

# The mean we commonly use (above) gives equal importance to each value. But we can manipulate the mean and give each item a different weight:
#
# $$
# \text{Weighted mean} = \frac{(x_1 \cdot w_1) + (x_2 \cdot w_2) + (x_3 \cdot w_3) + \ldots + (x_n \cdot w_n)}{w_1 + w_2 + w_3 + \ldots + w_n}
# $$
#
# This can be helpful when we want some values to contribute to the mean more than others. 

# +
# Three exams of .20 weight each and final exam of .40 weight
sample = [90, 80, 63, 87]
weights = [.20, .20, .20, .40]

weighted_mean = sum(s * w for s,w in zip(sample, weights)) / sum(weights)
weighted_mean
# -

# ### Median
#
# The median is the middlemost value in a set of ordered values. You sequentially order the values, and the median will be the centermost value. If you have an even number of values, you average the two centermost values.
#
# The median can be preferable in outlier-heavy situations (such as income-related data) over the mean, when your median is very different from your mean, that means you have a skewed dataset with outliers. 

# +
# Number of pets each person owns
sample = [0, 1, 5, 7, 9, 10, 14]

def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = int(n / 2) - 1 if n % 2 == 0 else int(n/2)

    if n % 2 == 0:
        return (ordered[mid] + ordered[mid+1]) / 2.0
    else:
        return ordered[mid]

median(sample)
# -

# ### Mode
#
# The mode is the most frequently occurring set of values. It primarily becomes useful when your data is repetitive and you want to find which values occur the most frequently.
#
# When no value occurs more than once, there is no mode. When two values occur with an equal amount of frequency, then the dataset is considered _bimodal_.
#
# In practicality, the mode is not used a lot unless your data is repetitive. This is commonly encountered with integers, categories, and other discrete variables.

# +
# Number of pets each person owns
from collections import defaultdict

sample = [1, 3, 2, 5, 7, 0, 2, 3]

def mode(values):
    counts = defaultdict(lambda: 0)

    for s in values:
        counts[s] += 1

    max_count = max(counts.values())
    modes = [v for v in set(values) if counts[v] == max_count]
    return modes

mode(sample)
# -

# ### Variance and Standard Deviation
#
# The _variance_ is a measure of how spread out our data is.
#
# #### Population Variance and Standard Deviation
#
# $$
# \text{Population variance} = \frac{(x_1 - mean)^2 + (x_2 - mean)^2 + \ldots + (x_n - mean)^2}{N}
# $$
#
# More formally:
#
# $$
# \sigma^2 = \frac{\sum(x_i - \mu)^2}{N}
# $$

# +
# Number of pets each person owns
data = [0, 1, 5, 7, 9, 10, 14]

def variance(values):
    mean = sum(values) / len(values)
    _variance = sum((v - mean) ** 2 for v in values) / len(values)
    return _variance

variance(data)
# -

# So the variance for number of pets owned by my office staff is 21.387755. OK, but what does it exactly mean?
#
# This number is larger than any of our observations because we did a lot squaring and summing, putting it on an entirely different metric. So how do we squeeze it back down so it’s back on the scale we started with?
#
# Let’s take the square root of the variance, which gives us the _standard deviation_.
#
# This is the variance scaled into a number expressed in terms of “number of pets,” which makes it a bit more meaningful:
#
# $$
# \sigma = \sqrt{\frac{\sum(x_i - \mu)^2}{N}}
# $$

# +
from math import sqrt

def std_dev(values):
    return sqrt(variance(values))

std_dev(data)
# -

# #### Sample Variance and Standard Deviation
#
# There is an important tweak we need to apply to the two formulas above when we calculate for a sample:
#
# $$
# s^2 = \frac{\sum{(x_i - \overline{x})^1}}{n - 1}
# $$
#
# $$
# s = \sqrt{\frac{\sum{(x_i - \overline{x})^1}}{n - 1}}
# $$
#
# When we average the squared differences, we divide by n–1 rather than the total number of items *n*.
#
# Why would we do this? We do this to decrease any bias in a sample and not underestimate the variance of the population based on our sample. 
#
# By counting values short of one item in our divisor, we increase the variance and therefore capture greater uncertainty in our sample.

# +
from math import sqrt

# Number of pets each person owns
data = [0, 1, 5, 7, 9, 10, 14]


def variance(values, is_sample: bool = False):
    mean = sum(values) / len(values)
    _variance = sum((v - mean) ** 2 for v in values) / (len(values) - (1 if is_sample else 0))

    return _variance


def std_dev(values, is_sample: bool = False):
    return sqrt(variance(values, is_sample))

print("VARIANCE = {}".format(variance(data, is_sample=True))) # 24.95238095238095
print("STD DEV = {}".format(std_dev(data, is_sample=True))) # 4


# -

# ## The Normal Distribution
#
# The *normal distribution*, also known as the *Gaussian distribution*, is a symmetrical bell-shaped distribution that has most mass around the mean, and its spread is defined as a standard deviation. The “tails” on either side become thinner as you move away from the mean.
#
#
# **Properties of a Normal Distribution**
#
# The normal distribution has several important properties that make it useful:
#
# - It’s symmetrical; both sides are identically mirrored at the mean, which is the center.
# - Most mass is at the center around the mean.
# - It has a spread (being narrow or wide) that is specified by standard deviation.
# - The “tails” are the least likely outcomes and approach zero infinitely but never touch zero.
# - It resembles a lot of phenomena in nature and daily life, and even generalizes nonnormal problems because of the central limit theorem, which we will talk about shortly.
#
# ### The Probability Density Function (PDF)
#
# $$
# f(x) = \frac{1}{\sigma} \cdot \sqrt{2\pi} \cdot e^{-\frac{1}{2}(\frac{x-\mu^2}{\sigma})}
# $$
#
# The normal distribution is continuous. This means to retrieve a probability we need to integrate a range of x values to find an area.

# normal distribution, returns likelihood
def normal_pdf(x: float, mean: float, std_dev: float) -> float:
    return (1.0 / (2.0 * math.pi * std_dev ** 2) ** 0.5)  * math.exp(-1.0 * ((x - mean) ** 2 / (2.0 * std_dev ** 2)))


# ### The Cumulative Distribution Function (CDF)
#
# With the normal distribution, the vertical axis is not the probability but rather the likelihood for the data. To find the probability we need to look at a given range, and then find the area under the curve for that range.

# +
from scipy.stats import norm

mean = 64.43
std_dev = 2.99

norm.cdf(64.43, mean, std_dev)
# -

# We can deductively find the area for a middle range by subtracting areas. If we wanted to find the probability of observing a golden retriever between 62 and 66 pounds, we would calculate the area up to 66 and subtract the area up to 62:

# +
from scipy.stats import norm

mean = 64.43
std_dev = 2.99

norm.cdf(66, mean, std_dev) - norm.cdf(62, mean, std_dev)
# -

# ### The Inverse CDF
#
# We will encounter situations where we need to look up an area on the CDF and then return the corresponding x-value. Of course this is a backward usage of the CDF, so we will need to use the inverse CDF, which flips the axes.
#
# For example, I want to find the weight that 95% of golden retrievers fall under:

# +
from scipy.stats import norm

norm.ppf(.95, loc=64.43, scale=2.99)
# -

# I find that 95% of golden retrievers are 69.348 or fewer pounds.
#
# You can also use the inverse CDF to generate random numbers that follow the normal distribution. If I want to create a simulation that generates one thousand realistic golden retriever weights, I just generate a random value between 0.0 and 1.0, pass it to the inverse CDF, and return the weight value:

# +
import random
from scipy.stats import norm

for i in range(0,1000):
    random_p = random.uniform(0.0, 1.0)
    random_weight = norm.ppf(random_p,  loc=64.43, scale=2.99)
    print(random_weight)


# -

# ### Z-Scores
#
# It is common to rescale a normal distribution so that the mean is 0 and the standard deviation is 1, which is known as the standard normal distribution. This makes it easy to compare the spread of one normal distribution to another normal distribution, even if they have different means and variances.
#
# Of particular importance with the standard normal distribution is it expresses all x-values in terms of standard deviations, known as *Z-scores*. Turning an x-value into a Z-score uses a basic scaling formula:
#
# $$
# z = \frac{x - \mu}{\sigma}
# $$
#
# Example:
#
# We have two homes from two different neighborhoods. Neighborhood A has a mean home value of $140,000 and standard deviation of $3,000. Neighborhood B has a mean home value of $800,000 and standard deviation of $10,000.
#
# $$
# \mu_A = 140,000
# \\\\
# \mu_B = 800,000
# \\\\
# \sigma_A = 3,000
# \\\\
# \sigma_B = 10,000
# $$
#
# Now we have two homes from each neighborhood. House A from neighborhood A is worth $150,000 and house B from neighborhood B is worth $815,000. Which home is more expensive relative to the average home in its neighborhood?
#
# $$
# x_A = 150,000
# \\\\
# x_B = 815,000
# $$
#
# Using $z = \frac{x - \mu}{\sigma}$:
#
# $$
# z_A = \frac{150000 - 140000}{3000} = 3.\overline{333}
# \\\\
# z_B = \frac{800000 - 140000}{10000} = 1.5
# $$
#
# So the house in neighborhood A is actually much more expensive relative to its neighborhood than the house in neighborhood B, as they have Z-scores of $3.\overline{333}$ and $1.5$, respectively.

# +
# Turn Z-scores into x-values and vice versa

def z_score(x, mean, std):
    return (x - mean) / std


def z_to_x(z, mean, std):
    return (z * std) + mean


mean = 140000
std_dev = 3000
x = 150000

# Convert to Z-score and then back to X
z = z_score(x, mean, std_dev)
back_to_x = z_to_x(z, mean, std_dev)

print("Z-Score: {}".format(z))  # Z-Score: 3.333
print("Back to X: {}".format(back_to_x))  # Back to X: 150000.0
# -

# ## Inferential Statistics
#
# We are wired as humans to be biased and quickly come to conclusions. Being a good data science professional requires you to suppress that primal desire and consider the possibility that other explanations can exist. It is acceptable (perhaps even enlightened) to theorize there is no explanation at all and a finding is just coincidental and random.

# ### The Central Limit Theorem
#
# When we start measuring large enough samples from a population, even if that population does not follow a normal distribution, the normal distribution still makes an appearance.

# +
# Exploring the central limit theorem in Python

# Samples of the uniform distribution will average out to a normal distribution.
import random
import plotly.express as px

sample_size = 31
sample_count = 1000

# Central limit theorem, 1000 samples each with 31
# random numbers between 0.0 and 1.0
x_values = [(sum([random.uniform(0.0, 1.0) for i in range(sample_size)]) / sample_size)
            for _ in range(sample_count)]

y_values = [1 for _ in range(sample_count)]

px.histogram(x=x_values, y = y_values, nbins=20).show()
# -

# This is because of the central limit theorem, which states that interesting things happen when we take large enough samples of a population, calculate the mean of each, and plot them as a distribution:
#
# - The mean of the sample means is equal to the population mean.
# - If the population is normal, then the sample means will be normal.
# - If the population is not normal, but the sample size is greater than 30, the sample means will still roughly form a normal distribution.
# - The standard deviation of the sample means equals the population standard deviation divided by the square root of n: $\text{sample standard deviation} = \frac{\text{population standard deviation}}{\sqrt{\text{sample size}}}$
#
# > While 31 is the textbook number of items you need in a sample to satisfy the central limit theorem and see a normal distribution, this sometimes is not the case. There are cases when you will need an even larger sample, such as when the underlying distribution is asymmetrical or multimodal (meaning it has several peaks rather than one at the mean).

# ### Confidence Intervals
#
# A *confidence interval* is a range calculation showing how confidently we believe a sample mean (or other parameter) falls in a range for the population mean.
#
# Example: *Based on a sample of 31 golden retrievers with a sample mean of 64.408 and a sample standard deviation of 2.05, I am 95% confident that the population mean lies between 63.686 and 65.1296.*
#
# First, I need the critical z-value which is the symmetrical range in a standard normal distribution that gives me 95% probability in the center.
#
# We need to leverage the inverse CDF. Logically, to get 95% of the symmetrical area in the center, we would chop off the tails that have the remaining 5% of area. Splitting that remaining 5% area in half would give us 2.5% area in each tail.

# +
# Retrieving a critical z-value

from scipy.stats import norm

def critical_z_value(p):
    norm_dist = norm(loc=0.0, scale=1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)

critical_z_value(p=.95)
# -

# Next, we're going to leverage the central limit theorem to produce the margin of error (E), which is the range around the sample mean that contains the population mean at that level of confidence. Recall that our sample of 31 golden retrievers has a mean of 64.408 and standard deviation of 2.05. The formula to get this margin of error is:
#
# $$
# E = \pm z_c \frac{s}{\sqrt{n}}
# \\\\
# E = \pm 1.95996 \cdot \frac{2.05}{\sqrt{31}}
# \\\\
# E = \pm 0.72164
# $$
#
# Apply that margin of error against the sample mean, and we get the confidence interval:
#
# $$
# \text{95% confidence interval} = 64.408 \pm 0.72164
# $$

# +
# Calculating a confidence interval in Python

from math import sqrt
from scipy.stats import norm


def critical_z_value(p):
    norm_dist = norm(loc=0.0, scale=1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)


def confidence_interval(p, sample_mean, sample_std, n):
    # Sample size must be greater than 30

    lower, upper = critical_z_value(p)
    lower_ci = lower * (sample_std / sqrt(n))
    upper_ci = upper * (sample_std / sqrt(n))

    return sample_mean + lower_ci, sample_mean + upper_ci

confidence_interval(p=.95, sample_mean=64.408, sample_std=2.05, n=31)
# -

# The way to interpret this is “based on my sample of 31 golden retriever weights with sample mean 64.408 and sample standard deviation of 2.05, I am 95% confident the population mean lies between 63.686 and 65.1296.”
#
# > One caveat to put here is that for this to work, our sample size must be at least 31 items. 

# ### Understanding P-Values
#
# *P-values* represent the probability of something occurring by chance rather than because of a hypothesized explanation.
#
# This helps us frame our null hypothesis ($H_0$), saying that the variable in question had no impact on the experiment and any positive results are just random luck. 
#
# The alternative hypothesis ($H_1$) poses that a variable in question (called the controlled variable) is causing a positive result.
#
# Traditionally, the threshold for statistical significance is a p-value of 5% or less, or .05. 

# ### Hypothesis Testing
#
# Past studies have shown that the mean recovery time for a cold is 18 days, with a standard deviation of 1.5 days, and follows a normal distribution.
#
# This means there is approximately 95% chance of recovery taking between 15 and 21 days.

# +
# Calculating the probability of recovery between 15 and 21 days

from scipy.stats import norm

# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5

# 95% probability recovery time takes between 15 and 21 days.
norm.cdf(21, mean, std_dev) - norm.cdf(15, mean, std_dev)
# -

# We can infer then from the remaining 5% probability that there’s a 2.5% chance of recovery taking longer than 21 days and a 2.5% chance of it taking fewer than 15 days.
#
# ---
#
# Now let’s say an experimental new drug was given to a test group of 40 people, and it took an average of 16 days for them to recover from the cold.
#
# Does the drug show a statistically signficant result? Or did the drug not work and the 16-day recovery was a coincidence with the test group? 
#
# That first question frames our alternative hypothesis, while the second question frames our null hypothesis.
#
# There are two ways we can calculate this: the one-tailed and two-tailed test. 
#
# #### One-Tailed Test
#
# When we approach the *one-tailed test*, we typically frame our null and alternative hypotheses using inequalities. 
#
# We hypothesize around the population mean and say that it either is greater than/equal to 18 (the null hypothesis $H_0$) or less than 18 (the alternative hypothesis $H_1$):
#
# $$
# H_0: \text{population mean} \geq 18
# \\\\
# H_1: \text{population mean} \leq 18
# $$
#
# To reject our null hypothesis, we need to show that our sample mean of the patients who took the drug is not likely to have been coincidental. Since a p-value of .05 or less is traditionally considered statistically significant, we will use that as our threshold.

# +
# Python code for getting x-value with 5% of area behind it

from scipy.stats import norm

# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5

# What x-value has 5% of area behind it?
norm.ppf(.05, mean, std_dev)
# -

# Therefore, if we achieve an average 15.53 or fewer days of recovery time in our sample group, our drug is considered statistically significant enough to have shown an impact. However, our sample mean of recovery time is actually 16 days and does not fall into this null hypothesis rejection zone. Therefore, the statistical significance test has failed.

# +
from scipy.stats import norm

# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5

# Probability of 16 or less days
p_value = norm.cdf(16, mean, std_dev)
p_value
# -

# Since the p-value of .0912 is greater than our statistical significance threshold of .05, we do not consider the drug trial a success and fail to reject our null hypothesis.
#
# #### Two-Tailed Test
#
# The previous test we performed is called the one-tailed test because it looks for statistical significance only on one tail. However, it is often safer and better practice to use a two-tailed test.
#
# To do a two-tailed test, we frame our null and alternative hypothesis in an “equal” and “not equal” structure:
#
# $$
# H_0: \text{population mean} = 18
# \\\\
# H_1: \text{population mean} \neq 18
# $$
#
# This has an important implication. We are structuring our alternative hypothesis to not test whether the drug improves cold recovery time, but if it had any impact. This includes testing if it increased the duration of the cold. 
#
# Naturally, this means we spread our p-value statistical significance threshold to both tails, not just one. If we are testing for a statistical significance of 5%, then we split it and give each 2.5% half to each tail.

# +
# Calculating a range for a statistical significance of 5%

from scipy.stats import norm

# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5

# What x-value has 2.5% of area behind it?
x1 = norm.ppf(.025, mean, std_dev)

# What x-value has 97.5% of area behind it
x2 = norm.ppf(.975, mean, std_dev)

print(x1) # 15.060054023189918
print(x2) # 20.93994597681008
# -

# The x-values for the lower tail and upper tail are 15.06 and 20.93, meaning if we are under or over, respectively, we reject the null hypothesis. 
#
# The sample mean value for the drug test group is 16, and 16 is not less than 15.06 nor greater than 20.9399. So like the one-tailed test, we still fail to reject the null hypothesis. 
#
# But what is the p-value? This is where it gets interesting with two-tailed tests. Our p-value is going to capture not just the area to the left of 16 but also the symmetrical equivalent area on the right tail. Since 16 is 4 days below the mean, we will also capture the area above 20, which is 4 days above the mean.

# +
# Calculating the two-tailed p-value

from scipy.stats import norm

# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5

# Probability of 16 or less days
p1 = norm.cdf(16, mean, std_dev)

# Probability of 20 or more days
p2 = 1.0 -  norm.cdf(20, mean, std_dev)

# P-value of both tails
p_value = p1 + p2
p_value
# -

# This is a lot greater than .05, so it definitely does not pass our p-value threshold of .05.
#
# If we are testing in an “equals 18” versus “not equals 18” capacity, we have to capture any probability that is of equal or less value on both sides. After all, we are trying to prove significance, and that includes anything that is equally or less likely to happen. We did not have this special consideration with the one-tailed test that used only “greater/less than” logic. But when we are dealing with “equals/not equals” our interest area goes in both directions.
#
# If our significance threshold is a p-value of .05 or less, our one-tailed test was closer to acceptance at p-value .0912 as opposed to the two-tailed test, which was about double that at p-value .182.
#
# This means the two-tailed test makes it harder to reject the null hypothesis and demands stronger evidence to pass a test. 

# ## The T-Distribution: Dealing with Small Samples
#
# Whether we are calculating confidence intervals or doing hypothesis testing, if we have 30 or fewer items in a sample we would opt to use a T-distribution instead of a normal distribution.
#
# The T-distribution is like a normal distribution but has fatter tails to reflect more variance and uncertainty.
#
# The smaller the sample size, the fatter the tails get in a T-distribution.

# +
# Getting a critical value range with a T-distribution

from scipy.stats import t

# get critical value range for 95% confidence
# with a sample size of 25

n = 25
lower = t.ppf(.025, df=n-1)
upper = t.ppf(.975, df=n-1)

print(lower, upper)
# -

# > Note that `df` is the “degrees of freedom” parameter, and as outlined earlier it should be one less of the sample size.

# ## Big Data Considerations and the Texas Sharpshooter Fallacy
#
# Let’s pretend I draw four playing cards from a fair deck. There’s no game or objective here other than to draw four cards and observe them. I get two 10s, a 3, and a 2. “This is interesting,” I say. “I got two 10s, a 3, and a 2. Is this meaningful? Are the next four cards I draw also going to be two consecutive numbers and a pair? What’s the underlying model here?”
#
# See what I did there? I took something that was completely random and I not only looked for patterns, but I tried to make a predictive model out of them. What has subtly happened here is I never made it my objective to get these four cards with these particular patterns. I observed them after they occurred.
#
# This is exactly what data mining falls victim to every day: finding coincidental patterns in random events.
#
# This is also analogous to me firing a gun at a wall. I then draw a target around the hole and bring my friends over to show off my amazing marksmanship. 
#
# Enter the *Texas Sharpshooter Fallacy*.
#
# The issue is this: the probability of a specific person winning the lottery is highly unlikely, but yet someone is going to win the lottery. Why should we be surprised when there is a winner?
#
# > This also applies to correlations.
#
# **So to prevent the Texas Sharpshooter Fallacy and falling victim to big data fallacies, try to use structured hypothesis testing and gather data for that objective. If you utilize data mining, try to obtain fresh data to see if your findings still hold up.**
#
# Finally, always consider the possibility that things can be coincidental; if there is not a commonsense explanation, then it probably was coincidental.

# ## Exercises

# 1. You bought a spool of 1.75 mm filament for your 3D printer. You want to measure how close the filament diameter really is to 1.75 mm. You use a caliper tool and sample the diameter five times on the spool:
#
# 1.78, 1.75, 1.72, 1.74, 1.77
#
# **Calculate the mean and standard deviation for this set of values.**

vals = [1.78, 1.75, 1.72, 1.74, 1.77]

mean = sum(vals) / len(vals)
mean


# +
def variance(values, is_sample: bool = False):
    mean = sum(values) / len(values)
    _variance = sum((v - mean) ** 2 for v in values) / (len(values) - (1 if is_sample else 0))

    return _variance


def std_dev(values, is_sample: bool = False):
    return sqrt(variance(values, is_sample))

std_dev(vals)
# -

# **Answer:**
#
# Mean: 1.752
# <br>
# Std: 0.02135415650406264

# 3. A manufacturer says the Z-Phone smart phone has a mean consumer life of 42 months with a standard deviation of 8 months. 
#
# **Assuming a normal distribution, what is the probability a given random Z-Phone will last between 20 and 30 months?**

# +
from scipy.stats import norm

mean = 42
std_dev = 8

norm.cdf(30, mean, std_dev) - norm.cdf(20, mean, std_dev)
# -

# **Answer:**
#
# 6.38%

# 3. I am skeptical that my 3D printer filament is not 1.75 mm in average diameter as advertised. I sampled 34 measurements with my tool. The sample mean is 1.715588 and the sample standard deviation is 0.029252.
#
# **What is the 99% confidence interval for the mean of my entire spool of filament?**

# +
from math import sqrt
from scipy.stats import norm


def critical_z_value(p):
    norm_dist = norm(loc=0.0, scale=1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)


def confidence_interval(p, sample_mean, sample_std, n):
    # Sample size must be greater than 30

    lower, upper = critical_z_value(p)
    lower_ci = lower * (sample_std / sqrt(n))
    upper_ci = upper * (sample_std / sqrt(n))

    return sample_mean + lower_ci, sample_mean + upper_ci

confidence_interval(p=.99, sample_mean=1.715588, sample_std=0.029252, n=34)
# -

# 4. Your marketing department has started a new advertising campaign and wants to know if it affected sales, which in the past averaged \\$10,345 a day with a standard deviation of \\$552. The new advertising campaign ran for 45 days and averaged \\$11,641 in sales.
#
# **Did the campaign affect sales? Why or why not? (Use a two-tailed test for more reliable significance.)**
#

# +
# Calculating a range for a statistical significance of 5%

from scipy.stats import norm

mean = 10345
std_dev = 552

# What x-value has 2.5% of area behind it?
x1 = norm.ppf(.025, mean, std_dev)

# What x-value has 97.5% of area behind it
x2 = norm.ppf(.975, mean, std_dev)

print(x1)
print(x2)
# -

# \\$11,641 is over \\$11,426.9 therefore the campaign was statistically significant.

# +
# Calculating the two-tailed p-value

from scipy.stats import norm

mean = 10345
std_dev = 552

p1 = 1.0 - norm.cdf(11641, mean, std_dev)

# Take advantage of symmetry
p2 = p1

# P-value of both tails
p_value = p1 + p2
p_value
