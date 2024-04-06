# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Coding Exercises (Part 2)

# Now, you will have the opportunity to practice what you have learned. <br>
# __Follow the instructions__ and insert your code! 

# The correct answer is provided below your coding cell. There you can check whether your code is correct.

# If you need some help or if you want to check your code, you can also have a look at the __solutions__.

# ### Have Fun!

# --------------------------------------------------------------------------------------------------------------

# ## Exercise 3: Sampling and Estimation

# 1. Import numpy and the S&P500 price returns (in %) for the year 2018 for the complete population (size = 500). 

# run the cell
import numpy as np
np.set_printoptions(precision=4, suppress= True)

# run the cell
pop = np.loadtxt("SP500_2018.csv", delimiter = ",", usecols = 1)
pop

# 2. Draw a __random sample__ with sample size __50__ and __save__ the sample. Use the random __seed 123__.









# ++++ __From this point, assume that you only have the sample. You don´t know anything about the population.__++++ 

# 3. Calculate and save the __sample mean__ (point estimate of the mean).



# (result: -3.8301530353346083)



# 4. Calculate and save the __sample standard deviation__ (point estimate of the std).



# (result: 24.56795822088298)



# 5. Calculate the __Standard Error__ of the sample mean.



# (result: 3.4744339715788284)



# 6. Estimate the (__equally-weighted__) mean return for the whole S&P 500 population for the year 2018 by constructing a __90% Confidence Interval__.







# (result: (-9.655218409282309, 1.9949123386130907))



# 7. __Same as Q6__. This time, construct the 90% Confidence Interval with __Bootstrapping__. Use __100,000 simulations__ and the random __seed 123__.









# (result: array([-9.404 ,  1.9449]))



# # Well Done!

# ---------------------------------------------------------------------------------------------------------------------

# # Solutions (Stop here if you want to code on your own!)

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Import numpy and the S&P500 price returns (in %) for the year 2018 for the complete population (size = 500). 

# run the cell
import numpy as np
np.set_printoptions(precision=4, suppress= True)

# run the cell
pop = np.loadtxt("SP500_2018.csv", delimiter = ",", usecols = 1)
pop

# 2. Draw a __random sample__ with sample size __50__ and __save__ the sample. Use the random __seed 123__.

sample_size = 50

np.random.seed(123)
sample = np.random.choice(pop, sample_size, replace = False)

sample



# ++++ __From this point, assume that you only have the sample. You don´t know anything about the population.__++++ 

# 3. Calculate and save the __sample mean__ (point estimate of the mean).

point_est_mean = sample.mean()
point_est_mean

# (result: -3.8301530353346083)



# 4. Calculate and save the __sample standard deviation__ (point estimate of the std).

point_est_std = sample.std(ddof = 1)
point_est_std

# (result: 24.56795822088298)



# 5. Calculate the __Standard Error__ of the sample mean.

standard_error = point_est_std / np.sqrt(sample_size)
standard_error

# (result: 3.4744339715788284)



# 6. Estimate the (__equally-weighted__) mean return for the whole S&P 500 population for the year 2018 by constructing a __90% Confidence Interval__.

import scipy.stats as stats

conf = 0.9

stats.t.interval(conf, loc = point_est_mean, scale = standard_error, df = sample_size - 1)

# (result: (-9.655218409282309, 1.9949123386130907))



# 7. __Same as Q6__. This time, construct the 90% Confidence Interval with __Bootstrapping__. Use __100,000 simulations__ and the random __seed 123__.

sims = 100000

np.random.seed(123)
bootstrap = []
for i in range(sims):
    bootstrap.append(np.random.choice(sample, size = sample_size, replace = True).mean())

bootstrap

np.percentile(bootstrap, [5, 95])

# (result: array([-9.404 ,  1.9449]))


