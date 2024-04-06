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

# # Coding Exercises (Part 2)

# Now, you will have the opportunity to practice what you have learned. <br>
# __Follow the instructions__ and insert your code! 

# The correct answer is provided below your coding cell. There you can check whether your code is correct.

# If you need some help or if you want to check your code, you can also have a look at the __solutions__.

# ### Have Fun!

# --------------------------------------------------------------------------------------------------------------

# ## Exercise 1: Descriptive Statistics

# 1. Import numpy and the S&P500 price returns (in %) for the year 2018 for the complete population (size = 500) as well as for a random sample (sample size = 50). 

# run the cell
import numpy as np
np.set_printoptions(precision=2, suppress= True)

# run the cell
pop = np.loadtxt("SP500_2018.csv", delimiter = ",", usecols = 1)

# run the cell
sample = np.loadtxt("sample_2018.csv", delimiter = ",", usecols = 1)



# 2. Double check the __size__ of pop and sample. 







# 3. Double check, whether all element of the sample are also in the population. 





# Calculate the following statistics for the population and the sample:

# 4. __mean__







# 5. __median__







# 6. __min__ 







# 7. __variance__







# 8. __standard deviation__







# 9. __25th__ and __75th percentile__







# 10. __skew__









# 11. __kurtosis__







# 12. Create a __histogram__ with __absolute frequencies__ for the __population__ (75 bins).







# # Well Done!

# ---------------------------------------------------------------------------------------------------------------------

# # Solutions (Stop here if you want to code on your own!)

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# 1. Import numpy and the S&P500 price returns (in %) for the year 2018 for the complete population (size = 500) as well as for a random sample (sample size = 50). 

# run the cell
import numpy as np
np.set_printoptions(precision=2, suppress= True)

# run the cell
pop = np.loadtxt("SP500_2018.csv", delimiter = ",", usecols = 1)

# run the cell
sample = np.loadtxt("sample_2018.csv", delimiter = ",", usecols = 1)



# 2. Double check the __size__ of pop and sample. 

pop.size

sample.mean()



# 3. Double check, whether all element of the sample are also in the population. 

np.isin(sample, pop)



# Calculate the following statistics for the population and the sample:

# 4. __mean__

pop.mean()

sample.mean()



# 5. __median__

np.median(pop)

np.median(sample)



# 6. __min__ 

pop.min()

sample.min()



# 7. __variance__

pop.var()

sample.var(ddof = 1)



# 8. __standard deviation__

pop.std()

sample.std(ddof = 1)



# 9. __25th__ and __75th percentile__

np.percentile(pop, [25, 75])

np.percentile(sample, [25, 75])



# 10. __skew__

import scipy.stats as stats

stats.skew(pop)

stats.skew(sample)



# 11. __kurtosis__

stats.kurtosis(pop)

stats.kurtosis(sample)



# 12. Create a __histogram__ with __absolute frequencies__ for the __population__ (75 bins).

import matplotlib.pyplot as plt

plt.figure(figsize = (12, 8))
plt.hist(pop, bins = 75)
plt.title("Absolute Frequencies - Population", fontsize = 20)
plt.xlabel("Stock Returns 2018 (in %)", fontsize = 15)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.xticks(np.arange(-100, 100, 10))
plt.show()




