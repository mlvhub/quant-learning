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

# ## Exercise 2: Probability Distributions

# run the cell
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. __Simulate__ playing Roulette __10 times__ (equally likely outcomes from __0 to 36__). Use a __random seed of 123__. <br>
# What is the value of the very first outcome?



# (result: 2)



# 2. __Simulate__ playing Roulette __1 million times__ (equally likely outcomes from __0 until 36__). Use a __random seed of 123__. __Save__ the results. <br>







# 3. Create a __histogram__ with __absolute frequencies__ for the outcome of question 2. Select an __appropriate number of bins__. 





# 4. Create a __histogram__ with __relative frequencies__ for the outcome of question 2. Select an __appropriate number of bins__. 





# 5. Let´s assume adult male heights in the US are on __average 70 inches__ with a __standard deviation of 4 inches__. Create an array with __1 million (random) observations__. Use a random __seed of 123__. __Save__ the results.





# 6. Create a __histogram__ with __absolute frequencies__ for the outcome of question 5. Use __1000 bins__. 





# 7. Adult women in the US are on average a bit shorter and less variable in height with a __mean height of 65 inches__ and __standard deviation of 3.5 inches__. <br>
# Construct a __90% Confidence Interval__ around the mean. 





# result: (59.24301230566984, 70.75698769433015)



# 8. __1%__ of all women in the US are __smaller than ... inches__? Calculate!



# (result: 56.85778244085706)



# 9. __2%__ of all women in the US are __taller than ... inches__? Calculate!



# (result: 72.18812118721138)



# 10. X percent of all women in the US are __taller than 70 inches__. Calculate!



# (result: 0.07656372550983481)



# 11. Mary is __62.4 inches__ tall. How many standard deviations is she away from the mean (__z-score__)?



# (result: -0.7428571428571432)

# # Well Done!

# ---------------------------------------------------------------------------------------------------------------------

# # Solutions (Stop here if you want to code on your own!)

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# run the cell
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. __Simulate__ playing Roulette __10 times__ (equally likely outcomes from __0 to 36__). Use a __random seed of 123__. <br>
# What is the value of the very first outcome?

np.random.seed(123)
np.random.randint(0, 37, 10)

# (result: 2)



# 2. __Simulate__ playing Roulette __1 million times__ (equally likely outcomes from __0 until 36__). Use a __random seed of 123__. __Save__ the results. <br>

np.random.seed(123)
a = np.random.randint(0, 37, 1000000)

a



# 3. Create a __histogram__ with __absolute frequencies__ for the outcome of question 2. Select an __appropriate number of bins__. 

plt.figure(figsize = (12, 8))
plt.hist(a, bins = 37, ec = "black")
plt.title("Discrete Uniform Distribution", fontsize = 20)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.show()



# 4. Create a __histogram__ with __relative frequencies__ for the outcome of question 2. Select an __appropriate number of bins__. 

plt.figure(figsize = (12, 8))
plt.hist(a, bins = 37, weights = np.ones(len(a)) / len(a), ec = "black")
plt.title("Discrete Uniform Distribution", fontsize = 20)
plt.ylabel("Relative Frequency", fontsize = 15)
plt.show()



# 5. Let´s assume adult male heights in the US are on __average 70 inches__ with a __standard deviation of 4 inches__. Create an array with __1 million (random) observations__. Use a random __seed of 123__. __Save__ the results.

np.random.seed(123)
a = np.random.normal(loc = 70, scale = 4, size = 1000000)



# 6. Create a __histogram__ with __absolute frequencies__ for the outcome of question 5. Use __1000 bins__. 

plt.figure(figsize = (12, 8))
plt.hist(a, bins = 1000,)
plt.title("Normal Distribution", fontsize = 20)
plt.ylabel("Absolute Frequency", fontsize = 15)
plt.show()



# 7. Adult women in the US are on average a bit shorter and less variable in height with a __mean height of 65 inches__ and __standard deviation of 3.5 inches__. <br>
# Construct a __90% Confidence Interval__ around the mean. 

mu = 65
sigma = 3.5

stats.norm.interval(0.9, loc = mu, scale = sigma)

# result: (59.24301230566984, 70.75698769433015)



# 8. __1%__ of all women in the US are __smaller than ... inches__? Calculate!

stats.norm.ppf(0.01, loc = mu, scale = sigma)

# (result: 56.85778244085706)



# 9. __2%__ of all women in the US are __taller than ... inches__? Calculate!

stats.norm.ppf(0.98, loc = mu, scale = sigma)

# (result: 72.18812118721138)



# 10. X percent of all women in the US are __taller than 70 inches__. Calculate!

1 - stats.norm.cdf(70, loc = mu, scale = sigma)

# (result: 0.07656372550983481)



# 11. Mary is __62.4 inches__ tall. How many standard deviations is she away from the mean (__z-score__)?

(62.4 - mu) / sigma 


