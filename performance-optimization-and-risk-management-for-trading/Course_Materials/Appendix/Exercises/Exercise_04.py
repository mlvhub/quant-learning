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

# ## Exercise 4: Hypothesis Testing

# You have a random sample (drawn from S&P 500) with 50 stocks/companies and their annual returns for the year 2018 (__sample size = 50__)

# 1. Import the sample. 

# run the cell
import numpy as np
import scipy.stats as stats
np.set_printoptions(precision=4, suppress= True)

# run the cell
sample = np.loadtxt("sample_2018.csv", delimiter = ",", usecols = 1)
sample



# 2. Test on a __10% level of significance__, whether the (equally-weighted) mean return for the whole S&P 500 population for the year 2018 is __equal to 0%__ or not. Formulate H0 and Ha and calculate the __t-statistic__ and the __p-value__ of your test.

#





# (result: Ttest_1sampResult(statistic=-1.1023818747645209, pvalue=0.27568087004548397)



# 3. Can you __reject H0__ (and conclude that the true mean is significantly different from 0%)?









# 4. Test on a __5% level of significance__, whether the (equally-weighted) mean return for the whole S&P 500 population for the year 2018 is __equal to or greater than 5%__ (H0). Can you conclude that the mean return is lower than 5% (Ha)?

#run the cell!
H0 = 5









# (result: pvalue = 0.007126848805366535, reject H0)



# 5. Import the full __S&P 500 population__ (annual returns) for the year 2018.

# run the cell
pop = np.loadtxt("SP500_2018.csv", delimiter = ",", usecols = 1)
pop



# 6. Calculate the population´s __skew__ and __kurtosis__.







# 7. Test for __skew__ on a __5% significance__ level. 



# (results: SkewtestResult(statistic=3.700813132575524, pvalue=0.00021490969981373073))



# 8. Test for __kurtosis__ on a __5% significance__ level.



# (results: KurtosistestResult(statistic=2.5240612693439624, pvalue=0.011600770790237903))



# 9. Test for __Normality__ on a __5% significance__ level!



# (results: NormaltestResult(statistic=20.066903133645717, pvalue=4.390635151897616e-05))



# # Well Done!

# ---------------------------------------------------------------------------------------------------------------------

# # Solutions (Stop here if you want to code on your own!)

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# You have a random sample (drawn from S&P 500) with 50 stocks/companies and their annual returns for the year 2018 (__sample size = 50__)

# 1. Import the sample. 

# run the cell
import numpy as np
import scipy.stats as stats
np.set_printoptions(precision=4, suppress= True)

# run the cell
sample = np.loadtxt("sample_2018.csv", delimiter = ",", usecols = 1)
sample



# 2. Test on a __10% level of significance__, whether the (equally-weighted) mean return for the whole S&P 500 population for the year 2018 is __equal to 0%__ or not. Formulate H0 and Ha and calculate the __t-statistic__ and the __p-value__ of your test.

# __H0:__ mean return == 0% <br> 
# __Ha:__ mean return != 0%

H0 = 0

stats.ttest_1samp(sample, H0)

# (result: Ttest_1sampResult(statistic=-1.1023818747645209, pvalue=0.27568087004548397)



# 3. Can you __reject H0__ (and conclude that the true mean is significantly different from 0%)?

conf = 0.9

t_stat, p_value = stats.ttest_1samp(sample, H0)

p_value < 1-conf



# 4. Test on a __5% level of significance__, whether the (equally-weighted) mean return for the whole S&P 500 population for the year 2018 is __equal to or greater than 5%__ (H0). Can you conclude that the mean return is lower than 5% (Ha)?

#run the cell!
H0 = 5

t_stat, p_value = stats.ttest_1samp(sample, H0)

t_stat

p_value = p_value / 2

p_value

# (result: pvalue = 0.007126848805366535, reject H0)



# 5. Import the full __S&P 500 population__ (annual returns) for the year 2018.

# run the cell
pop = np.loadtxt("SP500_2018.csv", delimiter = ",", usecols = 1)
pop



# 6. Calculate the population´s __skew__ and __kurtosis__.

stats.skew(pop)

stats.kurtosis(pop)



# 7. Test for __skew__ on a __5% significance__ level. 

stats.skewtest(pop)

# (results: SkewtestResult(statistic=3.700813132575524, pvalue=0.00021490969981373073))



# 8. Test for __kurtosis__ on a __5% significance__ level.

stats.kurtosistest(pop)

# (results: KurtosistestResult(statistic=2.5240612693439624, pvalue=0.011600770790237903))



# 9. Test for __Normality__ on a __5% significance__ level!

stats.normaltest(pop)

# (results: NormaltestResult(statistic=20.066903133645717, pvalue=4.390635151897616e-05))
