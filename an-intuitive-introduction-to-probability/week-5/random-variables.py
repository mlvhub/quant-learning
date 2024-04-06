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

import scipy
import numpy as np

scipy.stats.pdf(63, 5, )

1-scipy.stats.norm.cdf(63, 65, 5)

1-scipy.stats.norm.cdf(65, 60, 5)

1-scipy.stats.norm.cdf(65, 58, 5)

1-scipy.stats.norm.cdf(1, 0.8, 0.1)

scipy.stats.norm.cdf(1, 0.7, 0.3)

scipy.stats.norm.cdf(1, 0.6, 0.5)

scipy.stats.norm.ppf(0.7881446014166034, loc=0.6, scale=0.5)

scipy.stats.norm.cdf(1, 0.8, 0.2) - scipy.stats.norm.cdf(0.7, 0.8, 0.2)

scipy.stats.norm.cdf(1, 0.7, 0.1) - scipy.stats.norm.cdf(0.9, 0.7, 0.1)

# +
mu = 2250
sigma = 110
x = 2300

1 - scipy.stats.norm.cdf(x, mu, sigma)
# -

scipy.stats.norm.cdf(2400, mu, sigma) - scipy.stats.norm.cdf(2200, mu, sigma)

scipy.stats.norm.ppf(0.9, mu, sigma)

print(scipy.stats.norm.ppf(0.05, mu, sigma))
print(scipy.stats.norm.ppf(0.95, mu, sigma))


