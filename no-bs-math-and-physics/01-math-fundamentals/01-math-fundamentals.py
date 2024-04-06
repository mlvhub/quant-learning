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

# ## 1.17 Circles and polar coordinates

# ### Exercises

# #### E1.31

# a) (3,1)

# +
from sympy import *
from mpmath import *

def r(x, y):
    return (x**2 + y**2) ** (1/2)

def theta(x, y):
    return degrees(atan2(y, x))


# -

r(3, 1)

theta(3, 1)

# $$
# 3.16\angle{18.43}^\circ
# $$

# b) (-1, -2)

r(-1, -2)

360+theta(-1, -2)

# $$
# 2.24\angle{243.44}^\circ
# $$

# c) (0, -6)

r(0,-6)

360+theta(0,-6)


# $$
# 6\angle{270}^\circ
# $$

# ## 1.22 Compound interest
#
# ### Exercises

# #### E1.39
#
# a)

def monthly(apr, n, capital):
    monthly_rate = (1 + (apr/12)/100)**n
    return monthly_rate * capital


monthly(3, 120, 40000)

# b)

40000*(1.04)**10

# c)

# +
import math

math.exp(5/100)**10 *40000
# -

# #### E.140

first_apr=20000*(1.06)**5
second_apr=first_apr*(1.04)**5
second_apr


