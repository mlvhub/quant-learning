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

from sympy import *

# ## Sample Problem 1 - Total Change from a Table

year = [1960, 1970, 1980, 1990, 2000, 2010]
rate = [36, 40, 63, 64, 10, 8]
n = 5

left_rect = [rate[i]*(year[i+1]-year[i]) for i in range(0,n)]
LHS = sum(left_rect)
LHS

# **a)** The total change between 1960 and 2010 was approximately 2,130,000 people.

# **b)** The function is neither increasing nor decreasing, therefore it cannot be determined whether the approximation is an overestimate or an underestimate.

# ## Sample Problem 2 - Interpreting the Definite Integral

# **a)** The definite integrate from $a$ to $b$ represents the total value the machine has lost (in thousands of dollars) in the time delta $b - a$. 

# **b)**

# +
t = symbols('t')

r = 24*exp(-0.22*t)
r
# -

integrate(r, (t, 0, 5))

200_000 - (integrate(r, (t, 0, 5)).evalf() * 1000)

# The value of the machine 5 years after it was purchased is $127222.30003979

# ## Sample Problem 3 - Integrals and Area

# +
x = symbols('x')

h = 16 - x**2
h
# -

# **a)**

integrate(h, (x, 0, 6))

# **b)**

solveset(h, x)

# +
# 4 is inside our interval [0, 6]

# area above x axis
above = integrate(h, (x, 0, 4))
# area below x axis
below = -integrate(h, (x, 4, 6))
# total area
total_area = above + below
total_area.evalf()
# -


