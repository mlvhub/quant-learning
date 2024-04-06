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
from sympy.plotting import plot

# ## 1

# +
x = symbols('x')

g = 3 * sqrt(x) + (1/x**2) - x
g
# -

integrate(g, (x, 1, 4)).evalf()

# ## 2

# +
t = symbols('t')

f = 58 - 0.2*t + 0.09*t**2 - 0.0005*t**3 - 0.0001*t**4
f
# -

1/24 * integrate(f, (t, 0, 24))

# ## 3

7.2 - 5.2

# **a)** $2 million

# **b)**

# +
t = symbols('t')

v_prime = 500000 - 25*t**2
v_prime
# -

7_200_000 + integrate(v_prime, (t, 6, 10)).evalf()


