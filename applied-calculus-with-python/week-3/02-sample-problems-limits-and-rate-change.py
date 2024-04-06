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

# ### a)
#
# $$
# \text{slope} = \frac{y_2 - y_1}{x_2 - x_1}
# $$

slope = (6.84 - 3.40)/(1980 - 1970)
slope

3.4 + slope*10

# ### b)

# +
ref_x = 1980
ref_y = 6.84

x_y = [(1970, 3.40), (1975, 4.73), (1985, 8.73), (1990, 10.09), (1995, 11.64), (2000, 14.00)]
# -

[(ref_y - y)/(ref_x - x) for (x, y) in x_y]

# ## 2

# +
x = symbols('x')

T = 0.9*exp(0.7*x)
T
# -

# ### a)

m = (T.subs(x, 3) - T.subs(x, 0))/(3 - 0)
m

# ### b)
#
# Instantaneous rate of change:
#
# $$
# f'(c) = \lim_{⁡h \rightarrow 0} \frac{f(c+h)−f(c)}{h}
# $$

h = symbols('h')
f = (T.subs(x, (3 + h)) - T.subs(x, 3))/h
limit(f, h, 0).evalf()

# ### c)

limit(T, x, S.Infinity)

# The tumor will grow infinitely with time.
