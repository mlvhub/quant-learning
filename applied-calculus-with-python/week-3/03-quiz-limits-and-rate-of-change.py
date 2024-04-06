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

# ## 3

# +
t = symbols('t')

g = t/exp(t)

limit(g, t, S.Infinity)
# -

# ## 6

# +
x = symbols('x')

m = (2948-2530)/(x-26)
m.subs(x, 32)
# -

# ## 8

# +
x_y = {
    26: 2530,
    28: 2661,
    30: 2806,
    32: 2948,
    34: 3080,
}
pairs = ((26, 32), (28, 32), (30, 32), (32, 34))

[(x_y[x2] - x_y[x1])/(x2-x1)for (x1, x2) in pairs]
# -

(71+66)/2

# ## 9

# +
t = symbols('t')

c = 1.2*exp(-0.15*t)
c
# -

round((c.subs(t, 5)-c.subs(t, 0))/5-0, 4)

# ## 10

h = symbols('h')
f = (c.subs(t, (5 + h)) - c.subs(t, 5))/h
round(limit(f, h, 0).evalf(), 4)
