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

x = Symbol("x")

f = ((1+x)**0.5 - (1-x)**0.5) / x
f

# limit as x -> 0
plot(f, (x, -2, 2))

# manual way to calculate limit
xval = [-.1, -.01, -.001, -.0001, .0001, .001, .01, .01]

yval = [f.subs(x, i) for i in xval]
yval

rounded_yval = [round(i, 4) for i in yval]
rounded_yval

# exact way to calculate limits
limit(f, x, 0)

# one-sided limits
limit(f, x, 0, "-")

limit(f, x, 0, "+")

# limits with infinity
limit(1/x, x, S.Infinity)


