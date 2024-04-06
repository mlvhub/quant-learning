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

from sympy.plotting import *
from sympy import Symbol

x = Symbol("x")

plot(x**2)

plot1 = plot(x**2, (x, -3, 3))
plot1[0].line_color ='r'

f1 = 3*x+2
f2 = x**3
plot(f1, f2)

# piecewise function
plot((f1, (x, -10, 0)), (f2, (x, 0, 5)))
