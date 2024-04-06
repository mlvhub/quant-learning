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

# ## 2

# +
x = symbols('x')

f = 4*x - 3
f
# -

right_rect = [f.subs(x, i+1)*((i+1)-i) for i in range(2,5)]
RHS = sum(right_rect)
RHS

# ## 3

t = [ 0, 10, 20, 30, 40, 50, 60 ]
v = [ 81.8, 75.1, 47.6, 44.5, 55.7, 78.7, 78.5 ]
n = len(t) - 1

left_rect = [v[i]*(t[i+1]-t[i]) for i in range(0,n)]
LHS = sum(left_rect)
LHS

# ## 6

t = [ 0, 15, 30, 60 ]
f = [ 0, 360, 4800, 32000 ]
n = len(t) - 1

left_rect = [f[i]*(t[i+1]-t[i]) for i in range(0,n)]
LHS = sum(left_rect)
LHS

right_rect = [f[i+1]*(t[i+1]-t[i]) for i in range(0,n)]
RHS = sum(right_rect)
RHS

# ## 8

# +
x = symbols('x')

f = ln(x)
f
# -

integrate(f, (x, 1.3, 1.7))

# ## 9

g = 9 - x**2
g

solveset(g)

[(i, g.subs(x, i)) for i in range(1, 5)]

plot(g, (x, 1, 4))

(integrate(g, (x, 1, 3)) - integrate(g, (x, 3, 4))).evalf()

# ## 10

t = symbols('t')
v = 480*exp(0.06*t)
v

450 + integrate(v, (t, 0, 6))


