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

# ## 1
#
# Find the equation of the tangent line to the graph of $q(x)$ at the point where $x=1$.
#
# $$
# q(x)= \frac{x+1​}{e^x}
# $$

# +
x = symbols('x')

q = (x+1)/exp(x)
q
# -

q1 = q.subs(x, 1)
q1

dq = diff(q, x)
dq

m = dq.subs(x, 1)
m

m*(x - 1) + q1

# ## 2
#
# An object's position at time $t≥0$ is given by $p(t)=5t^4−3t^2+2$.
#

# **a)** Find the velocity as a function of $t$.

# +
t = symbols('t')

p = 5*t**4 - 3*t**2 + 2
p
# -

dp = diff(p, t)
dp

# **b)** Find the acceleration as a function of $t$.

dp2 = diff(p, t, 2)
dp2

# ## 3
#
# **a)** A/ it costs $1700 to produce 150kg of the chemical.
#
# **b)** A/ it costs $4 to produce every kilo beyond $x$ (150).

# ## 4
#
# First is increasing and therefore positive, second is decreasing and therefore negative.


