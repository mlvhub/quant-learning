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

# ## 3
#
# Point slope form:
#
# $$
# y - y_0 = m(x - x_0)
# $$
#
# $$
# p(15) = 100
# \\
# p'(1.5) = 6.2
# \\
# $$
#
# $$
# y - 100 = 6.2x - 15
# \\
# y = 6.2x - 115
# $$

# ## 4

# +
x = symbols('x')

diff(2*exp(-x) -x, x)
# -

# ## 5
#
# Find the equation of the tangent line to the graph of $f(x)=2x^2−3$ at the point where $x=1$.
#

# +
x = symbols('x')

f = 2*x**2 - 3
f
# -

f1 = f.subs(x, 1)
f1

df = diff(f, x)
df

m = df.subs(x, 1)
m

m*(x - 1) + f1

# ## 8
#
# An object’s position at time $t≥0t≥0$  is given by $p(t)=t^3−2t$. Use SymPy to find a formula for the acceleration of the object as a function of tt.

# +
t = symbols('t')

p = t**3 - 2*t
p
# -

diff(p, t, 2)
