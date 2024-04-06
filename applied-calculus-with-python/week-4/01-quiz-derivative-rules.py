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

# ## 2

# +
t = symbols('t')

D = (1/(4*t)**3)+(16*t**2)/(4*t)**3
D
# -

diff(D, t)

# ## 3

# +
x = symbols('x')

y = 3*exp(x) + x
y
# -

dy = diff(y, x)
dy

m = dy.subs(x, 0)
m

yf = y.subs(x, 0)
yf

# The equation of the tangent line to the curve where $x = 0$ is $y = 4x + 3$.

# ## 4

# +
x = symbols('x')

J = 5*exp(x) + (1/sqrt(x**5))
J
# -

diff(J, x)

# ## 5

# +
x = symbols('x')

f = (2*x**3 + 3)*(4* - 3)
f
# -

diff(f)

# ## 7

diff(1/f)

# ## 9

# +
x = symbols('x')

f = (exp(2*x) + ln(2*x))
f
# -

diff(f, x)


