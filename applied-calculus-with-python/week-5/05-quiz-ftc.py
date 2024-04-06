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
x = symbols('x')

integrate(2*x**4+sqrt(3)*x + 1)
# -

# ## 4

h = x**(1/2) + 1/x
h

integrate(h, (x, 4, 9)).evalf()

38/3 + ln(9/4)

# ## 5

f = exp(x)/(4 - exp(2*x))
f

integrate(f, x)

# ## 6

f = x**3/(1 + 2*x**2)**2
f

integrate(f, x)

# ## 7

x = symbols('x')
f = 3*x**2 + 2*x
f

1/5 * integrate(f, (x, 0, 5))

# ## 8

t = symbols('t')
P = 2560*exp(0.017*t)
P

1/50 * integrate(P, (t, 0, 50))

# ## 9

43750 - 28990

# ## 10

t = symbols('t')
N_prime = 4000 - 30*t**2
N_prime

43750 + integrate(N_prime, (t, 5, 6))


