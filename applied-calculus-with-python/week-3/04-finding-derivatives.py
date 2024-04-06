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

# +
# find the derivative of f(x) = x^2; f'(x) = 2x
# recall: f'(x) = lim h -> 0 (f(x+h) - f(x)) / h
x, h = symbols('x, h')
f = x**2

diff_quotient = (f.subs(x, x+h) - f)/h
diff_quotient.simplify()
# -

df = limit(diff_quotient, h, 0)
df

# find f'(a)
# find f'(0) of f(x) = x^2
df.subs(x, 0)

Derivative(x**2, x).doit()


def derivative_calculator(f, x):
    x = Symbol("x")
    d = Derivative(f, x).doit()
    return d


# +
f = x**3

derivative_calculator(f, "x")

# +
# simpler alternative to the first derivative
x = Symbol("x")

f = x**3
diff(f, x)
# -

# higher derivatives, option 1
diff(f, x, x)

diff(f, x, x, x)

# higher derivatives, option 2
diff(f, x, 2)

diff(f, x, 3)
