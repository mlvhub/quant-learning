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

# ## 1.

# +
x, y, z = symbols('x, y, z')

X = Matrix([x**2 * cos(y) + exp(z)*sin(y)])
Y = Matrix([x, y, z])
J = X.jacobian(Y)
J
# -

J.subs({x: pi, y: pi, z: 1})

# ## 2.

# +
x, y, z = symbols('x, y, z')

X = Matrix([x**2 * y - cos(x)*sin(y), exp(x+y)])
Y = Matrix([x, y])
J = X.jacobian(Y)
J
# -

J.subs({x: 0, y: pi})

# ## 3.

# +
x, y, z = symbols('x, y, z')

X = Matrix([x**3*cos(y) - x*sin(y)])
Y = Matrix([x, y])
H = X.jacobian(Y).jacobian(Y)
H
# -

# ## 4.

# +
x, y, z = symbols('x, y, z')

X = Matrix([x*y + sin(y)*sin(z) + z**3*exp(x)])
Y = Matrix([x, y, z])
H = X.jacobian(Y).jacobian(Y)
H
# -

# ## 5.

# +
x, y, z = symbols('x, y, z')

X = Matrix([x*y*cos(z) - sin(x)*exp(y)*x*z**3])
Y = Matrix([x, y, z])
H = X.jacobian(Y).jacobian(Y)
H
# -

H.subs({x: 0, y: 0, z: 0})


