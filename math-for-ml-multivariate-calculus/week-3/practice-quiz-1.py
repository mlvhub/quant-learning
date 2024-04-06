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
x1, x2 = symbols('x1, x2')

fx = x1**2*x2**2 + x1*x2


X = Matrix([fx])
Y = Matrix([x1, x2])
J = X.jacobian(Y)
J

# +
t = symbols('t')

x1 = 1 - t**2
x2 = 1 + t**2

X = Matrix([x1, x2])
Y = Matrix([t])
J = X.jacobian(Y)
J
# -

# ## 2.

# +
x1, x2, x3 = symbols('x1, x2, x3')

fx = x1**3*cos(x2)*exp(x3)


X = Matrix([fx])
Y = Matrix([x1, x2, x3])
J = X.jacobian(Y)
J

# +
t = symbols('t')

x1 = 2*t
x2 = 1 - t**2
x3 = exp(t)

X = Matrix([x1, x2, x3])
Y = Matrix([t])
J = X.jacobian(Y)
J
# -

# ## 3.

# +
x1, x2 = symbols('x1, x2')

fx = x1**2 - x2**2


X = Matrix([fx])
Y = Matrix([x1, x2])
J = X.jacobian(Y)
J

# +
u1, u2 = symbols('u1, u2')

x1 = 2*u1**2 + 3*u2**2 - u2
x2 = 2*u1 - 5*u2**3

X = Matrix([x1, x2])
Y = Matrix([u1, u2])
J = X.jacobian(Y)
J

# +
t = symbols('t')

u1 = exp(t/2)
u2 = exp(-2*t)

X = Matrix([u1, u2])
Y = Matrix([t])
J = X.jacobian(Y)
J
# -

# ## 5.

# +
x1, x2, x3 = symbols('x1, x2, x3')

fx = sin(x1)*cos(x2)*exp(x3)


X = Matrix([fx])
Y = Matrix([x1, x2, x3])
J = X.jacobian(Y)
J

# +
u1, u2 = symbols('u1, u2')

x1 = sin(u1) + cos(u2)
x2 = cos(u1) - sin(u2)
x3 = exp(u1 + u2)

X = Matrix([x1, x2, x3])
Y = Matrix([u1, u2])
J = X.jacobian(Y)
J

# +
t = symbols('t')

u1 = 1+t/2
u2 = 1-t/2

X = Matrix([u1, u2])
Y = Matrix([t])
J = X.jacobian(Y)
J
