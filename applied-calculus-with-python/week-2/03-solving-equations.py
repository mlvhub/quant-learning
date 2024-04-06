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

# # Solving Equations in SymPy

from sympy import *

x = symbols('x')

# define an equation
eq = Eq(5*x + 3, 1)
eq

# solve the equation
solveset(eq, x)

# multiple solutions
solveset(Eq(x**2, 1), x)

# ## Domains for solutions
#
# SymPy generally assumes that symbols are complex numbers.
#
# Exponential and logarithmic functions behave differently with complex numbers. For this reason, solving an exponential or logarithmic equation in SymPy can yield some strange-looking results.
#
# For example, the answer to $2^x = 8$ is clearly $x = 3$, but:

x = symbols('x')
solveset(Eq(2**x, 8), x)

# To keep this from happening, we can set the domain for solutions xx to be real numbers only:

x = symbols('x')
solveset(Eq(2**x, 8), x, domain = S.Reals)


