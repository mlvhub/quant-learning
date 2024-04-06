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

# ## 3.

# +
z1 = symbols('z1')

a1 = tanh(z1)

a1.diff(z1)

# +
a, b, c, d = symbols('a, b, c, d')


X = Matrix(
    [a, b],
    [c, d],
)
X.det()
# -


