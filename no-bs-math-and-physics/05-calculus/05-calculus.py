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

# # 5. Calculus

# ## E5.3

from sympy import *

x = symbols('x')
f = x**3 - 2*(x**2) + x
plot(f)

fp = 3*(x**2) - 4*x + 1
solve(fp, x)

f.subs(x, nsimplify(1/3))


