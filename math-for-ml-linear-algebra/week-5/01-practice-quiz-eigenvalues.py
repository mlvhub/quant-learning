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
l, a, b, c, d = symbols('l, a, b, c, d')

f = l**2 - (a + d)*l + (a*d - b*c)
f
# -

# ## 1

f_1 = f.subs({a: 1, b: 0, c: 0, d: 2})
f_1

solveset(f_1)

# ## 3

f_2 = f.subs({a: 3, b: 4, c: 0, d: 5})
f_2

solveset(f_2)

# ## 5

f_5 = f.subs({a: 1, b: 0, c: -1, d: 4})
f_5

solveset(f_5)

# ## 7

f_7 = f.subs({a: -3, b: 8, c: 2, d: 3})
f_7

solveset(f_7)

# ## 9

f_9 = f.subs({a: 5, b: 4, c: -4, d: -3})
f_9

solveset(f_9)

# ## 10

f_10 = f.subs({a: -2, b: -3, c: 1, d: 1})
f_10

solveset(f_10, domain=S.Reals)


