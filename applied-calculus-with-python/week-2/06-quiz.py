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
import numpy as np
import matplotlib.pyplot as plt

# ## 1

log(5) == ln(5)

# ## 2

x1 = np.linspace(1, 100)
x1

y1 = np.log10(x1)
y1

plt.plot(x1, y1)

y2 = np.emath.logn(0.9, x1)
y2

plt.plot(x1, y2)

log(1)

# ## 3

(1/2)*ln(exp(2/5))

# ## 4

# +
x, y, z = symbols('x y z', positive=True)

f4 = 2*ln(x) - ln(y+1) + 3*ln(z)
f4
# -

simplify(f4)

# ## 5

# +
x, y = symbols('x y', positive=True)

f5 = exp(2*ln(x) - ln(2*y))
f5
# -

simplify(f5)

# ## 6

# +
q = symbols('q')

eq = Eq(exp(2*q**2), exp(3*q + 2), domain=S.Reals)
eq
# -

solveset(eq)

# ## 7

# +
x = symbols('x')

eq = Eq(2*ln(x) - ln(2*x), 0)
eq
# -

solveset(eq, domain=S.Reals)

# ## 8

# +
k = symbols('k')

eq = Eq(2**(-12*k), 8)
eq
# -

solveset(eq, domain=S.Reals).evalf()

# ## 9

# +
t = symbols('t')

pt = 135120*exp(0.325*t)
pt
# -

pt.subs(t, 20).evalf()

# ## 10

# +
t = symbols('t')

eq = Eq(720*exp(0.03*t), 1400)
eq
# -

solveset(eq, domain=S.Reals).evalf()

# +
from sympy import *

x, y = symbols('x, y')

eq = Eq(6*x - 4*y, 14*x - 14*x)

solveset(eq, x)

# +
eq2 = Eq((4*y/3) - 2*y, 2)

solveset(eq2, y, domain=S.Reals)
