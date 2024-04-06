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

# # Sample Problems

from sympy import *

# ## Sample Problem 1 - Evaluating Logarithms

# **(a)** $ln (\frac{1}{e^3})$

ln(1/exp(3))

# **(b)** $e^{ln 2 - ln 4}$

exp(ln(2)-ln(4))

# ## Sample Problem 2 - Simplifying Exponential and Logarithmic Expressions
#
# Use the properties of logarithms to write the expression $ln⁡x−3ln⁡y+4ln⁡(z+1)$ as a single logarithm. You may use SymPy.

# +
x, y, z = symbols('x, y, z', positive=True)

f = ln(x)-3*ln(y)+4*ln(z+1)
f
# -

# can also use `logcombine`
simplify(f)

# ## Sample Problem 3 - Solving Exponential and Logarithmic Equations
#
# Solve each equation for $x$:

# **(a)** $e^{3x^2} = (e^4)^{4x + 3}$ 

# +
x = symbols('x')

eq = Eq(exp(3*x**2), exp(4)**(4*x+3))
eq
# -

solveset(eq, x, domain=S.Reals)

# **(b)** $ln⁡(x+1)−ln⁡(x)=1$

# +
x = symbols('x')

eq = Eq(ln(x+1)-ln(x), 1)
eq
# -

solveset(eq, x, domain=S.Reals)

# ## Sample Problem 4 - Finding an Exponential Function
#
# The function $g(x)$ has the form $g(x)=3^{kx}$, where $k$ is a constant. If $g(−4)=9$, then what is the value of $k$?
#
# $$
# g(-4) = 3^{-4k} = 9
# $$

# +
k = symbols('k')

eq = Eq(3**(-4*k), 9)
eq
# -

solveset(eq, k, domain=S.Reals)

# ## Sample Problem 5 - Predictions with an Exponential Function
#
# The population $P$ of Miami, Florida is given by the function $P(t)=362,000e^{0.01t}$, where $t$ is the number of years since 2000.

# **(a)** According to the function $P(t)$, what was the population of Miami in the year 2010?

# +
t = symbols('t')

f = 362000*exp(0.01*t)
f
# -

f.subs(t, 10)

# **(b)** According to the function $P(t)$, in what year will the population of Miami reach 450,000 people? 

eq = Eq(f, 450000)
eq

solveset(eq, t, domain=S.Reals).evalf()


