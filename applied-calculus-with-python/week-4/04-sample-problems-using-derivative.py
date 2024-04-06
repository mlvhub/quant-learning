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

# # Sample Problems

# ### Sample Problem 1 - First and Second Derivative Tests. 
#
# Let $f(x)=ln⁡(x^2+x+1)$

# +
x = symbols('x')

f = ln(x**2 + x + 1)
f

# +
# a)

f_prime = diff(f, x)
f_prime
# -

solveset(f_prime, x)

# +
# b)

sample = [-1, 0]
sample_values = [f_prime.subs(x, i) for i in sample]
sample_values
# -

# The function is decreasing on $(−∞,−\frac{1}{2})$ and increasing on $(-\frac{1}{2},∞)$.

# +
# c)

f_prime2 = diff(f, x, 2)
f_prime2
# -

critical_points = solve(f_prime2, x)
for p in critical_points:
    print(p.evalf())
critical_points

sample = [-5, 0, 5]
sample_values = [f_prime2.subs(x, i) for i in sample]
sample_values

# The graph of the function is concave up $(-\frac{1}{2}-\frac{\sqrt{3}}{2}, -\frac{1}{2}+\frac{\sqrt{3}}{2})$ and $f$ is concave down on $(−∞,-\frac{1}{2}-\frac{\sqrt{3}}{2})$ and $(-\frac{1}{2}+\frac{\sqrt{3}}{2},∞)$.

# **d)**
#
# From **b)**, we see the function changes from decreasing to increasing at $x=−\frac{1}{2}$, making it a **local minimum**.

# ## Sample Problem 2 - Critical Points vs. Inflection Points
#
# N(7)' > 0
# N(7)'' > 0
#
# N(9)' > 0
# N(9)'' < 0
#
# N(11)' < 0
# N(11)'' = 0
#
# Since the first derivative changes sign after 9, $t = 9$ is a **critical point**.
#
# Since the second derivative changes sign after 7, $t = 7$ is an **inflection point**.

# ## Sample Problem 3 - Profit, Revenue, and Cost
#
# A company’s revenue from selling $x$ items is given by the function $R(x)=360x−x**2$. The cost associated with selling $x$ items is given by $C(x)=12x+64$. 

# +
x = symbols('x')

R = 360*x - x**2
R
# -

C = 12*x + 64
C

P = R - C
P

P_prime = diff(P, x)
P_prime

solveset(P_prime, x)

sample = [173, 174, 175]
sample_values = [P_prime.subs(x, i) for i in sample]
sample_values

# To maximise profits, they need to sell 174 items.
