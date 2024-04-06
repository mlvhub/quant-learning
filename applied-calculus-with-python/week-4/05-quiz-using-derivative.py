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
from sympy.plotting import plot 

x = symbols('x')

# ## 1

f = x**4 - 6*x**2 + 8*x
f

f_prime = diff(f, x)
f_prime

solveset(f_prime, x)

# ## 2

h = exp(x)/(x**2 + 1)
h

h_prime = diff(h, x)
h_prime

sample = [0, 0.5, 0.9, 1, 1.1, 2, 3]
sample_values = [h_prime.subs(x, i).evalf() for i in sample]
sample_values

# ## 3

f = exp(-x**2)
f

f_prime = diff(f, x)
f_prime

solveset(f_prime, x)

sample = [exp(1), 1, 0, 1/exp(1)]
sample_values = [f_prime.subs(x, i).evalf() for i in sample]
sample_values

f.subs(x, 0)

# ## 4
#
# **Interval [−1,2]**

k = x**3 - 3*x**2
k

plot(k, (x, -1, 2))

k_prime = diff(k, x)
k_prime

solveset(k_prime, x)

sample = [-1, 0, 2]
sample_values = [k.subs(x, i).evalf() for i in sample]
sample_values

# ## 5

k_prime2 = diff(k, x, 2)
k_prime2

solveset(k_prime2, x)

sample = [-1, 1, 2]
sample_values = [k_prime2.subs(x, i).evalf() for i in sample]
sample_values

# ## 6
#
# T(6)' > 0
# T(6)'' > 0
#
# T(24)' > 0
# T(24)'' < 0

# ## 7
#
# T(6)' > 0
# T(6)'' > 0
#
# T(14)' > 0
# T(14)'' < 0
#
# T(32)' < 0
# T(32)'' = 0

# ## 8

R = 240*x - x**2
R

C = 30*x + 72
C

P = R - C
P

P_prime = diff(P, x)
P_prime

solveset(P_prime, x)

sample = [104, 105, 106]
sample_values = [P_prime.subs(x, i) for i in sample]
sample_values

# ## 9
#
# $$
# P = l + 2s = 160
# \\
# l = 160 - 2s
# \\
# A = l*s
# \\
# A = s(160 - 2s)
# \\
# A = 160s - 2s^2
# $$

A = 160*x - 2*x**2
A

A_prime = diff(A, x)
A_prime

solveset(A_prime)

80*40

# ## 10

I = 192*ln(x/1041) - x + 1042
I

I_prime = diff(I, x)
I_prime

y = solve(I_prime, x, domain=S.Reals)
y

xy = [(i, I.subs(x, i).evalf()) for i in range(1, 1043)]
max(xy, key=lambda item: item[1])
