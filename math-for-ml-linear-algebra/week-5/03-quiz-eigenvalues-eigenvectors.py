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

# +
from sympy import *

import numpy as np
np.set_printoptions(suppress=True)
# -

# ## 1

A = np.array([[4, -5, 6],
              [7, -8, 6],
              [3/2, -1/2, -2]])
vals, vecs = np.linalg.eig(A)
vecs

(2/6**(1/2))

# ## 2

L = np.array([[0,0,0,1],
              [1,0,0,0],
              [0,1,0,0],
              [0,0,1,0]])
vals, vecs = np.linalg.eig(L)
print(vals)
vecs

# ## 3

L_prime = np.array(
    [
        [0.1, 0.1, 0.1, 0.7],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ]
)
vals, vecs = np.linalg.eig(L_prime)
print(vals)
vecs

# ## 4

L = np.array([[0,1,0,0],
              [1,0,0,0],
              [0,0,0,1],
              [0,0,1,0]])
vals, vecs = np.linalg.eig(L)
print(vals)
vecs

# ## 5

# +

L_prime = np.array(
    [
        [0.1, 0.7, 0.1, 0.1],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.1, 0.7, 0.1],
    ]
)
vals, vecs = np.linalg.eig(L_prime)
print(vals)
vecs
# -

# ## 6

# +
l, a, b, c, d = symbols('l, a, b, c, d')

f = l**2 - (a + d)*l + (a*d - b*c)
f
# -

f_1 = f.subs({a: 3/2, b: -1, c: -1/2, d: 1/2})
f_1

# ## 7

solveset(f_1)

A = np.array([[3/2, -1],
              [-1/2, 1/2]])
vals, vecs = np.linalg.eig(A)
vals

3**(1/2)/2

# ## 8

# In Coursera's Jupyter:
#
# ```
# [[ 2.732  1.   ]
#  [-1.     1.366]]
# ```

vecs 

0.93/-0.34

0.59/0.80

-1-3**(1/2)

-1 + 3**(1/2)

# ## 9

C = np.array([
    [-1-3**(1/2), -1 + 3**(1/2)],
    [1, 1]
])
C

C_inv = np.linalg.inv(C)
C_inv

C_inv @ A @ C

1+3**(1/2)/2

1-3**(1/2)/2

# ## 10

D = np.array([
    [(-1-3**(1/2)/2)**2, 0],
    [0, (-1+3**(1/2)/2)**2]
])
D

C @ D @ C_inv
