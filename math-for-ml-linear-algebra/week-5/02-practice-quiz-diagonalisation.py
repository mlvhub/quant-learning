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

# +
l, a, b, c, d = symbols('l, a, b, c, d')

f = l**2 - (a + d)*l + (a*d - b*c)
f
# -

# ## 1

T = np.array([
    [6, -1],
    [2, 3]
])
C = np.array([
    [1, 1],
    [1, 2]
])
C_inv = np.linalg.inv(C)
C_inv

C_inv @ T @ C

# ## 2

T = np.array([
    [2, 7],
    [0, -1]
])
C = np.array([
    [7, 1],
    [-3, 0]
])
C_inv = np.linalg.inv(C)
C_inv

C_inv @ T @ C

# ## 3

T = np.array([
    [1, 0],
    [2, -1]
])
C = np.array([
    [1, 0],
    [1, 1]
])
C_inv = np.linalg.inv(C)
C_inv

C_inv @ T @ C

# ## 4

# +
a = symbols('a')

D = np.array([
    [a, 0],
    [0, a]
])
C = np.array([
    [1, 2],
    [0, 1]
])
C_inv = np.linalg.inv(C)
C_inv
# -

C @ D @ C_inv

# ## 5

n = 3
D = np.array([
    [5**n, 0],
    [0, 4**n]
])
C = np.array([
    [1, 1],
    [1, 2]
])
C_inv = np.linalg.inv(C)
C_inv

C @ D @ C_inv

# ## 6

n = 3
D = np.array([
    [(-1)**n, 0],
    [0, 2**n]
])
C = np.array([
    [7, 1],
    [-3, 0]
])
C_inv = np.linalg.inv(C)
C_inv

C @ D @ C_inv

# ## 7

n = 5
D = np.array([
    [1**n, 0],
    [0, (-1)**n]
])
C = np.array([
    [1, 0],
    [1, 1]
])
C_inv = np.linalg.inv(C)
C_inv

C @ D @ C_inv


