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

import numpy as np

# ## 5

A = np.array([
    [1, 0, -(4/13)/(-12/13)],
    [0, 1, -(-3/13)/(-12/13)]
])
A

r = np.array([
    [6],
    [2],
    [3]
])
r

A.dot(r)

# ## 6

R = np.array([
    [5, -1, -3, 7],
    [4, -4, 1, -2],
    [9, 3, 0, 12],
])
R

# +
np.set_printoptions(suppress=True)

A.dot(R)
