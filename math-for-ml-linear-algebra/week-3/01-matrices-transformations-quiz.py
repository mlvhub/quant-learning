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

# ## 1

# +
r = np.array([3, 2])
A = np.array([
    [1/2, -1],
    [0, 3/4]
])

A.dot(r)
# -

# ## 2

# +
s = np.array([-2, 4])
A = np.array([
    [1/2, -1],
    [0, 3/4]
])

A.dot(s)
# -

# ## 3

M = np.array([
    [-1/2, 1/2],
    [1/2, 1/2]
])

M.dot(np.array([1, 0]))

M.dot(np.array([0, 1]))

# ## 4

matrices = [
    [
        [np.sqrt(3/2), np.sqrt(3/2)],
        [1/2, 1/2]
    ],
    [
        [-1/2, 0],
        [0, np.sqrt(3/2)]
    ],
    [
        [np.sqrt(3/2), -1/2],
        [1/2, np.sqrt(3/2)]
    ],
    [
        [1/2, 0],
        [-np.sqrt(3/2), 1/2]
    ],
]

for m in matrices:
    print(np.array(m).dot(np.array([1, 0])))

for m in matrices:
    print(np.array(m).dot(np.array([0, 1])))

# ## 5

# +
M1 = np.array([
    [1, 0],
    [0, 8]
])
M2 = np.array([
    [1, 0],
    [-1/2, 1]
])

M1.dot(M2)
