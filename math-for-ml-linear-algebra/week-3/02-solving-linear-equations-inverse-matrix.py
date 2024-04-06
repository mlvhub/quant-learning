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

# ## 2

old_two = np.array([3, 4, 1])

old_one = np.array([1, 3/2, 1/2])

np.divide(old_two, 4 - old_one)

new_two = (old_two - 3*old_one) * -2
new_two

# ## 3

old_three = np.array([2, 8, 13])

old_three - 4*(old_one)

2-(2*9/4)

-(1/4) - 5*(-1/2)

(9/4)*(1/7)

# ## 7

# +
A = np.array([
    [1, 1, 1],
    [3, 2, 1],
    [2, 1, 2],
])

np.linalg.inv(A)
# -

A.dot(np.linalg.inv(A))

A_inv = np.array([
    [-1, 0, 1],
    [1, 1, -2],
    [1, -1, 1],
])
A.dot(A_inv)

# ## 8

# +
import numpy as np

A = [[1, 1, 3],
     [1, 2, 4],
     [1, 1, 2]]
Ainv = np.linalg.inv(A)
Ainv

# +
import numpy as np
A = [[4, 6, 2],
     [3, 4, 1],
     [2, 8, 13]]

s = [s1, s2, s3]

r = np.linalg.solve(A, s)
