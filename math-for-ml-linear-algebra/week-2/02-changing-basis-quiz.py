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
from fractions import Fraction


def change_basis(v, b1, b2):
    v_b1 = (v*b1).sum() / (b1*b1).sum()
    v_b2 = (v*b2).sum() / (b2*b2).sum()
    v_b1 = v_b1 if v_b1.is_integer() else Fraction(str(v_b1))
    v_b2 = v_b2 if v_b2.is_integer() else Fraction(str(v_b2))
    return np.array([v_b1, v_b2])


def change_basis(v, bn):
    vn = []
    for b in bn:
        v_b = (v*b).sum() / (b*b).sum()
        #v_b = v_b if v_b.is_integer() else Fraction(str(v_b))
        vn.append(v_b)
    return np.array(vn)


# ## 1

v = np.array([5, -1])
b1 = np.array([1, 1])
b2 = np.array([1, -1])

(v*b1).sum() / (b1*b1).sum()

(v*b2).sum() / (b2*b2).sum()

change_basis(v, b1, b2)

# Answer is $v_b = \begin{bmatrix}2\\3\end{bmatrix}$.

# ## 2

v = np.array([10, -5])
b1 = np.array([3, 4])
b2 = np.array([4, -3])

change_basis(v, b1, b2)

# Answer is $v_b = \begin{bmatrix}2/5\\11/5\end{bmatrix}$.

# ## 3

v = np.array([2, 2])
b1 = np.array([-3, 1])
b2 = np.array([1, 3])

change_basis(v, b1, b2)

# ## 4

v = np.array([1, 1, 1])
b1 = np.array([2, 1, 0])
b2 = np.array([1, -2, -1])
b3 = np.array([-1, 2, -5])

change_basis(v, [b1, b2, b3])

# Answer is $v_b = \begin{bmatrix}3/5\\-1/3\\-2/15\end{bmatrix}$

# ## 5

v = np.array([1, 1, 2, 3])
b1 = np.array([1, 0, 0, 0])
b2 = np.array([0, 2, -1, 0])
b3 = np.array([0, 1, 2, 0])
b4 = np.array([0, 0, 0, 3])

change_basis(v, [b1, b2, b3, b4])


