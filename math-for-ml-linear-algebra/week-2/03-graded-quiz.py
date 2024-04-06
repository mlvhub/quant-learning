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
v = np.array([1, 2])
c = np.array([1, 1])

prod = (v*c).sum()
size_c = np.sqrt((c * c).sum())

c * (prod / size_c**2)
# -

# ## 2

# +
v = np.array([2, 1])
c = np.array([3, -4])

prod = (v*c).sum()
size_c = np.sqrt((c * c).sum())

v_c = c * (prod / size_c**2)

np.sqrt((v_c * v_c).sum())


# -

# ## 3

def change_basis(v, bn):
    vn = []
    for b in bn:
        v_b = (v*b).sum() / (b*b).sum()
        #v_b = v_b if v_b.is_integer() else Fraction(str(v_b))
        vn.append(v_b)
    return np.array(vn)


v = np.array([-4, -3, 8])
b1 = np.array([1, 2, 3])
b2 = np.array([-2, 1, 0])
b3 = np.array([-3, -6, 5])

change_basis(v, [b1, b2, b3])

# ## 5

# +
p = np.array([3, 2, 4])
v = np.array([-1, 2, -3])

p + (2*v)
