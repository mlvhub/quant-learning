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
#
# Size of a vector $r = \begin{bmatrix}r_1\\r_2\end{bmatrix}$:
#
# $$
# |r| = \sqrt{r_1\smash{^2} + r_2\smash{^2}}
# $$

s = np.array([1,3,4,2])
s

prod = (s*s).sum()
print(prod)
np.sqrt(prod)

# Answer is $\sqrt{30}$.

# ## 2
#
# Dot product:
#
# $$
# a⋅b=a_1b_1​+a_2b_2​+\dots+a_n​b_n
# $$

# +
r = np.array([-5, 3, 2, 8])
s = np.array([1, 2, -1, 0])

(r*s).sum()
# -

# ## 3
#
# Scalar projection of $s$ onto $r$:
#
# $$
# \frac{r \cdot s}{|r|}
# $$

# +
r = np.array([3, -4, 0])
s = np.array([10, 5, -6])

prod = (r*s).sum()
size_r = np.sqrt((r * r).sum())

prod / size_r
# -

# ## 4
#
# Vector projection of $s$ onto $r$:
#
# $$
# r\frac{r \cdot s}{|r||r|} = \frac{r \cdot s}{r \cdot r} r
# $$

# +
r = np.array([3, -4, 0])
s = np.array([10, 5, -6])

prod = (r*s).sum()
size_r = np.sqrt((r * r).sum())

r * (prod / size_r**2)
# -

# ## 5

a = np.array([3, 0, 4])
b = np.array([0, 5, 12])

ab = a+b
np.sqrt((ab * ab).sum())

np.sqrt((a * a).sum()) + np.sqrt((b * b).sum())

# ## 6

# +
r = np.array([3, -4, 0])
s = np.array([10, 5, -6])

(r*s).sum() == (s*r).sum()

# +
r = np.array([3, -4, 0])
s = np.array([10, 5, -6])

prod_r = (r*s).sum()
size_r = np.sqrt((r * r).sum())

prod_s = (s*r).sum()
size_s = np.sqrt((s * s).sum())

prod_r / size_r == prod_s / size_s
