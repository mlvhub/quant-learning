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

# # Chapter 3. Vectors, Part 2

import numpy as np

# ## Vector Sets
#
# A *set* is a collection of vectors. We can describe them mathematically as following:
#
# $$
# V = {v_1, \dots, v_n}
# $$
#
# Vector sets can contain a finite or an infinite number of vectors. Vector sets with an infinite number of vectors may sound like a uselessly silly abstraction, but vector subspaces are infinite vector sets and have major implications for fitting statistical models to data.
#
# Vector sets can also be empty, and are indicated as $V = {}$.
#

# ## Linear Weighted Combination
#
# A *linear weighted combination* (or *linear mixture*, or *weighted combination*) is a way of mixing information from multiple variables, with some variables contributing more than others. Sometimes the term *coefficient* is used instead of weight.
#
# Linear weighted combination simply means scalar-vector multiplication and addition: take some set of vectors, multiply each vector by a scalar, and add them to produce a single vector.
#
# It's given by
#
# $$
# w = \lambda_1 v_1 + \lambda_2 v_2 + \dots + \lambda_n v_N
# \tag{3-1}
# $$
#
# It's assumed all vectors $v_i$ have the same dimensionality, otherwise the adition is invalid. The $\lambda$s can be any real number, including zero.
#
# Example for equation (3-1):
#
# $$
# \lambda_1 = 1, \lambda_2 = 2, \lambda_3 = -3, \space \space 
# v_1 = \begin{bmatrix} 4\\5\\1 \end{bmatrix},
# v_2 = \begin{bmatrix} -4\\0\\-4 \end{bmatrix},
# v_3 = \begin{bmatrix} 1\\3\\2 \end{bmatrix}
# $$
#
# $$
# w = \lambda_1 v_1 + \lambda_2 v_2 +\lambda_3 v_3 =
# \begin{bmatrix}
#     -7\\-4\\-13
# \end{bmatrix}
# $$

l1 = 1
l2 = 2
l3 = -3
v1 = np.array([4,5,1])
v2 = np.array([-4,0,-4])
v3 = np.array([1,3,2])
l1*v1 + l2*v2 + l3*v3

# Storing each vector and each coefficient as separate variables is tedious and does not scale up to larger problems. Therefore, in practice, linear weighted combinations are implemented via the compact and scalable matrix-vector multiplication method.

# ## Linear Independence
#
# A set of vectors is *linearly dependent* if at least one vector in the set can be expressed as a linear weighted combination of other vectors in that set. And thus, a set of vectors is *linearly independent* if no vector can be expressed as a linear weighted combination of other vectors in the set.
#
#
#
#
