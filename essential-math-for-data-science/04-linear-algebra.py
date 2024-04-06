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

# # Linear Algebra
#
# *Linear algebra* concerns itself with linear systems but represents them through vector spaces and matrices.
#
# ## What Is a Vector?
#
# A *vector* is an arrow in space with a specific direction and length, often representing a piece of data. It has no concept of location so always imagine its tail starts at the origin of a Cartesian plane (0,0).
#
# It is the central building block of linear algebra, including matrices and linear transformations.
#
# If you have a data record for the square footage of a house 18,000 square feet and its valuation \\$260,000, we could express that as a vector [18000, 2600000], stepping 18,000 steps in the horizontal direction and 260,000 steps in the vertical direction.
#
# We declare a vector mathematically like this:
#
# $$
# \overrightarrow{v} = \begin{bmatrix}
#    x \\
#    y
# \end{bmatrix}
# $$
#
# $$
# \overrightarrow{v} = \begin{bmatrix}
#    3 \\
#    2
# \end{bmatrix}
# $$

# +
# Declaring a vector in Python using NumPy

import numpy as np

v = np.array([3, 2])
v
# -

# Note also vectors can exist on more than two dimensions. Next we declare a three-dimensional vector along axes x, y, and z:
#
# $$
# \overrightarrow{v} 
# = \begin{bmatrix}
#    x \\
#    y \\
#    z
# \end{bmatrix}
# = \begin{bmatrix}
#    4 \\
#    1 \\
#    2
# \end{bmatrix}
# $$

# +
# Declaring a three-dimensional vector in Python using NumPy

import numpy as np

v = np.array([4, 1, 2])
v
# -

# ### Adding and Combining Vectors
#
# You simply add the respective x-values and then the y-values into a new vector:
#
# $$
# \overrightarrow{v} = \begin{bmatrix}
#    3 \\
#    2
# \end{bmatrix}
# $$
#
# $$
# \overrightarrow{w} = \begin{bmatrix}
#    2 \\
#    -1
# \end{bmatrix}
# $$
#
# $$
# \overrightarrow{v} + \overrightarrow{w}
# = \begin{bmatrix}
#    3 + 2 \\
#    2 + -1
# \end{bmatrix}
# = \begin{bmatrix}
#    5 \\
#    1
# \end{bmatrix}
# $$

# +
# Adding two vectors in Python using NumPy

from numpy import array

v = array([3,2])
w = array([2,-1])

# sum the vectors
v_plus_w = v + w
v_plus_w
# -

# To visually add these two vectors together, connect one vector after the other and walk to the tip of the last vector:

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [12, 8]

plt.arrow(0, 0, 3, 2, width = 0.02, fc = 'b', label = r'$\overrightarrow{v}=\begin{bmatrix}3\\2\end{bmatrix}$')
plt.arrow(3, 2, 2, -1, width = 0.02, fc = 'r', label = r'$\overrightarrow{w}=\begin{bmatrix}2\\-1\end{bmatrix}$')
plt.arrow(0, 0, 5, 1, width = 0.04, fc = 'g', ls = ':', fill = False, label = r'$\overrightarrow{v}+\overrightarrow{w}=\begin{bmatrix}5\\1\end{bmatrix}$')
plt.legend()
plt.grid()
# -

# ### Scaling Vectors
#
# *Scaling* is growing or shrinking a vector’s length. You can grow/shrink a vector by multiplying or scaling it with a single value, known as a *scalar*.

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [12, 8]

plt.arrow(0, 0, 3, 1, width = 0.02, fc = 'b', label = r'$\overrightarrow{v}=\begin{bmatrix}3\\1\end{bmatrix}$')
plt.arrow(0, 0, 6, 2, width = 0.04, fc = 'g', ls = ':', fill = False, label = r'$2\overrightarrow{v}=\begin{bmatrix}6\\2\end{bmatrix}$')
plt.legend()
plt.grid()
# -

# You multiply each element of the vector by the scalar value:
#
# $$
# \overrightarrow{v}=\begin{bmatrix}3\\1\end{bmatrix}
# $$
#
# $$
# 2\overrightarrow{v}=2\begin{bmatrix}3\\1\end{bmatrix} = \begin{bmatrix}3 \cdot 2 \\1 \cdot 2\end{bmatrix} = \begin{bmatrix}6\\2\end{bmatrix}
# $$

# +
# Scaling a number in Python using NumPy

from numpy import array

v = array([3,1])

# scale the vector
scaled_v = 2 * v
scaled_v
# -

# Scaling down a vector is similar:

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [12, 8]

plt.arrow(0, 0, 3, 1, width = 0.02, fc = 'b', label = r'$\overrightarrow{v}=\begin{bmatrix}3\\1\end{bmatrix}$')
plt.arrow(0, 0, 1.5, 0.5, width = 0.04, fc = 'g', ls = ':', fill = False, label = r'$0.5\overrightarrow{v}=\begin{bmatrix}1.5\\0.5\end{bmatrix}$')
plt.legend()
plt.grid()

# +
# Scaling down a number in Python using NumPy

from numpy import array

v = array([3,1])

# scale the vector
scaled_v = 0.5 * v
scaled_v
# -

# An important detail to note here is that scaling a vector does not necessarily change its direction, only its magnitude.
#
# When you multiply a vector by a negative number, it flips the direction of the vector:

# +
# Scaling down a number in Python using NumPy

from numpy import array

v = array([3,1])

# scale the vector
scaled_v = -1 * v
scaled_v

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [12, 8]
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

plt.arrow(0, 0, 3, 1, width = 0.02, fc = 'b', label = r'$\overrightarrow{v}=\begin{bmatrix}3\\1\end{bmatrix}$')
plt.arrow(0, 0, -3, -1, width = 0.04, fc = 'g', ls = ':', fill = False, label = r'$-1\overrightarrow{v}=\begin{bmatrix}-3\\-1\end{bmatrix}$')
plt.legend()
plt.grid()
# -

# ### Span and Linear Dependence
#
# These two operations, adding two vectors and scaling them, brings about a simple but powerful idea. With these two operations, we can combine two vectors and scale them to create any resulting vector we want.
#
# The whole space of possible vectors is called *span*, and in most cases our span can create unlimited vectors off those two vectors, simply by scaling and summing them.
#
# When we have two vectors in two different directions, they are *linearly independent* and have this unlimited span.
#
# But what happens when two vectors exist in the same direction, or exist on the same line? The combination of those vectors is also stuck on the same line, limiting our span to just that line. No matter how you scale it, the resulting sum vector is also stuck on that same line. This makes them *linearly dependent*.
#
# Why do we care whether two vectors are linearly dependent or independent? A lot of problems become difficult or unsolvable when they are linearly dependent.

# ## Linear Transformations
#
# This concept of adding two vectors with fixed direction, but scaling them to get different combined vectors, is hugely important.
#
# ### Basis Vectors
#
# Imagine we have two simple vectors $\hat{i}$ and $\hat{j}$ (“i-hat” and “j-hat”). These are known as basis vectors, which are used to describe transformations on other vectors.
#
# They typically have a length of 1 and point in perpendicular positive directions:

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [12, 8]
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

plt.arrow(0, 0, 1, 0, width = 0.02, fc = 'k', label = r'$\hat{i}$')
plt.arrow(0, 0, 0, 1, width = 0.02, fc = 'b', label = r'$\hat{j}$')
plt.legend()
plt.grid()
# -

# Our basis vector is expressed in a 2 × 2 matrix, where the first column is $\hat{i}$ and the second column is $\hat{j}$:
#
# $$
# \hat{i}=\begin{bmatrix}1\\0\end{bmatrix}
# $$
#
# $$
# \hat{j}=\begin{bmatrix}0\\1\end{bmatrix}
# $$
#
# $$
# \text{basis}=\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}
# $$
#
# #### Matrix
#
# A *matrix* is a collection of vectors (such as $\hat{i}$, $\hat{j}$) that can have multiple rows and columns and is a convenient way to package data. We can use and to create any vector we want by scaling and adding them.
#
# Let’s start with each having a length of 1 and showing the resulting vector $\overrightarrow{v}$:

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [12, 8]
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

plt.arrow(0, 0, 1, 0, width = 0.02, fc = 'k', label = r'$\hat{i}$')
plt.arrow(0, 0, 0, 1, width = 0.02, fc = 'b', label = r'$\hat{j}$')
plt.arrow(0, 0, 1, 1, width = 0.02, fc = 'g', label = r'$\overrightarrow{v}=\begin{bmatrix}1\\1\end{bmatrix}$')
plt.legend()
plt.grid()
# -

# I want vector $\overrightarrow{v}$ to land at [3, 2]. What happens to $\overrightarrow{v}$ if we stretch $\hat{i}$ by a factor of 3 and $\hat{j}$ by a factor of 2? First we scale them individually:
#
# $$
# \hat{3i}= 3\begin{bmatrix}3\\0\end{bmatrix} = \begin{bmatrix}3\\0\end{bmatrix}
# $$
#
# $$
# \hat{2j}= 2\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}0\\2\end{bmatrix}
# $$
#
# This is known as a *linear transformation*, where we transform a vector with stretching, squishing, sheering, or rotating by tracking basis vector movements.

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [12, 8]
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

plt.arrow(0, 0, 3, 0, width = 0.02, fc = 'k', label = r'$\hat{i}$')
plt.arrow(0, 0, 0, 2, width = 0.02, fc = 'b', label = r'$\hat{j}$')
plt.arrow(0, 0, 3, 2, width = 0.02, fc = 'g', label = r'$\overrightarrow{v}=\begin{bmatrix}3\\2\end{bmatrix}$')
plt.legend()
plt.grid()
# -

# **Recall that vector $\overrightarrow{v}$ is composed of adding $\hat{i}$ and $\hat{j}$:**
#
# $$
# \overrightarrow{v}= \begin{bmatrix}3\\0\end{bmatrix} + \begin{bmatrix}0\\2\end{bmatrix} = \begin{bmatrix}3\\2\end{bmatrix}
# $$
#
# Generally, with linear transformations, there are four movements you can achieve:

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["figure.figsize"] = [4, 4]
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

plt.arrow(0, 0, 1, 0, width = 0.02, fc = 'k', label = r'$\hat{i}$')
plt.arrow(0, 0, 0, 1, width = 0.02, fc = 'b', label = r'$\hat{j}$')
plt.arrow(0, 0, 1, 1, width = 0.02, fc = 'g', label = r'$\overrightarrow{v}=\begin{bmatrix}1\\1\end{bmatrix}$')
plt.title("Basis")
plt.legend()
plt.grid()

# +
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, axs = plt.subplots(2, 2, figsize=(8,8))
mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

axs[0, 0].arrow(0, 0, 1, 0, width = 0.02, fc = 'k', label = r'$\hat{i}$')
axs[0, 0].arrow(0, 0, 0, 2, width = 0.02, fc = 'b', label = r'$\hat{j}$')
axs[0, 0].arrow(0, 0, 1, 2, width = 0.02, fc = 'g', label = r'$\overrightarrow{v}=\begin{bmatrix}1\\2\end{bmatrix}$')
axs[0, 0].set_title('Scale')

axs[0, 1].arrow(0, 0, 0, -1, width = 0.02, fc = 'k', label = r'$\hat{i}$')
axs[0, 1].arrow(0, 0, 1, 0, width = 0.02, fc = 'b', label = r'$\hat{j}$')
axs[0, 1].arrow(0, 0, 1, -1, width = 0.02, fc = 'g', label = r'$\overrightarrow{v}=\begin{bmatrix}1\\-1\end{bmatrix}$')
axs[0, 1].set_title('Rotate')

axs[1, 0].arrow(0, 0, 1, 0, width = 0.02, fc = 'k', label = r'$\hat{i}$')
axs[1, 0].arrow(0, 0, 1, 1, width = 0.02, fc = 'b', label = r'$\hat{j}$')
axs[1, 0].arrow(0, 0, 2, 1, width = 0.02, fc = 'g', label = r'$\overrightarrow{v}=\begin{bmatrix}2\\1\end{bmatrix}$')
axs[1, 0].set_title('Shear')

axs[1, 1].arrow(0, 0, 0, 1, width = 0.02, fc = 'k', label = r'$\hat{i}$')
axs[1, 1].arrow(0, 0, 1, 0, width = 0.02, fc = 'b', label = r'$\hat{j}$')
axs[1, 1].arrow(0, 0, 1, 1, width = 0.02, fc = 'g', label = r'$\overrightarrow{v}=\begin{bmatrix}1\\1\end{bmatrix}$')
axs[1, 1].set_title('Inversion')


for ax in axs.flat:
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.legend()
    ax.grid()

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
# -

# > It is important to note that you cannot have transformations that are nonlinear, resulting in curvy or squiggly transformations that no longer respect a straight line. This is why we call it linear algebra, not nonlinear algebra!

# ### Matrix Vector Multiplication
#
# The formula to transform a vector $\overrightarrow{v}$ given basis vectors $\hat{i}$ and $\hat{j}$ packaged as a matrix is:
#
# $$
# \begin{bmatrix}x_{new}\\y_{new}\end{bmatrix} = \begin{bmatrix}a & b\\b & d\end{bmatrix} \begin{bmatrix}x\\y\end{bmatrix}
# $$
#
# $$
# \begin{bmatrix}x_{new}\\y_{new}\end{bmatrix} = \begin{bmatrix}ax + by\\cx + dy\end{bmatrix}
# $$
#
# $\hat{i}$ is the first column [a, c] and $\hat{j}$ is the column [b, d]. We package both of these basis vectors as a matrix, which again is a collection of vectors expressed as a grid of numbers in two or more dimensions.
#
# This formula is a shortcut for scaling and adding $\hat{i}$ and $\hat{j}$.
#
# In effect, a matrix really is a transformation expressed as basis vectors.

# +
# Matrix vector multiplication in NumPy

from numpy import array

# compose basis matrix with i-hat and j-hat
basis = array(
    [[3, 0],
     [0, 2]]
 )

# declare vector v
v = array([2,2])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)
new_v

# +
# Separating the basis vectors and applying them as a transformation

from numpy import array

# Declare i-hat and j-hat
i_hat = array([5, 7])
j_hat = array([4, 2])

# compose basis matrix using i-hat and j-hat
# also need to transpose rows into columns
basis = array([i_hat, j_hat]).transpose()
print(array([i_hat, j_hat]))
print(array([i_hat, j_hat]).transpose())

# declare vector v
v = array([3,2])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)
new_v

# +
# A more complicated transformation

from numpy import array

# Declare i-hat and j-hat
i_hat = array([2, 3])
j_hat = array([2, -1])

# compose basis matrix using i-hat and j-hat
# also need to transpose rows into columns
basis = array([i_hat, j_hat])#.transpose()
print(array([i_hat, j_hat]))
print(array([i_hat, j_hat]).transpose())

# declare vector v 0
v = array([2,1])

# create new vector
# by transforming v with dot product
new_v = basis.dot(v)
new_v
# -

# ## Matrix Multiplication
#
# Think of *matrix multiplication* as applying multiple transformations to a vector space. 
#
# Each transformation is like a function, where we apply the innermost first and then apply each subsequent transformation outward.
#
# Here is how we apply a rotation and then a shear to any vector $\overrightarrow{v}$ with value [x, y]:
#
# $$
# \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}
# \begin{bmatrix}0 & -1\\1 & 0\end{bmatrix}
# \begin{bmatrix}x\\y\end{bmatrix}
# $$
#
# You multiply and add each row from the first matrix to each respective column of the second matrix, in an “over-and-down! over-and-down!” pattern:
#
# $$
# \begin{bmatrix}a & b\\c & d\end{bmatrix}
# \begin{bmatrix}e & f\\g & h\end{bmatrix}
# =
# \begin{bmatrix}ae + bg & af + bh\\ce + dg & cf + dh\end{bmatrix}
# $$

# +
# Combining two transformations

from numpy import array

# Transformation 1
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

# Transformation 2
i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

# Combine Transformations
combined = transform2 @ transform1

# Test
print("COMBINED MATRIX:\n {}".format(combined))

v = array([1, 2])
print("RESULT:\n {}".format(combined.dot(v)))  # [-1, 1]
# -

# **Using `dot()` Versus `matmul()` and `@`**
#
# In general, you want to prefer `matmul()` and its shorthand `@` to combine matrices rather than the `dot()` operator in NumPy. The former generally has a preferable policy for higher-dimensional matrices and how the elements are broadcasted.
#
# ---

# Note that we also could have applied each transformation individually to vector $\overrightarrow{v}$ and still have gotten the same result:

rotated = transform1.dot(v)
sheered = transform2.dot(rotated)
sheered

# **Note that the order you apply each transformation matters.**
#
# Matrix dot products are not commutative, meaning you cannot flip the order and expect the same result:

# +
# Applying the transformations in reverse

from numpy import array

# Transformation 1
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

# Transformation 2
i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

# Combine Transformations, apply sheer first and then rotation
combined = transform1 @ transform2

# Test
print("COMBINED MATRIX:\n {}".format(combined))

v = array([1, 2])
print("RESULT:\n {}".format(combined.dot(v))) # [-2, 3]
# -

# > Think of each transformation as a function, and we apply them from the innermost to outermost just like nested function calls.

# ## Determinants
#
#
# *Determinants* describe how much a sampled area in a vector space changes in scale with linear transformations, and this can provide helpful information about the transformation.

# +
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches

fig, axs = plt.subplots(1, 2, figsize=(8,3))
mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

axs[0].arrow(0, 0, 1, 0, width = 0.06, fc = 'k', label = r'$\hat{i}$')
axs[0].arrow(0, 0, 0, 1, width = 0.06, fc = 'b', label = r'$\hat{j}$')
#axs[0].fill_between(1, 1, y2=0, color='g', alpha=0.3)

axs[1].arrow(0, 0, 3, 0, width = 0.06, fc = 'k', label = r'$3\hat{i}$')
axs[1].arrow(0, 0, 0, 2, width = 0.06, fc = 'b', label = r'$2\hat{j}$')
#axs[1].fill_between(3, 2, y2=0, color='g', alpha=0.3)

for ax in axs.flat:
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set(ylim=(0, 4), xlim=(0, 4))
    ax.legend()
    ax.grid()

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    


# +
# Calculating a determinant

from numpy.linalg import det
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)
determinant
# -

# Simple shears and rotations should not affect the determinant, as the area will not change:

# +
# A determinant for a shear

from numpy.linalg import det
from numpy import array

i_hat = array([1, 0])
j_hat = array([1, 1])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)
determinant
# -

# But scaling will increase or decrease the determinant, as that will increase/decrease the sampled area.
#
# When the orientation flips ($\hat{i}$, $\hat{j}$, swap clockwise positions), then the determinant will be negative:

# +
# A negative determinant

from numpy.linalg import det
from numpy import array

i_hat = array([-2, 1])
j_hat = array([1, 2])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)
determinant
# -

# **By far the most critical piece of information the determinant tells you is whether the transformation is linearly dependent.**
#
# If you have a determinant of 0, that means all of the space has been squished into a lesser dimension:

# +
# A determinant of zero

from numpy.linalg import det
from numpy import array

i_hat = array([-2, 1])
j_hat = array([3, -1.5])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)
determinant
# -

# > When you encounter this you will likely find a difficult or unsolvable problem on your hands.

# ## Special Types of Matrices
#
# ### Square Matrix
#
# The *square matrix* is a matrix that has an equal number of rows and columns:
#
# $$
# \begin{bmatrix}4 & 2 & 7\\5 & 1 & 9\\4 & 0 & 1\end{bmatrix}
# $$
#
# They are primarily used to represent linear transformations and are a requirement for many operations like eigendecomposition.
#
# ### Identity Matrix
#
# The *identity matrix* is a square matrix that has a diagonal of 1s while the other values are 0:
#
# $$
# \begin{bmatrix}1 & 0 & 0\\0 & 1 & 0\\0 & 0 & 1\end{bmatrix}
# $$
#
# When you have an identity matrix, you essentially have undone a transformation and found your starting basis vectors.
#
# ### Inverse Matrix
#
# An *inverse matrix* is a matrix that undoes the transformation of another matrix. Let’s say I have matrix $A$:
#
# $$
# A = \begin{bmatrix}4 & 2 & 4\\5 & 3 & 7\\9 & 3 & 6\end{bmatrix}
# $$
#
# The inverse of matrix $A$ is called $A^{-1}$:
#
# $$
# A^{-1} = \begin{bmatrix}-\frac{1}{2} & 0 & \frac{1}{3}\\5.5 & -2 & \frac{4}{3}\\-2 & 1 & \frac{1}{3}\end{bmatrix}
# $$
#
# When we perform matrix multiplication between $A^{-1}$ and $A$, we end up with an identity matrix.

# ### Diagonal Matrix
#
# The *diagonal matrix* has a diagonal of nonzero values while the rest of the values are 0. They represent simple scalars being applied to a vector space.
#
# $$
# \begin{bmatrix}4 & 0 & 0\\0 & 2 & 0\\0 & 0 & 5\end{bmatrix}
# $$

# ### Triangular Matrix
#
# The *triangular matrix* has a diagonal of nonzero values in front of a triangle of values, while the rest of the values are 0.
#
# They typically are easier to solve in systems of equations. They also show up in certain decomposition tasks like LU Decomposition.
#
# $$
# \begin{bmatrix}4 & 2 & 9\\0 & 1 & 6\\0 & 0 & 5\end{bmatrix}
# $$

# ### Sparse Matrix
#
# *Sparse matrices* are matrices that are mostly zeroes and have very few nonzero elements.
#
# From a computing standpoint, they provide opportunities to create efficiency. If a matrix has mostly 0s, a sparse matrix implementation will not waste space storing a bunch of 0s, and instead only keep track of the cells that are nonzero.
#
# $$
# sparse = \begin{bmatrix}0 & 0 & 0\\0 & 0 & 2\\0 & 0 & 0\\0 & 0 & 0\end{bmatrix}
# $$
#
# When you have large matrices that are sparse, you might explicitly use a sparse function to create your matrix.

# ## Systems of Equations and Inverse Matrices
#
#


