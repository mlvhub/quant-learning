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

# # 5. Linear Regression
#
# A *linear regression* fits a straight line to observed data, attempting to demonstrate a linear relationship between variables and make predictions on new data yet to be observed.
#
# One catch is we should not use the linear regression to make predictions outside the range of data we have.

# +
# Using scikit-learn to do a linear regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Import points
df = pd.read_csv('single_independent_variable_linear_small.csv', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Fit a line to the points
fit = LinearRegression().fit(X, Y)

# m = 1.7867224, b = -16.51923513
m = fit.coef_.flatten()
b = fit.intercept_.flatten()
print("m = {0}".format(m))
print("b = {0}".format(b))

# show in chart
plt.plot(X, Y, 'o') # scatterplot
plt.plot(X, m*X+b) # line
plt.show()
# -

# ## Residuals and Squared Errors
#
# The residual is the numeric difference between the line (predicted y-values) and the points (ACTUAL y-values),
#
# Another name for residuals are errors, because they reflect how wrong our line is in predicting the data.

# +
# Calculating the residuals for a given line and data

import pandas as pd

# Import points
points = pd.read_csv('single_independent_variable_linear_small.csv', delimiter=",").itertuples()

# Test with a given line
m = 1.93939
b = 4.73333

# Calculate the residuals
for p in points:
    y_actual = p.y
    y_predict = m*p.x + b
    residual = y_actual - y_predict
    print(residual)
# -

# We want to minimize these residuals in total so there is the least gap possible between the line and points. 
#
# The best approach is to take the sum of squares, which simply squares each residual, or multiplies each residual by itself, and sums them. 

# +
# Calculating the sum of squares for a given line and data

# Calculating the residuals for a given line and data

import pandas as pd

# Import points
points = pd.read_csv('single_independent_variable_linear_small.csv', delimiter=",").itertuples()

# Test with a given line
m = 1.93939
b = 4.73333

sum_of_squares = 0.0

# calculate sum of squares
for p in points:
    y_actual = p.y
    y_predict = m*p.x + b
    residual_squared = (y_predict - y_actual)**2
    sum_of_squares += residual_squared

print("sum of squares = {}".format(sum_of_squares))
# -

# ## Finding the Best Fit Line
#
# > This is the heart of “training” a machine learning algorithm. We provide some data and an objective function (the sum of squares) and it will find the right coefficients m and b to fulfill that objective. So when we “train” a machine learning model we really are minimizing a loss function.
#
# ### Closed Form Equation
#
# For a simple linear regression with only one input and one output variable, here are the closed form equations to calculate $m$ and $b$:
#
# $$
# m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}
# $$
#
# $$
# b = \frac{\sum y}{n} - m \frac{\sum x}{n}
# $$

# +
# Calculating m and b for a simple linear regression

import pandas as pd

# Load the data
points = list(pd.read_csv('single_independent_variable_linear_small.csv', delimiter=",").itertuples())

n = len(points)

m = (n*sum(p.x*p.y for p in points) - sum(p.x for p in points) *
    sum(p.y for p in points)) / (n*sum(p.x**2 for p in points) -
    sum(p.x for p in points)**2)

b = (sum(p.y for p in points) / n) - m * sum(p.x for p in points) / n

print(m, b)
# -

# The reason the closed form equations do not scale well with larger datasets is due to a computer science concept called *computational complexity*, which measures how long an algorithm takes as a problem size grows. 

# ### Inverse Matrix Techniques
#
# $$
# b = (X^T \cdot X)^{-1} \cdot X^T \cdot y
# $$

# +
# Using inverse and transposed matrices to fit a linear regression

import pandas as pd
from numpy.linalg import inv
import numpy as np

# Import points
points = pd.read_csv('single_independent_variable_linear_small.csv', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1].flatten()

# Add placeholder "1" column to generate intercept
X_1 = np.vstack([X, np.ones(len(X))]).T

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Calculate coefficents for slope and intercept
b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
print(b) # [1.93939394, 4.73333333]

# Predict against the y-values
y_predict = X_1.dot(b)
y_predict
# -

# When you have a lot of data with a lot of dimensions, computers can start to choke and produce unstable results.
#
# This is a use case for matrix decomposition, in this specific case, we take our matrix $X$, append an additional column of 1s to generate the intercept $\beta_0$ just like before, and then decompose it into two component matrices $Q$ and $R$:
#
# $$
# X = Q \cdot R
# $$
#
# $$
# b = R^{-1} \cdot Q^T \cdot y
# $$

# +
# Using QR decomposition to perform a linear regression

import pandas as pd
from numpy.linalg import qr, inv
import numpy as np

# Import points
points = pd.read_csv('single_independent_variable_linear_small.csv', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1].flatten()

# Add placeholder "1" column to generate intercept
X_1 = np.vstack([X, np.ones(len(X))]).transpose()

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# calculate coefficents for slope and intercept
# using QR decomposition
Q, R = qr(X_1)
b = inv(R).dot(Q.transpose()).dot(Y)
b
# -

# Typically, QR decomposition is the method used by many scientific libraries for linear regression because it copes with large amounts of data more easily and is more stable.
#
# > Remember that computers work only to so many decimal places and have to approximate, so it becomes important our algorithms do not deteriorate with compounding errors in those approximations.

# ### Gradient Descent
#
# *Gradient descent* is an optimization technique that uses derivatives and iterations to minimize/maximize a set of parameters against an objective.
#
# For the function $f(x) = (x - 3)^2 + 4$, let’s find the x-value that produces the lowest point of that function:

# +
# Using gradient descent to find the minimum of a parabola

import random

def f(x):
    return (x - 3) ** 2 + 4

def dx_f(x):
    return 2*(x - 3)

# The learning rate
L = 0.001

# The number of iterations to perform gradient descent
iterations = 100_000

 # start at a random x
x = random.randint(-15,15)

for i in range(iterations):

    # get slope
    d_x = dx_f(x)

    # update x by subtracting the (learning rate) * (slope)
    x -= L * d_x

print(x, f(x))
# -

# #### Gradient Descent and Linear Regression
#
#

# +
# Performing gradient descent for a linear regression

import pandas as pd

# Import points from CSV
points = list(pd.read_csv("single_independent_variable_linear_small.csv").itertuples())

# Building the model
m = 0.0
b = 0.0


# The learning Rate
L = .001

# The number of iterations
iterations = 100_000

n = float(len(points))  # Number of elements in X

# Perform Gradient Descent
for i in range(iterations):

    # slope with respect to m
    D_m = sum(2 * p.x * ((m * p.x + b) - p.y) for p in points)

    # slope with respect to b
    D_b = sum(2 * ((m * p.x + b) - p.y) for p in points)

    # update m and b
    m -= L * D_m
    b -= L * D_b

print("y = {0}x + {1}".format(m, b))
# -

# #### Gradient Descent for Linear Regression Using SymPy

# +
# Calculating partial derivatives for m and b

from sympy import *

m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n))

d_m = diff(sum_of_squares, m)
d_b = diff(sum_of_squares, b)
print(d_m)
print(d_b)

# +
# Solving linear regression using SymPy

import pandas as pd
from sympy import *

# Import points from CSV
points = list(pd.read_csv("single_independent_variable_linear_small.csv").itertuples())

m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n))

d_m = diff(sum_of_squares, m) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)

d_b = diff(sum_of_squares, b) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)

# compile using lambdify for faster computation
d_m = lambdify([m, b], d_m)
d_b = lambdify([m, b], d_b)

# Building the model
m = 0.0
b = 0.0

# The learning Rate
L = .001

# The number of iterations
iterations = 100_000

# Perform Gradient Descent
for i in range(iterations):

    # update m and b
    m -= d_m(m,b) * L
    b -= d_b(m,b) * L

print("y = {0}x + {1}".format(m, b))
# -


