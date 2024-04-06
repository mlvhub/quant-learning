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

from sympy import *

# ## Finding Local Extrema With SymPy
#
# ### Steps
#
# 1. Find the derivative $f′(x)$.
#
# 2. Find critical points: Locate all points (if any) where $f′(x)$ is undefined, then set $f′(x)=0$ and solve for $x$.
#
# 3. Divide the domain of $f(x)$ into subintervals using the critical points. 
#
# 4. Choose a sample point in each subinterval and test the value of $f′(x)$ at that point. The sign of $f′(x)$ at the sample point (positive or negative) is the sign of $f′(x)$ on the entire subinterval.
#
# 5. List the critical points where $f′(x)$ changes from positive to negative (local maxima) and the critical points where $f′(x)$ changes from negative to positive (local minima).

# ### Example
#
# Find the local extrema of the function $f(x)=x3+x2−x$.
#
# > Note that for this function, there are no points where the derivative is undefined. Therefore, we find all critical points by solving $f′(x)=0$. 

# +
x = symbols('x')

f = x**3 + x**2 - x
f
# -

f_prime = diff(f, x)
f_prime

solveset(f_prime, x)

sample = [-2, 0, 1]
sample_values = [f_prime.subs(x, i) for i in sample]
sample_values

# We conclude that:
#
# - Since $f′$ changes from positive to negative at $x=−1$, this is the location of a local maximum.
# - Since $f′$ changes from negative to positive at $x=\frac{1}{3}$​, this is the location of a local minimum.

# ## Finding Global Extrema With SymPy
#
# ### The Closed Interval Method
#
# 1. Find critical points. Identify points where $f′$ does not exist to find critical points. Then solve $f′(x)=0$ for $x$ to find more critical points.
#
# 2. Find function values at critical points. Evaluate the function value $f(x)$ at each of the critical points you found.
#
# 3. Find function values at endpoints. Evaluate the function values at the endpoints of the interval, $f(a)$ and $f(b)$.
#
# 4. Compare values. Compare the values computed in the previous steps. The highest value is the absolute maximum, and the lowest value is the absolute minimum.
#
# ### Example
#
# Find the global maximum and minimum values of the function $f(x)=x3+x2−x$ on the interval $[−2,1]$.

# +
x = symbols('x')

f = x**3 + x**2 - x
f
# -

x_values = [-2, -1, 1/3, 1]
f_values = [f.subs(x, i) for i in x_values]
f_values

max(f_values)

min(f_values)

# Comparing the values in the list f_values, we determine that the maximum value achieved by $f$ on $[−2,1]$ is $1$, and the minimum value is $−2$.

# ## Increase, Decrease, and Concavity
#
# ### Increasing and Decreasing
#
# 1. Use the `diff()` function to take the derivative.
# 2. Use `solveset()` to solve for critical points.
# 3. Divide the domain into subintervals at the critical points.
# 4. Use the `subs()` function with a list comprehension to compute derivative values at sample points taken from each subinterval.
#
# ### Concavity
#
# We use the same method as for increase/decrease, but with the second derivative instead of the first.
#
# 1. Use `diff(f, x, 2)` to take the second derivative of $f$.
# 2. Use `solveset()` to solve for possible inflection points.
# 3. Divide the domain into subintervals at the possible inflection points.
# 4. Use the `subs()` function with a list comprehension to compute second derivative values at sample points taken from each subinterval.

#
