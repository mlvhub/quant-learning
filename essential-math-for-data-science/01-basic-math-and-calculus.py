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

# # Basic Math and Calculus Review

# +
# Charting a linear function in Python using SymPy

from sympy import *

x = symbols('x')
f = 2*x + 1
plot(f)

# +
# Charting an exponential function

from sympy import *

x = symbols('x')
f = x**2 + 1
plot(f)

# +
# Declaring a function with two independent variables in Python

from sympy import *
from sympy.plotting import plot3d

x, y = symbols('x y')
f = 2*x + 3*y
plot3d(f)
# -

# ## Summation

# $$
# \sum_{i=1}^n 10x_i
# $$

# +
x = [1, 4, 6, 2]
n = len(x)

summation = sum(10*x[i] for i in range(0,n))
summation

# +
# summation in SymPy

from sympy import symbols, Sum

i,n = symbols('i n')

# iterate each element i from 1 to n,
# then multiply and sum
summation = Sum(2*i,(i,1,n))

# specify n as 5,
# iterating the numbers 1 through 5
up_to_5 = summation.subs(n, 5)
up_to_5.doit() # 30
# -

# ## Logarithms

# A logarithm is a math function that finds a power for a specific number and base. 

# $$
# 2^x=8
# $$
#
# $$
# {\log_2 8} = x
# $$
#
# More generally:
#
# $$
# a^x=b
# $$
#
# $$
# {\log_a b} = x
# $$

# +
# Using the log function in Python

from math import log

# 2 raised to what power gives me 8?
log(8, 2) # prints 3.0
# -

# | Operator       | Exponent property             | Logarithm property                           |
# |----------------|-------------------------------|----------------------------------------------|
# | Multiplication | $$x^m * x^n = x^{m+n}$$       | $$log(a * b) = log(a) + log(b)$$             |
# | Division       | $$\frac{x^m}{x^n} = m^{m-n}$$ | $$log(\frac{a}{b}) = log(a) - log(b)$$       |
# | Exponentiation | $$(x^m)^n = x^{mn}$$          | $$log(a^n) = n * log(a)$$                    |
# | Zero exponent  | $$x^0 = 1$$                   | $$log(1) = 0$$                               |
# | Inverse        | $$x^{-1} = \frac{1}{x}$$      | $$log(x^{-1}) = log(\frac{1}{x}) = -log(x)$$ |

# ## Euler’s Number and Natural Logarithms

# ### Euler’s Number
#
# A property of Euler’s number is its exponential function is a derivative to itself.
#
# TODO: add more content here

# ### Natural Logarithms
#
# $$log_e 10 = ln(10)$$

# +
# Calculating the natural logarithm of 10 in Python 

from math import log

# e raised to what power gives us 10?
x = log(10) # e is the default base for `log` in Python
x # 2.302585092994046
# -

# ## Limits

# +
# A function that forever approaches 0 but never reaches 0

from sympy import *

x = Symbol('x')
f = 1 / x
plot(f)
# -

# $$\lim_{x\to\infty} \frac{1}{x} = 0$$

# +
# Using SymPy to calculate limits

from sympy import *

x = symbols('x')
f = 1 / x
limit(f, x, oo)
# -

# We can discover the Euler number using the limit:
#
# $$\lim_{n\to\infty} (1 + \frac{1}{n}) = e = 2.71828169413...$$

# +
from sympy import *

n = symbols('n')
f = (1 + (1/n))**n
result = limit(f, n, oo)

result
# -

result.evalf()


# ## Derivatives
#
# A derivative tells the slope of a function, and it is useful to measure the rate of change at any point in a function.
#
# When the slope is 0, that means we are at the minimum or maximum of an output variable.

# To get the slope of the function $f(x) = x^2$, we can use the formula:
#
# $$
# m = \frac{y_2 - y_1}{x_2 - x_1}
# $$
#
# $$
# m = \frac{4.41 - 4.0}{2.1 - 2.0} = 4.1
# $$

# +
# A derivative calculator in Python

def derivative_x(f, x, step_size):
    m = (f(x + step_size) - f(x)) / ((x + step_size) - x)
    return m


def my_function(x):
    return x**2

slope_at_2 = derivative_x(my_function, 2, .00001)
slope_at_2
# -

# With a function: $f(x) = x^2$
#
# The derivative function will make the exponent a multiplier and then decrement the exponent by 1:
#
# $$\frac{d}{dx} x^2 = 2x$$

# +
# Calculating a derivative in SymPy

from sympy import *

# Declare 'x' to SymPy
x = symbols('x')

# Now just use Python syntax to declare function
f = x**2

# Calculate the derivative of the function
dx_f = diff(f)
dx_f


# +
# A derivative calculator in Python

def f(x):
    return x**2

def dx_f(x):
    return 2*x

slope_at_2 = dx_f(2.0)
slope_at_2
# -

# ## Partial Derivatives
#
# These are derivatives of functions that have multiple input variables.
#
# Rather than finding the slope on a one-dimensional function, we have slopes with respect to multiple variables in several directions. For each given variable derivative, we assume the other variables are held constant. 
#
# Let’s take the function $f(x, y) = 2x^3 + 3y^3$. The x and y variable each get their own derivatives $\frac{d}{dx}$ and $\frac{d}{dy}$. These represent the slope values with respect to each variable on a multidimensional surface. We technically call these “slopes” gradients when dealing with multiple dimensions.
#
# $f(x, y) = 2x^3 + 3y^3$
#
# $\frac{d}{dx}2x^3 + 3y^3 = 6x^2$
#
# $\frac{d}{dy}2x^3 + 3y^3 = 9y^2$

# +
# Calculating partial derivatives with SymPy

from sympy import *
from sympy.plotting import plot3d

# Declare x and y to SymPy
x,y = symbols('x y')

# Now just use Python syntax to declare function
f = 2*x**3 + 3*y**3

# Calculate the partial derivatives for x and y
dx_f = diff(f, x)
dy_f = diff(f, y)

print(dx_f) # prints 6*x**2
print(dy_f) # prints 9*y**2

# plot the function
plot3d(f)
# -

# #### Using Limits to Calculate Derivatives

# SymPy allows us to do some interesting explorations about math. Take our function $f(x) = x^2$; we approximated a slope for $x = 2$ by drawing a line through a close neighboring point = 2.0001 by adding a step 0.0001. Why not use a limit to forever decrease that step s and see what slope it approaches?
#
# $\lim_{x\to\infty} \frac{(x + s)^2 - x^2}{(x + s) - x}$
#
# In our example, we are interested in the slope where so let’s substitute that:
#
# $\lim_{x\to\infty} \frac{(2 + s)^2 - 2^2}{(2 + s) - 2} = 4$

# +
# Using limits to calculate a slope

from sympy import *

# "x" and step size "s"
x, s = symbols('x s')

# declare function
f = x**2

# slope between two points with gap "s"
# substitute into rise-over-run formula
slope_f = (f.subs(x, x + s) - f) / ((x+s) - x)

# substitute 2 for x
slope_2 = slope_f.subs(x, 2)

# calculate slope at x = 2
# infinitely approach step size _s_ to 0
limit(slope_2, s, 0)
# -

# ## The Chain Rule
#
# Given:
#
# $y = x^2 + 1$
#
# $z = y^3 - 2$
#
# We have:
#
# $z = (x^2 + 1)^3 - 2$
#
# So what is the derivative for z with respect to x? 

# +
# Finding the derivative of z with respect to x



# +
from sympy import *

z = (x**2 + 1)**3 - 2
dz_dx = diff(z, x)
dz_dx
# -

# $\frac{dz}{dx}((x^2 + 1)^3 - 2) = 6x(x^2 + 1)^2$

# But look at this. Let’s start over and take a different approach. If we take the derivatives of the y and z functions separately, and then multiply them together, this also produces the derivative of z with respect to x! 
#
#

# $\frac{dy}{dx}(x^2 + 1) = 2x$
#
# $\frac{dz}{dy}(y^3 - 2) = 3y^2$
#
# $\frac{dz}{dx}(2x)(3y^2) = 6xy^2$

# This is the chain rule, which says that for a given function y (with input variable x) composed into another function z (with input variable y), we can find the derivative of z with respect to x by multiplying the two respective derivatives together:
#
# $\frac{dz}{dx} = \frac{dz}{dy} * \frac{dy}{dx}$

# +
# Calculating the derivative dz/dx with and without the chain rule, but still getting the same answer

from sympy import *

x, y = symbols('x y')


# derivative for first function
# need to underscore y to prevent variable clash
_y = x**2 + 1
dy_dx = diff(_y)

# derivative for second function
z = y**3 - 2
dz_dy = diff(z)

# Calculate derivative with and without
# chain rule, substitute y function
dz_dx_chain = (dy_dx * dz_dy).subs(y, _y)
dz_dx_no_chain = diff(z.subs(y, _y))

# Prove chain rule by showing both are equal
print(dz_dx_chain) # 6*x*(x**2 + 1)**2
print(dz_dx_no_chain) # 6*x*(x**2 + 1)**2

dz_dx_chain == dz_dx_no_chain
# -

# ## Integrals

# The opposite of a derivative is an integral, which finds the area under the curve for a given range. (They measure accumulation as opposed to rate of change, i.e. how much of something there is at given point in time.

# +
# Using SymPy to perform integration

from sympy import *

# Declare 'x' to SymPy
x = symbols('x')

# Now just use Python syntax to declare function
f = x**2 + 1

# Calculate the integral of the function with respect to x
# for the area between x = 0 and 1
area = integrate(f, (x, 0, 1))
area
# -

# #### Calculating Integrals with Limits

# +
# Using limits to calculate integrals

from sympy import *

# Declare variables to SymPy
x, i, n = symbols('x i n')

# Declare function and range
f = x**2 + 1
lower, upper = 0, 1

# Calculate width and each rectangle height at index "i"
delta_x = ((upper - lower) / n)
x_i = (lower + delta_x * i)
fx_i = f.subs(x, x_i)

# Iterate all "n" rectangles and sum their areas
n_rectangles = Sum(delta_x * fx_i, (i, 1, n)).doit()

# Calculate the area by approaching the number
# of rectangles "n" to infinity
area = limit(n_rectangles, n, oo)

area
# -

# ## Exercises

# #### 1. Is the value 62.6738 rational or irrational? Why or why not?
#
# Rational as it can be expressed as a fraction.

# #### 2. Evaluate the expression: $10^710^{-5}$
#
# $10^2$

expr = 10**7 * 10**-5
N(expr)

# #### 3. Evaluate the expression: $81^{\frac{1}{2}}$
#
# $81^{\frac{1}{2}} = \sqrt{81} = 9$

expr = 81**(1/2)
N(expr)

# #### 4. Evaluate the expression: $25^{\frac{3}{2}}$
#
# $25^{\frac{3}{2}} = \sqrt{25}^3 = 125$

expr = 25**(3/2)
N(expr)

# #### 5. Assuming no payments are made, how much would a $1,000 loan be worth at 5% interest compounded monthly after 3 years?

# +
from math import exp

"""
a: balance
p: starting_investment
r: interest rate
t: timespan
n: periods
"""

p = 1000
r = .05
t = 3
n = 12

a = p * (1 + (r/n))**(n * t)
a
# -

# #### 6. Assuming no payments are made, how much would a $1,000 loan be worth at 5% interest compounded continuously after 3 years?
#

# +
from math import exp

p = 1000 # principal, starting amount
r = .05 # interest rate, by year
t = 3 # time, number of years

a = p * exp(r*t)
a
# -

# #### 7. For the function $f(x) = 3x^2 + 1$ what is the slope at x = 3?
#
# $\frac{d}{dx}3x^2 + 1= 6x$
#
# $f'(3) = 6(3) = 18$

# +
x = symbols('x')
f = 3*x**2

dx_f = diff(f)
dx_f
# -

dx_f.subs(x, 3)

# #### 8. For the function  $f(x) = 3x^2 + 1$ what is the area under the curve for x between 0 and 2?
#
# $\int{3x^2 + 1} = \frac{3x^3}{3} + \frac{1x}{1} = x^3 + x$
#
# $\int_{0}^{2}{3x^2} = 2^3 + 2 - 0 ^ 3 + 0 = 10$

# +
x = symbols('x')
f = 3*x**2 + 1

area = integrate(f, (x, 0, 2))
area
