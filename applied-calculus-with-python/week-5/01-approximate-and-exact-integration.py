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
from sympy.parsing.sympy_parser import parse_expr


# ## Left Riemann Sum Calculator

def left_riemann_sum(f, n):
    x = Symbol('x')

    # define the limits
    a = 0
    b = 1
    width = (b-a)/n
    height = []

    for i in range(n):
        height.append(float(parse_expr(f).subs({x: a+width*i}).evalf()))
    
    # find the individual areas of each rectangle
    area = [(width*height[i]) for i in range(n)]

    # add the areas together
    print(f'The Left Riemann Sum is {sum(area)}')

    # compute the actual integral
    print(f'The actual integral is {Integral(f, (x,a,b)).doit()}')


left_riemann_sum('x**2', 10)

left_riemann_sum('x**2', 10000)

left_riemann_sum('x**3', 500)

# example of an indefinite integral
x = Symbol('x')
integrate(x**3)

# ## Riemann Sum from table

x = [ 0, 2, 5, 6, 10 ]
f = [ 1.5, 4, 7.2, 8, 9.1 ]
n = 4
left_rect = [f[i]*(x[i+1]-x[i]) for i in range(0,n)]
right_rect = [f[i+1]*(x[i+1]-x[i]) for i in range(0,n)]
LHS = sum(left_rect)
RHS = sum(right_rect)
print(f'The left-hand sum is {LHS} and the right-hand sum is {RHS}')


