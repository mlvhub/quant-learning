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

from numpy import sqrt
from sympy.plotting import *
from sympy import Symbol


# ax^2 + bx + c
def zeros(a, b, c):
    D = sqrt(b*b-4*a*c)
    x1 = (-b + D) / (2 * a)
    x2 = (-b - D) / (2 * a)

    return (x1, x2)


def print_graph(a, b, c):
    x = Symbol("x")
    plot(a*x**2 + b*x + c)


# +
a = 1
b = 2
c = 1

zeros(a, b, c)
# -

print_graph(a, b, c)

zeros(1, 0, 1)

# k(x) = 5x - 9
(6 + 4) / (3 - 1)
# y = mx + b
# y = 5x + b
# b = y - 5x = 6 - 5(3) = = -9
#            = = -4 - 5(1) = -9

def linear(x1, y1, x2, y2):
    x = Symbol('x')
    m = (y2-y1)/(x2-x1)
    return m*(x - x1) + y1


linear(1, -4, 3, 6)

linear(4, 9, 1.5, 7)
