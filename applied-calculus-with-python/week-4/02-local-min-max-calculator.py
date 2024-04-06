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


def local_extrema_calculator(f, x):
    x = Symbol("x")
    j = 0

    min_list = []
    max_list = []

    # find f'
    dy = Derivative(f, x).doit()
    print(f'First derivative: {dy}')

    # find critical values
    critical_points = solve(dy, x)
    print(f'Critical points: {critical_points}')

    # check if min/max using second derivative test
    d2f = Derivative(dy, x).doit()
    print(f'Second derivative: {d2f}')

    # review second derivative test
    for i in critical_points:
        cp = d2f.subs({x: critical_points[j]}).evalf()
        if cp > 0:
            print(f'x = {critical_points[j].evalf(3)} is a local minimum')
            y = float(parse_expr(f).subs({x: critical_points[j]}).evalf())
            min_list.append(y)
        elif cp < 0:
            print(f'x = {critical_points[j].evalf(3)} is a local minimum')
            y = float(parse_expr(f).subs({x: critical_points[j]}).evalf())
            max_list.append(y)
        else:
            print(f'Unable to determine if {cp} is min or max')
        j = j + 1

    # find local min/max
    print(f'Local mins of f(x) = {f}: {min_list}')
    print(f'Local maxes of f(x) = {f}: {max_list}')



local_extrema_calculator("3*x**4-16*x**3+18*x**2", "x")


