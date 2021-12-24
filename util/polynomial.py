# -*- coding: utf-8 -*-
"""Script with polynomial functions.

These functions are used in curve approximations.

Written by: Dmitry Duplyakin (dmitry.duplyakin@nrel.gov)
in collaboration with the National Renewable Energy Laboratories.
"""

import numpy as np


def get_polynomial_func(degree=1):
    """Return polynomial function of specified degree."""
    if degree == 3:
        return polynomial3
    elif degree == 4:
        return polynomial4
    elif degree == 5:
        return polynomial5
    elif degree == 6:
        return polynomial6
    elif degree == 7:
        return polynomial7


def polynomial3(x, a, b, c, d):
    return a + b*x + c*(x**2) + d*(x**3)


def polynomial4(x, a, b, c, d, f):
    return a + b*x + c*(x**2) + d*(x**3) + f*(x**4)


def polynomial5(x, a, b, c, d, f, g):
    return a + b*x + c*(x**2) + d*(x**3) + f*(x**4) + g*(x**5)


def polynomial6(x, a, b, c, d, f, g, h):
    return a + b*x + c*(x**2) + d*(x**3) + f*(x**4) + g*(x**5) + h*(x**6)


def polynomial7(x, a, b, c, d, f, g, h, i):
    return a + b*x + c*(x**2) + d*(x**3) + f*(x**4) + g*(x**5) + h*(x**6) + \
           i*(x**7)

# This is a start of a more polished code for handling polynomials -- needs more work
# 
# def polynomial(*args):
#     """First arg -- x, following -- coeffecients for x^0, x^1, x^2, etc."""
#     x = args[0]
#     degree = len(args) - 2  # minus x, minus intercept (a*x^0)
#     coeffs = args[1:]
#     return np.inner([x**d for d in range(degree + 1)], coeffs)
