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

# ## Binary Search

numbers = list(range(0, 1000))


# version with recursion
def binary_search(numbers, n, start=None, end=None):
    if start is not None and start > end:
        return None

    if start is None and end is None:
        start = 0
        end = len(numbers) - 1

    mid = (start + end) // 2
    mid_value = numbers[mid]

    if mid_value == n:
        return mid
    elif mid_value < n:
        return binary_search(numbers, n, start=mid + 1, end=end)
    else:
        return binary_search(numbers, n, start=start, end=mid - 1)


binary_search(numbers, 100)

# +
import numpy as np

results = [binary_search(numbers, n) == n for n in range(0, 1000)]
np.alltrue(results)
# -

# ### Exercises

# #### 1.1

# +
from sympy import log

log(128, 2)
# -

# #### 1.2

log(256, 2)

# ## Running time
#
# `n`: linear time.
#
# Binary search runs in logarithmic time.

# ## Big O Notation
#
# Tells you the maximum number of operations an algorithm will make. It tells the worst case scenario.
#
# Common Big O Notations:
#
# $O(n)$: linear time, e.g. simple search.
#
# $O(log \space n)$: logarithmic time, e.g. binary search.
#
# $O(n * log \space n)$: a fast sorting algorithm, e.g. quicksort.
#
# $O(n^2)$: quadratic time, a slow sorting algorithm, e.g. selection sort.
#
# $O(n!)$: factorial time, a really slow algorithm, e.g. traveling salesperson.
#
