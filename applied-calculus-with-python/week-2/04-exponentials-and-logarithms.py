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

import math

math.exp(1)

# +
# natural log is `log`

math.log(1)
# -

math.log(10)

# to change base
math.log(10, 5) # base 5

math.log10(4.34) # base 10

# ## Graphing with Logarithms

import numpy as np
import matplotlib.pyplot as plt
import math


def create_graph():
    x = np.linspace(0.001,16,2000)
    y_e = np.log(x) # natural log
    y_2 = np.log2(x) # base 2
    y_10 = np.log10(x) # base 10

    plt.plot(x, y_e)
    plt.plot(x, y_10)
    plt.plot(x, y_2)

    plt.legend(['ln', 'log10', 'log2'], loc='lower right')
    plt.xticks(range(math.floor(min(x)), math.ceil(max(x))+1))
    plt.axhline(0, color='black', linewidth='0.5')
    plt.axvline(0, color='black', linewidth='0.5')


create_graph()

# ## Example to find digits

# +
import math

def digit_counter(n):
    return int(math.log10(n)) + 1


# -

digit_counter(23421) == 5


