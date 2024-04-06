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

import itertools
from sympy import *
import matplotlib.pyplot as plt
import numpy as np

# +
import base64
from IPython.display import Image, display

def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))


# -

# # 1.5 End of Chapter Problems

# ## 13

p_a = 0.5
p_d = 0.25

# a
p_b = 1 - (p_a + p_d)
p_b

# b
p_b_d = p_b + p_d
p_b_d

# ## 14

p_a = 0.4
p_b = 0.7
p_a_u_b = 0.9

# a
p_a_i_b = p_a + p_b - p_a_u_b
p_a_i_b

# b
p_ac_i_b = p_b - p_a_i_b
p_ac_i_b

# c
p_a_m_b = p_a - p_a_i_b
p_a_m_b

# d
p_ac_m_b = 1 - p_a_u_b
p_ac_m_b

# e
p_ac_u_b = p_ac_m_b + p_b
p_ac_u_b

# f
# there is no intersection between $a$ and $a^c$
p_a_i_b

# ## 15

# a
1/6

trials = list(itertools.product(range(1, 7), repeat=2))
print(len(trials))
#trials

[(x, y) for (x, y) in trials if x + y == 7]

# b
6 / 36

# c
(5/6) * (3/6)

# ## 17

# p_a = p_b
# p_c = 2 * p_d
# p_a_u_c = p_a + p_c
# p_a + p_b + p_c + p_d = 1
p_a_u_c = 0.6
p_a = 0.2
p_b = 0.2
p_c = 0.4
p_d = 0.2

# 1a - 1b = 0
# 1c - 2d = 0
# 1a + 1c = 0.6
# 1a + 1b + 1c + 1d = 1 
A = np.array([[1, -1, 0, 0], [0, 0, 1, -2], [1, 0, 1, 0], [1, 1, 1, 1]])
b = np.array([0, 0, 0.6, 1])
x = np.linalg.solve(A, b)
x

# ## 18

# a
1/16

# b
1 - (1/16)*2**2

(1/16)*3**2 - (1/16)*1**2

# ## 22

# +
p_coffee = 0.7
p_cake = 0.4
p_both = 0.2

p_coffee_given_cake = p_both / p_cake
p_coffee_given_cake
# -

# # 23

p_a = 0.5
p_b = 0.35
p_c = 0.4

# +
# a
p_a_i_b = 0.2

p_a_given_b = p_a_i_b / p_b
p_a_given_b

# +
# b
p_c_i_b = 0.15

p_c_given_b =  p_c_i_b / p_b
p_c_given_b

# +
# c
p_a_i_c = 0.2
p_a_u_c = p_a + p_c - p_a_i_c
p_b_i_a_u_c = 0.25

p_b_given_a_u_c = p_b_i_a_u_c / p_a_u_c
p_b_given_a_u_c
# -

# d
p_b_i_a_i_c = 0.1
p_b_given_a_i_c = p_b_i_a_i_c / p_a_i_c
p_b_given_a_i_c

# ## 24

# a
3/10

# b
2/5

4/6

# ## 25
#
# 120/600 got As.
# 200/600 lived on campus.
# 80/120 lived off campus given they got an A.
#
# More students lived off campus that got an A.
#
# A. Independent 

# ## 27

# +
# a

mm("""
flowchart TB
    first["`1`"]
    g["`P(G) = 0.8`"]
    eg["`P(G ⋂ E) = 0.08`"]
    ecg["`P(G ⋂ E^c) = 0.72`"]
    gc["`P(G^c) = 0.2`"]
    egc["`P(G^c ⋂ E) = 0.06`"]
    ecgc["`P(G^c ⋂ E^c) = 0.14`"]
    first -->|0.8| g
    first -->|0.2| gc
    g -->|"P(E|G) = 0.1"| eg
    g -->|"P(E^c|G) = 0.9"| ecg
    gc -->|"P(E|G^c) = 0.3"| egc
    gc -->|"P(E^c|G^c) = 0.7"| ecgc
""")

# +
# b

p_e = 0.08 + 0.06
p_e
# -

# c
p_g_ec = 0.72 / (1 - p_e)
p_g_ec

# ## 28

# +
p_a1 = (5/100) * (95/99) * (94/98)
p_a2 = (95/100) * (5/99) * (94/98)
p_a3 = (95/100) * (94/99) * (5/98)

p_a1 + p_a2 + p_a3
# -

# ## 29

# **a**
#
# $$
# P(f) = P_{c1} \cdot P_{c2} \cdot P_{c3}
# $$

# **b**
#
# $$
# P(f) = P_{c1} + P_{c2} + P_{c3} + P_{c1}P_{c2} + P_{c1}P_{c3} + P_{c2}P_{c3} + P_{c1}P_{c2}P_{c3}
# \\
# P(f) = 1 - P_{c1}^cP_{c2}^cP_{c3}^c
# $$

# **c**
#
# $$
# P(C_1, C_2) = P_{C_1} + P_{C_2} + P_{C_1}P_{C_2}
# \\
# P(C_1, C_2) = 1 - P_{C_1}^cP_{C_2}^c
# \\
# P(f) = P(C_1, C_2)P_{C_3}
# \\
# P(f) = (1 - P_{C_1}^cP_{C_2}^c)P_{C_3}
# \\
# P(f) = (1 - (1 - P_{C_1})(1 - P_{C_2}))P_{C_3}
# $$

# **d**
#
# $$
# P(C_1, C_2) = P_{C_1}P_{C_2}
# \\
# P(f) = P(C_1, C_2) + P_{C_3} + P(C_1, C_2)P_{C_3}
# \\
# P(f) = (1 - P(C_1, C_2)^c)(1 - P_{C_3}^c)
# \\
# P(f) = 1 - (1 - P(C_1, C_2))(1 - P_{C_3})
# $$

# **e**
#
# $$
# P(C_1, C_2) = P_{C_1}P_{C_2}
# \\
# P(C_3, C_4) = P_{C_3}P_{C_4}
# \\
# P(C_1, C_2, C_3, C_4) = (1 - P_{C_1}P_{C_2}^c)(1 - P_{C_3}P_{C_4}^c)
# \\
# P(C_1, C_2, C_3, C_4) = 1 - (1 - P_{C_1}P_{C_2})(1 - P_{C_3}P_{C_4})
# \\
# P(f) = P(C_1, C_2, C_3, C_4)P_{C_5}
# \\
# P(f) = (1 - (1 - P_{C_1}P_{C_2})(1 - P_{C_3}P_{C_4}))P_{C_5}
# $$

# ## 31

# +
p_s = 0.5
p_r_g_s = 0.01
p_r_g_sc = 0.00001
p_r = p_r_g_s*p_s + p_r_g_sc*(1-p_s)

p_s_g_r = (p_r_g_s * p_s) / p_r
p_s_g_r
# -

# ## 33
#
# It is to my advantage, given where I chose the first door, there was a $1/3$ probability of getting the car, but if I switch I have a $2/3$ chance of getting it right.
#
# TODO: do the long calculation myself, to practice, calculate the probabilities of the different scenarios and then use Bayes to calculate the final probability to determine whether it is advantageous to switch or not.

# ## 34
#
# TODO: verify
#
# **a.**
# P(A) = 1/6
# P(B) = |{1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}|/36 = 6/36 = 1/6
# P(A, B) = 1/36
# P(A)P(B) = 1/36
#
# Since P(A)P(B) = P(A, B), the events are independent.
#
# **b.**
# P(C) = 1/6
# P(A, C) = 1/36
# P(A)P(B) = 1/36
#
# Similar to the above, P(A)P(C) = P(A, C) = 1/36, so the events are independent.
#
# **c.**
#
# Similar to the above, P(B)P(C) = P(B, C) = 1/36, so the events are independent.
#
# **d.**
#
# P(A)P(B)P(C) = 216
# P(A, B, C) = 0
#
# Since P(A)(B)P(C) != P(A,B, C), so the events are dependent.
