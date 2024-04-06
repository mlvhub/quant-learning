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

# +
import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt

def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))


# -

# ## 1.4.0 Conditional Probability
#
# Notation:
#
# _What is the probability that it rains **given** that it is cloudy?_
#
# $$
# \begin{aligned}
# R&: \text{rainy} \\
# C&: \text{cloudy} \\
# P&(R|C)
# \end{aligned}
# $$

# ## Example 1.15
#
# $$
# \begin{aligned}
# A &= \{1, 3, 5\} \\
# B &= \{1, 2, 3\} \\
# P(A) &= \frac{1}{3} \\ 
# P(A|B) &= \frac{P(A \cap B)}{|B|} \\ 
# P(A|B) &= \frac{2}{3} \\ 
# \end{aligned}
#
# $$

# ### Example 1.17

trials = list(itertools.product([1,2,3,4,5,6], repeat=2))
len(trials)

# +
s = [(x1, x2) for (x1, x2) in trials if x1 + x2 == 7]
a = [(x1, x2) for (x1, x2) in s if x1 == 4 or x2 == 4]

len(a)/len(s)
# -

(95/100) * (94/99) * (93/98)

# **Chain rule for conditional probability:**
#
# Two events: 
# $$
# \begin{aligned}
# P(A \cap B) &= P(A)P(B|A) \\
# &= P(B)P(A|B)
# \end{aligned}
# $$
#
# $n$ events:
#
# $$
# \begin{aligned}
# P(A_1 \cap A_2 \cap \cdots \cap A_n) &= P(A_1)P(A_2|A_1)P(A_3|A_2,A_1) \cdots P(A_n|A_{n-1}A_{n-2}\cdots A_1)
# \end{aligned}
# $$

# ### Example 1.19
#
# $$
# \begin{aligned}
# P(A_1) &= \frac{95}{100} \\
# P(A_2|A_1) &= \frac{94}{99} \\
# P(A_3|A_1, A_2) &= \frac{93}{98} \\
# P(A_1 \cap A_2 \cap A_3) &= P(A_1)P(A_2|A_1)P(A_3|A_2,A_1) \\
# &= \frac{95}{100}\frac{94}{99}\frac{93}{98} \\
# \end{aligned}
# $$

(95/100)*(94/99)*(93/98)

# ## 1.4.1 Independence
#
# Two events $A$ and $B$ are independent if and only if $P(A \cap B)=P(A)P(B)$. 

# ## 1.4.3 Bayes' Rule

# ### Example 1.25

# +
p_r_b1 = 0.75

p_b1 = 1/3

p_r = 0.6
# -

p_b1_r = (p_r_b1 * p_b1) / p_r
p_b1_r

# ## 1.4.4 Conditional Independence

# ### Example 1.27

p_a_c = 1/2
print(p_a_c)
p_b_c = 1/2
print(p_b_c)
p_a_u_b_c = (1/2) * (1/2)
print(p_a_u_b_c)
p_a = ((1/2) * (1/2)) + (1 * (1/2))
print(p_a)
p_b = ((1/2) * (1/2)) + (1 * (1/2))
print(p_b)
# not sure about this last one
p_a_i_b = 0

# ## 1.4.5 Solved Problems: Conditional Probability

# ### Problem 1

# +
t = symbols('t')

f = exp(-t/5)
f
# -

p_a = (f.subs(t, 2) - f.subs(t, 3)).evalf()
p_a

p_b = f.subs(t, 2).evalf()
p_b

# Since $A \subset B$, we have $A \cap B = A$:

p_a_b = p_a / p_b
p_a_b

# ### Problem 2

# +
# a.

(1/2)**3
# -

# b.
trials = list(itertools.product([1,2], repeat=3))
print(len(trials))
trials

3/len(trials)

# c.
4/(len(trials) - 1)

# ### Problem 3

# See https://www.probabilitycourse.com/chapter1/1_4_5_solved3.php

# ### Problem 4
#
# See https://www.probabilitycourse.com/chapter1/1_4_5_solved3.php

# ### Problem 5

# +
# TODO: move decision tree from handwritten notes to here

mm("""
flowchart LR
    markdown["`This **is** _Markdown_`"]
    newLines["`Line1
    Line 2
    Line 3`"]
    markdown --> newLines
""")
# -

# a.
1/8

# b.
print(11/48)
p_l = (1/12)+(1/24)+(1/24)+(1/16)
p_l

# c.
print(1/8)
p_r_i_l = (1/12)+(1/24)
p_r_i_l

print(6/11)
p_r_l = p_r_i_l / p_l
p_r_l

# ### Problem 6

mm("""
flowchart TB
    first["`1`"]
    rc1["`P(RC1) = 1/3`"]
    rc1h["`P(RC1H) = 1/3 * 1/2 = 1/6`"]
    rc1t["`P(RC1T) = 1/3 * 1/2 = 1/6`"]
    rc2["`P(RC2) = 1/3`"]
    rc2h["`P(RC2H) = 1/3 * 1/2 = 1/6`"]
    rc2t["`P(RC2T) = 1/3 * 1/2 = 1/6`"]
    fc["`P(FC) = 1/3`"]
    fch["`P(FCH) = 1/3 * 1 = 1/3`"]
    fct["`P(RCT) = 1/3 * 0 = 0`"]
    first --> rc1
    first --> rc2
    first --> fc
    rc1 --> rc1h
    rc1 --> rc1t
    rc2 --> rc2h
    rc2 --> rc2t
    fc --> fch
    fc --> fct
""")

# You pick a coin at random and toss it. What is the probability that it lands heads up?
print(2/3)
p_h = (1/6) + (1/6) + (1/3)
p_h

# You pick a coin at random and toss it, and get heads. What is the probability that it is the two-headed coin?
p_h_fc = 1
p_fc_h = (1 * 1/3)/p_h
p_fc_h


