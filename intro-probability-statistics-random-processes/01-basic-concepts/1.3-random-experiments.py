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
import numpy as np
import itertools

# ## 1.3.1 Random Experiments
#
# **Outcome:** A result of a random experiment.<br>
# **Sample** Space: The set of all possible outcomes.<br>
# **Event:** A subset of the sample space.
#

# ## 1.3.2 Probability

# Axioms of probability:
#
# - Axiom 1: for any event $A$, $P(A) \geq 0$
# - Axiom 2: probability of the sample $S$ is $P(S) = 1$
# - Axiom 3: if $A_1, A_2, A_3 \cdots$ are disjoint events, then $P(A_1 \cup A_2 \cup A_3 \cdots) = P(A_1) + P(A_2) + P(A_3) + \cdots$

# Notation:
#
# $$
# P(A \cap B) = P(A \text{ and } B) = P(A, B)
# \\
# P(A \cup B) = P(A \text{ or } B)
# $$

# ## 1.3.3 Finding Probabilities

# Using the axioms of probability we can prove (see example 1.10 for proofs):
# - for any event $A$, $P(A^c) = 1 - P(A)$
# - the probability of the empty set is zero, i.e. $P(\empty) = 0$
# - for any event $A$, $P(A) \leq 1$
# - $P(A - B) = P(A) - P(A \cap B)$
# - $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (inclusion-exclusion principle for $n=2$)
# - if $A \subset B$ then $P(A) \leq P(B)$
#

# **Example 1.11**
#
# $A$: rain today<br>
# $B$: rain tomorrow<br>
# $P(A^c \cap B^c)$: no rain either day
#
# $$
# P(A) = 0.6
# \\
# P(B) = 0.5
# \\
# P(A^c \cap B^c) = 0.3
# $$

# **a.** The probability that it will rain today or tomorrow:
#
# $$
# \begin{aligned}
# P(A \cup B) &= 1 - P(A \cup B)^c \\
# &= 1 - P(A^c \cap B^c) && \text{by DeMorgan's Law} \\
# &= 1 - 0.3 \\
# &= 0.7
# \end{aligned}
# $$

# **b.** The probability that it will rain today and tomorrow.
#
# $$
# \begin{aligned}
# P(A \cap B) &= P(A) + P(B) - P(A \cup B) \\
# &= 0.6 + 0.5 - 0.7 \\
# &= 0.4 \\
# \end{aligned}
# $$

# **c.** The probability that it will rain today but not tomorrow.
#
# $$
# \begin{aligned}
# P(A \cap B^c) &= P(A - B) \\
# &= P(A) - P(A \cap B) \\
# &= 0.6 - 0.4 \\
# &= 0.2
# \end{aligned}
# $$

# **d.** The probability that it either will rain today or tomorrow, but not both.
#
# $$
# \begin{aligned}
# P(A - B) + P(B - A) &= 0.2 + P(B) - P(B \cap A) \\
# &= 0.2 + 0.5 - 0.4  \\
# &= 0.3
# \end{aligned}
# $$

# **Inclusion-Exclusion Principle:**
#
# $$
# P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) - P(A \cap B \cap C)
# $$

# ## 1.3.4 Discrete Probability Models
#
# If a sample space $S$ is a countable set, this refers to a **discrete** probability model. 
#
# In a countable sample space, to find probability of an event, all we need to do is sum the probability of individual elements in that set.
#
# **Finite Sample Spaces with Equally Likely Outcomes:**
#
# $$
# P(A) = \frac{|A|}{|S|}
# $$
#
# Thus, finding probability of $A$ reduces to a counting problem in which we need to count how many elements are in $A$ and $S$.

# ## 1.3.5 Continuous Probability Models

# ## 1.3.6 Solved Problems: Random Experiments and Probabilities

# ## 4

trials = list(itertools.product([1,2,3,4,5,6], repeat=2))
trials

print(5/12)
len([(x1, x2) for (x1, x2) in trials if x1 < x2])/len(trials)

print(11/36)    
len([(x1, x2) for (x1, x2) in trials if x1 == 6 or x2 == 6])/len(trials)

# ## 5

# +
t = symbols('t')

f = exp(-t/5)
f
# -

f.subs(t, 0)

limit(f, t, S.Infinity)

for t1 in range(1, 10000):
    t2 = t1 + np.random.randint(1, 100)
    if f.subs(t, t2) > f.subs(t, t1):
        print("FAILED: ", t1, t2)

(1 - f.subs(t, 3)).evalf()

(f.subs(t, 1) - f.subs(t, 2)).evalf()

# ## 6


