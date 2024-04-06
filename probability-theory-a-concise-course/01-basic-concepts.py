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

# # 1. Basic Concepts

# ## 1.1 Probability and Relative Frequency
#
# The event of tossing an unbiased coin is is said to be "equiprobable", that is its outcomes are equally likely to happen.
#
# The probably of getting heads or getting tails is the same, $\frac{1}{2}$.
#
# But what really is *probability*?
#
# Let $A$ denote some event associated with the possible outcomes of the experiment. Then the *probability* $P(A)$ of the event $A$ is defined as the fraction of outcomes in which $A$ occurs:
#
# $$
# P(A) = \frac{N(A)}{N}
# \tag{1.1}
# $$
#
# Where $N$ is the **total** number of outcomes of the experiment and $N(A)$ is the number of outcomes leading to the occurrence of the event $A$.
#

# **Example 1: well-balanced coin toss**
#
# $$
# A = \text{Getting heads or tails}
# \\
# N = 2 \space \text{(All outcomes, heads AND tails)}
# \\
# N(A) = 1 \space \text{(Heads OR tails)}
# \\
# P(A) = \frac{1}{2}
# $$

# **Example 2: a single unbiased die**
#
# $$
# A = \text{An even number of spots}
# \\
# N = 6 \space \text{(All the die's faces)}
# \\
# N(A) = 3 \space \text{Faces with even number of spots}
# \\
# P(A) = \frac{3}{6} = \frac{1}{2}
# $$

# **Example 3: throwing a pair of dice**
#
# $$
# A = \text{Both dice show the same number of spots}
# \\
# N = 6 * 6 = 36 \space \text{(The combination of both dice's faces)}
# \\
# N(A) = 6 \space \text{The combination of both dice's having the same spots}
# \\
# P(A) = \frac{6}{36} = \frac{1}{6}
# $$

# Relative frequency of event $A$ (in a given series of trials):
#
# $$
# \frac{n(A)}{n}
# $$
#
# Where:
#
# $$
# n: \text{total number of experiments}
# \\
# n(A): \text{number of experiments in which }A\text{ occurs}
# $$

# Probability of event A:
#
# $$
# P(A) = \lim\limits_{n\rightarrow\infty} \frac{n(A)}{n}
# \tag{1.3}
# $$
#
# Meaning, for a large series of trials (large $n$) the probability of event A is the same as the fraction of experiments leading to the occurence of $A$.

# + [markdown] vscode={"languageId": "plaintext"}
# ## 1.2 Rudiments of Combinatorial Analysis 
#
# -

# **Theorem 1.1**
#
# Given $n_1$ elements $a_1, a_2, \dots, a_{n_1}$ and $n_2$ elements $b_1, b_2, \dots, b_{n_2}$, there are precisely $n_1n_2$ distinct ordered pairs $(a_i, b_j)$ containing one element of each kind.

# +
import numpy as np

n1 = 3
n2 = 6

a = np.random.randn(n1)
b = np.random.randn(n2)

pairs = []
for a_i in a:
    for b_j in b:
        pairs.append((a_i, b_j))

len(pairs) == n1*n2
# -

# **Theorem 1.2**
#
# More generally, given $n_1$ elements $a_1, a_2, \dots, a_{n_1}$, $n_2$ elements $b_1, b_2, \dots, b_{n_2}$, up to $n_r$ elements $x_1, x_2, \dots, x_{n_r}$, there are precisely $n_1 n_2 \dots n_r$ distinct ordered r-tuples $(a_i, b_j, \dots, x_{i_r})$ containing one element of each kind.

# **Example 1: what is the probability of getting three sixes in a throw of three dice?**
#
# $$
# A = \text{Three sixes}
# \\
# N = 6 * 6 * 6 = 216 \space \text{(The combination of all the dice's faces)}
# \\
# N(A) = 1 \space \text{(A combination of three sixes)}
# \\
# P(A) = \frac{1}{216} = 0.462962962\%
# $$

# **Example 2: sampling with replacement**
#
# Suppose we choose $r$ objects in succession from a "population" (i.e. "set") of $n$ distinct objects $a_1, a_2, \dots, a_n$, in such a way that after choosing each object and recording the choice, we return the object to the population before making the next choice.
#
# This gives us an ordered sample of the form:
#
# $$
# (a_{i_1}, a_{i_2}, \dots, a_{i_r})
# \tag{1.4}
# $$
#
# Setting $n_1 = n_2 = \dots = n_r = n$ in Theorem 1.2 we find that there are precisely
#
# $$
# N = n^r
# \tag{1.5}
# $$
#
# distinct ordered samples of the form:
#
# $$
# (a_{i_1}, a_{i_2}, \dots, a_{i_r})
# $$

# **Example 3: sampling without replacement**
#
# Next, suppose we choose $r$ objects in succession from a population of $n$ distinct objects $a_1, a_2, \dots, a_n$, in such a way that an object once chosen is removed from the population.
#
# Then, we again get an ordered sample of the form
#
# $$
# (a_{i_1}, a_{i_2}, \dots, a_{i_r})
# $$
#
# but now there are $n - 1$ objects left after the first choice, $n - 2$ objects left after the second choice, and so on. Clearly this corresponds to setting:
#
# $$
# n_1 = n, n_2 = n - 1, \dots, n_r = n - r + 1
# $$
#
# in Theorem 1.2.
#
# Hence, instead of $n^r$ distinct samples as in the case of sampling with replacement, there are now only
#
# $$
# N = n(n - 1) \cdots (n - r + 1)
# \tag{1.6}
# $$
#
# distinct samples.
#
# If $r = n$ then the above reduces to
#
# $$
# N = n(n - 1) \cdots 2 \cdot 1 = n!
# \tag{1.7}
# $$
#
# being the total number of permutations of $n$ objects.

# **Example 4**
#
# [See page 6]

# **Example 5**
#
# A subway train made up of $n$ cars is boarded by $r$ passengers $(r \leq n)$, each entering a car completely at random. 
#
# What is the probability of the passengers all ending up in different cars?
#
# ---
# Every car has the same probability of being entered by a passenger.
#
# Every choice for a passenger includes all cars (sampling with replacement), therefore:
#
# $$
# N = n^r
# $$
#
# However, for $A$ to happen, every passenger loses the option of entering the previous passenger's car (sampling without replacement), therefore:
#
# $$
# N(A) = n(n - 1) \cdots (n - 1 + 1)
# $$
#
# Therefore,the probability of $A$ occurring is given by:
#
# $$
# P(A) = \frac{n(n - 1) \cdots (n - 1 + 1)}{n^r}
# $$

# **Theorem 1.3**
#
# A population of $n$ elements has precisely
#
# $$
# C_r^n = \frac{n!}{r!(n - r)!}
# \tag{1.8}
# $$
#
# subpopulations of size $r \leq n$.
#
# > An expression of the form (1.8) is called a *binomial coefficient*, often denoted by $\begin{pmatrix}n\\r\end{pmatrix}$ instead of $C_r^n$.
#
# The number $C_r^n$ is sometimes called the *number of combinations of $n$ things taken $r$ at a time* (without regard for order).

# **Theorem 1.4**
#
# A generalisation of Theorem 1.3:
#
# Given a population of $n$ elements, let $n_1, n_2, \cdots, n_k$ be positive integers such that
#
# $$
# n_1 + n_2 + \cdots + n_k = n
# $$
#
# Then there are precisely
#
# $$
# N = \frac{n!}{n_1! \space n_2! \cdots n_k!}
# \tag{1.9}
# $$
#
# ways of partitioning the population into $k$ subpopulations, of sizes $n_1, n_2, \cdots, n_k$, respectively.
#
#

# **Example 6: quality control**
#
# A batch of 100 manufactured items is checked by an inspector, who examines 10 items selected at random. If none of the 10 items is defective, he accepts the whole batch. Otherwise,  the batch is subjected to further inspection.
#
# What is the probability that a batch containing 10 defective items will be accepted?
#
# ---
#
# The number of ways of selecting 10 items out of a batch of 100 items equals the number of combinations of 100 things taken 10 at a time, which is given by (1.8):
#
# $$
# N = C_{10}^{100} = \frac{100!}{10! \space 90!}
# $$
#
# By hypothesis, these combinations are all equiprobable.
#
# Let $A$ be the event that "the batch of items accepted by the inspector". Then $A$ occurs whenever all 10 items belong to the set of 90 items of acceptable quality. Hence the number of combinations favorable to $A$ is
#
# $$
# N(A) = C_{10}^{90} = \frac{90!}{10! \space 80!}
# $$
#
# Then, from (1.1) the probability of event $A$, i.e. a batch containing 10 defective items will be accepted, equals
#
# $$
# P(A) = \frac{N(A)}{N} = \frac{90!}{80!} \frac{90!}{100!} = \frac{81 \cdot 82 \cdots 90}{91 \cdot 92 \cdots 100} \approx (1 - \frac{1}{10})^{10} \approx \frac{1}{e}
# $$

# **Example 7**
#
# What is the probability that two playing cards picked at random from a full deck are both aces?
#
# ---
# My solution:
#
# $$
# N = C_2^{52} = \frac{52!}{2! \space 50!}
# $$
#
# $$
# N(A) = C_2^{4} = \frac{4!}{2! \space 2!}
# $$
#
# $$
# P(A) = \frac{N(A)}{N} = \frac{\frac{4!}{2! \space 2!}}{\frac{52!}{2! \space 50!}} = \frac{4! \space 2! \space 50!}{2! \space 2! \space 52!} = \frac{1}{221} \approx 0.452\%
# $$

# **Example 8**
#
# What is the probability that each of four bridge players holds an ace?
#
# ---
# Book solution:
#
# (In bridge players are dealt 13 cards)
#
# Applying Theorem 1.4 with $n = 52$ and $n_1 = n_2 = n_3 = n_4 = 13$, we find that there are 
#
# $$
# \frac{52!}{13! \space 13! \space 13! \space 13!}
# $$
#
# distinct deals of bridge.
#
# There are $4! = 24$ ways of giving an ace to each player, and then the remaining 48 cards can be dealt out in
#
# $$
# \frac{48!}{12! \space 12! \space 12! \space 12!}
# $$
#
# distinct ways. Hence, there are
#
# $$
# 24 \frac{48!}{(12!)^4}
# $$
#
# distinct deals of bridge such that each player receives an ace.
#
# Therefore, the probability of each player receing an ace is just
#
# $$
# 24 \frac{48!}{(12!)^4} \frac{(13!)^4}{52!} = \frac{24(13)^4}{52 \cdot 51 \cdot 50 \cdot 49} \approx 0.105
# $$

# **Stirling's formula**
#
# Most of the above formulas contain the quantity
#
# $$
# n! = n(n-1) \cdots 2 \cdot 1
# $$
#
# called $n$ *factorial*. For large $n$, it can be shown that
#
# $$
# n \backsim \sqrt{2 \pi n} n^n e^{-n}
# $$
#
# Where $\backsim$ between two symbols $\alpha_n$ and $\beta_n$ mean that the ratio $\alpha_n/\beta_n \rightarrow 1$ as $n \rightarrow \infty$.

# ## Problems

# 1. A four-volume work is placed in random order on a bookshelf. What is the probability of the volumes being in proper order left to right or from right to left?
#
# Sampling without replacement, from (1.7)
#
# $$
# C_r^n = \frac{n!}{r!(n - r)!}
# $$

# Solution:
#
# $$
# n = 4
# \\
# N = n! = 4! = 24
# \\
# N(A) = 2 \text{(We want either (1, 2, 3, 4) or (4, 3, 2, 1))}
# \\
# P(A) = \frac{2}{24} = \frac{1}{12} \approx 0.0833 \approx 8.33\%
# $$

# 2. A wooden cube with painted faces is sawed up into 1000 little cubes, all of the same size. The little cubes are then mixed up, and one is chosen at random.
#
# What is the probability of it having just 2 painted faces?
#
# *Ans. 0.096*

# Solution:
#
# 1000 cubes (volume) = $\sqrt[3]{1000} = 10$ cubes in a "row" = $10^2 = 100$ cubes per face.
#
# Cubes with two painted faces $= (8 * 4)+(8 * 3)+(8 * 3)+(8 * 2) = 96$
#
# $$
# P(A) = \frac{96}{1000} = 0.096 = 9.6\%
# $$

# 3. A batch of $n$ manufactured items contains $k$ defective items. Suppose $m$ items are selected at random from the batch. 
#
# What is the probability that $l$ of these items are defective?

# Solution:
#
# $$
# N = C_{m}^{n} = \frac{n!}{m! \space (n-m)!}
# $$
#
# $$
# N(A) = C_{l}^{k} = \frac{k!}{l! \space (k-l)!}
# $$
#
# $$
# P(A) = \frac{C_{l}^{k}}{C_{m}^{n}}
# $$

# 4. Ten books are placed in random order on a bookshelf. Find the probability of three given books being side by side.
#
# *Ans. $\frac{1}{15}$*

# Solution:
#
# $3! = 6$ combinations for (1, 2, 3) being side by side, which can appear in 8 different places in a set of 10 elements. The other 7 sevens books have $7!$ possible combinations.
#
# $$
# N = n! = 10!
# \\
# N(A) = 6 * 8 * 7!
# \\
# P(A) = \frac{6 * 8 * 7!}{10!} = \frac{1}{15}
# $$

#
