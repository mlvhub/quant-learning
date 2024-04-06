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

# # Vectors and matrices

# ## 1.1 Column and row vectors. Column form and row form of a matrix.
#
# An $n$-dimensional vector $v \in \reals^n$ is denoted by $v = (v_i)_{i=1:n}$  and has $n$ components $v_i \in \reals$, for $i = 1 : n$.
#
# The vector $v = (v_i)_{i=1:n}$ is a **column vector** of size $n$ if
#
# $$
# \begin{equation}
#     v = \begin{pmatrix}
#         v_1
#         \\
#         v_2
#         \\
#         \vdots
#         \\
#         v_n
#     \end{pmatrix}
#     \tag{1.1}
# \end{equation}
# $$
#
# A column vector is also called an $n \times 1$ vector.

# The vector is a **row vector** of size $n$ if
#
# $$
# \begin{equation}
#     v^t = (v_1 \space v_2 \times v_n)
#     \tag{1.2}
# \end{equation}
# $$
#
# A row vector is also called an $1 \times n$ vector.

# An $m \times n$ matrix $A = (A(j,k))_{j=1:m,k=1:n}$ has $m$ rows and $n$ columns. Rather than using the entry by entry notation above for the matrix $A$, we will use either a column-based notation (more often), or a row-based notation, both being better suited for numerical computations.
#
# The **column form** of the matrix $A$ is
#
# $$
# \begin{equation}
#     A = (a_1 | a_2 | \cdots | a_n) = \text{col}(a_k)_{k=1:n}
#     \tag{1.3}
# \end{equation}
# $$
#
# where $a_k$ is the $k$-th column of $A$, $k = 1 : n$.

# The **row form** of the matrix $A$ is
#
# $$
# \begin{equation}
#     A = \begin{pmatrix}
#         r_1
#         \\
#         --
#         \\
#         r_2
#         \\
#         --
#         \\
#         \vdots
#         \\
#         --
#         \\
#         r_m
#     \end{pmatrix} = \text{row}(r_j)_{j=1:m}
#     \tag{1.4}
# \end{equation}
# $$
#
# where $r_j$ is the $j$-th column of $A$, $j = 1 : m$.

# **Row Vector - Column Vector multiplication:**
#
# Let $v = (v_i)_{i=1:n}$ be a column vector of size $n$, and let $w^t = (w_1 \space w_2 \cdots w_n)$ be a row vector of size $n$. Then,
#
# $$
# \begin{equation}
#
# w^t v = \sum_{i=1}^n w_i v_i
#     \tag{1.5}
# \end{equation}
# $$
#

# **Column Vector - Row multiplication:**
#
# Let $v = (v_j)_{j=i:m}$ be a column vector of size $m$, and let $w^t = (w_1 \space w_2 \cdots w_n)$ be a row vector of size $n$. Then, $vw^t$ is an $m \times n$ matrix with the following entries:
#
# $$
# (vw^t)(j, k) = v_jw_k, \forall \space j=1:m, k=1:n
# \tag{1.6}
# $$

# **Matrix - Column Vector multiplication:**
#
# Let $A = col(a_k)_{k=1:n}$ be an $m \times n$ matrix given by the column form (1.3), and let $v = (v_k)_{k=1:n}$ be a column vector of size $n$ given by (1.1). Then,
#
# $$
# Av = \sum_{k=1}^n v_k a_k
# \tag{1.7}
# $$
#
# The result of the multiplication of column vector $v$ by the matrix $A$ is a column vector $Av$, which is the *linear combination* of the columns of $A$ with coefficients equal to the corresponding entries of $v$.
#
# If $A = row(r_j)_{j=1:m}$ is the row form of $A$, then the $j$-th entry of the $m \times 1$ column vector $Av$ is
#
# $$
# (Av)(j) = r_j v_j, \forall \space 1 \leq j \leq m
# \tag{1.8}
# $$
#
# > Since $r_j$ is a $1 \times n$ row vector and $v$ is a $n \times 1$ column vector, it follows from (1.5) that the multiplication from (1.8) can be performed.

# **Row Vector - Matrix multiplication:**
#
# Let $A = row(r_j)_{j=i}$ be an $m \times n$ matrix given by the row form (1.4), and let $w^t = (w_1 \space w_2 \cdots w_m)$ be a row vector of size $m$. Then,
#
# $$
# w^tA = \sum_{j=1}^m w_j r_j 
# \tag{1.9}
# $$
#
# The result of multiplication of the row vector $w^t$ by the matrix $A$ (from the right) is a row vector $w^t A$ which is the linear combination of the rows of $A$ with coefficients equal to the corresponding entries of $w^t$.
#
# If $A = col(a_k)_{k=1:n}$ is the column form of $A$, then the $k$-th entry of the $1 \times n$ row vector $w^t A$ is
#
# $$
# (w^t A)(k) = w^t a_k, \forall \space 1 \leq k \leq n
# \tag{1.10}
# $$

# **Matrix - Matrix multiplication:**
#
# (i) Let $A$ be an $m \times n$ matrix, and let $B$ be an $n \times p$ matrix given by $B = col(b_k)_{k=1:p}$. Then, $AB$ is the $m \times p$ matrix given by
#
# $$
# AB = col(Ab_k)_{k=1:p} = (Ab_1 | Ab_2 | \cdots | Ab_p)
# \tag{1.11}
# $$
#
# The result of multiplying the matrices $A$ and $B$ is a matrix whose columns are the columns of $B$ multiplied by the matrix $A$.
#
# (i) Let $A$ be an $m \times n$ matrix given by $A = row(r_j)_{j=1:m}$, and let $B$ be an $n \times p$ matrix. Then, $AB$ is the $m \times p$ matrix given by
#
# $$
# AB = row(r_jB)_{j=1:m} = \begin{pmatrix}
#     r_1B \\
#     -- \\
#     r_2B \\
#     -- \\
#     \vdots \\
#     -- \\
#     r_mB \\
# \end{pmatrix}
# \tag{1.12}
# $$
#
# The result of multiplying the matrices $A$ and $B$ is a matrix whose rows are the rows of $A$ multiplied by the matrix $B$.
#
# (iii) Let $A$ be an $m \times b$ matrix given $A = row(r_j)_{j=1:m}$, and let $B$ be an $n \times p$ matrix given by $B = col(b_k)_{k=1:p}$. Then, $AB$ is the $m \times p$ matrix whose entries are given by
#
# $$
# (AB)(j,k) = r_jb_k, \forall \space j=1:m, k = 1:p
# \tag{1.13}
# $$
#

# **Matrix - Matrix - Matrix multiplication:**
#
# Let $A$ be an $m \times n$ matrix given by $A = row(r_j)_{j=i:m}$, let $B$ be an $n \times p$ matrix, and let $C$ be a $p \times l$ matrix given by $C = col(c_k)_{k=1:l}$. Then, $ABC$ is the $m \times l$ matrix whose entries are given by
#
# $$
# (ABC)(j,k) = r_j B_{c_k}, \forall \space j=1:m,k=1:l
# \tag{1.14}
# $$
#
# > Matrix multiplication is associative, i.e., $ABC$ = (AB)C = A(BC).

# **Definition 1.1.**
#
# The transpose of an $n \times 1$ column vector $v = (v_i)_{i=1:n}$ is the $1 \times n$ row vector $v^t = (v_1 \space v_2 \dots v_n)$. The transpose of an $1 \times n$ row vector $r = (r_1 \space r_2 \dots r_n)$ is the $n \times 1$ column vector $r^t = (r_i)_{i=1:n}$.
#
# Note that
#
# $$
# (cv)^t = cv^t, \forall \space v \in \reals^n, c \in \reals
# $$

# **Definition 1.2.**
#
# The transpose matrix $A^t$ of an $m \times n$ matrix $A$ is an $n \times m$ matrix given by
#
# $$
# A^t(k, j) = A(j,k), \forall \space k = 1:n, j = 1:m
# \tag{1.16}
# $$
#
# Transposing a column for matrix switches it to row form, and viceversa, as follows:
#
# $$
# A = col(a_k)_{k=1:n} \iff A^t = row(a^t_k)_{k=1:n}
# \tag{1.17}
# $$
#
# $$
# A = row(r_j)_{j=1:m} \iff A^t = col(r^t_j)_{j=1:m}
# \tag{1.18}
# $$
#
# From (1.16), we find that, for any matrix $A$:,
#
# $$
# (A^t)^t = A
# \tag{1.19}
# $$
#
# and, for any matrices $A$ an $B$ of the same size,
#
# $$
# (A + B)^t = A^t + B^t
# \tag{1.20}
# $$

# **Lemma 1.1.**
#
# Let $A$ be an $m \times n$ matrix and let $v$ be a column vector of size $n$. Then,
#
# $$
# (Av)^t = v^tA^t
# \tag{1.21}
# $$
#
# *Proof.* Let $A = col(a_k)_{k=1:n}$ and $v = (v_i)_{i=1:n}$. Then, $Av = \sum_{i=1}^n v_a a_i$, and
#
# $$
# (Av)^t = (\sum_{i=1}^n v_i a_i)^t = \sum_{i=1}^n (v_i a_i)^t = \sum_{i=1}^n v_i a_i^t
# \tag{1.22}
# $$
#
# since $v_i \in \reals$; see (1.15).
#
# Note that $A^t = row(a_k^t)_{k=1:n}$; cf. (1.17). Then, from (1.9), it follows that
#
# $$
# v^t A^t = \sum_{i=1}^v v_i a_i^t
# \tag{1.23}
# $$
#
# From (1.22) and (1.23), we conclude that $(Av)^t = v^tA^t$.

# **Lemma 1.2.**
# Let $A$ be an $m \times n$ matrix and let $B$ be an $n \times p$ matrix. Then,
#
# $$
# (AB)^t = B^tA^t
# \tag{1.24}
# $$
#
# **Proof.** (Page 5)

# **Definition 1.3.**
#
# A matrix with the same number of rows and columns is called a *square matrix*.
#
# > An $n \times n$ matrix is also called a square matrix of size $n$.

# **Definition 1.4.**
#
# A square matrix is symmetric if and only if the matrix and its transpose are the same, i.e. $A = A^t$,
#
# $$
# A(j, k) = A(k, j), \forall 1 \leq j \leq k \leq n;
# $$
#
# > The product of two symmetric matrices is not necessarily a symmetric matrix. See example on page 6.
#
# The *identity matrix*, denoted by $I$, is a square matrix with entries equal to 1 on the main diagonal and equal to 0 elsewhere, i.e.,
#
# $$
# I = \begin{pmatrix}
# 1 & 0 & \dots & 0 \\
# 0 & 1 & \dots & 0 \\
# \vdots & \vdots & \ddots & \vdots \\
# 0 & 0 & \dots & 1 \\
# \end{pmatrix}
# $$
