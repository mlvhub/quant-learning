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

def birthday_prob(n_people):
    prob = 1
    for i in range(1, n_people + 1):
        prob *= (366 - i)/365
    return prob


birthday_prob(12)

1 - birthday_prob(12)


