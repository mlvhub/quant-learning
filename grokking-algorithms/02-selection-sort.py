# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Selection sort

# %%
def find_smallest(elems):
    smallest = elems[0]
    index = 0
    for i in range(1, len(elems)):
        if elems[i] < smallest:
            smallest = elems[i]
            index = i
    return index


# %%
elems = [3,5,6,3,2,4,5,8,3]
find_smallest(elems)


# %%
def selection_sort(elems):
    arr = []

    for _ in range(0, len(elems)):
        smallest = find_smallest(elems)
        arr.append(elems.pop(smallest))

    return arr


# %%
selection_sort(elems)
