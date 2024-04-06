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

# # Quicksort

def split_farm(width, height):
    if width == 0 or height == 0:
        return max(width, height)

    square = min(width, height)
    if height > width:
        num_squares = height // width
        new_height = height - (square * num_squares)
        return split_farm(width, new_height)
    else:
        num_squares = width // height
        new_width = width - (square * num_squares)
        return split_farm(new_width, height)


# +
width = 1680
height = 640

split_farm(width, height)


# -

# ChatGPT
def split_farm(width, height):
    if height == 0:
        return width
    elif width == 0:
        return height
    else:
        return split_farm(height, width % height)


# +
width = 1680
height = 640

split_farm(width, height)


# -

# ## Exercises

# 4.1
def sum_numbers(numbers):   
    if len(numbers) == 0:
        return 0
    return numbers[0] + sum(numbers[1:])


sum_numbers(list(range(1, 4)))


# 4.2
def count(numbers):   
    if len(numbers) == 0:
        return 0
    return 1 + count(numbers[1:])


count(list(range(1, 4)))


# 4.3
def maximum(numbers, max_number=None):   
    if len(numbers) == 0:
        return max_number
    
    if max_number is None:
        max_number = numbers[0]
    
    if numbers[0] > max_number:
        max_number = numbers[0]
    return maximum(numbers[1:], max_number)


maximum([1,3,5,2,6,4])


# +
# 4.4 already done in 01
# -

# ## Quicksort

def quicksort(array):
    if len(array) <= 1:
        return array
    
    pivot = array[0]
    lt = []
    gt = []
    for x in array[1:]:
        if x > pivot:
            gt.append(x)
        else:
            lt.append(x)
    return quicksort(lt) + [pivot] + quicksort(gt)


quicksort([1,3,5,2,6,4])
