# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Coding Exercises

# ## Exercise 3: Create a Rectangle Class

# ![image.png](attachment:image.png)

# ### The Rectangle Class live in action

rec = Rectangle(a = 3, b = 4)

rec

rec.

rec.a

rec._a

rec._b

rec.area

rec.calc_area()

rec.calc_diagonal()

rec.calc_perimeter()

rec.set_parameters(b = 5)

rec

rec.area

rec.calc_perimeter()



# ### Option 1: Self_guided -> Create the Rectangle Class!









# ### Option 2: Some Help

# ## STOP HERE, IF YOU WANT TO DO THE EXERCISE WITHOUT HINTS!

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



import numpy as np


# +
# complete!

class Rectangle():
    
    def __init__(self, a, b):
        self._a = a
        self._b = b
        self.calc_area()
        
    def __repr__(self):
        
        
    def calc_area(self):
        
    
    def calc_perimeter(self):
        
    
    def calc_diagonal(self):
          
    
    def set_parameters(self, a = None, b = None):
        
# -







# # Solutions (Stop here if you want to code on your own!)

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



import numpy as np


class Rectangle():
    
    def __init__(self, a, b):
        self._a = a
        self._b = b
        self.calc_area()
    
    def __repr__(self):
        return "Rectangle with a = {} and b = {}.".format(self._a, self._b)
        
    def calc_area(self):
        self.area = self._a * self._b
        return self.area
    
    def calc_perimeter(self):
        return 2 * self._a + 2 * self._b
    
    def calc_diagonal(self):
        return np.sqrt(self._a**2 + self._b ** 2)  
    
    def set_parameters(self, a = None, b = None):
        if a is not None:
            self._a = a
            self.calc_area()
        if b is not None:
            self._b = b
            self.calc_area()


