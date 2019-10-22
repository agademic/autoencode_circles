#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:44:02 2019

@author: a.gogohia
"""

import numpy as np
import random
from skimage import draw

def create_circles(amount=100, res=30, stroke=3, rad=(3,8)):
    """ A function to create circles as numpy arrays and return them as flattened
    collection of arrays for the different circles.
    Variables:
        amount = int, How many circles to generate, default 100
        res = int, resolution of the arrays (squares), default 30
        stroke = int, Width of the circles, default 3
        rad = tuple, Range of radius of the circles randomly chosen, default (3, 8)
    """
    arr = np.zeros((res, res))
    stroke = stroke # set the width of the circle
    arr1 = arr
    i = 1
    while i < (amount+1):
        
    # Create an outer and inner circle. Then subtract the inner from the outer.
        arr = np.zeros((res, res))
        radius = random.randint(rad[0],rad[1])
        inner_radius = radius - (stroke // 2) + (stroke % 2) - 1 
        outer_radius = radius + ((stroke + 1) // 2)
        x = random.randint(1+radius,res-radius)
        y = random.randint(1+radius,res-radius)
        ri, ci = draw.circle(x, y, radius=inner_radius, shape=arr.shape)
        ro, co = draw.circle(x, y, radius=outer_radius, shape=arr.shape)
        arr[ro, co] = 1
        arr[ri, ci] = 0
        arr1 = np.append(arr1, arr, axis=0)
        i += 1
    
    
    arr1 = arr1[res:,:]
    
    arr2 = arr1.flatten()
    arr2 = arr2.reshape(int(arr1.shape[0]/res),res*res)
    
    return(arr2)
    
#test = create_circles(amount=10, res=50, rad=(10,25), stroke=8)


#n = 10  
#plt.figure(figsize=(10, 4))
#for i in range(n):
    # display original
#    ax = plt.subplot(2, n, i + 1)
#    plt.imshow(test[i].reshape(100, 100))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()