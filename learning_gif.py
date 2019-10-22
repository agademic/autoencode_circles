#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:38:58 2019

@author: a.gogohia
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

fig = plt.figure()

ims = []
for i in range(100):
    im = plt.imshow(predictions.predhis[i][0].reshape(res, res))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=500)

writer = PillowWriter(fps=2)
ani.save("demo2.gif", writer=writer)

plt.show()