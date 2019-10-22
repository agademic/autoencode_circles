#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:25:40 2019

@author: a.gogohia
"""

import os
os.chdir('/Users/a.gogohia/Desktop/IAV/Auto Encoder')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from CircleGenArray_func import create_circles
import matplotlib.pyplot as plt
from keras.callbacks import Callback



#Creating a Callback subclass that stores each epoch prediction
class prediction_history(Callback):
    def __init__(self):
        self.predhis = []
    def on_epoch_end(self, epoch, logs={}):
        self.predhis.append(self.model.predict(train_arr))
        
#Calling the subclass
predictions=prediction_history()

res = 50 # give resolution for array
train_arr = create_circles(amount=1000, stroke=5, rad=(5,15), res=res)
test_arr = create_circles(amount=5, stroke=5, rad=(5,15), res=res)

autoencoder = Sequential()
autoencoder.add(Dense(128,  activation='elu', input_shape=(res*res,)))
#autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(5,    activation='linear', name="bottleneck"))
autoencoder.add(Dense(128,  activation='elu'))
#autoencoder.add(Dense(512,  activation='elu'))
autoencoder.add(Dense(res*res,  activation='sigmoid'))
autoencoder.compile(loss='mean_squared_error', optimizer = Adam())

trained_model = autoencoder.fit(train_arr, train_arr, batch_size=32, epochs=200,
                                callbacks=[predictions])
                                

decoded_output = autoencoder.predict(train_arr)        # reconstruction


n = 5  
plt.figure(figsize=(10, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(train_arr[-i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_output[-i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


decoded_test = autoencoder.predict(test_arr)

n = 5  
plt.figure(figsize=(10, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_arr[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_test[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


n = 10
plt.figure(figsize=(10, 5))
for i in range(n):
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(predictions.predhis[i][0].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    