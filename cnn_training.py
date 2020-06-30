#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:14:23 2020

Train a CNN to pick P and S wave arrivals

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle

p_data, _ = pickle.load( open( 'pnsn_P_training_data.pkl', 'rb' ) )
s_data, _ = pickle.load( open( 'pnsn_S_training_data.pkl', 'rb' ) )
n_data, _ = pickle.load( open( 'pnsn_N_training_data.pkl', 'rb' ) )

# some params
plots=0
train=1
ponly=1

# plot the data
if plots:
    # plot ps to check
    plt.figure()
    for ii in range(10):
        plt.plot(p_data[ii,:])
        
    # plot ss to check
    plt.figure()
    for ii in range(10):
        plt.plot(s_data[ii,:])
        
    # plot noise to check
    plt.figure()
    for ii in range(10):
        plt.plot(n_data[ii,:])

# make inputs
if ponly:
    features=np.concatenate((n_data,p_data))
else:
    features=np.concatenate((n_data,p_data,s_data))
    
# make target vector N=0, P=1, S=2
if ponly:
    target=np.concatenate((np.zeros(n_data.shape[0]),600*np.ones(p_data.shape[0])))
else:
    target=np.concatenate((np.zeros(n_data.shape[0]),np.ones(p_data.shape[0]),2*np.ones(s_data.shape[0])))

# make training and testing data
inds=np.arange(target.shape[0])
np.random.shuffle(inds)
train_inds=inds[:int(0.75*len(inds))]
test_inds=inds[int(0.75*len(inds)):]
x_train=features[train_inds,:]
y_train=target[train_inds]
x_test=features[test_inds,:]
y_test=target[test_inds]

# do the shifts and make batches
def my_data_generator(batch_size,dataset,targets):
    while True:
        start_of_batch=np.random.choice(dataset.shape[0]-batch_size)
        # grab batch
        batch=dataset[start_of_batch:start_of_batch+batch_size,:]
        # make target data for batch
        batch_target=np.zeros_like(batch)
        batch_target[:, batch_target.shape[1]//2]=targets[start_of_batch:start_of_batch+batch_size]
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        winsize=15 # winsize in seconds
        sr=40 # sample rate
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        new_batch=np.zeros((batch_size,int(winsize*sr)))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) # keep 4s worth of samples
            new_batch[ii,:]=batch[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        yield(new_batch,new_batch_target)

# generate batch data
my_data=my_data_generator(32,x_train,y_train)
x,y=next(my_data)

# PLOT GENERATOR RESULTS
if plots:
    for ind in range(10):
        fig, ax1 = plt.subplots()
        t=1/40*np.arange(x.shape[1])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude', color='tab:red')
        ax1.plot(t, x[ind,:], color='tab:red') #, label='data')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Prediction', color='tab:blue')  # we already handled the x-label with ax1
        ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
        ax2.tick_params(axis='y')
        ax2.set_ylim((-0.1,2.1))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(('prediction','target'))
        plt.show()

# BUILD THE MODEL
#These models start with an input
input_layer=tf.keras.layers.Input(shape=(600,)) # 1 Channel seismic data

#These Convolutional blocks expect 2D data (time-steps x channels)
#This is just one channel, but if you wanted to add more stations as extra channels you can
network=tf.keras.layers.Reshape((600,1))(input_layer)

# Batch normalization
network=tf.keras.layers.BatchNormalization()(network)

# build the network, here is your first convolution layer
network=tf.keras.layers.Conv1D(32,21,activation='relu')(network) # N filters, Filter Size, Stride, padding

# This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
network=tf.keras.layers.BatchNormalization()(network)

#Max Pooling Layer
network=tf.keras.layers.MaxPooling1D()(network)

#Next Block
network=tf.keras.layers.Conv1D(64,15,activation='relu')(network)
network=tf.keras.layers.BatchNormalization()(network)
network=tf.keras.layers.MaxPooling1D()(network)

#Next Block
network=tf.keras.layers.Conv1D(128,11,activation='relu')(network)
network=tf.keras.layers.BatchNormalization()(network)
network=tf.keras.layers.MaxPooling1D()(network)

#Dense end of network
network=tf.keras.layers.Flatten()(network)
network=tf.keras.layers.Dense(512,activation='relu')(network)
network=tf.keras.layers.BatchNormalization()(network)

network=tf.keras.layers.Dense(512,activation='relu')(network)
network=tf.keras.layers.BatchNormalization()(network)
output=tf.keras.layers.Dense(600, activation='sigmoid')(network)

model=tf.keras.models.Model(input_layer,output)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

# TRAIN THE MODEL
if train:
    batch_size=480
    history=model.fit_generator(my_data_generator(batch_size,x_train,y_train),
                        steps_per_epoch=len(x_train)//batch_size,
                        validation_data=my_data_generator(batch_size,x_test,y_test),
                        validation_steps=len(x_test)//batch_size,
                        epochs=40)
    model.save_weights("pick.tf")
else:
    model.load_weights("pick.tf")

# See how things went
my_test_data=my_data_generator(50,x_test,y_test)
x,y=next(my_test_data)

test_predictions=model.predict(x)

# PLOT A FEW EXAMPLES
for ind in range(10):
    fig, ax1 = plt.subplots()
    t=1/40*np.arange(x.shape[1])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.plot(t, x[ind,:], color='tab:red') #, label='data')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Prediction')  # we already handled the x-label with ax1
    ax2.plot(t, test_predictions[ind,:], color='tab:blue') #, label='prediction')
    ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
    ax2.tick_params(axis='y')
    ax2.set_ylim((-0.1,2.1))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(('prediction','target'))
    plt.show()
    
# PLOT PREDICTIONS
plt.figure()
for ind in range(10):
    plt.plot(t,test_predictions[ind,:])